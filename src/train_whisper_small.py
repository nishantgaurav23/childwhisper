"""Whisper-small full fine-tuning script for children's speech.

Loads config from YAML, sets up model with SpecAugment + gradient checkpointing,
trains with Seq2SeqTrainer, evaluates with WER, and optionally pushes to HF Hub.
Designed for Kaggle T4 GPU; testable on MacBook with --dry-run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

try:
    from src.augment import create_augmentation
    from src.dataset import (
        WhisperDataset, WhisperDataCollator, create_train_val_split, stratified_subset,
    )
    from src.evaluate import compute_wer
    from src.preprocess import load_metadata
except ModuleNotFoundError:
    from augment import create_augmentation
    from dataset import (
        WhisperDataset, WhisperDataCollator, create_train_val_split, stratified_subset,
    )
    from evaluate import compute_wer
    from preprocess import load_metadata

logger = logging.getLogger(__name__)


def load_training_config(config_path: str) -> dict:
    """Load training config from YAML, merging common + whisper_small sections."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = {}
    if "common" in raw:
        config.update(raw["common"])
    if "whisper_small" in raw:
        config.update(raw["whisper_small"])
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on children's speech")
    parser.add_argument("--metadata-path", type=str, required=True,
                        help="Path to metadata JSONL file")
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--config", type=str,
                        default="configs/training_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/whisper-small",
                        help="Output directory for checkpoints")
    parser.add_argument("--num-train-epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 training step + 1 eval step for testing")
    parser.add_argument("--no-push-to-hub", action="store_true",
                        help="Disable pushing checkpoints to HF Hub")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Path to MUSAN noise directory for augmentation")
    parser.add_argument("--realclass-dir", type=str, default=None,
                        help="Path to RealClass noise directory for augmentation")
    parser.add_argument("--hub-model-id", type=str, default=None,
                        help="Override HuggingFace Hub model ID")
    parser.add_argument("--subset-fraction", type=float, default=None,
                        help="Override data subset fraction (0.0-1.0, default from config)")
    args = parser.parse_args(argv)
    args.push_to_hub = not args.no_push_to_hub
    return args


def create_augment_fn(
    noise_dir: str | None,
    realclass_dir: str | None,
    config: dict,
):
    """Create augmentation function from CLI args and config.

    Returns None if no noise dirs provided. Raises ValueError if only one provided.
    """
    if noise_dir is None and realclass_dir is None:
        return None
    if (noise_dir is None) != (realclass_dir is None):
        raise ValueError(
            "Both --noise-dir and --realclass-dir must be provided together"
        )
    aug_cfg = config.get("augmentation", {})
    return create_augmentation(
        noise_dir=noise_dir,
        realclass_dir=realclass_dir,
        realclass_min_snr=aug_cfg.get("realclass_min_snr_db", 5.0),
        realclass_max_snr=aug_cfg.get("realclass_max_snr_db", 20.0),
        musan_min_snr=aug_cfg.get("musan_min_snr_db", 0.0),
        musan_max_snr=aug_cfg.get("musan_max_snr_db", 15.0),
    )


def setup_model(config: dict, dry_run: bool = False) -> tuple:
    """Load Whisper-small model and processor with SpecAugment + gradient checkpointing."""
    model_name = config["model_name"]

    #use_fp16 = config.get("fp16", True) and not dry_run
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    processor = WhisperProcessor.from_pretrained(model_name)

    # Enable gradient checkpointing
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Enable SpecAugment
    sa = config.get("spec_augment", {})
    if sa.get("apply", True):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = sa.get("mask_time_prob", 0.05)
        model.config.mask_time_length = sa.get("mask_time_length", 10)
        model.config.mask_feature_prob = sa.get("mask_feature_prob", 0.04)
        model.config.mask_feature_length = sa.get("mask_feature_length", 10)

    # Disable forced_decoder_ids and suppress_tokens via generation_config
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    return model, processor


def setup_training_args(
    config: dict,
    output_dir: str,
    push_to_hub: bool = True,
    dry_run: bool = False,
    num_train_epochs: int | None = None,
) -> Seq2SeqTrainingArguments:
    """Build Seq2SeqTrainingArguments from config dict."""
    epochs = num_train_epochs or config.get("num_train_epochs", 3)

    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 2),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 8),
        "learning_rate": config.get("learning_rate", 1e-5),
        "warmup_steps": config.get("warmup_steps", 500),
        "num_train_epochs": epochs,
        "fp16": config.get("fp16", True),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "eval_strategy": "steps",
        "eval_steps": config.get("eval_steps", 500),
        "save_steps": config.get("save_steps", 500),
        "save_total_limit": config.get("save_total_limit", 3),
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "predict_with_generate": True,
        "generation_max_length": config.get("generation_max_length", 225),
        "logging_steps": config.get("logging_steps", 50),
        "dataloader_num_workers": config.get("dataloader_num_workers", 4),
        "push_to_hub": push_to_hub,
        "remove_unused_columns": False,
    }

    if push_to_hub:
        kwargs["hub_model_id"] = config.get("hub_model_id")
        kwargs["hub_private_repo"] = config.get("hub_private_repo", True)

    if dry_run:
        kwargs["max_steps"] = 1
        kwargs["eval_steps"] = 1
        kwargs["save_steps"] = 1
        kwargs["logging_steps"] = 1
        kwargs["push_to_hub"] = False
        kwargs["fp16"] = False
        kwargs["gradient_checkpointing"] = False
        kwargs["dataloader_num_workers"] = 0

    return Seq2SeqTrainingArguments(**kwargs)


def make_compute_metrics(tokenizer):
    """Create a compute_metrics function for Seq2SeqTrainer.

    Decodes predictions and labels, then computes WER.
    """

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id for decoding
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        label_ids = np.where(label_ids == -100, pad_id, label_ids)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = compute_wer(label_str, pred_str)
        return {"wer": wer}

    return compute_metrics


def build_datasets(
    config: dict,
    metadata_path: str,
    audio_dir: str,
    augment_fn=None,
) -> tuple:
    """Build train and val WhisperDatasets with child_id split and optional subsetting."""
    metadata = load_metadata(metadata_path)

    # Apply stratified subset before train/val split
    subset_cfg = config.get("data_subset", {})
    fraction = subset_cfg.get("fraction", 1.0)
    if fraction < 1.0:
        metadata = stratified_subset(
            metadata,
            fraction=fraction,
            split_by=config.get("validation", {}).get("split_by", "child_id"),
            stratify_by=config.get("validation", {}).get("stratify_by", "age_bucket"),
            seed=subset_cfg.get("seed", 42),
        )
        logger.info("Data subset: %d samples (%.0f%%)", len(metadata), fraction * 100)

    val_cfg = config.get("validation", {})
    train_meta, val_meta = create_train_val_split(
        metadata,
        val_ratio=val_cfg.get("split_ratio", 0.1),
        split_by=val_cfg.get("split_by", "child_id"),
        stratify_by=val_cfg.get("stratify_by", "age_bucket"),
    )

    logger.info("Train: %d utterances, Val: %d utterances", len(train_meta), len(val_meta))

    # Write split metadata to temp files for WhisperDataset
    import json
    import tempfile

    train_meta_fd = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    train_meta_path = Path(train_meta_fd.name)
    train_meta_fd.close()
    val_meta_fd = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    val_meta_path = Path(val_meta_fd.name)
    val_meta_fd.close()

    train_meta_path.write_text("\n".join(json.dumps(e) for e in train_meta))
    val_meta_path.write_text("\n".join(json.dumps(e) for e in val_meta))

    model_name = config.get("model_name", "openai/whisper-small")
    min_dur = config.get("min_duration_sec", 0.3)
    max_dur = config.get("max_duration_sec", 30.0)

    train_ds = WhisperDataset(
        metadata_path=str(train_meta_path),
        audio_dir=audio_dir,
        model_name=model_name,
        augment_fn=augment_fn,
        min_duration=min_dur,
        max_duration=max_dur,
    )
    val_ds = WhisperDataset(
        metadata_path=str(val_meta_path),
        audio_dir=audio_dir,
        model_name=model_name,
        augment_fn=None,  # No augmentation for validation
        min_duration=min_dur,
        max_duration=max_dur,
    )

    return train_ds, val_ds


def main(argv: list[str] | None = None) -> float:
    """Run Whisper-small fine-tuning.

    Returns validation WER.
    """
    logging.basicConfig(level=logging.INFO)

    args = parse_args(argv)
    config = load_training_config(args.config)

    # CLI overrides
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.hub_model_id is not None:
        config["hub_model_id"] = args.hub_model_id
    if args.subset_fraction is not None:
        config.setdefault("data_subset", {})["fraction"] = args.subset_fraction

    # Setup augmentation
    augment_fn = create_augment_fn(
        noise_dir=args.noise_dir,
        realclass_dir=args.realclass_dir,
        config=config,
    )
    if augment_fn is not None:
        logger.info("Augmentation enabled (noise_dir=%s, realclass_dir=%s)",
                     args.noise_dir, args.realclass_dir)

    # Setup
    model, processor = setup_model(config, dry_run=args.dry_run)
    training_args = setup_training_args(
        config,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        dry_run=args.dry_run,
        num_train_epochs=config.get("num_train_epochs"),
    )

    train_ds, val_ds = build_datasets(
        config, args.metadata_path, args.audio_dir, augment_fn=augment_fn
    )

    collator = WhisperDataCollator(pad_token_id=processor.tokenizer.pad_token_id)
    compute_metrics_fn = make_compute_metrics(processor.tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor.feature_extractor,
    )

    # Train
    logger.info("Starting training (dry_run=%s)", args.dry_run)
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    wer = metrics.get("eval_wer", -1.0)
    logger.info("Validation WER: %.4f", wer)

    # Save final model
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    return wer


if __name__ == "__main__":
    main()
