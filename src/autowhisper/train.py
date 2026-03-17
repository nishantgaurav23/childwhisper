"""Mutable training script for AutoWhisper experiment loop.

THIS IS THE ONLY FILE THE AI AGENT MAY MODIFY.
It must print `val_wer: X.XXXX` and `peak_vram_mb: XXXX` to stdout.
Training must complete within TIME_BUDGET seconds.
"""

import os
import sys
import time
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.autowhisper.prepare import (
    EVAL_SAMPLES,
    TIME_BUDGET,
    evaluate_wer,
    load_fast_eval_set,
)
from src.dataset import WhisperDataCollator
from src.preprocess import load_audio

# ── Configuration ──────────────────────────────────────────────────────────────
# The agent modifies these values each experiment.

CONFIG = {
    "model_name": "openai/whisper-small",
    "mode": "full",  # "full" or "lora"
    "learning_rate": 3.0e-5,
    "warmup_steps": 300,
    "max_steps": 500,
    "per_device_batch_size": 4,
    "gradient_accumulation": 8,
    "fp16": True,
    "gradient_checkpointing": True,
    "num_beams": 5,
    "max_new_tokens": 225,
    "spec_augment_mask_time_prob": 0.05,
    "spec_augment_mask_feature_prob": 0.04,
    # LoRA config (used only if mode == "lora")
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
}


def get_peak_vram_mb() -> int:
    """Return peak GPU VRAM usage in MB, or -1 if no GPU."""
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / 1024 / 1024)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return -1  # MPS doesn't track peak memory
    return -1


def main():
    start_time = time.time()

    # ── Data ───────────────────────────────────────────────────────────────────
    data_dir = os.environ.get("AUTOWHISPER_DATA_DIR", "data")
    output_dir = os.environ.get("AUTOWHISPER_OUTPUT_DIR", "/tmp/autowhisper_output")

    eval_set = load_fast_eval_set(data_dir, n_samples=EVAL_SAMPLES)

    # ── Model ──────────────────────────────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(CONFIG["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model_name"])

    if CONFIG["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # SpecAugment
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = CONFIG["spec_augment_mask_time_prob"]
    model.config.mask_feature_prob = CONFIG["spec_augment_mask_feature_prob"]

    # Disable forced decoder IDs
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # LoRA setup
    if CONFIG["mode"] == "lora":
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=CONFIG["lora_r"],
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=CONFIG["lora_target_modules"],
            lora_dropout=CONFIG["lora_dropout"],
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_config)

    # ── Training ───────────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        max_steps=CONFIG["max_steps"],
        fp16=CONFIG["fp16"] and torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=CONFIG["max_new_tokens"],
        report_to="none",
    )

    # Build dataset from eval_set metadata (in a real run, this would be
    # the full training set; for autowhisper we train on a subset for speed)
    collator = WhisperDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    _trainer = Seq2SeqTrainer(  # noqa: F841 — setup for agent modifications
        model=model,
        args=training_args,
        data_collator=collator,
    )

    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > TIME_BUDGET:
        print("val_wer: -1.0")
        print(f"peak_vram_mb: {get_peak_vram_mb()}")
        sys.exit(1)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    model.eval()
    predictions = []
    references = []

    data_path = Path(data_dir)
    with torch.no_grad():
        for item in eval_set:
            audio_path = data_path / item["audio_path"]
            audio, sr = load_audio(audio_path)
            input_features = processor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).input_features

            device = next(model.parameters()).device
            input_features = input_features.to(device)

            predicted_ids = model.generate(
                input_features,
                num_beams=CONFIG["num_beams"],
                max_new_tokens=CONFIG["max_new_tokens"],
            )
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            predictions.append(transcription)
            references.append(item.get("orthographic_text", ""))

    # ── Results ────────────────────────────────────────────────────────────────
    result = evaluate_wer(predictions, references)
    peak_vram = get_peak_vram_mb()

    print(f"val_wer: {result['wer']:.4f}")
    print(f"peak_vram_mb: {peak_vram}")


if __name__ == "__main__":
    main()
