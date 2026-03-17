"""Inference pipeline for children's speech transcription.

Competition submission entrypoint. Two-model ensemble: Whisper-large-v3 + LoRA
(Model A) and Whisper-small fine-tuned (Model B), run sequentially on A100
with confidence-based prediction merging and time budget management.
Falls back to single model when adapter/weights are unavailable.
Runs on A100 (80 GB VRAM), no network, 2-hour limit.
Also works on MacBook (MPS/CPU) for local testing.

Specs: S1.4 (zero-shot), S2.4 (fine-tuned), S3.3 (ensemble), S5.2 (faster inference)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Make src/ importable from submission/
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.preprocess import is_silence, load_audio  # noqa: E402
from src.utils import normalize_and_postprocess  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

# --- Constants ---
DATA_DIR = Path("/code_execution/data")
OUTPUT_DIR = Path("/code_execution/submission")
SAMPLE_RATE = 16000
BATCH_SIZE = 16
DEFAULT_MODEL = "openai/whisper-small"
LARGE_MODEL = "openai/whisper-large-v3"
FINETUNED_WEIGHTS_DIR = Path(__file__).resolve().parent / "model_weights" / "whisper_small_ft"
LORA_ADAPTER_DIR = Path(__file__).resolve().parent / "model_weights" / "lora_large_v3"

# Time budget for 2-hour inference limit
TIME_LIMIT_SEC = 7200
SAFETY_MARGIN_SEC = 300
MODEL_B_CUTOFF_SEC = 5400  # Don't start Model B after 90 minutes


def get_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_optimal_batch_size(device: str, model_size: str = "small") -> int:
    """Return optimal batch size based on device and model size.

    A100 (80GB) can handle much larger batches than T4/MPS.
    """
    batch_sizes = {
        "cuda": {"large": 32, "small": 64},
        "mps": {"large": 16, "small": 16},
        "cpu": {"large": 4, "small": 4},
    }
    device_map = batch_sizes.get(device, {"large": 4, "small": 4})
    return device_map.get(model_size, 16)


def maybe_compile(model, device: str):
    """Optionally compile model with torch.compile on CUDA for faster inference.

    Skips on CPU/MPS where compile is unreliable or unsupported.
    Falls back to original model if compile fails.
    """
    if device != "cuda":
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        logger.warning("torch.compile failed, using uncompiled model")
        return model


def get_beam_config(model_size: str = "small") -> dict:
    """Return beam search configuration based on model size.

    Large model gets num_beams=8 (speed gains from SDPA offset the cost).
    Small model keeps num_beams=5 (diminishing returns).
    """
    if model_size == "large":
        return {"num_beams": 8, "length_penalty": 1.0, "max_new_tokens": 225}
    return {"num_beams": 5, "length_penalty": 1.0, "max_new_tokens": 225}


def resolve_model_path(
    finetuned_path: str | Path,
    default: str = DEFAULT_MODEL,
) -> str:
    """Resolve model path: use fine-tuned weights if valid, otherwise default.

    A valid fine-tuned weights directory must exist and contain a config.json file
    (written by Seq2SeqTrainer.save_model()).
    """
    path = Path(finetuned_path)
    if path.is_dir() and (path / "config.json").exists():
        logger.info("Using fine-tuned weights from %s", path)
        return str(path)
    logger.info("Fine-tuned weights not found at %s, falling back to %s", path, default)
    return default


def load_metadata(data_dir: Path) -> list[dict]:
    """Load utterance_metadata.jsonl from data_dir. Returns list of dicts."""
    meta_path = Path(data_dir) / "utterance_metadata.jsonl"
    text = meta_path.read_text().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n")]


def load_model(
    device: str,
    model_name_or_path: str = DEFAULT_MODEL,
) -> tuple:
    """Load Whisper-small model + processor. Returns (model, processor).

    Args:
        device: Target device (cuda, mps, cpu).
        model_name_or_path: HF Hub model ID or local directory path.
            Defaults to "openai/whisper-small" (zero-shot).
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    # Load processor from same path; fall back to default if not found
    try:
        processor = WhisperProcessor.from_pretrained(model_name_or_path)
    except (OSError, Exception):
        logger.warning(
            "Processor not found at %s, falling back to %s",
            model_name_or_path,
            DEFAULT_MODEL,
        )
        processor = WhisperProcessor.from_pretrained(DEFAULT_MODEL)

    return model, processor


def transcribe_batch(
    model,
    processor,
    audio_arrays: list[np.ndarray],
    device: str,
    num_beams: int = 5,
    length_penalty: float = 1.0,
    max_new_tokens: int = 225,
) -> list[str]:
    """Transcribe a batch of audio arrays. Returns list of raw text predictions."""
    if not audio_arrays:
        return []

    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device)
    if device == "cuda":
        input_features = input_features.to(torch.float16)

    with torch.no_grad():
        generated = model.generate(
            input_features,
            language="en",
            task="transcribe",
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    texts = processor.batch_decode(generated, skip_special_tokens=True)
    return texts


def run_inference(
    model,
    processor,
    utterances: list[dict],
    data_dir: Path,
    device: str,
    batch_size: int = 16,
) -> dict[str, str]:
    """Run inference on all utterances. Returns {utterance_id: normalized_text}."""
    # Sort by duration (longest first) for efficient batching
    sorted_utts = sorted(
        utterances, key=lambda u: u.get("audio_duration_sec", 0), reverse=True
    )

    predictions: dict[str, str] = {}

    for i in range(0, len(sorted_utts), batch_size):
        batch = sorted_utts[i : i + batch_size]
        audio_arrays = []
        batch_ids = []
        silent_ids = []

        for u in batch:
            uid = u["utterance_id"]
            audio_path = Path(data_dir) / u["audio_path"]
            try:
                audio, _ = load_audio(audio_path, target_sr=SAMPLE_RATE)
            except Exception:
                predictions[uid] = ""
                continue

            if is_silence(audio):
                predictions[uid] = ""
                silent_ids.append(uid)
                continue

            audio_arrays.append(audio)
            batch_ids.append(uid)

        if audio_arrays:
            texts = transcribe_batch(model, processor, audio_arrays, device)
            for uid, text in zip(batch_ids, texts):
                predictions[uid] = normalize_and_postprocess(text)

    return predictions


def write_submission(
    predictions: dict[str, str], utterances: list[dict], output_dir: Path
) -> Path:
    """Write submission.jsonl. Returns path to output file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.jsonl"

    with open(output_path, "w") as f:
        for u in utterances:
            uid = u["utterance_id"]
            line = {
                "utterance_id": uid,
                "orthographic_text": predictions.get(uid, ""),
            }
            f.write(json.dumps(line) + "\n")

    return output_path


def load_large_model(
    device: str,
    base_model: str = LARGE_MODEL,
    adapter_path: str | Path = LORA_ADAPTER_DIR,
) -> tuple:
    """Load Whisper-large-v3 base model + LoRA adapter. Returns (model, processor)."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    processor = WhisperProcessor.from_pretrained(base_model)
    return model, processor


def merge_predictions(
    preds_a: dict[str, str],
    preds_b: dict[str, str] | None,
) -> dict[str, str]:
    """Merge predictions from two models.

    Prefer Model A predictions. Fall back to Model B when A is empty/whitespace.
    If preds_b is None (Model B didn't run), return preds_a as-is.
    """
    if preds_b is None:
        return dict(preds_a)

    merged: dict[str, str] = {}
    for uid, text_a in preds_a.items():
        if text_a.strip():
            merged[uid] = text_a
        else:
            text_b = preds_b.get(uid, "")
            merged[uid] = text_b
    return merged


def check_time_budget(
    elapsed_sec: float,
    cutoff_sec: float = MODEL_B_CUTOFF_SEC,
) -> bool:
    """Return True if there's enough time to run Model B."""
    return elapsed_sec < cutoff_sec


def run_ensemble_inference(
    utterances: list[dict],
    data_dir: Path,
    device: str,
    adapter_path: str | Path = LORA_ADAPTER_DIR,
    small_model_path: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> dict[str, str]:
    """Run two-model ensemble inference with time budget management.

    1. Load and run Model A (large-v3 + LoRA)
    2. If time allows, load and run Model B (small fine-tuned)
    3. Merge predictions (prefer A, fall back to B on empty)

    Falls back to small-only if large model / adapter not available.
    """
    t0 = time.time()
    preds_a = None
    preds_b = None

    # Try Model A (large-v3 + LoRA)
    try:
        logger.info("Loading Model A (Whisper-large-v3 + LoRA)...")
        model_a, proc_a = load_large_model(device, adapter_path=adapter_path)
        logger.info("Model A loaded in %.1fs", time.time() - t0)

        preds_a = run_inference(model_a, proc_a, utterances, data_dir, device, batch_size)
        logger.info("Model A inference done in %.1fs", time.time() - t0)

        # Free VRAM before loading Model B
        del model_a, proc_a
        if device == "cuda":
            torch.cuda.empty_cache()

    except (FileNotFoundError, OSError) as exc:
        logger.warning("Model A unavailable (%s), falling back to small model only", exc)

    elapsed = time.time() - t0

    # Model B: run if Model A failed OR time budget allows for ensemble
    if preds_a is None or check_time_budget(elapsed):
        try:
            logger.info("Loading Model B (Whisper-small fine-tuned)...")
            model_b, proc_b = load_model(device, model_name_or_path=small_model_path)
            preds_b = run_inference(
                model_b, proc_b, utterances, data_dir, device, batch_size
            )
            logger.info("Model B inference done in %.1fs", time.time() - t0)
            del model_b, proc_b
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Model B failed: %s", exc)

    # Merge or use whichever is available
    if preds_a is not None:
        return merge_predictions(preds_a, preds_b)
    if preds_b is not None:
        return preds_b
    # Both failed — return empty predictions
    return {u["utterance_id"]: "" for u in utterances}


def main():
    """Entrypoint: load data, run ensemble inference, write output."""
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()

    device = get_device()
    print(f"Device: {device}")

    # Load metadata
    utterances = load_metadata(DATA_DIR)
    print(f"Loaded {len(utterances)} utterances")

    # Resolve small model path
    small_model_path = resolve_model_path(FINETUNED_WEIGHTS_DIR)
    print(f"Small model source: {small_model_path}")

    # Run ensemble inference
    predictions = run_ensemble_inference(
        utterances=utterances,
        data_dir=DATA_DIR,
        device=device,
        adapter_path=LORA_ADAPTER_DIR,
        small_model_path=small_model_path,
        batch_size=BATCH_SIZE,
    )
    print(f"Ensemble inference done in {time.time() - t0:.1f}s")

    # Write output
    output_path = write_submission(predictions, utterances, OUTPUT_DIR)
    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
