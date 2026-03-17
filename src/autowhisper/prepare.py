"""Fixed evaluation harness for AutoWhisper experiment loop.

This module is IMMUTABLE — the AI agent must not modify it.
All evaluation goes through this harness to ensure fair comparison.
"""

import hashlib
from pathlib import Path

import jiwer

from src.evaluate import split_by_child_id
from src.preprocess import load_metadata
from src.utils import get_normalizer

TIME_BUDGET = 900  # seconds per experiment (15 min on T4)
EVAL_SAMPLES = 200  # number of validation samples for fast eval


def load_validation_metadata(data_dir: str) -> list[dict]:
    """Load validation-only metadata using child_id split from S1.5."""
    data_path = Path(data_dir)
    metadata_path = data_path / "train_word_transcripts.jsonl"
    all_metadata = load_metadata(str(metadata_path))
    _, val_metadata = split_by_child_id(all_metadata, val_ratio=0.1, seed=42)
    return val_metadata


def load_fast_eval_set(data_dir: str, n_samples: int = EVAL_SAMPLES) -> list[dict]:
    """Load a fixed, deterministic subset of validation data for fast evaluation.

    Biased toward shorter utterances for speed. Deterministic via child_id hash.
    """
    val_metadata = load_validation_metadata(data_dir)

    # Sort by duration (shorter first) for speed bias, with deterministic tiebreaker
    val_metadata.sort(
        key=lambda e: (
            e.get("audio_duration_sec", 999),
            hashlib.md5(e.get("child_id", "").encode()).hexdigest(),
        )
    )

    # Take at most n_samples
    n = min(n_samples, len(val_metadata))
    return val_metadata[:n]


def evaluate_wer(predictions: list[str], references: list[str]) -> dict:
    """Compute WER with EnglishTextNormalizer applied to both sides.

    Returns dict with wer, substitutions, deletions, insertions, n_samples.
    Raises ValueError if inputs are empty or mismatched length.
    """
    if len(predictions) == 0 or len(references) == 0:
        raise ValueError("Predictions and references must be non-empty")
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    normalizer = get_normalizer()
    norm_preds = [normalizer(p) for p in predictions]
    norm_refs = [normalizer(r) for r in references]

    # Filter out empty references after normalization
    pairs = [
        (p, r) for p, r in zip(norm_preds, norm_refs) if r.strip()
    ]
    if not pairs:
        raise ValueError("All references are empty after normalization")

    filtered_preds, filtered_refs = zip(*pairs)
    filtered_preds = list(filtered_preds)
    filtered_refs = list(filtered_refs)

    output = jiwer.process_words(filtered_refs, filtered_preds)

    return {
        "wer": output.wer,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "n_samples": len(filtered_refs),
    }


def evaluate_wer_by_age(
    predictions: list[str],
    references: list[str],
    age_buckets: list[str],
) -> dict:
    """Compute per-age-bucket WER breakdown.

    Returns dict mapping age_bucket -> wer float.
    """
    if len(predictions) == 0:
        raise ValueError("Predictions must be non-empty")
    if len(predictions) != len(references) or len(predictions) != len(age_buckets):
        raise ValueError("All inputs must have the same length")

    normalizer = get_normalizer()

    buckets: dict[str, tuple[list[str], list[str]]] = {}
    for pred, ref, age in zip(predictions, references, age_buckets):
        if age not in buckets:
            buckets[age] = ([], [])
        buckets[age][0].append(normalizer(pred))
        buckets[age][1].append(normalizer(ref))

    result = {}
    for age, (preds, refs) in buckets.items():
        # Filter empty refs
        pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
        if pairs:
            fp, fr = zip(*pairs)
            output = jiwer.process_words(list(fr), list(fp))
            result[age] = output.wer
        else:
            result[age] = -1.0  # no valid samples

    return result
