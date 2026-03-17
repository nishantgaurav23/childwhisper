"""Local validation framework for ChildWhisper.

Provides child_id-based train/val splitting (no speaker leakage),
WER computation with Whisper EnglishTextNormalizer, and per-age-bucket breakdown.
"""

from __future__ import annotations

from collections import defaultdict

import jiwer

from src.utils import normalize_text


def split_by_child_id(
    metadata: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split metadata by child_id with age-bucket stratification.

    All utterances from a given child go into the same split (no speaker leakage).
    Stratifies by age_bucket so each split has proportional representation.

    Returns (train_metadata, val_metadata).
    """
    if not metadata:
        return [], []

    # Group child_ids by age_bucket
    child_to_bucket: dict[str, str] = {}
    for m in metadata:
        child_to_bucket[m["child_id"]] = m["age_bucket"]

    bucket_to_children: dict[str, list[str]] = defaultdict(list)
    for child, bucket in child_to_bucket.items():
        bucket_to_children[bucket].append(child)

    # Sort for determinism
    for bucket in bucket_to_children:
        bucket_to_children[bucket].sort()

    # Stratified split: pick val children from each bucket
    import random

    rng = random.Random(seed)
    val_children: set[str] = set()

    for bucket in sorted(bucket_to_children.keys()):
        children = bucket_to_children[bucket][:]
        rng.shuffle(children)
        n_val = max(0, round(len(children) * val_ratio))
        val_children.update(children[:n_val])

    # If no val children selected (too few per bucket), pick at least from global pool
    if not val_children and len(child_to_bucket) > 1:
        all_children = sorted(child_to_bucket.keys())
        rng2 = random.Random(seed)
        rng2.shuffle(all_children)
        n_val = max(1, round(len(all_children) * val_ratio))
        val_children.update(all_children[:n_val])

    train = [m for m in metadata if m["child_id"] not in val_children]
    val = [m for m in metadata if m["child_id"] in val_children]
    return train, val


def compute_wer(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """Compute WER with Whisper EnglishTextNormalizer applied to both sides.

    Skips pairs where the reference is empty (after normalization).
    Returns 0.0 if no valid pairs remain.
    """
    filtered_refs = []
    filtered_hyps = []

    for ref, hyp in zip(references, hypotheses):
        norm_ref = normalize_text(ref)
        if not norm_ref.strip():
            continue
        norm_hyp = normalize_text(hyp)
        filtered_refs.append(norm_ref)
        filtered_hyps.append(norm_hyp)

    if not filtered_refs:
        return 0.0

    return jiwer.wer(filtered_refs, filtered_hyps)


def compute_per_age_wer(
    references: list[str],
    hypotheses: list[str],
    age_buckets: list[str],
) -> dict[str, float]:
    """Compute WER separately for each age bucket.

    Returns dict mapping age_bucket → WER.
    """
    bucket_refs: dict[str, list[str]] = defaultdict(list)
    bucket_hyps: dict[str, list[str]] = defaultdict(list)

    for ref, hyp, age in zip(references, hypotheses, age_buckets):
        bucket_refs[age].append(ref)
        bucket_hyps[age].append(hyp)

    result = {}
    for bucket in sorted(bucket_refs.keys()):
        result[bucket] = compute_wer(bucket_refs[bucket], bucket_hyps[bucket])
    return result


def validation_summary(
    references: list[str],
    hypotheses: list[str],
    age_buckets: list[str],
) -> dict:
    """Compute a full validation summary.

    Returns dict with: overall_wer, per_age_wer, num_utterances,
    num_empty_refs_skipped, num_empty_preds.
    """
    num_empty_refs = sum(
        1 for ref in references if not normalize_text(ref).strip()
    )
    num_empty_preds = sum(
        1 for ref, hyp in zip(references, hypotheses)
        if normalize_text(ref).strip() and not normalize_text(hyp).strip()
    )

    return {
        "overall_wer": compute_wer(references, hypotheses),
        "per_age_wer": compute_per_age_wer(references, hypotheses, age_buckets),
        "num_utterances": len(references),
        "num_empty_refs_skipped": num_empty_refs,
        "num_empty_preds": num_empty_preds,
    }
