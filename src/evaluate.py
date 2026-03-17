"""Local validation framework for ChildWhisper.

Provides child_id-based train/val splitting (no speaker leakage),
WER computation with Whisper EnglishTextNormalizer, per-age-bucket breakdown,
and synthetic noisy validation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np

try:
    from src.augment import create_noise_only_augmentation
    from src.utils import normalize_text
except ModuleNotFoundError:
    from augment import create_noise_only_augmentation
    from utils import normalize_text


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
    rng = random.Random(seed)
    val_children: set[str] = set()

    for bucket in sorted(bucket_to_children.keys()):
        children = bucket_to_children[bucket][:]
        rng.shuffle(children)
        n_val = max(1, round(len(children) * val_ratio))
        # Don't take all children as val — leave at least one for train
        n_val = min(n_val, len(children) - 1) if len(children) > 1 else 0
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


def apply_noise_to_val(
    audio_list: list[np.ndarray],
    noise_dir: str | Path,
    snr_db: float = 10.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> list[np.ndarray]:
    """Apply noise augmentation to a list of validation audio arrays.

    Uses create_noise_only_augmentation from src/augment.py with a fixed SNR
    and p=1.0 so every sample gets noise applied.

    Args:
        audio_list: List of 1D float32 audio arrays.
        noise_dir: Path to directory containing noise audio files (e.g. RealClass).
        snr_db: Signal-to-noise ratio in dB (default 10.0).
        sample_rate: Audio sample rate (default 16000).
        seed: Random seed for reproducibility.

    Returns:
        List of noisy audio arrays, same length/shape/dtype as input.
    """
    if not audio_list:
        return []

    augment_fn = create_noise_only_augmentation(
        noise_dir=noise_dir,
        sample_rate=sample_rate,
        min_snr=snr_db,
        max_snr=snr_db,
        p=1.0,
    )

    rng = np.random.RandomState(seed)
    noisy_list = []
    old_state = np.random.get_state()
    for audio in audio_list:
        # Set a per-sample seed derived from the main seed for reproducibility
        np.random.seed(rng.randint(0, 2**31))
        noisy = augment_fn(audio, sample_rate=sample_rate)
        noisy_list.append(noisy)
    np.random.set_state(old_state)

    return noisy_list


def combined_validation_summary(
    references: list[str],
    clean_hyps: list[str],
    noisy_hyps: list[str],
    age_buckets: list[str],
) -> dict:
    """Compute combined clean and noisy validation summary.

    Returns dict with:
        clean: validation_summary for clean hypotheses
        noisy: validation_summary for noisy hypotheses
        wer_degradation: noisy_wer - clean_wer
        relative_degradation: (noisy - clean) / clean if clean > 0, else 0.0
    """
    clean_summary = validation_summary(references, clean_hyps, age_buckets)
    noisy_summary = validation_summary(references, noisy_hyps, age_buckets)

    clean_wer = clean_summary["overall_wer"]
    noisy_wer = noisy_summary["overall_wer"]
    degradation = noisy_wer - clean_wer

    if clean_wer > 0:
        relative = degradation / clean_wer
    else:
        relative = 0.0

    return {
        "clean": clean_summary,
        "noisy": noisy_summary,
        "wer_degradation": degradation,
        "relative_degradation": relative,
    }


def format_validation_report(combined_summary: dict) -> str:
    """Format a combined validation summary as a human-readable report.

    Args:
        combined_summary: Dict from combined_validation_summary.

    Returns:
        Formatted string with clean/noisy WER comparison table.
    """
    clean = combined_summary["clean"]
    noisy = combined_summary["noisy"]

    lines = []
    lines.append("Validation Report")
    lines.append("=" * 50)
    lines.append(f"{'Metric':<20} {'Clean':>10} {'Noisy':>10}")
    lines.append("-" * 50)
    lines.append(
        f"{'Overall WER':<20} {clean['overall_wer']:>10.4f} {noisy['overall_wer']:>10.4f}"
    )

    # Per-age-bucket WER
    all_buckets = sorted(
        set(clean["per_age_wer"].keys()) | set(noisy["per_age_wer"].keys())
    )
    for bucket in all_buckets:
        c_wer = clean["per_age_wer"].get(bucket, 0.0)
        n_wer = noisy["per_age_wer"].get(bucket, 0.0)
        lines.append(f"{'WER (' + bucket + ')':<20} {c_wer:>10.4f} {n_wer:>10.4f}")

    lines.append("-" * 50)
    lines.append(f"{'Empty preds':<20} {clean['num_empty_preds']:>10d} "
                 f"{noisy['num_empty_preds']:>10d}")
    lines.append(f"{'Utterances':<20} {clean['num_utterances']:>10d} "
                 f"{noisy['num_utterances']:>10d}")
    lines.append("-" * 50)
    lines.append(f"Degradation (abs):  {combined_summary['wer_degradation']:.4f}")
    lines.append(f"Degradation (rel):  {combined_summary['relative_degradation']:.4f}")

    return "\n".join(lines)


def compute_error_breakdown(
    references: list[str],
    hypotheses: list[str],
) -> dict:
    """Compute substitution/insertion/deletion breakdown with WER.

    Normalizes both sides with EnglishTextNormalizer. Skips pairs where
    the normalized reference is empty.

    Returns dict with: substitutions, insertions, deletions, hits,
    total_ref_words, wer, sub_rate, ins_rate, del_rate.
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
        return {
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "hits": 0,
            "total_ref_words": 0,
            "wer": 0.0,
            "sub_rate": 0.0,
            "ins_rate": 0.0,
            "del_rate": 0.0,
        }

    output = jiwer.process_words(filtered_refs, filtered_hyps)
    total_ref = sum(len(r.split()) for r in filtered_refs)

    subs = output.substitutions
    ins = output.insertions
    dels = output.deletions
    hits = output.hits

    return {
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "hits": hits,
        "total_ref_words": total_ref,
        "wer": (subs + ins + dels) / total_ref if total_ref > 0 else 0.0,
        "sub_rate": subs / total_ref if total_ref > 0 else 0.0,
        "ins_rate": ins / total_ref if total_ref > 0 else 0.0,
        "del_rate": dels / total_ref if total_ref > 0 else 0.0,
    }


def compute_per_age_error_breakdown(
    references: list[str],
    hypotheses: list[str],
    age_buckets: list[str],
) -> dict[str, dict]:
    """Compute error breakdown separately for each age bucket.

    Returns dict mapping age_bucket to error breakdown dict.
    """
    bucket_refs: dict[str, list[str]] = defaultdict(list)
    bucket_hyps: dict[str, list[str]] = defaultdict(list)

    for ref, hyp, age in zip(references, hypotheses, age_buckets):
        bucket_refs[age].append(ref)
        bucket_hyps[age].append(hyp)

    result = {}
    for bucket in sorted(bucket_refs.keys()):
        result[bucket] = compute_error_breakdown(bucket_refs[bucket], bucket_hyps[bucket])
    return result


def detect_hallucinations(
    references: list[str],
    hypotheses: list[str],
    threshold: float = 3.0,
) -> list[dict]:
    """Detect hallucinated predictions where hyp word count >> ref word count.

    Flags utterances where hyp_word_count > threshold * ref_word_count.
    Also flags non-empty hypothesis when reference is empty (ratio = inf).
    Empty hypotheses are never flagged.

    Returns list of dicts with: index, reference, hypothesis,
    ref_word_count, hyp_word_count, ratio.
    """
    flagged = []
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        norm_ref = normalize_text(ref)
        norm_hyp = normalize_text(hyp)

        hyp_words = len(norm_hyp.split()) if norm_hyp.strip() else 0
        ref_words = len(norm_ref.split()) if norm_ref.strip() else 0

        if hyp_words == 0:
            continue

        if ref_words == 0:
            ratio = float("inf")
        else:
            ratio = hyp_words / ref_words

        if ratio > threshold:
            flagged.append({
                "index": i,
                "reference": norm_ref,
                "hypothesis": norm_hyp,
                "ref_word_count": ref_words,
                "hyp_word_count": hyp_words,
                "ratio": ratio,
            })

    return flagged


def error_analysis_summary(
    references: list[str],
    hypotheses: list[str],
    age_buckets: list[str],
    hallucination_threshold: float = 3.0,
) -> dict:
    """Compute a full error analysis summary.

    Returns dict with: overall_breakdown, per_age_breakdown,
    hallucinations, hallucination_count, hallucination_rate.
    """
    overall = compute_error_breakdown(references, hypotheses)
    per_age = compute_per_age_error_breakdown(references, hypotheses, age_buckets)
    hallucinations = detect_hallucinations(references, hypotheses, hallucination_threshold)
    total = len(references)

    return {
        "overall_breakdown": overall,
        "per_age_breakdown": per_age,
        "hallucinations": hallucinations,
        "hallucination_count": len(hallucinations),
        "hallucination_rate": len(hallucinations) / total if total > 0 else 0.0,
    }


def format_error_analysis_report(summary: dict) -> str:
    """Format an error analysis summary as a human-readable report.

    Includes S/I/D breakdown table, per-age breakdown, and hallucination summary.
    """
    lines = []
    overall = summary["overall_breakdown"]

    lines.append("Error Analysis Report")
    lines.append("=" * 60)

    # Overall breakdown
    lines.append(f"Overall WER:       {overall['wer']:.4f}")
    lines.append(f"Total Ref Words:   {overall['total_ref_words']}")
    lines.append("")
    lines.append(f"  Substitutions:   {overall['substitutions']:>6d}  "
                 f"(Sub rate: {overall['sub_rate']:.4f})")
    lines.append(f"  Insertions:      {overall['insertions']:>6d}  "
                 f"(Ins rate: {overall['ins_rate']:.4f})")
    lines.append(f"  Deletions:       {overall['deletions']:>6d}  "
                 f"(Del rate: {overall['del_rate']:.4f})")
    lines.append(f"  Hits:            {overall['hits']:>6d}")

    # Per-age breakdown
    per_age = summary["per_age_breakdown"]
    if per_age:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"{'Age':<10} {'WER':>8} {'Sub':>6} {'Ins':>6} {'Del':>6} {'Refs':>6}")
        lines.append("-" * 60)
        for bucket in sorted(per_age.keys()):
            b = per_age[bucket]
            lines.append(
                f"{bucket:<10} {b['wer']:>8.4f} {b['substitutions']:>6d} "
                f"{b['insertions']:>6d} {b['deletions']:>6d} {b['total_ref_words']:>6d}"
            )

    # Hallucination summary
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Hallucinations:    {summary['hallucination_count']} "
                 f"({summary['hallucination_rate']:.2%} of utterances)")
    if summary["hallucinations"]:
        lines.append("")
        for h in summary["hallucinations"][:10]:  # show at most 10
            ratio_str = f"{h['ratio']:.1f}x" if h["ratio"] != float("inf") else "inf"
            lines.append(f"  [{h['index']}] ratio={ratio_str}: "
                         f"ref='{h['reference']}' -> hyp='{h['hypothesis']}'")
        if len(summary["hallucinations"]) > 10:
            lines.append(f"  ... and {len(summary['hallucinations']) - 10} more")

    return "\n".join(lines)
