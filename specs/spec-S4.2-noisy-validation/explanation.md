# Explanation S4.2 — Noisy Validation Set

## Why
The competition scores models on both clean WER and Noisy WER (classroom environments). Without a noisy validation set, we can't measure how well our models handle background noise locally — we'd only learn about noise robustness after submitting. A synthetic noisy val set lets us iterate on augmentation strategy and noise robustness offline.

## What
Three new functions added to `src/evaluate.py`:

1. **`apply_noise_to_val(audio_list, noise_dir, snr_db=10.0, sample_rate=16000, seed=42)`** — Takes clean validation audio arrays and applies RealClass classroom noise at a fixed SNR. Uses `create_noise_only_augmentation` from `src/augment.py` with p=1.0 (noise always applied). Deterministic via seed for reproducible evaluation.

2. **`combined_validation_summary(references, clean_hyps, noisy_hyps, age_buckets)`** — Runs `validation_summary` on both clean and noisy hypotheses, then computes absolute and relative WER degradation. Returns a dict with `clean`, `noisy`, `wer_degradation`, and `relative_degradation` keys.

3. **`format_validation_report(combined_summary)`** — Formats the combined summary as a human-readable table showing clean vs noisy WER side by side, per-age-bucket breakdown, and degradation metrics.

## How
- `apply_noise_to_val` delegates to `create_noise_only_augmentation` (S4.1) with `min_snr=max_snr=snr_db` for a fixed SNR level. Per-sample seeds are derived from the main seed via `np.random.RandomState` for reproducibility.
- `combined_validation_summary` reuses the existing `validation_summary` function (S1.5) for both clean and noisy predictions, avoiding code duplication.
- `format_validation_report` produces a text table matching the dashboard design in `design.md` section 10.

## Connections
- **Depends on S4.1** (`create_noise_only_augmentation`) for the noise mixing
- **Depends on S1.5** (`validation_summary`, `compute_wer`) for WER computation
- **Used by S4.3** (retrain with augmented data) to measure noise robustness improvements after retraining
- **Used by S5.3** (error analysis) to break down clean vs noisy error patterns
- Enables the Phase 4 workflow: augment training data → retrain → measure noisy val WER improvement

## Test Coverage
- 14 tests in `tests/test_noisy_validation.py`, all passing
- 21 existing tests in `tests/test_evaluate.py` still passing (no regressions)
- Tests cover: shape/dtype/determinism for audio noise, summary structure/values, report formatting
