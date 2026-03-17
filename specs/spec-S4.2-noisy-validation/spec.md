# Spec S4.2 — Noisy Validation Set

## Overview
Extend `src/evaluate.py` with functions to create synthetic noisy validation audio and produce combined clean/noisy validation reports. Uses `create_noise_only_augmentation` from `src/augment.py` to mix RealClass classroom noise into validation audio at a fixed SNR (default 10 dB).

## Dependencies
- **S4.1** (Noise augmentation pipeline) — done: provides `create_noise_only_augmentation`
- **S1.5** (Local validation framework) — done: provides `validation_summary`, `compute_wer`, etc.

## Location
- `src/evaluate.py` — new functions added to existing module

## Functional Requirements

### NV-1: Apply Noise to Validation Audio
- `apply_noise_to_val(audio_list, noise_dir, snr_db=10.0, sample_rate=16000, seed=42)` → list of noisy np.ndarrays
- Takes a list of clean audio arrays (1D float32 np.ndarray) and a noise directory path
- Uses `create_noise_only_augmentation` from `src/augment.py` with `min_snr=snr_db, max_snr=snr_db` (fixed SNR)
- Probability p=1.0 (always apply noise to all val samples)
- Seeds the noise augmentation for reproducibility
- Returns list of noisy audio arrays, same length/dtype as input

### NV-2: Combined Validation Summary
- `combined_validation_summary(references, clean_hyps, noisy_hyps, age_buckets)` → dict
- Returns dict with:
  - `clean`: validation_summary result for clean hypotheses
  - `noisy`: validation_summary result for noisy hypotheses
  - `wer_degradation`: noisy_wer - clean_wer (absolute difference)
  - `relative_degradation`: (noisy_wer - clean_wer) / clean_wer if clean_wer > 0, else 0.0

### NV-3: Format Validation Report
- `format_validation_report(combined_summary)` → str
- Produces a human-readable table string (as in design.md section 10)
- Shows clean vs noisy WER side by side, per age bucket
- Shows degradation metrics

## Non-Functional Requirements
- No dependency on model inference — operates on pre-computed hypotheses and audio arrays
- Audio processing (noise mixing) works on CPU (MacBook compatible)
- All functions importable and testable independently
- Existing functions in evaluate.py remain unchanged

## TDD Notes

### Tests to Write First
1. **test_apply_noise_returns_list** — returns list of same length as input
2. **test_apply_noise_output_shape** — each output array same shape as corresponding input
3. **test_apply_noise_output_dtype** — output arrays are float
4. **test_apply_noise_modifies_audio** — noisy audio differs from clean
5. **test_apply_noise_deterministic** — same seed → same output
6. **test_apply_noise_raises_missing_dir** — FileNotFoundError for bad noise_dir
7. **test_apply_noise_empty_list** — empty input → empty output
8. **test_combined_summary_structure** — has clean, noisy, wer_degradation, relative_degradation keys
9. **test_combined_summary_clean_matches** — clean sub-dict matches validation_summary
10. **test_combined_summary_degradation** — degradation = noisy_wer - clean_wer
11. **test_combined_summary_zero_clean_wer** — relative_degradation is 0.0 when clean WER is 0
12. **test_format_report_contains_headers** — output string contains "Clean" and "Noisy"
13. **test_format_report_contains_age_buckets** — output contains age bucket labels
14. **test_format_report_contains_degradation** — output contains degradation info

## Outcomes
- `src/evaluate.py` has `apply_noise_to_val`, `combined_validation_summary`, `format_validation_report`
- `tests/test_noisy_validation.py` has ≥14 tests, all passing
- `ruff` lint passes
- Existing evaluate.py tests still pass
- Functions integrate with augment.py's `create_noise_only_augmentation`
