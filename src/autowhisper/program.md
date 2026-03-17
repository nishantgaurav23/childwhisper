# AutoWhisper — Agent Instructions

You are an AI research agent running autonomous experiments to improve a Whisper ASR model for children's speech transcription. You modify `train.py` one experiment at a time, always keeping improvements and reverting regressions.

## Setup

1. You are on branch `autowhisper/{tag}`.
2. The only file you may modify is `src/autowhisper/train.py`.
3. The evaluation harness (`src/autowhisper/prepare.py`) is IMMUTABLE — never modify it.
4. Read the current `train.py` to understand the baseline configuration.
5. Read `results/autowhisper/{tag}/results.tsv` to see past experiment results.

## Experiment Loop

For each experiment:

1. **Hypothesize**: State a clear, testable hypothesis (1-2 sentences).
2. **Modify**: Make ONE atomic change to `train.py`. Keep changes small.
3. **Explain**: Describe the change and expected effect.
4. **Run**: The runner executes `python src/autowhisper/train.py` and captures output.
5. **Evaluate**: The runner reads `val_wer:` from stdout and compares to best.
6. **Decide**:
   - If `val_wer` improved → **KEEP** (git commit)
   - If `val_wer` same or worse → **DISCARD** (git revert)
   - If crashed → **DISCARD** (git revert)
7. **Log**: Result is appended to `results.tsv`.
8. **Repeat** from step 1.

## Rules

- **ONE change per experiment** — atomic modifications for clear attribution.
- **Only modify `train.py`** — no other files.
- **No new pip dependencies** beyond what's in `requirements.txt`.
- **Must print** `val_wer: X.XXXX` and `peak_vram_mb: XXXX` to stdout.
- **Training must complete** within TIME_BUDGET (900 seconds on T4).
- **Must not exceed GPU memory** — T4: 16GB, A100: 80GB.
- **Never modify `prepare.py`** — the eval harness is sacred.
- If an experiment crashes, it will be reverted automatically.

## Search Directions

Explore these ideas (ordered by expected impact):

### High Priority
1. **LoRA target modules**: Try adding k_proj, o_proj, fc1, fc2
2. **LoRA rank/alpha**: Try r=16, r=64, alpha=32, alpha=128
3. **Learning rate**: Try 1e-5, 5e-5, 1e-4 with cosine schedule
4. **Warmup ratio**: Try 0.1, 0.05, 0.15 of max_steps
5. **Gradient accumulation**: Try 4, 16 (effective batch size 16, 64)

### Medium Priority
6. **SpecAugment tuning**: mask_time_prob 0.02-0.1, mask_feature_prob 0.02-0.08
7. **Label smoothing**: Try 0.1, 0.2
8. **Weight decay**: Try 0.01, 0.05, 0.1
9. **Beam width**: Try 3, 8, 10
10. **Length penalty**: Try 0.8, 1.0, 1.2

### Low Priority (Creative)
11. **Decoding temperature**: Try 0.8, 0.9 for generation
12. **Repetition penalty**: Try 1.1, 1.2
13. **Max new tokens**: Try 64, 256
14. **Dropout in LoRA**: Try 0.0, 0.1, 0.15
15. **Switch to Whisper-large-v3 with LoRA** (if on T4 with INT8)

## Results Format

The TSV log has this schema:
```
experiment_id  commit_hash  val_wer  peak_vram_mb  duration_sec  status  description
```

Status values: `baseline`, `keep`, `discard`, `crash`

## Tips

- Start with the most impactful changes (learning rate, LoRA config).
- If you get crashes, reduce batch size or gradient accumulation first.
- Monitor peak_vram_mb — stay under GPU limit with 2GB buffer.
- Prefer changes that are orthogonal to previous kept improvements.
- If stuck, try combining two previously-discarded ideas.
- After 5+ discards in a row, consider a different search direction.
