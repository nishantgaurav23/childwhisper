# Explanation S6.1 — AutoWhisper: Autonomous AI Experiment Loop

## Why

Manual hyperparameter tuning and the grid/random sweep from S5.5 are systematic but constrained to predefined search spaces. AutoWhisper enables **creative, open-ended experimentation** — an AI agent can propose architecture tweaks, decoding strategies, augmentation recipes, and loss function changes that a grid search would never explore. The keep/revert discipline prevents regressions while allowing bold exploration.

This is inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), adapted for Whisper fine-tuning on children's speech within Kaggle's ephemeral GPU sessions.

## What

A three-file system plus supporting infrastructure:

| Component | File | Role |
|-----------|------|------|
| Eval Harness | `src/autowhisper/prepare.py` | **Immutable** — loads 200-sample fast-eval subset, computes WER via jiwer + EnglishTextNormalizer |
| Training Script | `src/autowhisper/train.py` | **Mutable** — the only file the agent modifies; prints `val_wer:` and `peak_vram_mb:` |
| Agent Instructions | `src/autowhisper/program.md` | **Immutable** — rules, search directions, loop specification for the AI agent |
| Runner | `src/autowhisper/runner.py` | Orchestrates git commit/revert, subprocess execution, timeout handling |
| Logger | `src/autowhisper/logger.py` | TSV results logging, frontier tracking, matplotlib visualization |
| CLI | `src/autowhisper/__main__.py` | `python -m src.autowhisper init/run/revert/summary/plot` subcommands |
| Config | `configs/autowhisper/base_config.yaml` | Starting configuration from best S5.5 result |
| Notebook | `notebooks/05_autowhisper.ipynb` | Kaggle-ready with scripted mode (no API) and agent mode (Claude API) |

## How

### The Experiment Loop

1. Agent reads `program.md` + current `train.py` + past results
2. Proposes ONE atomic modification to `train.py`
3. Runner executes `python train.py`, captures stdout (time-bounded)
4. Parses `val_wer:` from output, compares to best
5. **Keep** (git commit) if improved, **Revert** (git checkout) if not
6. Logs to `results.tsv`, repeats

### Key Design Decisions

- **Fixed eval harness**: `prepare.py` uses the same child_id-split validation from S1.5, selecting a deterministic 200-sample subset biased toward shorter utterances for speed. This ensures fair comparison across all experiments.
- **Single mutable file**: Constraining the agent to only modify `train.py` prevents scope creep and makes attribution trivial — each git commit is one experiment.
- **Dual mode notebook**: Scripted mode applies pre-defined patches (zero API cost), agent mode uses Claude API for creative modifications. Both share the same runner/logger infrastructure.
- **Session-aware**: The Kaggle notebook detects remaining session time and stops with a 30-minute buffer for checkpoint upload to HF Hub.

### Integration Points

- `prepare.py` → `src/evaluate.py` (split_by_child_id), `src/preprocess.py` (load_metadata), `src/utils.py` (get_normalizer)
- `train.py` → `src/dataset.py` (WhisperDataCollator), `src/preprocess.py` (load_audio)
- `runner.py` → `logger.py` (append_result)
- `__main__.py` → `runner.py` + `logger.py` (full CLI orchestration)

## Connections

| Spec | Relationship |
|------|-------------|
| S1.5 | Validation split and WER computation reused in `prepare.py` |
| S2.1 | WhisperDataCollator used in `train.py` |
| S2.2 | Whisper-small training pattern replicated in `train.py` |
| S3.1 | LoRA configuration option in `train.py` CONFIG |
| S4.1 | Augmentation explored as search direction in `program.md` |
| S5.5 | Sweep infrastructure patterns (subprocess execution, results logging) carried forward; best sweep config → `base_config.yaml` |

## Test Coverage

- **47 tests** across 3 test files:
  - `test_autowhisper_prepare.py`: 17 tests (eval harness, load_fast_eval_set, WER computation)
  - `test_autowhisper_logger.py`: 16 tests (TSV I/O, best WER, frontier, summary, plot)
  - `test_autowhisper_runner.py`: 14 tests (git ops, subprocess parsing, timeout, keep/revert)
- All git and subprocess operations mocked — no real GPU or model downloads in tests
- Full test suite: 520 tests passing (no regressions)
