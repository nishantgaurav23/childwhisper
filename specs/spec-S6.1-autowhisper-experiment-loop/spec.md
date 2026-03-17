# Spec S6.1 — AutoWhisper: Autonomous AI Experiment Loop

## Overview
- **Spec ID**: S6.1
- **Phase**: 6 (Autonomous Optimization)
- **Depends On**: S2.2 (Whisper-small training), S3.1 (LoRA training), S4.1 (Augmentation), S1.5 (Validation), S5.5 (Sweep infrastructure)
- **Location**: `src/autowhisper/`, `configs/autowhisper/`, `notebooks/05_autowhisper.ipynb`
- **Status**: done

## Motivation
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), which lets an LLM agent autonomously run ~100 ML experiments overnight on a single GPU. The agent modifies a training script, runs a time-boxed experiment, evaluates the result, keeps improvements, reverts regressions, and loops indefinitely.

Our S5.5 sweep is good for grid/random search over predefined hyperparameter ranges. But AutoWhisper goes further — it lets an AI agent make **creative, open-ended modifications** (architecture tweaks, decoding strategies, augmentation recipes, loss functions) that a grid search would never explore, while maintaining a disciplined keep/revert loop that prevents regressions.

## Problem
We have a working two-model ensemble (Whisper-large-v3 LoRA + Whisper-small) with augmentation. But squeezing the last WER points requires exploring a vast, irregular search space:
- LoRA target modules (beyond q_proj/v_proj — try k_proj, o_proj, fc1, fc2)
- LoRA rank/alpha combinations with dropout
- Learning rate schedules (cosine, linear, warmup ratios)
- Augmentation intensity and mixing strategies
- SpecAugment parameter tuning
- Decoding parameters (beam width, length penalty, repetition penalty)
- Preprocessing changes (trim aggressiveness, silence thresholds)
- Label smoothing, weight decay tuning
- Gradient accumulation / effective batch size

Manual experimentation is slow. S5.5 grid/random sweep is systematic but uncreative. AutoWhisper bridges the gap.

## Architecture

### Core Concept
Three files define the system (mirroring autoresearch):

| File | Role | Mutable? |
|------|------|----------|
| `src/autowhisper/prepare.py` | Fixed eval harness: loads data, computes WER on fast-eval subset | **No** (agent cannot modify) |
| `src/autowhisper/train.py` | Training script the agent modifies each experiment | **Yes** (only file agent touches) |
| `src/autowhisper/program.md` | Instructions for the AI agent running the loop | **No** (human edits to steer research) |

Plus supporting infrastructure:
| File | Role |
|------|------|
| `src/autowhisper/runner.py` | Experiment loop orchestrator (git commit, run, eval, keep/revert) |
| `src/autowhisper/logger.py` | Results TSV logger |
| `configs/autowhisper/base_config.yaml` | Base training config |
| `notebooks/05_autowhisper.ipynb` | Kaggle notebook to run the loop on T4 |

### The Experiment Loop

```
┌─────────────────────────────────────────────────┐
│  1. Agent reads program.md + current train.py    │
│  2. Agent proposes a modification to train.py    │
│  3. Git commit the change                        │
│  4. Run: python train.py (budget: N steps)       │
│  5. Read: grep val_wer from run.log              │
│  6. If val_wer improved → KEEP commit            │
│     If val_wer worsened → REVERT (git reset)     │
│  7. Log to results.tsv                           │
│  8. Repeat from step 1                           │
└─────────────────────────────────────────────────┘
```

## Requirements

### FR-1: Fixed Evaluation Harness (`prepare.py`)
- **What**: Immutable module providing data loading and WER evaluation.
- **Functions**:
  - `load_fast_eval_set(data_dir, n_samples=200) -> list[dict]`: Loads a fixed subset of validation data (by child_id split). Returns list of `{audio_path, transcript, child_id, age_bucket}`. Subset is deterministic (seeded by child_id hash). Biased toward shorter utterances for speed.
  - `evaluate_wer(predictions: list[str], references: list[str]) -> dict`: Computes WER using jiwer with EnglishTextNormalizer. Returns `{wer, substitutions, deletions, insertions, n_samples}`.
  - `evaluate_wer_by_age(predictions, references, age_buckets) -> dict`: Per-age-bucket WER breakdown.
  - `TIME_BUDGET: int = 900`: Default time budget per experiment in seconds (15 min on T4).
  - `EVAL_SAMPLES: int = 200`: Number of validation samples for fast eval.
- **Constraints**: Agent MUST NOT modify this file. All evaluation goes through this harness to ensure fair comparison across experiments.
- **Data**: Uses the existing child_id-split validation set from S1.5. The 200-sample fast-eval subset is selected once and frozen.

### FR-2: Mutable Training Script (`train.py`)
- **What**: The only file the AI agent modifies. Contains model loading, fine-tuning config, training loop, and inference.
- **Initial state**: A self-contained script that:
  1. Loads Whisper-small (or large-v3 with LoRA) from HF Hub or local path
  2. Configures fine-tuning (LoRA params, optimizer, scheduler, augmentation)
  3. Trains for a fixed number of steps (derived from TIME_BUDGET)
  4. Runs inference on the fast-eval set
  5. Prints `val_wer: 0.XXXX` and `peak_vram_mb: XXXX` to stdout (parseable by runner)
- **Rules for the agent** (enforced in program.md):
  - Only modify `train.py` — no other files
  - No new pip dependencies beyond `requirements.txt`
  - Must print `val_wer:` and `peak_vram_mb:` lines to stdout
  - Training must complete within TIME_BUDGET seconds
  - Must not exceed GPU memory (T4: 16GB, A100: 80GB)
- **Starting baseline**: Copies the best config from S5.5 sweep results as the initial `train.py`.

### FR-3: Agent Instructions (`program.md`)
- **What**: Markdown file containing complete instructions for the AI agent.
- **Sections**:
  1. **Setup**: How to initialize a run (create branch, read files, run baseline)
  2. **Experiment Loop**: Step-by-step loop specification
  3. **Rules**: What can/cannot be modified, output format, time budget
  4. **Search Directions**: Human-curated list of ideas to explore (updated between runs)
  5. **Results Format**: TSV schema, keep/revert criteria
- **Key rules**:
  - One modification per experiment (atomic changes for clear attribution)
  - Always explain the hypothesis before modifying
  - Keep changes small — prefer simplicity
  - If an experiment crashes, revert and log as "crash"
  - Never modify prepare.py

### FR-4: Experiment Runner (`runner.py`)
- **What**: Orchestrates the experiment loop programmatically (alternative to relying on the AI agent to manage git).
- **Functions**:
  - `init_run(tag: str, base_branch: str = "main") -> str`: Creates branch `autowhisper/{tag}`, runs baseline, returns branch name.
  - `run_experiment(train_script: str, time_budget: int) -> dict`: Executes train.py, captures stdout/stderr, parses val_wer and peak_vram_mb. Returns `{val_wer, peak_vram_mb, duration_sec, status, stdout, stderr}`.
  - `evaluate_and_decide(result: dict, best_wer: float) -> str`: Returns "keep" if val_wer < best_wer, "discard" if val_wer >= best_wer, "crash" if status != 0.
  - `keep_experiment(description: str)`: Git commit with description.
  - `revert_experiment()`: Git reset to previous commit.
  - `log_result(result: dict, decision: str, description: str, log_path: str)`: Append to results.tsv.
- **Timeout handling**: Kills training process if it exceeds `time_budget + 60s` grace period.
- **Crash recovery**: If train.py crashes (non-zero exit), revert and log as crash.

### FR-5: Results Logger (`logger.py`)
- **What**: Manages the results.tsv experiment log.
- **TSV Schema**:
  ```
  experiment_id  commit_hash  val_wer  peak_vram_mb  duration_sec  status  description
  ```
  - `experiment_id`: Sequential integer (001, 002, ...)
  - `commit_hash`: Short git SHA (or "baseline" for first run)
  - `val_wer`: Float, -1.0 for crashes
  - `peak_vram_mb`: Integer, -1 for crashes
  - `duration_sec`: Wall clock time
  - `status`: "keep" | "discard" | "crash" | "baseline"
  - `description`: One-line description of the change
- **Functions**:
  - `init_log(path: str)`: Write TSV header.
  - `append_result(path: str, result: dict)`: Append one row.
  - `load_results(path: str) -> list[dict]`: Read all results.
  - `get_best_wer(path: str) -> float`: Return lowest val_wer from "keep" or "baseline" rows.
  - `get_frontier(path: str) -> list[dict]`: Return monotonically-improving "keep" results (the improvement frontier).

### FR-6: Base Config (`base_config.yaml`)
- **What**: Starting configuration for train.py, derived from best S5.5 sweep result.
- **Sections**:
  ```yaml
  model:
    name: "openai/whisper-small"  # or whisper-large-v3
    mode: "full"                   # or "lora"
    lora:
      r: 32
      alpha: 64
      target_modules: ["q_proj", "v_proj"]
      dropout: 0.05

  training:
    learning_rate: 3.0e-5
    warmup_steps: 300
    max_steps: 500           # per-experiment step budget
    per_device_batch_size: 4
    gradient_accumulation: 8
    fp16: true
    gradient_checkpointing: true

  augmentation:
    enabled: true
    spec_augment:
      mask_time_prob: 0.05
      mask_feature_prob: 0.04

  decoding:
    num_beams: 5
    max_new_tokens: 128

  budget:
    time_limit_sec: 900       # 15 min per experiment
    eval_samples: 200
    gpu: "t4"                 # t4 or a100
  ```

### FR-7: Kaggle Notebook (`05_autowhisper.ipynb`)
- **What**: Self-contained notebook to run the AutoWhisper loop on Kaggle T4.
- **Structure**:
  1. Install deps, clone repo, setup git
  2. Download competition data + noise data
  3. Initialize run with tag (e.g., `run_mar17`)
  4. Run baseline experiment
  5. Loop: agent proposes modification → run → eval → keep/revert (for N experiments or until session timeout)
  6. Push results.tsv and best checkpoint to HF Hub
  7. Print summary table
- **Session management**: Detect remaining Kaggle session time, stop loop with 30 min buffer for checkpoint upload.
- **Two modes**:
  - **Agent mode**: Full AI agent loop (requires Claude API key as Kaggle secret). The agent reads program.md, modifies train.py, and the runner executes.
  - **Scripted mode**: Pre-defined modification sequence (no API needed). A list of patches applied one-by-one. Good for running known experiments without API cost.

### FR-8: Analysis & Visualization
- **What**: Post-run analysis of experiment results.
- **Functions in `logger.py`**:
  - `plot_progress(results_path: str, output_path: str)`: Scatter plot of all experiments (x=experiment_id, y=val_wer), colored by status (green=keep, red=discard, gray=crash). Overlay the improvement frontier line.
  - `print_summary(results_path: str)`: Print to stdout: total experiments, keeps, discards, crashes, best WER, total GPU time, improvement over baseline.
- **Output**: `results/autowhisper/{tag}/progress.png` and stdout summary.

## Adaptation Details (vs. Karpathy's autoresearch)

| autoresearch | AutoWhisper | Reason |
|---|---|---|
| Metric: val_bpb | Metric: val_wer | ASR task |
| 5-min budget (H100) | 15-min budget (T4) | Whisper fine-tune is slower, T4 is slower |
| Infinite loop | Session-bounded (12hr Kaggle, or ~30 experiments) | Ephemeral GPU |
| Single file (train.py) | Single file (train.py) | Same pattern |
| Git branch per run | Git branch per run | Same pattern |
| Claude as agent | Claude as agent (or scripted mode) | API cost fallback |
| ~50M param GPT | ~244M Whisper-small or LoRA adapter | Different model |
| Train from scratch | Fine-tune pretrained | Transfer learning |
| No checkpointing | HF Hub checkpointing | Kaggle sessions are ephemeral |

## Integration Points
- Uses validation split from S1.5 (`src/evaluate.py`)
- Uses augmentation pipeline from S4.1 (`src/augment.py`)
- Uses dataset class from S2.1 (`src/dataset.py`)
- Uses text normalization from S1.3 (`src/utils.py`)
- Best configs feed back into S5.4 final submission
- Results complement S5.5 sweep results (creative search vs. systematic search)

## Tangible Outcomes
- [ ] `python -m src.autowhisper.runner init --tag run_mar17` creates branch and runs baseline
- [ ] `python -m src.autowhisper.runner run --tag run_mar17 --train src/autowhisper/train.py` executes one experiment and logs result
- [ ] `python -m src.autowhisper.runner revert --tag run_mar17` reverts last experiment
- [ ] `python -m src.autowhisper.prepare` validates the eval harness loads data and computes WER
- [ ] `results.tsv` logs all experiments with correct schema
- [ ] `python -m src.autowhisper.logger summary results/autowhisper/run_mar17/results.tsv` prints summary
- [ ] `python -m src.autowhisper.logger plot results/autowhisper/run_mar17/results.tsv` generates progress.png
- [ ] Kaggle notebook `05_autowhisper.ipynb` runs in scripted mode (no API key needed) for at least 5 experiments
- [ ] Agent mode works with Claude API key: agent reads program.md, modifies train.py, runner executes
- [ ] Best experiment WER is lower than starting baseline
- [ ] All experiments are git-committed (keeps) or reverted (discards) — clean git history
- [ ] All runner/logger functions have unit tests (mocked git, mocked subprocess)
- [ ] ruff passes on all new code

## Test-Driven Requirements

### Tests to Write First

#### prepare.py tests (`tests/test_autowhisper_prepare.py`)
1. `test_load_fast_eval_set_returns_correct_count`: Returns exactly EVAL_SAMPLES items
2. `test_load_fast_eval_set_deterministic`: Same data_dir + n_samples always returns same items
3. `test_load_fast_eval_set_no_speaker_leakage`: All returned child_ids are in validation split only
4. `test_evaluate_wer_perfect`: All correct predictions → WER = 0.0
5. `test_evaluate_wer_all_wrong`: All wrong predictions → WER > 0
6. `test_evaluate_wer_applies_normalizer`: "IT'S" vs "its" → WER = 0.0 (normalizer handles this)
7. `test_evaluate_wer_by_age_returns_all_buckets`: All age buckets present in output
8. `test_evaluate_wer_empty_predictions`: Empty list raises ValueError

#### runner.py tests (`tests/test_autowhisper_runner.py`)
9. `test_init_run_creates_branch`: Mock git — verify branch `autowhisper/{tag}` is created
10. `test_run_experiment_parses_wer`: Mock subprocess with stdout containing `val_wer: 0.1523` → result['val_wer'] == 0.1523
11. `test_run_experiment_parses_vram`: Mock subprocess with stdout containing `peak_vram_mb: 14200` → result['peak_vram_mb'] == 14200
12. `test_run_experiment_timeout_kills_process`: Mock subprocess that hangs → killed after budget + grace
13. `test_run_experiment_crash_returns_crash_status`: Mock subprocess exit code 1 → status = "crash"
14. `test_evaluate_and_decide_keep`: val_wer < best_wer → "keep"
15. `test_evaluate_and_decide_discard`: val_wer >= best_wer → "discard"
16. `test_evaluate_and_decide_crash`: status != 0 → "crash"
17. `test_keep_experiment_commits`: Mock git — verify git commit is called
18. `test_revert_experiment_resets`: Mock git — verify git reset is called

#### logger.py tests (`tests/test_autowhisper_logger.py`)
19. `test_init_log_creates_header`: TSV file starts with correct header
20. `test_append_result_adds_row`: After append, file has header + 1 data row
21. `test_load_results_parses_all_fields`: All fields parsed with correct types
22. `test_get_best_wer_ignores_crashes`: Crash rows (wer=-1) are excluded
23. `test_get_best_wer_includes_baseline`: Baseline row is considered
24. `test_get_frontier_monotonic`: Frontier is strictly decreasing
25. `test_print_summary_format`: Output contains total, keeps, discards, best WER
26. `test_plot_progress_creates_file`: Mock matplotlib — verify file is written

### Mocking Strategy
- Mock `subprocess.run` / `subprocess.Popen` for all training runs and git commands
- Mock `librosa.load` and audio I/O for eval harness tests
- Mock `transformers` model loading (no real model downloads in tests)
- Mock `matplotlib.pyplot` for plot tests
- Mock git operations (commit, reset, branch) via `unittest.mock.patch`
- No real GPU training in any test
- No real API calls in any test
- Use small synthetic audio fixtures (sine waves) for eval harness integration tests

### Coverage
- All public functions in prepare.py, runner.py, logger.py must have tests
- All edge cases (timeout, crash, empty results, speaker leakage) must have tests
- Target: >85% code coverage

## Out of Scope
- Actual model training (runs on Kaggle/Colab GPU)
- Claude API integration for agent mode (that's runtime config, not code)
- Modifying existing training scripts in `src/` (autowhisper has its own self-contained train.py)
- Updating submission/main.py (winning config is manually applied after analysis)
- Cost tracking for Claude API usage
- Multi-GPU or distributed training
