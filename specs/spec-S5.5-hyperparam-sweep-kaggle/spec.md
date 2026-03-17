# Spec S5.5 -- Hyperparameter Sweep with Kaggle API Integration

## Overview
Build a hyperparameter sweep runner that generates training configs, packages them into Kaggle-compatible notebooks, and manages remote GPU runs via the Kaggle CLI — all from the local terminal. This enables systematic tuning of learning rate, batch size, LoRA rank, and augmentation parameters without manual notebook editing, using Kaggle's free T4 GPU quota.

## Dependencies
| Spec | Feature | Status |
|------|---------|--------|
| S2.2 | Whisper-small training script | done |
| S3.1 | LoRA configuration & training | done |
| S4.1 | Noise augmentation pipeline | done |

## Target Location
| File | Action |
|------|--------|
| `src/sweep.py` | Create — sweep config generator, results parser |
| `src/kaggle_runner.py` | Create — Kaggle API wrapper (push/status/pull) |
| `scripts/sweep_whisper_small.sh` | Create — shell launcher for whisper-small sweep |
| `scripts/sweep_whisper_lora.sh` | Create — shell launcher for whisper-lora sweep |
| `scripts/kaggle_push.sh` | Create — push notebook + start run |
| `scripts/kaggle_status.sh` | Create — poll run status |
| `scripts/kaggle_pull.sh` | Create — pull results back locally |
| `configs/sweep_configs/` | Create — generated YAML configs per trial |
| `tests/test_sweep.py` | Create — tests for sweep logic |
| `tests/test_kaggle_runner.py` | Create — tests for Kaggle runner (mocked API) |

## Functional Requirements

### FR-1: Sweep Configuration Generator
- **What**: Generate a list of hyperparameter trial configs from a search space definition.
- **Inputs**: Base config YAML path, search space dict (param name -> list of values or range), search strategy ("grid", "random"), max_trials (int), seed (int).
- **Outputs**: List of config dicts, each a complete training config with one combo of hyperparams. Also writes each config to `configs/sweep_configs/trial_{N}.yaml`.
- **Edge cases**: Empty search space returns single trial with base config. max_trials=0 raises ValueError. Conflicting params (e.g., batch_size > GPU memory) are not validated here — training will fail fast.

### FR-2: Notebook Generator
- **What**: Convert a training script + config into a self-contained Kaggle notebook (.ipynb) that trains for 1 epoch, evaluates on validation set, and writes results to a CSV.
- **Inputs**: Training script path, config YAML path, model type ("whisper-small" or "whisper-lora"), num_epochs (default 1 for sweep).
- **Outputs**: A `.ipynb` file in `notebooks/sweep/` ready for Kaggle push. The notebook: (1) installs deps, (2) downloads data from Kaggle dataset, (3) runs training with the config, (4) evaluates val WER, (5) writes `results.json` with {trial_id, config, val_wer, train_loss, duration_sec}, (6) saves best checkpoint to HF Hub if WER < threshold.
- **Edge cases**: Script import errors should be caught and logged in results.json with wer=-1.

### FR-3: Kaggle API Wrapper
- **What**: Programmatic interface to push notebooks, check status, and pull outputs via `kaggle` CLI.
- **Inputs**: Notebook path, Kaggle kernel slug, dataset slugs.
- **Outputs**: Status enum (queued, running, complete, error), output files downloaded to local dir.
- **Edge cases**: Missing kaggle.json credentials raises clear error. Network failures retry 3 times with backoff. Kernel not found returns appropriate error.

### FR-4: Results Aggregator
- **What**: Collect results.json from all completed trials, rank by val WER, and output a summary table + best config.
- **Inputs**: List of trial output directories or a sweep output root dir.
- **Outputs**: `sweep_results.csv` (sorted by WER ascending), `best_config.yaml` (copy of winning trial's config), printed summary table to stdout.
- **Edge cases**: Failed trials (wer=-1) are listed last. No successful trials prints warning.

### FR-5: Shell Script Launchers
- **What**: Convenience scripts that orchestrate the full sweep workflow.
- **Inputs**: Optional overrides (--max-trials, --search-strategy, --model-type).
- **Outputs**: Pushes all trials to Kaggle, polls until complete, pulls results, prints summary.
- **Edge cases**: Ctrl+C should not leave orphan kernels (print instructions to stop them manually).

## Data Models

### Search Space Definition (YAML)
```yaml
search_space:
  learning_rate: [5.0e-6, 1.0e-5, 3.0e-5, 5.0e-5, 1.0e-4]
  warmup_steps: [100, 300, 500]
  per_device_train_batch_size: [2, 4]
  gradient_accumulation_steps: [4, 8, 16]
  # LoRA-specific (ignored for whisper-small)
  lora_r: [16, 32, 64]
  lora_alpha: [32, 64, 128]
  spec_augment_mask_time_prob: [0.02, 0.05, 0.08]

strategy: random   # or "grid"
max_trials: 10
num_epochs: 1      # quick sweep; full training uses best config with 3 epochs
seed: 42
```

### Trial Result (JSON)
```json
{
  "trial_id": "trial_003",
  "config": {"learning_rate": 3e-5, "warmup_steps": 300, "...": "..."},
  "val_wer": 0.1834,
  "train_loss": 2.41,
  "duration_sec": 2340,
  "status": "complete",
  "error": null
}
```

### Kaggle kernel-metadata.json
```json
{
  "id": "nishantgaurav23/childwhisper-sweep-trial-003",
  "title": "ChildWhisper Sweep Trial 003",
  "code_file": "sweep_trial_003.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["nishantgaurav23/pasketti-audio"],
  "competition_sources": [],
  "kernel_sources": []
}
```

## Integration Points
- Reads base configs from `configs/training_config.yaml`
- Uses `src/train_whisper_small.py` and `src/train_whisper_lora.py` as training entrypoints
- Checkpoints saved to HuggingFace Hub (same repos as regular training)
- Best config output can be directly passed to `scripts/train_small_dry.sh` or full training scripts
- Kaggle dataset must be pre-uploaded (audio + metadata) — referenced by slug in kernel metadata

## Tangible Outcomes
- [ ] `python src/sweep.py --config configs/training_config.yaml --search-space configs/sweep_space.yaml --model whisper-small` generates N trial configs in `configs/sweep_configs/`
- [ ] `python src/kaggle_runner.py push notebooks/sweep/sweep_trial_001.ipynb` pushes and starts a Kaggle kernel
- [ ] `python src/kaggle_runner.py status nishantgaurav23/childwhisper-sweep-trial-001` returns run status
- [ ] `python src/kaggle_runner.py pull nishantgaurav23/childwhisper-sweep-trial-001 -o output/sweep/trial_001/` downloads results
- [ ] `python src/sweep.py aggregate output/sweep/` produces `sweep_results.csv` and `best_config.yaml`
- [ ] `./scripts/sweep_whisper_small.sh` runs end-to-end: generate configs -> push to Kaggle -> poll -> pull -> summarize
- [ ] All Kaggle API calls are mocked in tests — no real API hits
- [ ] Sweep of 10 trials fits within one Kaggle session (~9 hrs with internet)

## Test-Driven Requirements

### Tests to Write First
1. `test_generate_grid_configs`: Grid search space produces correct number of combos
2. `test_generate_random_configs`: Random search with max_trials caps output, seed is deterministic
3. `test_generate_configs_empty_space`: Empty search space returns base config only
4. `test_generate_configs_lora_params_ignored_for_small`: LoRA params stripped for whisper-small
5. `test_notebook_generator_creates_valid_ipynb`: Output is valid JSON with correct cell structure
6. `test_notebook_generator_embeds_config`: Generated notebook contains the trial config
7. `test_kaggle_push_calls_cli`: Mock subprocess — verify `kaggle kernels push` is called correctly
8. `test_kaggle_status_parses_output`: Mock subprocess — verify status parsing for queued/running/complete/error
9. `test_kaggle_pull_downloads_files`: Mock subprocess — verify output files are saved to correct dir
10. `test_kaggle_missing_credentials`: Raises clear error when kaggle.json is missing
11. `test_aggregate_results_sorts_by_wer`: Results CSV is sorted ascending by val_wer
12. `test_aggregate_results_handles_failures`: Failed trials (wer=-1) appear last
13. `test_aggregate_produces_best_config`: best_config.yaml matches the lowest-WER trial
14. `test_kernel_metadata_format`: Generated kernel-metadata.json matches Kaggle schema

### Mocking Strategy
- Mock `subprocess.run` for all `kaggle` CLI calls (push, status, output)
- Mock file I/O for notebook generation (verify content, not actual file writes)
- Mock `yaml.safe_load` for testing config loading edge cases
- No real Kaggle API calls in any test
- No real GPU training in any test

### Coverage
- All public functions must have tests
- All edge cases from FR edge cases must have tests
- Target: >80% code coverage
