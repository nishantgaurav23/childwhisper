"""Tests for hyperparameter sweep config generator, notebook generator, and results aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


class TestGenerateConfigs:
    """Tests for FR-1: Sweep Configuration Generator."""

    def _base_config(self, tmp_path: Path) -> Path:
        """Create a minimal base config YAML."""
        cfg = {
            "common": {"sample_rate": 16000, "min_duration_sec": 0.3},
            "whisper_small": {
                "model_name": "openai/whisper-small",
                "learning_rate": 1e-5,
                "warmup_steps": 500,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "num_train_epochs": 3,
            },
            "whisper_large_v3": {
                "model_name": "openai/whisper-large-v3",
                "learning_rate": 1e-3,
                "lora": {"r": 32, "alpha": 64},
            },
        }
        p = tmp_path / "base_config.yaml"
        p.write_text(yaml.dump(cfg))
        return p

    def test_generate_grid_configs(self, tmp_path):
        """Grid search space produces correct number of combos."""
        from src.sweep import generate_configs

        base_cfg = self._base_config(tmp_path)
        search_space = {
            "learning_rate": [1e-5, 3e-5],
            "warmup_steps": [100, 500],
        }
        output_dir = tmp_path / "sweep_configs"
        configs = generate_configs(
            base_config_path=str(base_cfg),
            search_space=search_space,
            strategy="grid",
            max_trials=100,
            seed=42,
            model_type="whisper-small",
            output_dir=str(output_dir),
        )
        # 2 x 2 = 4 combos
        assert len(configs) == 4
        # Verify each config has the overridden values
        lrs = sorted(c["learning_rate"] for c in configs)
        assert lrs == sorted([1e-5, 1e-5, 3e-5, 3e-5])
        # Verify YAML files written
        assert len(list(output_dir.glob("trial_*.yaml"))) == 4

    def test_generate_random_configs(self, tmp_path):
        """Random search with max_trials caps output, seed is deterministic."""
        from src.sweep import generate_configs

        base_cfg = self._base_config(tmp_path)
        search_space = {
            "learning_rate": [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4],
            "warmup_steps": [100, 300, 500, 700, 1000],
        }
        output_dir = tmp_path / "sweep_configs"
        configs1 = generate_configs(
            base_config_path=str(base_cfg),
            search_space=search_space,
            strategy="random",
            max_trials=3,
            seed=42,
            model_type="whisper-small",
            output_dir=str(output_dir),
        )
        assert len(configs1) == 3

        # Same seed produces same configs
        output_dir2 = tmp_path / "sweep_configs2"
        configs2 = generate_configs(
            base_config_path=str(base_cfg),
            search_space=search_space,
            strategy="random",
            max_trials=3,
            seed=42,
            model_type="whisper-small",
            output_dir=str(output_dir2),
        )
        assert configs1 == configs2

    def test_generate_configs_empty_space(self, tmp_path):
        """Empty search space returns single trial with base config."""
        from src.sweep import generate_configs

        base_cfg = self._base_config(tmp_path)
        output_dir = tmp_path / "sweep_configs"
        configs = generate_configs(
            base_config_path=str(base_cfg),
            search_space={},
            strategy="grid",
            max_trials=10,
            seed=42,
            model_type="whisper-small",
            output_dir=str(output_dir),
        )
        assert len(configs) == 1
        assert configs[0]["model_name"] == "openai/whisper-small"

    def test_generate_configs_max_trials_zero_raises(self, tmp_path):
        """max_trials=0 raises ValueError."""
        from src.sweep import generate_configs

        base_cfg = self._base_config(tmp_path)
        with pytest.raises(ValueError, match="max_trials"):
            generate_configs(
                base_config_path=str(base_cfg),
                search_space={"learning_rate": [1e-5]},
                strategy="grid",
                max_trials=0,
                seed=42,
                model_type="whisper-small",
                output_dir=str(tmp_path / "out"),
            )

    def test_generate_configs_lora_params_ignored_for_small(self, tmp_path):
        """LoRA params stripped for whisper-small model type."""
        from src.sweep import generate_configs

        base_cfg = self._base_config(tmp_path)
        search_space = {
            "learning_rate": [1e-5],
            "lora_r": [16, 32],
            "lora_alpha": [32, 64],
        }
        output_dir = tmp_path / "sweep_configs"
        configs = generate_configs(
            base_config_path=str(base_cfg),
            search_space=search_space,
            strategy="grid",
            max_trials=100,
            seed=42,
            model_type="whisper-small",
            output_dir=str(output_dir),
        )
        # LoRA params should be ignored, so only learning_rate varies -> 1 config
        assert len(configs) == 1
        assert "lora_r" not in configs[0]
        assert "lora_alpha" not in configs[0]

    def test_generate_configs_lora_params_kept_for_lora(self, tmp_path):
        """LoRA params kept for whisper-lora model type."""
        from src.sweep import generate_configs

        base_cfg = self._base_config(tmp_path)
        search_space = {
            "learning_rate": [1e-3],
            "lora_r": [16, 32],
        }
        output_dir = tmp_path / "sweep_configs"
        configs = generate_configs(
            base_config_path=str(base_cfg),
            search_space=search_space,
            strategy="grid",
            max_trials=100,
            seed=42,
            model_type="whisper-lora",
            output_dir=str(output_dir),
        )
        assert len(configs) == 2
        rs = sorted(c["lora_r"] for c in configs)
        assert rs == [16, 32]


class TestNotebookGenerator:
    """Tests for FR-2: Notebook Generator."""

    def test_notebook_generator_creates_valid_ipynb(self, tmp_path):
        """Output is valid JSON with correct cell structure."""
        from src.sweep import generate_notebook

        config = {
            "model_name": "openai/whisper-small",
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
        }
        nb_path = generate_notebook(
            trial_id="trial_001",
            config=config,
            model_type="whisper-small",
            output_dir=str(tmp_path),
            kaggle_dataset_slug="nishantgaurav23/pasketti-audio",
        )
        assert Path(nb_path).exists()
        with open(nb_path) as f:
            nb = json.load(f)
        # Valid notebook structure
        assert "cells" in nb
        assert "metadata" in nb
        assert "nbformat" in nb
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) > 0
        # Has at least one code cell
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 1

    def test_notebook_generator_embeds_config(self, tmp_path):
        """Generated notebook contains the trial config."""
        from src.sweep import generate_notebook

        config = {
            "model_name": "openai/whisper-small",
            "learning_rate": 3e-5,
            "warmup_steps": 300,
        }
        nb_path = generate_notebook(
            trial_id="trial_042",
            config=config,
            model_type="whisper-small",
            output_dir=str(tmp_path),
            kaggle_dataset_slug="nishantgaurav23/pasketti-audio",
        )
        with open(nb_path) as f:
            content = f.read()
        # Config values should appear in the notebook
        assert "3e-05" in content or "3.0e-05" in content or "0.00003" in content
        assert "300" in content
        assert "trial_042" in content

    def test_notebook_generator_creates_kernel_metadata(self, tmp_path):
        """Kernel metadata JSON is created alongside notebook."""
        from src.sweep import generate_notebook

        config = {"model_name": "openai/whisper-small", "learning_rate": 1e-5}
        nb_path = generate_notebook(
            trial_id="trial_001",
            config=config,
            model_type="whisper-small",
            output_dir=str(tmp_path),
            kaggle_dataset_slug="nishantgaurav23/pasketti-audio",
            kaggle_username="nishantgaurav23",
        )
        meta_path = Path(nb_path).parent / "kernel-metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "id" in meta
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["enable_gpu"] is True


class TestAggregateResults:
    """Tests for FR-4: Results Aggregator."""

    def _create_trial_results(self, base_dir: Path, trials: list[dict]):
        """Helper to create trial result directories."""
        for trial in trials:
            trial_dir = base_dir / trial["trial_id"]
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "results.json").write_text(json.dumps(trial))

    def test_aggregate_results_sorts_by_wer(self, tmp_path):
        """Results CSV is sorted ascending by val_wer."""
        from src.sweep import aggregate_results

        trials = [
            {
                "trial_id": "trial_001",
                "val_wer": 0.25,
                "train_loss": 3.0,
                "duration_sec": 100,
                "status": "complete",
                "error": None,
                "config": {"learning_rate": 1e-5},
            },
            {
                "trial_id": "trial_002",
                "val_wer": 0.15,
                "train_loss": 2.0,
                "duration_sec": 120,
                "status": "complete",
                "error": None,
                "config": {"learning_rate": 3e-5},
            },
            {
                "trial_id": "trial_003",
                "val_wer": 0.20,
                "train_loss": 2.5,
                "duration_sec": 110,
                "status": "complete",
                "error": None,
                "config": {"learning_rate": 5e-5},
            },
        ]
        self._create_trial_results(tmp_path, trials)

        csv_path, best_cfg_path = aggregate_results(str(tmp_path))
        assert Path(csv_path).exists()
        assert Path(best_cfg_path).exists()

        # Read CSV and check sorting
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        wers = [float(r["val_wer"]) for r in rows]
        assert wers == sorted(wers)
        assert wers[0] == 0.15

    def test_aggregate_results_handles_failures(self, tmp_path):
        """Failed trials (wer=-1) appear last."""
        from src.sweep import aggregate_results

        trials = [
            {
                "trial_id": "trial_001",
                "val_wer": -1,
                "train_loss": -1,
                "duration_sec": 10,
                "status": "error",
                "error": "OOM",
                "config": {"learning_rate": 1e-4},
            },
            {
                "trial_id": "trial_002",
                "val_wer": 0.18,
                "train_loss": 2.1,
                "duration_sec": 120,
                "status": "complete",
                "error": None,
                "config": {"learning_rate": 3e-5},
            },
        ]
        self._create_trial_results(tmp_path, trials)

        csv_path, best_cfg_path = aggregate_results(str(tmp_path))
        import csv

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        # Successful trial first, failed last
        assert rows[0]["trial_id"] == "trial_002"
        assert rows[-1]["trial_id"] == "trial_001"

    def test_aggregate_produces_best_config(self, tmp_path):
        """best_config.yaml matches the lowest-WER trial."""
        from src.sweep import aggregate_results

        trials = [
            {
                "trial_id": "trial_001",
                "val_wer": 0.22,
                "train_loss": 2.8,
                "duration_sec": 100,
                "status": "complete",
                "error": None,
                "config": {"learning_rate": 1e-5, "warmup_steps": 500},
            },
            {
                "trial_id": "trial_002",
                "val_wer": 0.14,
                "train_loss": 1.9,
                "duration_sec": 130,
                "status": "complete",
                "error": None,
                "config": {"learning_rate": 3e-5, "warmup_steps": 300},
            },
        ]
        self._create_trial_results(tmp_path, trials)

        _, best_cfg_path = aggregate_results(str(tmp_path))
        with open(best_cfg_path) as f:
            best = yaml.safe_load(f)
        assert best["learning_rate"] == 3e-5
        assert best["warmup_steps"] == 300

    def test_aggregate_no_successful_trials(self, tmp_path):
        """Warning printed when no successful trials."""
        from src.sweep import aggregate_results

        trials = [
            {
                "trial_id": "trial_001",
                "val_wer": -1,
                "train_loss": -1,
                "duration_sec": 5,
                "status": "error",
                "error": "crash",
                "config": {"learning_rate": 1e-4},
            },
        ]
        self._create_trial_results(tmp_path, trials)

        csv_path, best_cfg_path = aggregate_results(str(tmp_path))
        assert Path(csv_path).exists()
        # No best config when all failed
        assert best_cfg_path is None
