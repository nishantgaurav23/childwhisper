"""Hyperparameter sweep config generator, notebook generator, and results aggregator.

Spec: S5.5
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import random
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

LORA_PARAMS = {"lora_r", "lora_alpha", "lora_dropout"}


def _load_base_config(base_config_path: str, model_type: str) -> dict:
    """Load base config and merge common + model-specific sections."""
    with open(base_config_path) as f:
        raw = yaml.safe_load(f)

    config = {}
    if "common" in raw:
        config.update(raw["common"])

    section = "whisper_small" if model_type == "whisper-small" else "whisper_large_v3"
    if section in raw:
        config.update(raw[section])
    return config


def _filter_search_space(search_space: dict, model_type: str) -> dict:
    """Remove LoRA params if model_type is whisper-small."""
    if model_type == "whisper-small":
        return {k: v for k, v in search_space.items() if k not in LORA_PARAMS}
    return dict(search_space)


def generate_configs(
    base_config_path: str,
    search_space: dict,
    strategy: str,
    max_trials: int,
    seed: int,
    model_type: str,
    output_dir: str,
) -> list[dict]:
    """Generate trial configs from a search space.

    Args:
        base_config_path: Path to base training config YAML.
        search_space: Dict of param name -> list of values.
        strategy: "grid" or "random".
        max_trials: Max number of trials to generate.
        seed: Random seed for reproducibility.
        model_type: "whisper-small" or "whisper-lora".
        output_dir: Directory to write trial YAML files.

    Returns:
        List of config dicts, one per trial.
    """
    if max_trials <= 0:
        raise ValueError("max_trials must be a positive integer")

    base = _load_base_config(base_config_path, model_type)
    space = _filter_search_space(search_space, model_type)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not space:
        trial = copy.deepcopy(base)
        trial_path = out / "trial_001.yaml"
        with open(trial_path, "w") as f:
            yaml.dump(trial, f)
        return [trial]

    param_names = sorted(space.keys())
    param_values = [space[k] for k in param_names]

    if strategy == "grid":
        combos = list(itertools.product(*param_values))
        combos = combos[:max_trials]
    elif strategy == "random":
        rng = random.Random(seed)
        combos = []
        for _ in range(max_trials):
            combo = tuple(rng.choice(vals) for vals in param_values)
            combos.append(combo)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    configs = []
    for i, combo in enumerate(combos):
        trial = copy.deepcopy(base)
        for name, val in zip(param_names, combo):
            trial[name] = val
        trial_path = out / f"trial_{i + 1:03d}.yaml"
        with open(trial_path, "w") as f:
            yaml.dump(trial, f)
        configs.append(trial)

    return configs


def generate_notebook(
    trial_id: str,
    config: dict,
    model_type: str,
    output_dir: str,
    kaggle_dataset_slug: str,
    kaggle_username: str = "nishantgaurav23",
) -> str:
    """Generate a Kaggle-compatible notebook for a single sweep trial.

    Returns path to the generated .ipynb file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_json = json.dumps(config, indent=2)

    train_script = (
        "src/train_whisper_small.py"
        if model_type == "whisper-small"
        else "src/train_whisper_lora.py"
    )

    cells = [
        _markdown_cell(
            f"# ChildWhisper Sweep — {trial_id}\n\nAuto-generated sweep trial."
        ),
        _code_cell(
            "import json, time, yaml, subprocess, sys\n"
            "from pathlib import Path\n\n"
            f"TRIAL_ID = {trial_id!r}\n"
            f"MODEL_TYPE = {model_type!r}\n"
            f"TRAIN_SCRIPT = {train_script!r}\n"
        ),
        _code_cell(
            f"CONFIG = json.loads({config_json!r})\n"
            "print('Trial config:', json.dumps(CONFIG, indent=2))\n"
        ),
        _code_cell(
            "# Write config to YAML for training script\n"
            "config_path = Path('/kaggle/working/trial_config.yaml')\n"
            "full_cfg = {'common': {}, 'whisper_small': CONFIG, 'whisper_large_v3': CONFIG}\n"
            "config_path.write_text(yaml.dump(full_cfg))\n"
        ),
        _code_cell(
            "# Run training\n"
            "t0 = time.time()\n"
            "try:\n"
            "    result = subprocess.run(\n"
            "        ['python', TRAIN_SCRIPT,\n"
            f"         '--metadata-path', '/kaggle/input/{kaggle_dataset_slug.split('/')[-1]}"
            "/train_word_transcripts.jsonl',\n"
            f"         '--audio-dir', '/kaggle/input/{kaggle_dataset_slug.split('/')[-1]}',\n"
            "         '--config', str(config_path),\n"
            "         '--output-dir', '/kaggle/working/checkpoints',\n"
            "         '--no-push-to-hub',\n"
            "         '--num-train-epochs', '1'],\n"
            "        capture_output=True, text=True, timeout=28800\n"
            "    )\n"
            "    duration = time.time() - t0\n"
            "    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)\n"
            "    if result.returncode != 0:\n"
            "        raise RuntimeError(result.stderr[-1000:])\n"
            "    # Parse WER from trainer output\n"
            "    import re\n"
            "    wer_match = re.search(r'eval_wer.*?([0-9.]+)', result.stdout)\n"
            "    val_wer = float(wer_match.group(1)) if wer_match else -1\n"
            "    loss_match = re.search(r\"'train_loss':\\s*([0-9.]+)\", result.stdout)\n"
            "    train_loss = float(loss_match.group(1)) if loss_match else -1\n"
            "    status = 'complete'\n"
            "    error = None\n"
            "except Exception as e:\n"
            "    duration = time.time() - t0\n"
            "    val_wer = -1\n"
            "    train_loss = -1\n"
            "    status = 'error'\n"
            "    error = str(e)\n"
        ),
        _code_cell(
            "# Write results\n"
            "results = {\n"
            "    'trial_id': TRIAL_ID,\n"
            "    'config': CONFIG,\n"
            "    'val_wer': val_wer,\n"
            "    'train_loss': train_loss,\n"
            "    'duration_sec': duration,\n"
            "    'status': status,\n"
            "    'error': error,\n"
            "}\n"
            "Path('/kaggle/working/results.json').write_text(json.dumps(results, indent=2))\n"
            "print(json.dumps(results, indent=2))\n"
        ),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    nb_filename = f"sweep_{trial_id}.ipynb"
    nb_path = out / nb_filename
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=2)

    # Also create kernel-metadata.json
    try:
        from src.kaggle_runner import create_kernel_metadata
    except ModuleNotFoundError:
        from kaggle_runner import create_kernel_metadata

    kernel_slug = f"childwhisper-sweep-{trial_id}"
    meta = create_kernel_metadata(
        kernel_slug=kernel_slug,
        notebook_filename=nb_filename,
        kaggle_username=kaggle_username,
        dataset_slugs=[kaggle_dataset_slug],
        title=f"ChildWhisper Sweep {trial_id.replace('_', ' ').title()}",
    )
    meta_path = out / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(nb_path)


def _markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": [],
        "execution_count": None,
    }


def aggregate_results(sweep_dir: str) -> tuple[str, str | None]:
    """Collect results.json from trial subdirs, rank by WER, output summary.

    Returns (csv_path, best_config_yaml_path or None).
    """
    root = Path(sweep_dir)
    results = []

    for results_file in sorted(root.rglob("results.json")):
        with open(results_file) as f:
            data = json.load(f)
        results.append(data)

    if not results:
        logger.warning("No trial results found in %s", sweep_dir)
        csv_path = root / "sweep_results.csv"
        csv_path.write_text("")
        return str(csv_path), None

    # Sort: successful trials by WER ascending, failed trials last
    successful = [r for r in results if r["val_wer"] >= 0]
    failed = [r for r in results if r["val_wer"] < 0]
    successful.sort(key=lambda r: r["val_wer"])
    sorted_results = successful + failed

    # Write CSV
    csv_path = root / "sweep_results.csv"
    fieldnames = [
        "trial_id",
        "val_wer",
        "train_loss",
        "duration_sec",
        "status",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Write best config
    best_cfg_path = None
    if successful:
        best = successful[0]
        best_cfg_path = root / "best_config.yaml"
        with open(best_cfg_path, "w") as f:
            yaml.dump(best["config"], f)
        logger.info("Best trial: %s (WER=%.4f)", best["trial_id"], best["val_wer"])
    else:
        logger.warning("No successful trials — cannot produce best_config.yaml")

    return str(csv_path), str(best_cfg_path) if best_cfg_path else None


def main():
    """CLI entrypoint for sweep operations."""
    parser = argparse.ArgumentParser(description="ChildWhisper hyperparameter sweep")
    sub = parser.add_subparsers(dest="command")

    # generate subcommand
    gen = sub.add_parser("generate", help="Generate sweep trial configs")
    gen.add_argument("--config", required=True, help="Base config YAML")
    gen.add_argument("--search-space", required=True, help="Search space YAML")
    gen.add_argument(
        "--model", required=True, choices=["whisper-small", "whisper-lora"]
    )
    gen.add_argument("--strategy", default="random", choices=["grid", "random"])
    gen.add_argument("--max-trials", type=int, default=10)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output-dir", default="configs/sweep_configs")

    # notebook subcommand
    nb = sub.add_parser("notebook", help="Generate Kaggle notebook for a trial")
    nb.add_argument("--trial-id", required=True)
    nb.add_argument("--trial-config", required=True, help="Trial config YAML path")
    nb.add_argument("--model", required=True, choices=["whisper-small", "whisper-lora"])
    nb.add_argument("--output-dir", default="notebooks/sweep")
    nb.add_argument("--kaggle-dataset", default="nishantgaurav23/pasketti-audio")
    nb.add_argument("--kaggle-username", default="nishantgaurav23")

    # aggregate subcommand
    agg = sub.add_parser("aggregate", help="Aggregate sweep results")
    agg.add_argument("sweep_dir", help="Directory containing trial results")

    args = parser.parse_args()

    if args.command == "generate":
        with open(args.search_space) as f:
            space_cfg = yaml.safe_load(f)
        search_space = space_cfg.get("search_space", {})
        configs = generate_configs(
            base_config_path=args.config,
            search_space=search_space,
            strategy=args.strategy,
            max_trials=args.max_trials,
            seed=args.seed,
            model_type=args.model,
            output_dir=args.output_dir,
        )
        print(f"Generated {len(configs)} trial configs in {args.output_dir}")

    elif args.command == "notebook":
        with open(args.trial_config) as f:
            config = yaml.safe_load(f)
        nb_path = generate_notebook(
            trial_id=args.trial_id,
            config=config,
            model_type=args.model,
            output_dir=args.output_dir,
            kaggle_dataset_slug=args.kaggle_dataset,
            kaggle_username=args.kaggle_username,
        )
        print(f"Generated notebook: {nb_path}")

    elif args.command == "aggregate":
        csv_path, best_path = aggregate_results(args.sweep_dir)
        print(f"Results: {csv_path}")
        if best_path:
            print(f"Best config: {best_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
