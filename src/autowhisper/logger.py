"""Results TSV logger for AutoWhisper experiment loop."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TSV_FIELDS = [
    "experiment_id",
    "commit_hash",
    "val_wer",
    "peak_vram_mb",
    "duration_sec",
    "status",
    "description",
]


def init_log(path: str) -> None:
    """Create a new results TSV file with header."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("\t".join(TSV_FIELDS) + "\n")


def append_result(path: str, result: dict) -> None:
    """Append one experiment result row to the TSV file."""
    with open(path, "a", newline="") as f:
        row = "\t".join(str(result[field]) for field in TSV_FIELDS)
        f.write(row + "\n")


def load_results(path: str) -> list[dict]:
    """Read all results from TSV, parsing types correctly."""
    results = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(
                {
                    "experiment_id": row["experiment_id"],
                    "commit_hash": row["commit_hash"],
                    "val_wer": float(row["val_wer"]),
                    "peak_vram_mb": int(float(row["peak_vram_mb"])),
                    "duration_sec": int(float(row["duration_sec"])),
                    "status": row["status"],
                    "description": row["description"],
                }
            )
    return results


def get_best_wer(path: str) -> float:
    """Return lowest val_wer from 'keep' or 'baseline' rows. Returns inf if none."""
    results = load_results(path)
    valid = [
        r["val_wer"]
        for r in results
        if r["status"] in ("keep", "baseline") and r["val_wer"] >= 0
    ]
    return min(valid) if valid else float("inf")


def get_frontier(path: str) -> list[dict]:
    """Return monotonically-improving results (the improvement frontier).

    Considers 'keep' and 'baseline' rows in order, keeping only those
    that improve upon the previous best.
    """
    results = load_results(path)
    frontier = []
    best_so_far = float("inf")
    for r in results:
        if r["status"] in ("keep", "baseline") and r["val_wer"] >= 0:
            if r["val_wer"] < best_so_far:
                best_so_far = r["val_wer"]
                frontier.append(r)
    return frontier


def print_summary(path: str) -> None:
    """Print experiment summary to stdout."""
    results = load_results(path)
    total = len(results)
    keeps = sum(1 for r in results if r["status"] == "keep")
    discards = sum(1 for r in results if r["status"] == "discard")
    crashes = sum(1 for r in results if r["status"] == "crash")
    baselines = sum(1 for r in results if r["status"] == "baseline")
    best = get_best_wer(path)
    total_time = sum(r["duration_sec"] for r in results)

    baseline_wer = None
    for r in results:
        if r["status"] == "baseline":
            baseline_wer = r["val_wer"]
            break

    print("AutoWhisper Experiment Summary")
    print(f"{'=' * 40}")
    print(f"Total experiments: {total}")
    print(f"  Baseline: {baselines}")
    print(f"  Keep: {keeps}")
    print(f"  Discard: {discards}")
    print(f"  Crash: {crashes}")
    print(f"Best WER: {best:.4f}")
    if baseline_wer is not None:
        improvement = baseline_wer - best
        print(f"Improvement over baseline: {improvement:.4f} ({improvement / baseline_wer * 100:.1f}%)")
    print(f"Total GPU time: {total_time // 60}m {total_time % 60}s")


def plot_progress(results_path: str, output_path: str) -> None:
    """Scatter plot of experiments colored by status, with frontier line overlay."""
    results = load_results(results_path)

    colors = {"keep": "green", "discard": "red", "crash": "gray", "baseline": "blue"}

    fig, ax = plt.subplots(figsize=(12, 6))

    for status, color in colors.items():
        subset = [r for r in results if r["status"] == status]
        if subset:
            x = [int(r["experiment_id"]) for r in subset]
            y = [r["val_wer"] for r in subset]
            ax.scatter(x, y, c=color, label=status, s=60, alpha=0.7)

    # Overlay frontier line
    frontier = get_frontier(results_path)
    if frontier:
        fx = [int(r["experiment_id"]) for r in frontier]
        fy = [r["val_wer"] for r in frontier]
        ax.plot(fx, fy, "k--", linewidth=1.5, label="frontier", alpha=0.8)

    ax.set_xlabel("Experiment ID")
    ax.set_ylabel("Validation WER")
    ax.set_title("AutoWhisper Experiment Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
