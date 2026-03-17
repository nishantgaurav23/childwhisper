"""CLI entry point for AutoWhisper runner.

Usage:
    python -m src.autowhisper init --tag run_mar17
    python -m src.autowhisper run --tag run_mar17 --train src/autowhisper/train.py
    python -m src.autowhisper revert --tag run_mar17
    python -m src.autowhisper summary --results results/autowhisper/run_mar17/results.tsv
    python -m src.autowhisper plot --results results/autowhisper/run_mar17/results.tsv
"""

import argparse
import os
import sys

from src.autowhisper.logger import init_log, get_best_wer, print_summary, plot_progress
from src.autowhisper.runner import (
    evaluate_and_decide,
    init_run,
    keep_experiment,
    log_result,
    revert_experiment,
    run_experiment,
)


def get_commit_hash() -> str:
    """Get short git commit hash."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def get_experiment_id(log_path: str) -> str:
    """Get next experiment ID from log file."""
    if not os.path.exists(log_path):
        return "000"
    with open(log_path) as f:
        lines = f.readlines()
    return f"{len(lines) - 1:03d}"  # subtract header


def cmd_init(args):
    results_dir = f"results/autowhisper/{args.tag}"
    os.makedirs(results_dir, exist_ok=True)
    log_path = f"{results_dir}/results.tsv"

    branch = init_run(args.tag, base_branch=args.base_branch)
    init_log(log_path)
    print(f"Initialized AutoWhisper run '{args.tag}' on branch {branch}")
    print(f"Results log: {log_path}")


def cmd_run(args):
    results_dir = f"results/autowhisper/{args.tag}"
    log_path = f"{results_dir}/results.tsv"

    if not os.path.exists(log_path):
        print(f"Error: Run '{args.tag}' not initialized. Run 'init' first.")
        sys.exit(1)

    exp_id = get_experiment_id(log_path)
    print(f"Running experiment {exp_id}: {args.description}")

    result = run_experiment(args.train, time_budget=args.budget)
    best_wer = get_best_wer(log_path)
    decision = evaluate_and_decide(result, best_wer)

    if decision == "keep":
        keep_experiment(args.description)
        commit = get_commit_hash()
        print(f"KEEP — val_wer: {result['val_wer']:.4f} (improved from {best_wer:.4f})")
    elif decision == "discard":
        revert_experiment()
        commit = "reverted"
        print(f"DISCARD — val_wer: {result['val_wer']:.4f} (best: {best_wer:.4f})")
    else:
        revert_experiment()
        commit = "crashed"
        print(f"CRASH — {result.get('stderr', 'unknown error')[:100]}")

    log_result(
        result=result,
        decision=decision,
        description=args.description,
        experiment_id=exp_id,
        commit_hash=commit,
        log_path=log_path,
    )


def cmd_revert(args):
    revert_experiment()
    print("Reverted last experiment changes.")


def cmd_summary(args):
    print_summary(args.results)


def cmd_plot(args):
    output = args.output or args.results.replace(".tsv", "_progress.png")
    plot_progress(args.results, output)
    print(f"Plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="AutoWhisper experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new experiment run")
    p_init.add_argument("--tag", required=True, help="Run tag (e.g., run_mar17)")
    p_init.add_argument("--base-branch", default="main", help="Base branch")

    # run
    p_run = subparsers.add_parser("run", help="Run one experiment")
    p_run.add_argument("--tag", required=True, help="Run tag")
    p_run.add_argument("--train", default="src/autowhisper/train.py", help="Training script")
    p_run.add_argument("--budget", type=int, default=900, help="Time budget (sec)")
    p_run.add_argument("--description", default="Experiment", help="One-line description")

    # revert
    p_revert = subparsers.add_parser("revert", help="Revert last experiment")
    p_revert.add_argument("--tag", required=True, help="Run tag")

    # summary
    p_summary = subparsers.add_parser("summary", help="Print results summary")
    p_summary.add_argument("--results", required=True, help="Path to results.tsv")

    # plot
    p_plot = subparsers.add_parser("plot", help="Generate progress plot")
    p_plot.add_argument("--results", required=True, help="Path to results.tsv")
    p_plot.add_argument("--output", help="Output PNG path")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "revert": cmd_revert,
        "summary": cmd_summary,
        "plot": cmd_plot,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
