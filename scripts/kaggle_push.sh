#!/usr/bin/env bash
# Push a notebook directory to Kaggle
# Usage: ./scripts/kaggle_push.sh notebooks/sweep/trial_001
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "Usage: $0 <notebook-dir>"
    echo "  notebook-dir: Directory containing kernel-metadata.json + .ipynb"
    exit 1
fi

python src/kaggle_runner.py push "$1"
