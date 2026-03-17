#!/usr/bin/env bash
# Pull output from a completed Kaggle kernel
# Usage: ./scripts/kaggle_pull.sh nishantgaurav23/childwhisper-sweep-trial-001 -o output/sweep/trial_001
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "Usage: $0 <kernel-slug> [-o output-dir]"
    exit 1
fi

python src/kaggle_runner.py pull "$@"
