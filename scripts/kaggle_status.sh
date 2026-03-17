#!/usr/bin/env bash
# Check status of a Kaggle kernel
# Usage: ./scripts/kaggle_status.sh nishantgaurav23/childwhisper-sweep-trial-001
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "Usage: $0 <kernel-slug>"
    exit 1
fi

python src/kaggle_runner.py status "$1"
