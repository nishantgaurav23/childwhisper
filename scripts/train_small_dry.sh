#!/usr/bin/env bash
# Dry-run Whisper-small training for local testing on MacBook
# Usage:
#   ./scripts/train_small_dry.sh
#   ./scripts/train_small_dry.sh --num-train-epochs 5
#   ./scripts/train_small_dry.sh --output-dir output/experiment1 --num-train-epochs 2
#
# Any extra arguments are forwarded to train_whisper_small.py
set -euo pipefail

cd "$(dirname "$0")/.."

python src/train_whisper_small.py \
    --metadata-path data/train_word_transcripts.jsonl \
    --audio-dir data \
    --config configs/training_config.yaml \
    --output-dir output/whisper-small-dry \
    --dry-run \
    --no-push-to-hub \
    "$@"
