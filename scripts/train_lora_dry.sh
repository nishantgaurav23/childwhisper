#!/usr/bin/env bash
# Dry-run Whisper-large-v3 LoRA training (1 step) for local testing on MacBook
set -euo pipefail

cd "$(dirname "$0")/.."

python src/train_whisper_lora.py \
    --metadata-path data/train_word_transcripts.jsonl \
    --audio-dir data/audio_sample \
    --config configs/training_config.yaml \
    --output-dir output/whisper-lora-dry \
    --dry-run \
    --no-push-to-hub
