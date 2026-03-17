#!/usr/bin/env bash
# Download model weights from HuggingFace Hub
# Prerequisites: huggingface-cli authenticated
set -euo pipefail

WEIGHTS_DIR="${1:-submission/model_weights}"

echo "=== ChildWhisper Weight Download ==="
echo "Target directory: ${WEIGHTS_DIR}"

mkdir -p "${WEIGHTS_DIR}"

echo ""
echo "Downloading LoRA adapter for Whisper-large-v3..."
echo "  huggingface-cli download nishantgaurav23/pasketti-whisper-lora --local-dir ${WEIGHTS_DIR}/lora_large_v3"
echo ""
echo "Downloading fine-tuned Whisper-small..."
echo "  huggingface-cli download nishantgaurav23/pasketti-whisper-small --local-dir ${WEIGHTS_DIR}/whisper_small_ft"
echo ""
echo "NOTE: Run the above commands manually after models are trained and uploaded."
echo ""
echo "Done."
