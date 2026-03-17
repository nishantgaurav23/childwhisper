#!/usr/bin/env bash
# Package submission.zip for DrivenData upload
set -euo pipefail

OUTPUT="${1:-submission.zip}"

echo "=== ChildWhisper Submission Builder ==="

# Verify required files exist
if [ ! -f "submission/main.py" ]; then
    echo "ERROR: submission/main.py not found"
    exit 1
fi

if [ ! -d "submission/model_weights" ]; then
    echo "WARNING: submission/model_weights/ not found — weights may not be downloaded yet"
fi

# Build zip
echo "Creating ${OUTPUT}..."
cd submission
zip -r "../${OUTPUT}" . -x "__pycache__/*" "*.pyc" ".DS_Store"
cd ..

# Report size
SIZE=$(du -h "${OUTPUT}" | cut -f1)
echo ""
echo "Submission package: ${OUTPUT} (${SIZE})"
echo "Upload at: https://www.drivendata.org/competitions/308/childrens-word-asr/"
echo ""
echo "Done."
