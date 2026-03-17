#!/usr/bin/env bash
# Sync data from Google Drive to a Kaggle Dataset (one-time setup).
#
# Prerequisites:
#   1. pip install gdown kaggle
#   2. kaggle.json configured (~/.kaggle/kaggle.json)
#   3. Google Drive folder shared with "Anyone with the link"
#
# Usage:
#   ./scripts/gdrive_to_kaggle_dataset.sh <GDRIVE_FOLDER_ID>
#
# This downloads from Google Drive → local staging → uploads as Kaggle Dataset.
# After this, notebooks access data at /kaggle/input/pasketti-audio/
set -euo pipefail

GDRIVE_FOLDER_ID="${1:?Usage: $0 <GDRIVE_FOLDER_ID>}"
KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-nishantgaurav23/pasketti-audio}"
STAGING_DIR="${STAGING_DIR:-/tmp/childwhisper-kaggle-dataset}"

echo "=== Google Drive → Kaggle Dataset Sync ==="
echo "Drive folder: $GDRIVE_FOLDER_ID"
echo "Kaggle dataset: $KAGGLE_DATASET_SLUG"
echo "Staging dir: $STAGING_DIR"
echo ""

# Step 1: Download from Google Drive
echo "Step 1/4: Downloading from Google Drive..."
mkdir -p "$STAGING_DIR"
python -c "
from src.gdrive_utils import download_folder
download_folder('${GDRIVE_FOLDER_ID}', '${STAGING_DIR}')
print('Download complete.')
"

# Step 2: Validate downloaded data
echo ""
echo "Step 2/4: Validating data..."
if [ ! -f "$STAGING_DIR/train_word_transcripts.jsonl" ]; then
    echo "ERROR: train_word_transcripts.jsonl not found in downloaded data"
    echo "Contents of $STAGING_DIR:"
    ls -la "$STAGING_DIR/"
    exit 1
fi

AUDIO_COUNT=$(find "$STAGING_DIR" -name "*.flac" -o -name "*.wav" | wc -l | tr -d ' ')
echo "Found: train_word_transcripts.jsonl + $AUDIO_COUNT audio files"

if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "WARNING: No audio files found. Check your Google Drive folder structure."
    echo "Expected: folder/audio/*.flac (or .wav)"
fi

# Step 3: Create Kaggle dataset metadata
echo ""
echo "Step 3/4: Creating Kaggle dataset metadata..."

DATASET_OWNER=$(echo "$KAGGLE_DATASET_SLUG" | cut -d'/' -f1)
DATASET_NAME=$(echo "$KAGGLE_DATASET_SLUG" | cut -d'/' -f2)

cat > "$STAGING_DIR/dataset-metadata.json" << EOF
{
  "title": "Pasketti Children ASR Audio",
  "id": "$KAGGLE_DATASET_SLUG",
  "licenses": [{"name": "CC-BY-4.0"}]
}
EOF

# Step 4: Upload to Kaggle
echo ""
echo "Step 4/4: Uploading to Kaggle..."

# Check if dataset already exists
if kaggle datasets status "$KAGGLE_DATASET_SLUG" 2>/dev/null; then
    echo "Dataset exists, creating new version..."
    kaggle datasets version -p "$STAGING_DIR" -m "Updated from Google Drive" --dir-mode zip
else
    echo "Creating new dataset..."
    kaggle datasets create -p "$STAGING_DIR" --dir-mode zip
fi

echo ""
echo "=== Done! ==="
echo "Dataset available at: https://www.kaggle.com/datasets/$KAGGLE_DATASET_SLUG"
echo ""
echo "In your Kaggle notebook, data will be at:"
echo "  /kaggle/input/$DATASET_NAME/audio/"
echo "  /kaggle/input/$DATASET_NAME/train_word_transcripts.jsonl"
echo ""
echo "To clean up staging dir: rm -rf $STAGING_DIR"
