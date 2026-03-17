#!/usr/bin/env bash
# Download competition data from DrivenData/Kaggle
# Prerequisites: kaggle CLI authenticated, DrivenData account
set -euo pipefail

DATA_DIR="${1:-data}"

echo "=== ChildWhisper Data Download ==="
echo "Target directory: ${DATA_DIR}"

mkdir -p "${DATA_DIR}/audio_sample"

echo ""
echo "Manual steps required:"
echo "1. Download audio zips from DrivenData competition page"
echo "2. Download train_word_transcripts.jsonl from DrivenData"
echo "3. Request TalkBank access and download additional audio"
echo "4. Extract audio files into ${DATA_DIR}/"
echo "5. Copy ~100 random FLAC files to ${DATA_DIR}/audio_sample/ for local testing"
echo ""
echo "Competition URL: https://www.drivendata.org/competitions/308/childrens-word-asr/"
echo ""
echo "Done."
