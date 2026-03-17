#!/usr/bin/env bash
# End-to-end hyperparameter sweep for Whisper-large-v3 LoRA via Kaggle API
# Usage:
#   ./scripts/sweep_whisper_lora.sh
#   ./scripts/sweep_whisper_lora.sh --max-trials 5 --strategy grid
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_TRIALS="${MAX_TRIALS:-10}"
STRATEGY="${STRATEGY:-random}"
SEED="${SEED:-42}"
KAGGLE_DATASET="${KAGGLE_DATASET:-nishantgaurav23/pasketti-audio}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-nishantgaurav23}"
OUTPUT_DIR="output/sweep/whisper-lora"
POLL_INTERVAL=60

echo "=== ChildWhisper Sweep: whisper-lora ==="
echo "Trials: $MAX_TRIALS | Strategy: $STRATEGY | Seed: $SEED"

# Step 1: Generate configs
python src/sweep.py generate \
    --config configs/training_config.yaml \
    --search-space configs/sweep_space.yaml \
    --model whisper-lora \
    --strategy "$STRATEGY" \
    --max-trials "$MAX_TRIALS" \
    --seed "$SEED" \
    --output-dir configs/sweep_configs

# Step 2: Generate notebooks and push to Kaggle
mkdir -p "$OUTPUT_DIR"
KERNEL_SLUGS=()
for cfg in configs/sweep_configs/trial_*.yaml; do
    trial_id=$(basename "$cfg" .yaml)
    nb_dir="notebooks/sweep/${trial_id}"
    mkdir -p "$nb_dir"

    python src/sweep.py notebook \
        --trial-id "$trial_id" \
        --trial-config "$cfg" \
        --model whisper-lora \
        --output-dir "$nb_dir" \
        --kaggle-dataset "$KAGGLE_DATASET" \
        --kaggle-username "$KAGGLE_USERNAME"

    echo "Pushing $trial_id to Kaggle..."
    python src/kaggle_runner.py push "$nb_dir"

    slug="${KAGGLE_USERNAME}/childwhisper-sweep-${trial_id}"
    KERNEL_SLUGS+=("$slug")
done

echo ""
echo "=== All ${#KERNEL_SLUGS[@]} trials pushed ==="
echo "Polling for completion every ${POLL_INTERVAL}s..."
echo "Press Ctrl+C to stop polling (kernels will continue on Kaggle)"
echo ""

trap 'echo ""; echo "Polling stopped. Kernels still running on Kaggle."; exit 0' INT

all_done=false
while [ "$all_done" = false ]; do
    all_done=true
    for slug in "${KERNEL_SLUGS[@]}"; do
        status=$(python src/kaggle_runner.py status "$slug" 2>/dev/null | sed -n 's/.*Status: \([a-zA-Z]*\).*/\1/p' || echo "unknown")
        [ -z "$status" ] && status="unknown"
        echo "  $slug: $status"
        if [ "$status" != "complete" ] && [ "$status" != "error" ]; then
            all_done=false
        fi
    done
    if [ "$all_done" = false ]; then
        echo "  Waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    fi
done

# Pull results
echo ""
echo "=== Pulling results ==="
for slug in "${KERNEL_SLUGS[@]}"; do
    trial_id=$(echo "$slug" | sed 's/.*childwhisper-sweep-//')
    python src/kaggle_runner.py pull "$slug" -o "$OUTPUT_DIR/$trial_id"
done

# Aggregate
echo ""
echo "=== Aggregating results ==="
python src/sweep.py aggregate "$OUTPUT_DIR"

echo ""
echo "Done! Results in $OUTPUT_DIR/sweep_results.csv"
echo "Best config in $OUTPUT_DIR/best_config.yaml"
