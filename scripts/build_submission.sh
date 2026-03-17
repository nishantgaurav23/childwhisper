#!/usr/bin/env bash
# Package submission.zip for DrivenData upload.
# Copies required src/ files into submission/, validates, and builds zip.
#
# Usage:
#   ./scripts/build_submission.sh [--output FILE] [--dry-run]
#
# Spec: S3.4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
OUTPUT="${PROJECT_ROOT}/submission.zip"
DRY_RUN="False"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="True"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--output FILE] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --output FILE   Output zip path (default: submission.zip)"
            echo "  --dry-run       Validate without building"
            exit 0
            ;;
        *)
            # Legacy positional arg for output path
            OUTPUT="$1"
            shift
            ;;
    esac
done

echo "=== ChildWhisper Submission Builder ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Output: ${OUTPUT}"
echo ""

cd "${PROJECT_ROOT}"

# Bundle src/ into submission/ (competition runtime needs it alongside main.py)
echo "Bundling src/ into submission/..."
mkdir -p submission/src
cp src/__init__.py submission/src/ 2>/dev/null || touch submission/src/__init__.py
cp src/preprocess.py submission/src/
cp src/utils.py submission/src/
echo "  Copied preprocess.py, utils.py"

# Run Python builder
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.submission_builder import (
    validate_submission_dir,
    compute_size_budget,
    get_submission_manifest,
    build_submission_zip,
)

sub_dir = Path('submission')
dry_run = ${DRY_RUN}
output = Path('${OUTPUT}')

# Validate
print('Validating submission directory...')
errors = validate_submission_dir(sub_dir)
hard_errors = [e for e in errors if 'model_weights' not in e]
warnings = [e for e in errors if 'model_weights' in e]

for w in warnings:
    print(f'  WARNING: {w}')
if hard_errors:
    for e in hard_errors:
        print(f'  ERROR: {e}')
    sys.exit(1)
print('  Structure OK')

# Size budget
budget = compute_size_budget(sub_dir)
print()
print('Size budget:')
print(f'  Code:    {budget[\"code_bytes\"]:>12,} bytes')
print(f'  Weights: {budget[\"weights_bytes\"]:>12,} bytes')
print(f'  Total:   {budget[\"total_human\"]}')
if budget['warning']:
    print(f'  WARNING: {budget[\"warning\"]}')

# Manifest
manifest = get_submission_manifest(sub_dir)
print()
print(f'Files: {len(manifest)}')

if dry_run:
    print()
    print('Dry run — no zip created.')
    print('Manifest:')
    for entry in manifest:
        print(f'  {entry[\"size\"]:>10,}  {entry[\"path\"]}')
    sys.exit(0)

# Build
print()
print('Building zip...')
result = build_submission_zip(sub_dir, output)
import os
size = os.path.getsize(result)
print(f'Created: {result} ({size:,} bytes)')
print()
print('Upload at: https://www.drivendata.org/competitions/308/childrens-word-asr/')
print('Done.')
"
