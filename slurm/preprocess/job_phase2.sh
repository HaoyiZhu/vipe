#!/bin/bash
#
# Phase 2: package all staging zips into final output zips (max 10k clips each).
# Runs after all Phase 1 jobs complete.
#

export MYHOME="$HOME"
MINIFORGE_ROOT="$MYHOME/miniforge3"
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
conda activate vipe

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$MYHOME/projects/vipe}"
INPUT_DIR="$PROJECT_DIR/miradata"
OUTPUT_DIR="$PROJECT_DIR/miradata_processed"

echo "=== Phase 2 packaging starting ==="
echo "Host: $(hostname)"

# Check how many staging manifests exist
n_manifests=$(find "$OUTPUT_DIR/.staging" -name "_manifest.json" 2>/dev/null | wc -l)
echo "Staging manifests found: $n_manifests"

python "$PROJECT_DIR/scripts/preprocess_miradata.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --phase 2 \
    --p2-dover-min 0.35 \
    --p2-saturation-min 0.0 --p2-saturation-max 180.0 \
    --p2-vmafmotion-min 0.5 --p2-vmafmotion-max 50.0 \
    --p2-unimatch-min 3.0 --p2-unimatch-max 50.0

echo "=== Phase 2 packaging done ==="
