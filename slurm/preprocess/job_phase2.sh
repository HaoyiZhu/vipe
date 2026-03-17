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
n_manifests=$(ls "$OUTPUT_DIR/.staging/"*_manifest.json 2>/dev/null | wc -l)
echo "Staging manifests found: $n_manifests"

python "$PROJECT_DIR/scripts/preprocess_miradata.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --phase 2

echo "=== Phase 2 packaging done ==="
