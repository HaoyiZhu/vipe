#!/bin/bash
#SBATCH -A nvr_elm_llm
#SBATCH -p cpu_long
#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -J mira-pack
#SBATCH --array=0-12
#SBATCH -o slurm/preprocess/logs/phase2_pack_%A_%a.out
#SBATCH -e slurm/preprocess/logs/phase2_pack_%A_%a.err

export MYHOME="$HOME"
MINIFORGE_ROOT="$MYHOME/miniforge3"
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
conda activate vipe

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$MYHOME/projects/vipe}"
MANIFEST_DIR="$PROJECT_DIR/miradata_processed/phase2_manifests"
OUTPUT_DIR="$PROJECT_DIR/miradata_processed"

echo "=== Phase 2 pack starting ($(date)) ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Zip index: ${SLURM_ARRAY_TASK_ID}"

python "$PROJECT_DIR/scripts/phase2_pack_one.py" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --zip-index "${SLURM_ARRAY_TASK_ID}"

echo "=== Phase 2 pack done ($(date)) ==="
