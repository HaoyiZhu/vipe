#!/bin/bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch_singlenode,batch_block1,batch_block3,batch_block4,backfill_singlenode,backfill_block1,backfill_block3,backfill_block4
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --nice=100
#SBATCH -J scene_cut_stg
#SBATCH --array=0-117%60
#SBATCH -o slurm/preprocess/logs/scene_cut_%A_%a.out
#SBATCH -e slurm/preprocess/logs/scene_cut_%A_%a.err

export MYHOME="$HOME"
MINIFORGE_ROOT="$MYHOME/miniforge3"
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
conda activate sana

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$MYHOME/projects/vipe}"
STAGING_ROOT="$PROJECT_DIR/miradata_processed/.staging"

echo "=== Scene cut annotation starting ($(date)) ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python "$PROJECT_DIR/scripts/scene_cut_staging.py" \
    --staging-root "$STAGING_ROOT" \
    --job-id "${SLURM_ARRAY_TASK_ID}" \
    --threshold 0.5 \
    --device cuda \
    --save-interval 50 \
    --fresh

echo "=== Scene cut annotation done ($(date)) ==="
