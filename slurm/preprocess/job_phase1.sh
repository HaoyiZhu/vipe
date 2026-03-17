#!/bin/bash
#
# Phase 1 worker: process a subset of input zips into staging.
# Called by submit_all.sh with: job_phase1.sh <zip_list_file> <num_workers>
#
# Self-resubmit: if not all zips finish within the time limit, USR1 is
# caught 120s before timeout; we stop Python and resubmit the same job.
# On the next run, --resume skips finished zips.
#

ZIP_LIST="$1"
NUM_WORKERS="${2:-1}"

export MYHOME="$HOME"
MINIFORGE_ROOT="$MYHOME/miniforge3"
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"
conda activate vipe

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$MYHOME/projects/vipe}"
INPUT_DIR="$PROJECT_DIR/miradata"
OUTPUT_DIR="$PROJECT_DIR/miradata_processed"
LOG_DIR="$PROJECT_DIR/slurm/preprocess/logs"
STAGING_DIR="$OUTPUT_DIR/.staging"

# Use local SSD for temp files instead of Lustre
LOCAL_TMP="/tmp/miradata_prep_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL_TMP"

PY_PID=""

cleanup_and_resubmit() {
    echo "$(date): USR1 – time limit approaching"
    if [ -n "$PY_PID" ] && kill -0 "$PY_PID" 2>/dev/null; then
        kill -TERM "$PY_PID"
        wait "$PY_PID" 2>/dev/null || true
    fi
    rm -rf "$LOCAL_TMP"

    # Check if all assigned zips are done
    all_done=true
    while IFS= read -r zipname || [ -n "$zipname" ]; do
        [ -z "$zipname" ] && continue
        base="${zipname%.zip}"
        if [ ! -f "$STAGING_DIR/${base}_manifest.json" ]; then
            all_done=false
            break
        fi
    done < "$ZIP_LIST"

    if [ "$all_done" = true ]; then
        echo "$(date): All zips in this group are done, no resubmit needed"
    else
        # Clean incomplete staging zips before resubmit
        while IFS= read -r zipname || [ -n "$zipname" ]; do
            [ -z "$zipname" ] && continue
            base="${zipname%.zip}"
            if [ ! -f "$STAGING_DIR/${base}_manifest.json" ] && [ -f "$STAGING_DIR/${base}.zip" ]; then
                rm -f "$STAGING_DIR/${base}.zip"
                echo "  Cleaned incomplete: ${base}.zip"
            fi
        done < "$ZIP_LIST"

        gid="${SLURM_JOB_NAME##*-}"
        echo "$(date): Resubmitting group $gid..."
        sbatch \
            --account=nvr_elm_llm \
            --partition=cpu_short \
            --time=04:00:00 \
            --nodes=1 \
            --cpus-per-task=8 \
            --mem=32G \
            --signal=B:USR1@120 \
            --open-mode=append \
            --job-name="${SLURM_JOB_NAME}" \
            --output="$LOG_DIR/phase1_${gid}_%j.out" \
            --error="$LOG_DIR/phase1_${gid}_%j.err" \
            --export=ALL \
            "$PROJECT_DIR/slurm/preprocess/job_phase1.sh" "$ZIP_LIST" "$NUM_WORKERS"
    fi
    exit 0
}
trap cleanup_and_resubmit USR1

echo "=== Phase 1 worker starting ($(date)) ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Zip list: $ZIP_LIST"
echo "Workers: $NUM_WORKERS"
echo "Temp dir: $LOCAL_TMP"
echo "Zips to process:"
cat "$ZIP_LIST"
echo ""

python "$PROJECT_DIR/scripts/preprocess_miradata.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$NUM_WORKERS" \
    --phase 1 \
    --zip-list "$ZIP_LIST" \
    --tmp-dir "$LOCAL_TMP" \
    --resume &
PY_PID=$!
wait "$PY_PID"
PY_EXIT=$?

rm -rf "$LOCAL_TMP"
echo "=== Phase 1 worker done ($(date), exit=$PY_EXIT) ==="
exit $PY_EXIT
