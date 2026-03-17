#!/bin/bash
set -euo pipefail
#
# Submit distributed miradata preprocessing to Slurm.
#
# Phase 1: Split 125 input zips across N Slurm jobs on cpu_short (4h, up to 960 CPUs).
#           Each job processes its assigned zips with local multi-threading.
# Phase 2: After all Phase 1 jobs finish, one job packages staging → final output zips.
#
# Usage:  bash slurm/preprocess/submit_all.sh [ZIPS_PER_JOB]
#         ZIPS_PER_JOB defaults to 5.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/miradata"
OUTPUT_DIR="$PROJECT_DIR/miradata_processed"
LISTS_DIR="$SCRIPT_DIR/zip_lists"
LOG_DIR="$SCRIPT_DIR/logs"

ZIPS_PER_JOB="${1:-5}"

mkdir -p "$LISTS_DIR" "$LOG_DIR" "$OUTPUT_DIR/.staging"

# ── Clean up old incomplete staging (no manifest = was interrupted) ──
echo "Cleaning incomplete staging zips..."
cleaned=0
for z in "$OUTPUT_DIR/.staging/"*.zip; do
    [ -f "$z" ] || continue
    base="$(basename "$z" .zip)"
    if [ ! -f "$OUTPUT_DIR/.staging/${base}_manifest.json" ]; then
        rm -f "$z"
        cleaned=$((cleaned + 1))
    fi
done
echo "  Removed $cleaned incomplete staging zips"

# ── Discover input zips ──
mapfile -t ALL_ZIPS < <(ls "$INPUT_DIR"/*.zip 2>/dev/null | xargs -n1 basename | sort)
TOTAL=${#ALL_ZIPS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No zip files found in $INPUT_DIR"
    exit 1
fi
echo "Found $TOTAL input zips"

# ── Split into groups ──
rm -f "$LISTS_DIR"/group_*.txt
group_idx=0
for ((i = 0; i < TOTAL; i += ZIPS_PER_JOB)); do
    list_file="$LISTS_DIR/group_$(printf '%04d' $group_idx).txt"
    for ((j = i; j < i + ZIPS_PER_JOB && j < TOTAL; j++)); do
        echo "${ALL_ZIPS[$j]}" >> "$list_file"
    done
    group_idx=$((group_idx + 1))
done
NUM_GROUPS=$group_idx
echo "Split into $NUM_GROUPS groups ($ZIPS_PER_JOB zips/group)"

# ── Submit Phase 1 jobs ──
PHASE1_IDS=()
for ((g = 0; g < NUM_GROUPS; g++)); do
    gid="$(printf '%04d' $g)"
    list_file="$LISTS_DIR/group_${gid}.txt"
    n_zips=$(wc -l < "$list_file")
    workers=1  # sequential: one zip at a time to minimize Lustre I/O contention

    job_id=$(sbatch \
        --parsable \
        --account=nvr_elm_llm \
        --partition=cpu_short \
        --time=04:00:00 \
        --nodes=1 \
        --cpus-per-task=8 \
        --mem=32G \
        --signal=B:USR1@120 \
        --open-mode=append \
        --job-name="mira-prep-${gid}" \
        --output="$LOG_DIR/phase1_${gid}_%j.out" \
        --error="$LOG_DIR/phase1_${gid}_%j.err" \
        --export=ALL \
        "$SCRIPT_DIR/job_phase1.sh" "$list_file" "$workers"
    )
    PHASE1_IDS+=("$job_id")
    echo "  Phase 1 group $gid ($n_zips zips, $workers workers): job $job_id"
done

# ── Submit Phase 2 job (depends on all Phase 1) ──
dep_str=$(IFS=:; echo "${PHASE1_IDS[*]}")

echo ""
echo "NOTE: Phase 1 jobs auto-requeue if they hit the 4h limit."
echo "      Phase 2 must be submitted manually after ALL Phase 1 jobs finish."
echo "      Run:  sbatch --account=nvr_elm_llm --partition=cpu_short --time=04:00:00 \\"
echo "                   --nodes=1 --cpus-per-task=8 --mem=64G \\"
echo "                   --job-name=mira-prep-pkg \\"
echo "                   --output=$LOG_DIR/phase2_%j.out --error=$LOG_DIR/phase2_%j.err \\"
echo "                   $SCRIPT_DIR/job_phase2.sh"
echo ""
echo "Monitor:  squeue -u \$(whoami) | grep mira-prep"
echo "Logs:     ls $LOG_DIR/"
