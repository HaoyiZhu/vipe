#!/usr/bin/env bash
# Monitor S3 uploads and delete local files confirmed as uploaded.
#
# Runs as a long-lived Slurm job on cpu_datamover. Periodically compares
# local files against S3 using rclone, and deletes local files whose S3
# counterpart exists with matching size.
#
# Usage:
#   # Monitor and clean up SpatialVID-HQ (dry-run first!)
#   bash scripts/s3_cleanup_monitor.sh \
#       --src ~/data/SpatialVID-HQ --s3-prefix SpatialVID-HQ --dry-run
#
#   # Actually delete confirmed uploads, check every 10 minutes
#   bash scripts/s3_cleanup_monitor.sh \
#       --src ~/data/SpatialVID-HQ --s3-prefix SpatialVID-HQ --interval 600
#
#   # Monitor both directories (run two instances)
#   bash scripts/s3_cleanup_monitor.sh \
#       --src ~/data/SpatialVID-HQ --s3-prefix SpatialVID-HQ \
#       --job-name s3_clean_hq
#   bash scripts/s3_cleanup_monitor.sh \
#       --src ~/data/SpatialVID-HQ-tar --s3-prefix SpatialVID-HQ-tar \
#       --job-name s3_clean_tar

set -euo pipefail

# ---- Defaults ----
S3_ENDPOINT="${S3_ENDPOINT:-https://pdx.s8k.io}"
S3_BUCKET="${S3_BUCKET:-sana}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-team-elm}"
S3_SECRET_KEY="${S3_SECRET_KEY:-4b3d0cdc9e7441557f7c51c300ab79d6}"
RCLONE_BIN="${RCLONE_BIN:-$(command -v rclone 2>/dev/null || echo "")}"

SRC_DIR=""
S3_PREFIX=""
PARTITION="cpu_datamover"
TIME_LIMIT="7-00:00:00"
ACCOUNT="${ACCOUNT:-nvr_elm_llm}"
INTERVAL=600          # seconds between checks
DRY_RUN=""
JOB_NAME="s3_cleanup"
CPUS_PER_TASK=4
DELETE_EMPTY_DIRS="1" # remove empty directories after file cleanup
MODE="submit"         # "submit" to submit slurm job, "worker" for internal use

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)             SRC_DIR="$2"; shift 2 ;;
        --s3-prefix)       S3_PREFIX="$2"; shift 2 ;;
        --bucket)          S3_BUCKET="$2"; shift 2 ;;
        --endpoint)        S3_ENDPOINT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN="1"; shift ;;
        --interval)        INTERVAL="$2"; shift 2 ;;
        --partition)       PARTITION="$2"; shift 2 ;;
        --time)            TIME_LIMIT="$2"; shift 2 ;;
        --account)         ACCOUNT="$2"; shift 2 ;;
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --cpus-per-task)   CPUS_PER_TASK="$2"; shift 2 ;;
        --no-delete-dirs)  DELETE_EMPTY_DIRS=""; shift ;;
        --worker)          MODE="worker"; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: s3_cleanup_monitor.sh [OPTIONS]

Monitor S3 uploads and delete local files confirmed as uploaded (size-matched).
Submits a long-running Slurm job that periodically checks and cleans up.

Required:
  --src PATH             Local source directory being uploaded
  --s3-prefix PREFIX     S3 destination prefix (under bucket)

Options:
  --bucket NAME          S3 bucket name (default: sana)
  --endpoint URL         S3 endpoint (default: https://pdx.s8k.io)
  --dry-run              Log what would be deleted but don't actually delete
  --interval SECS        Seconds between checks (default: 600 = 10 min)
  --partition NAME       Slurm partition (default: cpu_datamover)
  --time LIMIT           Slurm time limit (default: 7-00:00:00)
  --account NAME         Slurm account (default: nvr_elm_llm)
  --job-name NAME        Slurm job name (default: s3_cleanup)
  --cpus-per-task N      CPUs per Slurm task (default: 4)
  --no-delete-dirs       Don't remove empty directories after cleanup
  -h, --help             Show this help
HELP
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validation ----
if [[ -z "$SRC_DIR" ]]; then
    echo "ERROR: --src is required."; exit 1
fi
SRC_DIR="$(realpath "$SRC_DIR")"
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: Source directory '$SRC_DIR' not found."; exit 1
fi
if [[ -z "$S3_PREFIX" ]]; then
    S3_PREFIX="$(basename "$SRC_DIR")"
fi
if [[ -z "$RCLONE_BIN" ]]; then
    echo "ERROR: rclone not found. Install rclone or set RCLONE_BIN."; exit 1
fi

# ==================== Worker mode (runs inside Slurm) ====================
if [[ "$MODE" == "worker" ]]; then

    RCLONE_REMOTE="s3cleanup"
    export RCLONE_CONFIG_S3CLEANUP_TYPE=s3
    export RCLONE_CONFIG_S3CLEANUP_PROVIDER=AWS
    export RCLONE_CONFIG_S3CLEANUP_ENV_AUTH=false
    export RCLONE_CONFIG_S3CLEANUP_ACCESS_KEY_ID="$S3_ACCESS_KEY"
    export RCLONE_CONFIG_S3CLEANUP_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
    export RCLONE_CONFIG_S3CLEANUP_REGION=us-east-1
    export RCLONE_CONFIG_S3CLEANUP_ENDPOINT="$S3_ENDPOINT"

    S3_DEST="${RCLONE_REMOTE}:${S3_BUCKET}/${S3_PREFIX}"

    echo "======================================"
    echo "S3 Cleanup Monitor"
    echo "======================================"
    echo "Node:         $(hostname)"
    echo "Local:        $SRC_DIR"
    echo "S3:           s3://${S3_BUCKET}/${S3_PREFIX}"
    echo "Interval:     ${INTERVAL}s"
    echo "Dry run:      ${DRY_RUN:-no}"
    echo "Delete dirs:  ${DELETE_EMPTY_DIRS:-no}"
    echo "Started:      $(date)"
    echo "======================================"

    ROUND=0
    while true; do
        ROUND=$((ROUND + 1))
        echo ""
        echo "========== Round $ROUND | $(date) =========="

        WORK_TMP=$(mktemp -d)
        LOCAL_LIST="${WORK_TMP}/local.txt"
        S3_LIST="${WORK_TMP}/s3.txt"
        MATCHED="${WORK_TMP}/matched.txt"

        # List local files: relative_path<TAB>size_bytes
        echo "[1/4] Listing local files..."
        (cd "$SRC_DIR" && find . -type f -printf '%P\t%s\n' | sort) > "$LOCAL_LIST"
        LOCAL_COUNT=$(wc -l < "$LOCAL_LIST")
        LOCAL_SIZE=$(awk '{s+=$2} END {printf "%.2f", s/1073741824}' "$LOCAL_LIST")
        echo "  Local: $LOCAL_COUNT files ($LOCAL_SIZE GiB)"

        if [[ "$LOCAL_COUNT" -eq 0 ]]; then
            echo "  No local files remaining. All cleaned up!"
            rm -rf "$WORK_TMP"

            if [[ -n "${DELETE_EMPTY_DIRS:-}" ]]; then
                echo "  Removing empty directory tree: $SRC_DIR"
                if [[ -z "${DRY_RUN:-}" ]]; then
                    find "$SRC_DIR" -type d -empty -delete 2>/dev/null || true
                else
                    echo "  [DRY RUN] Would remove empty dirs under $SRC_DIR"
                fi
            fi

            echo "  Done! Exiting."
            exit 0
        fi

        # List S3 files: relative_path<TAB>size_bytes
        echo "[2/4] Listing S3 files..."
        "$RCLONE_BIN" lsf -R --format "ps" --separator $'\t' "$S3_DEST" 2>/dev/null \
            | sort > "$S3_LIST"
        S3_COUNT=$(wc -l < "$S3_LIST")
        S3_SIZE=$(awk -F'\t' '{s+=$2} END {printf "%.2f", s/1073741824}' "$S3_LIST")
        echo "  S3:    $S3_COUNT files ($S3_SIZE GiB)"

        # Find files present in both with matching size
        echo "[3/4] Comparing..."
        # Both files are sorted; join on full line (path\tsize) gives confirmed matches
        comm -12 "$LOCAL_LIST" "$S3_LIST" > "$MATCHED"
        MATCH_COUNT=$(wc -l < "$MATCHED")
        MATCH_SIZE=$(awk -F'\t' '{s+=$2} END {printf "%.2f", s/1073741824}' "$MATCHED")
        echo "  Confirmed on S3 (size-matched): $MATCH_COUNT files ($MATCH_SIZE GiB)"

        REMAINING=$((LOCAL_COUNT - MATCH_COUNT))
        echo "  Still uploading / not yet on S3: $REMAINING files"

        # Delete confirmed files
        if [[ "$MATCH_COUNT" -gt 0 ]]; then
            echo "[4/4] Deleting confirmed local files..."
            DELETED=0
            FAILED=0
            while IFS=$'\t' read -r rel_path size; do
                local_file="${SRC_DIR}/${rel_path}"
                if [[ -z "${DRY_RUN:-}" ]]; then
                    if rm -f "$local_file" 2>/dev/null; then
                        DELETED=$((DELETED + 1))
                    else
                        echo "  WARN: Failed to delete $local_file"
                        FAILED=$((FAILED + 1))
                    fi
                else
                    echo "  [DRY RUN] Would delete: $rel_path ($size bytes)"
                    DELETED=$((DELETED + 1))
                fi
            done < "$MATCHED"
            echo "  Deleted: $DELETED files, Failed: $FAILED files"

            # Clean up empty directories
            if [[ -n "${DELETE_EMPTY_DIRS:-}" && -z "${DRY_RUN:-}" ]]; then
                echo "  Cleaning empty directories..."
                find "$SRC_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
            fi
        else
            echo "[4/4] No files to delete this round."
        fi

        rm -rf "$WORK_TMP"

        if [[ "$REMAINING" -eq 0 && "$MATCH_COUNT" -gt 0 && -z "${DRY_RUN:-}" ]]; then
            echo ""
            echo "All files uploaded and cleaned up! Exiting."
            exit 0
        fi

        echo ""
        echo "Sleeping ${INTERVAL}s until next check..."
        sleep "$INTERVAL"
    done

    exit 0
fi

# ==================== Submit mode (submits Slurm job) ====================
LOG_DIR="output/slurm_log"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "S3 Cleanup Monitor - Submitting"
echo "======================================"
echo "Source:       $SRC_DIR"
echo "S3 prefix:    s3://${S3_BUCKET}/${S3_PREFIX}"
echo "Interval:     ${INTERVAL}s"
echo "Dry run:      ${DRY_RUN:-no}"
echo "Partition:    $PARTITION"
echo "Time limit:   $TIME_LIMIT"
echo "======================================"

# Build the worker command with all arguments forwarded
WORKER_ARGS=(
    bash "$(realpath "$0")" --worker
    --src "$SRC_DIR"
    --s3-prefix "$S3_PREFIX"
    --bucket "$S3_BUCKET"
    --endpoint "$S3_ENDPOINT"
    --interval "$INTERVAL"
    --job-name "$JOB_NAME"
)
if [[ -n "${DRY_RUN:-}" ]]; then
    WORKER_ARGS+=(--dry-run)
fi
if [[ -z "${DELETE_EMPTY_DIRS:-}" ]]; then
    WORKER_ARGS+=(--no-delete-dirs)
fi

SBATCH_CMD=(
    sbatch
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --time="$TIME_LIMIT"
    --job-name="$JOB_NAME"
    --output="${LOG_DIR}/${JOB_NAME}_%j.log"
    --error="${LOG_DIR}/${JOB_NAME}_%j.log"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --wrap="${WORKER_ARGS[*]}"
)

echo ""
echo "Submitting Slurm job..."
echo "  ${SBATCH_CMD[*]}"
echo ""

if "${SBATCH_CMD[@]}"; then
    echo ""
    echo "Job submitted successfully!"
    echo "  Monitor: squeue -u \$USER -n ${JOB_NAME}"
    echo "  Logs:    tail -f ${LOG_DIR}/${JOB_NAME}_*.log"
else
    echo "ERROR: Failed to submit Slurm job."
    exit 1
fi
