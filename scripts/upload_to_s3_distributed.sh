#!/usr/bin/env bash
# Distributed S3 upload: recursively uploads a local directory to S3 via Slurm job array.
# Shards top-level items (files/dirs) across workers for parallel uploads.
#
# Usage:
#   # Upload a directory to S3 (recursive)
#   bash scripts/upload_to_s3_distributed.sh --src ~/data/SpatialVID-HQ --s3-prefix SpatialVID-HQ
#
#   # Preview what would be uploaded
#   bash scripts/upload_to_s3_distributed.sh --src ~/data/SpatialVID-HQ --s3-prefix SpatialVID-HQ --dry-run
#
#   # More workers, custom bucket
#   bash scripts/upload_to_s3_distributed.sh --src ~/data/SpatialVID-HQ-tar --s3-prefix SpatialVID-HQ-tar \
#       --num-workers 10 --bucket my-bucket

set -euo pipefail

# ---- Defaults ----
S3_ENDPOINT="${S3_ENDPOINT:-https://pdx.s8k.io}"
S3_BUCKET="${S3_BUCKET:-sana}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-team-elm}"
S3_SECRET_KEY="${S3_SECRET_KEY:-4b3d0cdc9e7441557f7c51c300ab79d6}"
RCLONE_BIN="${RCLONE_BIN:-$(command -v rclone 2>/dev/null || echo "")}"

SRC_DIR=""
S3_PREFIX=""
NUM_WORKERS=5
PARTITION="cpu_datamover"
TIME_LIMIT="7-00:00:00"
ACCOUNT="${ACCOUNT:-nvr_elm_llm}"
TRANSFERS=32
MULTI_THREAD_STREAMS=4
DRY_RUN=""
CPUS_PER_TASK=16
JOB_NAME="s3_upload"
BANDWIDTH=""

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)             SRC_DIR="$2"; shift 2 ;;
        --s3-prefix)       S3_PREFIX="$2"; shift 2 ;;
        --bucket)          S3_BUCKET="$2"; shift 2 ;;
        --endpoint)        S3_ENDPOINT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN="1"; shift ;;
        --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
        --partition)       PARTITION="$2"; shift 2 ;;
        --time)            TIME_LIMIT="$2"; shift 2 ;;
        --account)         ACCOUNT="$2"; shift 2 ;;
        --transfers)       TRANSFERS="$2"; shift 2 ;;
        --bandwidth)       BANDWIDTH="$2"; shift 2 ;;
        --cpus-per-task)   CPUS_PER_TASK="$2"; shift 2 ;;
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --help|-h)
            cat <<'HELP'
Usage: upload_to_s3_distributed.sh [OPTIONS]

Distributed S3 upload via Slurm job array. Shards top-level entries across workers.

Required:
  --src PATH             Local source directory to upload
  --s3-prefix PREFIX     S3 destination prefix (under bucket)

Options:
  --bucket NAME          S3 bucket name (default: sana)
  --endpoint URL         S3 endpoint (default: https://pdx.s8k.io)
  --dry-run              Preview without uploading
  --num-workers N        Number of parallel Slurm tasks (default: 5)
  --partition NAME       Slurm partition (default: cpu_datamover)
  --time LIMIT           Slurm time limit (default: 7-00:00:00)
  --account NAME         Slurm account (default: nvr_elm_llm)
  --transfers N          rclone parallel transfers per worker (default: 32)
  --bandwidth LIMIT      Bandwidth limit per worker, e.g. '100M'
  --cpus-per-task N      CPUs per Slurm task (default: 16)
  --job-name NAME        Slurm job name prefix (default: s3_upload)
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
    echo "INFO: No --s3-prefix specified, using '${S3_PREFIX}'"
fi
if [[ -z "$RCLONE_BIN" ]]; then
    echo "ERROR: rclone not found. Install rclone or set RCLONE_BIN."; exit 1
fi

# ---- Step 1: Enumerate top-level entries for sharding ----
WORK_DIR="output/.s3_upload_workdir_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$WORK_DIR"
MANIFEST="${WORK_DIR}/manifest.txt"

echo "======================================"
echo "Distributed S3 Upload"
echo "======================================"
echo "Source:       $SRC_DIR"
echo "Destination:  s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "Endpoint:     $S3_ENDPOINT"
echo "Partition:    $PARTITION"
echo "Workers:      $NUM_WORKERS"
echo "Time limit:   $TIME_LIMIT"
echo "Account:      $ACCOUNT"
echo "Transfers:    $TRANSFERS (per worker)"
echo "Dry run:      ${DRY_RUN:-no}"
echo "Work dir:     $WORK_DIR"
echo "======================================"
echo ""

# List top-level entries (files and directories) for sharding
echo "Building manifest of top-level entries..."
> "$MANIFEST"
for entry in "$SRC_DIR"/*; do
    [[ -e "$entry" ]] || continue
    basename "$entry" >> "$MANIFEST"
done

TOTAL_ENTRIES=$(wc -l < "$MANIFEST")
echo "Total top-level entries: $TOTAL_ENTRIES"

if [[ "$TOTAL_ENTRIES" -eq 0 ]]; then
    echo "No entries to upload. Exiting."
    rm -rf "$WORK_DIR"
    exit 0
fi

if [[ "$NUM_WORKERS" -gt "$TOTAL_ENTRIES" ]]; then
    NUM_WORKERS="$TOTAL_ENTRIES"
    echo "Adjusted workers to $NUM_WORKERS (capped to entry count)."
fi

# ---- Step 2: Round-robin shard entries across workers ----
echo "Sharding $TOTAL_ENTRIES entries across $NUM_WORKERS workers..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    > "${WORK_DIR}/shard_${i}.txt"
done

line_num=0
while IFS= read -r line; do
    worker_id=$((line_num % NUM_WORKERS))
    echo "$line" >> "${WORK_DIR}/shard_${worker_id}.txt"
    line_num=$((line_num + 1))
done < "$MANIFEST"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    count=$(wc -l < "${WORK_DIR}/shard_${i}.txt")
    echo "  Worker $i: $count entries"
done

# ---- Step 3: Write config for workers ----
echo "$NUM_WORKERS" > "${WORK_DIR}/num_workers"

CONFIG_FILE="${WORK_DIR}/config.env"
cat > "$CONFIG_FILE" << CFGEOF
S3_ENDPOINT="$S3_ENDPOINT"
S3_BUCKET="$S3_BUCKET"
S3_ACCESS_KEY="$S3_ACCESS_KEY"
S3_SECRET_KEY="$S3_SECRET_KEY"
RCLONE_BIN="$RCLONE_BIN"
SRC_DIR="$SRC_DIR"
S3_PREFIX="$S3_PREFIX"
TRANSFERS=$TRANSFERS
MULTI_THREAD_STREAMS=$MULTI_THREAD_STREAMS
DRY_RUN="${DRY_RUN}"
BANDWIDTH="${BANDWIDTH}"
CFGEOF

# ---- Step 4: Write the worker script ----
WORKER_SCRIPT="${WORK_DIR}/worker.sh"
cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$1"

NUM_WORKERS_VAL=$(cat "${WORK_DIR}/num_workers")
WORKER_ID=$((SLURM_ARRAY_TASK_ID % NUM_WORKERS_VAL))
SHARD_FILE="${WORK_DIR}/shard_${WORKER_ID}.txt"

if [[ ! -f "$SHARD_FILE" ]]; then
    echo "ERROR: Shard file not found: $SHARD_FILE"
    exit 1
fi

source "${WORK_DIR}/config.env"

RCLONE_REMOTE="s3upload"
export RCLONE_CONFIG_S3UPLOAD_TYPE=s3
export RCLONE_CONFIG_S3UPLOAD_PROVIDER=AWS
export RCLONE_CONFIG_S3UPLOAD_ENV_AUTH=false
export RCLONE_CONFIG_S3UPLOAD_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export RCLONE_CONFIG_S3UPLOAD_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
export RCLONE_CONFIG_S3UPLOAD_REGION=us-east-1
export RCLONE_CONFIG_S3UPLOAD_ENDPOINT="$S3_ENDPOINT"

TOTAL_ENTRIES=$(wc -l < "$SHARD_FILE")
echo "======================================"
echo "Worker ${WORKER_ID} | Task ${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | Entries: $TOTAL_ENTRIES"
echo "======================================"

UPLOADED=0
FAILED=0

while IFS= read -r entry_name; do
    local_path="${SRC_DIR}/${entry_name}"
    s3_dest="${RCLONE_REMOTE}:${S3_BUCKET}/${S3_PREFIX}/${entry_name}"

    if [[ -d "$local_path" ]]; then
        # Directory: copy recursively
        s3_dest="${s3_dest}/"
        echo ""
        echo "--- [DIR] ${entry_name} ---"
    else
        # File: copy to parent prefix
        s3_dest="${RCLONE_REMOTE}:${S3_BUCKET}/${S3_PREFIX}/"
        echo ""
        echo "--- [FILE] ${entry_name} ---"
    fi

    echo "  Local:  $local_path"
    echo "  S3:     ${s3_dest}"

    RCLONE_ARGS=(
        "$RCLONE_BIN" copy
        "$local_path" "$s3_dest"
        --transfers "$TRANSFERS"
        --multi-thread-streams "$MULTI_THREAD_STREAMS"
        --progress --stats 30s
        --log-level INFO
    )

    if [[ -n "${DRY_RUN:-}" ]]; then
        RCLONE_ARGS+=(--dry-run)
    fi
    if [[ -n "${BANDWIDTH:-}" ]]; then
        RCLONE_ARGS+=(--bwlimit "$BANDWIDTH")
    fi

    echo "  Command: ${RCLONE_ARGS[*]}"

    if "${RCLONE_ARGS[@]}"; then
        echo "  => ${entry_name}: complete."
        UPLOADED=$((UPLOADED + 1))
    else
        echo "  => ${entry_name}: FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done < "$SHARD_FILE"

echo ""
echo "======================================"
echo "Worker ${WORKER_ID} Summary"
echo "  Uploaded: $UPLOADED entries"
echo "  Failed:   $FAILED entries"
echo "======================================"
WORKER_EOF
chmod +x "$WORKER_SCRIPT"

# ---- Step 5: Submit Slurm job array ----
LOG_DIR="output/slurm_log"
mkdir -p "$LOG_DIR"

SBATCH_CMD=(
    sbatch
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --time="$TIME_LIMIT"
    --job-name="${JOB_NAME}"
    --output="${LOG_DIR}/${JOB_NAME}_%A_%a.log"
    --error="${LOG_DIR}/${JOB_NAME}_%A_%a.log"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    "--array=0-$((NUM_WORKERS - 1))"
    "$WORKER_SCRIPT" "$WORK_DIR"
)

echo ""
echo "Submitting Slurm job array..."
echo "  ${SBATCH_CMD[*]}"
echo ""

if "${SBATCH_CMD[@]}"; then
    echo ""
    echo "Job submitted successfully!"
    echo "  Monitor: squeue -u \$USER -n ${JOB_NAME}"
    echo "  Logs:    tail -f ${LOG_DIR}/${JOB_NAME}_*.log"
    echo "  Work dir: $WORK_DIR (clean up manually after completion)"
else
    echo "ERROR: Failed to submit Slurm job."
    exit 1
fi
