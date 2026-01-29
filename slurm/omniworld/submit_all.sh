#!/bin/bash
# ============================================================================
# OmniWorld Large-Scale Annotation Job Submission Script
# 
# This script:
# 1. Computes total number of chunks across all scenes (cached to file)
# 2. Divides chunks into job groups (each job processes ~400 chunks)
# 3. Checks status of existing jobs (skip completed, re-submit failed/partial)
# 4. Generates and submits SLURM job scripts
#
# Usage:
#   ./submit_all.sh [--dry-run] [--force] [--chunks-per-job N] [--recompute-chunks]
#
# Options:
#   --dry-run            Show what would be submitted without actually submitting
#   --force              Re-submit all jobs, ignoring status
#   --chunks-per-job N   Number of chunks per job group (default: 400)
#   --recompute-chunks   Force recomputation of chunk list (ignore cache)
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/haozhu/data/OmniWorld-Game"
JOBS_DIR="$SCRIPT_DIR/jobs"
STATUS_DIR="$SCRIPT_DIR/status"
LOGS_DIR="$SCRIPT_DIR/logs"
TEMPLATE="$SCRIPT_DIR/job_template.sh"
CHUNK_CACHE="$SCRIPT_DIR/chunk_list.json"

# Default settings
DRY_RUN=false
FORCE=false
RECOMPUTE_CHUNKS=false
CHUNKS_PER_JOB=400
WINDOW_SIZE=960
OVERLAP=480

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --chunks-per-job)
            CHUNKS_PER_JOB="$2"
            shift 2
            ;;
        --recompute-chunks)
            RECOMPUTE_CHUNKS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$JOBS_DIR" "$STATUS_DIR" "$LOGS_DIR"

echo "=============================================="
echo "OmniWorld Large-Scale Annotation Submission"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Data root: $DATA_ROOT"
echo "Chunks per job: $CHUNKS_PER_JOB"
echo "Dry run: $DRY_RUN"
echo "Force re-submit: $FORCE"
echo ""

# Step 1: Get total number of chunks (use cache if available)
if [ -f "$CHUNK_CACHE" ] && [ "$RECOMPUTE_CHUNKS" = false ]; then
    echo "Step 1: Loading chunk list from cache..."
    TOTAL_CHUNKS=$(python3 -c "import json; print(json.load(open('$CHUNK_CACHE'))['total_chunks'])")
    echo "Total chunks: $TOTAL_CHUNKS (cached)"
else
    echo "Step 1: Computing chunk list (this may take a while)..."
    cd "$PROJECT_DIR"
    
    # Compute and cache chunk list
    python3 << EOF
import json
from pathlib import Path

# Lightweight chunk computation without importing vipe
# This mimics the logic in OmniWorldSlidingWindowStreamList._build_chunks

data_root = Path("$DATA_ROOT")
window_size = $WINDOW_SIZE
overlap = $OVERLAP
stride = window_size - overlap

chunks = []

# Discover all scenes
scenes = sorted([d.name for d in data_root.iterdir() 
                 if d.is_dir() and (d / "color").exists()])

for scene_id in scenes:
    scene_dir = data_root / scene_id
    color_dir = scene_dir / "color"
    
    # Count frames
    total_frames = len(list(color_dir.glob("*.png")))
    if total_frames == 0:
        continue
    
    # Generate chunks
    if total_frames <= window_size:
        if total_frames >= 10:
            chunks.append({
                "scene_id": scene_id,
                "frame_start": 0,
                "frame_end": total_frames,
                "chunk_name": f"{scene_id}_frames0000_{total_frames:04d}"
            })
        continue
    
    # Compute chunk starts
    chunk_starts = []
    frame_start = 0
    while frame_start + window_size <= total_frames:
        chunk_starts.append(frame_start)
        frame_start += stride
    
    # Check if we need final chunk
    if chunk_starts:
        last_chunk_end = chunk_starts[-1] + window_size
        if last_chunk_end < total_frames:
            final_start = total_frames - window_size
            if final_start > chunk_starts[-1]:
                chunk_starts.append(final_start)
    
    # Create chunks
    for fs in chunk_starts:
        fe = fs + window_size
        chunks.append({
            "scene_id": scene_id,
            "frame_start": fs,
            "frame_end": fe,
            "chunk_name": f"{scene_id}_frames{fs:04d}_{fe:04d}"
        })

# Save to cache
cache_data = {
    "total_chunks": len(chunks),
    "window_size": window_size,
    "overlap": overlap,
    "num_scenes": len(scenes),
    "chunks": chunks
}

with open("$CHUNK_CACHE", "w") as f:
    json.dump(cache_data, f, indent=2)

print(f"Computed {len(chunks)} chunks from {len(scenes)} scenes")
print(f"Cached to $CHUNK_CACHE")
EOF
    
    TOTAL_CHUNKS=$(python3 -c "import json; print(json.load(open('$CHUNK_CACHE'))['total_chunks'])")
    echo "Total chunks: $TOTAL_CHUNKS"
fi

# Step 2: Calculate number of job groups
NUM_JOBS=$(( (TOTAL_CHUNKS + CHUNKS_PER_JOB - 1) / CHUNKS_PER_JOB ))
echo "Number of job groups: $NUM_JOBS"
echo ""

# Step 3: Check job statuses and submit
echo "Step 3: Checking job statuses and submitting..."

submitted=0
skipped_completed=0
skipped_running=0
resubmitted=0

for ((job_idx=0; job_idx<NUM_JOBS; job_idx++)); do
    # Calculate chunk range for this job
    chunk_start=$((job_idx * CHUNKS_PER_JOB))
    chunk_end=$(( (job_idx + 1) * CHUNKS_PER_JOB ))
    if [ $chunk_end -gt $TOTAL_CHUNKS ]; then
        chunk_end=$TOTAL_CHUNKS
    fi
    
    # Format group ID
    group_id=$(printf "group_%04d" $job_idx)
    
    # Paths
    job_script="$JOBS_DIR/${group_id}.sh"
    status_file="$STATUS_DIR/${group_id}.json"
    
    # Check status
    should_submit=true
    reason=""
    
    if [ -f "$status_file" ] && [ "$FORCE" = false ]; then
        # Read status from file
        state=$(python3 -c "import json; print(json.load(open('$status_file'))['state'])" 2>/dev/null || echo "unknown")
        slurm_job_id=$(python3 -c "import json; print(json.load(open('$status_file')).get('slurm_job_id', ''))" 2>/dev/null || echo "")
        
        case $state in
            completed)
                should_submit=false
                reason="already completed"
                skipped_completed=$((skipped_completed + 1))
                ;;
            running)
                # Check if job is actually still running in SLURM
                if [ -n "$slurm_job_id" ]; then
                    job_running=$(squeue -j "$slurm_job_id" 2>/dev/null | grep -c "$slurm_job_id" || echo "0")
                    if [ "$job_running" -gt 0 ]; then
                        should_submit=false
                        reason="still running (job $slurm_job_id)"
                        skipped_running=$((skipped_running + 1))
                    else
                        reason="previous job $slurm_job_id no longer running"
                        resubmitted=$((resubmitted + 1))
                    fi
                fi
                ;;
            failed|partial)
                reason="re-submitting ($state)"
                resubmitted=$((resubmitted + 1))
                ;;
            *)
                reason="unknown status, submitting"
                ;;
        esac
    fi
    
    if [ "$should_submit" = false ]; then
        echo "[$group_id] SKIP: $reason"
        continue
    fi
    
    # Generate job script
    sed -e "s/GROUP_ID/${group_id}/g" \
        -e "s/CHUNK_START/${chunk_start}/g" \
        -e "s/CHUNK_END/${chunk_end}/g" \
        "$TEMPLATE" > "$job_script"
    chmod +x "$job_script"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[$group_id] DRY-RUN: Would submit chunks [$chunk_start, $chunk_end) - $reason"
    else
        # Submit job
        sbatch_output=$(sbatch "$job_script")
        job_id=$(echo "$sbatch_output" | grep -oP '\d+')
        echo "[$group_id] SUBMITTED: chunks [$chunk_start, $chunk_end), job_id=$job_id - $reason"
        
        # Initialize status file
        python3 -c "
import json
status = {
    'state': 'running',
    'completed_chunks': [],
    'failed_chunks': [],
    'total_chunks': $((chunk_end - chunk_start)),
    'slurm_job_id': '$job_id',
    'exit_code': None,
    'chunk_range': [$chunk_start, $chunk_end]
}
with open('$status_file', 'w') as f:
    json.dump(status, f, indent=2)
"
        submitted=$((submitted + 1))
    fi
done

echo ""
echo "=============================================="
echo "Submission Summary"
echo "=============================================="
echo "Total job groups: $NUM_JOBS"
echo "Submitted: $submitted"
echo "Re-submitted (failed/partial): $resubmitted"
echo "Skipped (completed): $skipped_completed"
echo "Skipped (running): $skipped_running"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a dry run. Use without --dry-run to actually submit jobs."
fi
