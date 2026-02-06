#!/bin/bash
# ============================================================================
# Sekai (sekai-real-walking-hq) Large-Scale Annotation Job Submission Script
#
# This script:
# 1. Lists all MP4 videos in the sekai data directory
# 2. Splits videos into groups (default: 200 per group)
# 3. Generates a CSV file per group for RawMP4StreamList
# 4. Generates per-group SLURM job scripts from job_template.sh
# 5. Submits each group via sbatch
#
# Usage:
#   ./submit_all.sh [--dry-run] [--force] [--videos-per-group N] [--recompute-lists]
#
# Options:
#   --dry-run            Show what would be submitted without actually submitting
#   --force              Re-submit all jobs, ignoring status
#   --videos-per-group N Number of videos per job group (default: 200)
#   --recompute-lists    Force regeneration of video list CSV files
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
MYHOME="/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/haozhu"
DATA_DIR="$MYHOME/data/sekai-real-walking-hq"
OUTPUT_ROOT="$DATA_DIR/vipe_results"
JOBS_DIR="$SCRIPT_DIR/jobs"
LISTS_DIR="$SCRIPT_DIR/video_lists"
LOGS_DIR="$SCRIPT_DIR/logs"
TEMPLATE="$SCRIPT_DIR/job_template.sh"

# Default settings
DRY_RUN=false
FORCE=false
RECOMPUTE_LISTS=false
VIDEOS_PER_GROUP=200

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
        --videos-per-group)
            VIDEOS_PER_GROUP="$2"
            shift 2
            ;;
        --recompute-lists)
            RECOMPUTE_LISTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$JOBS_DIR" "$LISTS_DIR" "$LOGS_DIR"

echo "=============================================="
echo "Sekai Large-Scale Annotation Submission"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Data directory: $DATA_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Videos per group: $VIDEOS_PER_GROUP"
echo "Dry run: $DRY_RUN"
echo "Force re-submit: $FORCE"
echo ""

# Step 1: List all MP4 files and split into groups
echo "Step 1: Discovering videos..."

# Get sorted list of all mp4 files
ALL_VIDEOS=()
while IFS= read -r -d '' file; do
    ALL_VIDEOS+=("$file")
done < <(find "$DATA_DIR" -maxdepth 1 -name "*.mp4" -print0 | sort -z)

TOTAL_VIDEOS=${#ALL_VIDEOS[@]}
echo "Total videos found: $TOTAL_VIDEOS"

if [ "$TOTAL_VIDEOS" -eq 0 ]; then
    echo "ERROR: No MP4 files found in $DATA_DIR"
    exit 1
fi

# Calculate number of groups
NUM_GROUPS=$(( (TOTAL_VIDEOS + VIDEOS_PER_GROUP - 1) / VIDEOS_PER_GROUP ))
echo "Number of job groups: $NUM_GROUPS (at $VIDEOS_PER_GROUP videos/group)"
echo ""

# Step 2: Generate CSV files for each group
echo "Step 2: Generating video list CSV files..."

csv_generated=0
csv_skipped=0

for ((group_idx=0; group_idx<NUM_GROUPS; group_idx++)); do
    group_id=$(printf "group_%04d" $group_idx)
    csv_file="$LISTS_DIR/${group_id}.csv"

    # Skip if CSV already exists and --recompute-lists not given
    if [ -f "$csv_file" ] && [ "$RECOMPUTE_LISTS" = false ]; then
        csv_skipped=$((csv_skipped + 1))
        continue
    fi

    # Calculate video range for this group
    vid_start=$((group_idx * VIDEOS_PER_GROUP))
    vid_end=$(( (group_idx + 1) * VIDEOS_PER_GROUP ))
    if [ "$vid_end" -gt "$TOTAL_VIDEOS" ]; then
        vid_end=$TOTAL_VIDEOS
    fi

    # Write CSV (one video path per line)
    > "$csv_file"
    for ((v=vid_start; v<vid_end; v++)); do
        echo "${ALL_VIDEOS[$v]}" >> "$csv_file"
    done

    csv_generated=$((csv_generated + 1))
done

echo "  CSV files generated: $csv_generated"
echo "  CSV files skipped (already exist): $csv_skipped"
echo ""

# Step 3: Generate and submit SLURM jobs
echo "Step 3: Generating and submitting jobs..."

submitted=0
skipped_completed=0
skipped_running=0
resubmitted=0

for ((group_idx=0; group_idx<NUM_GROUPS; group_idx++)); do
    group_id=$(printf "group_%04d" $group_idx)

    # Calculate video range for logging
    vid_start=$((group_idx * VIDEOS_PER_GROUP))
    vid_end=$(( (group_idx + 1) * VIDEOS_PER_GROUP ))
    if [ "$vid_end" -gt "$TOTAL_VIDEOS" ]; then
        vid_end=$TOTAL_VIDEOS
    fi
    num_videos=$((vid_end - vid_start))

    # Paths
    job_script="$JOBS_DIR/${group_id}.sh"
    csv_file="$LISTS_DIR/${group_id}.csv"
    output_dir="$OUTPUT_ROOT/${group_id}"

    # Check if group is already done by counting completed outputs
    should_submit=true
    reason=""

    if [ "$FORCE" = false ]; then
        # Check if output vipe directory has results for all videos in this group
        if [ -d "$output_dir/vipe" ]; then
            completed_count=$(find "$output_dir/vipe" -name "*_info.pkl" 2>/dev/null | wc -l)
            if [ "$completed_count" -ge "$num_videos" ]; then
                should_submit=false
                reason="already completed ($completed_count/$num_videos videos)"
                skipped_completed=$((skipped_completed + 1))
            elif [ "$completed_count" -gt 0 ]; then
                reason="partially done ($completed_count/$num_videos), re-submitting"
                resubmitted=$((resubmitted + 1))
            fi
        fi

        # Check if a SLURM job is currently running for this group
        if [ "$should_submit" = true ]; then
            running_jobs=$(squeue -u "$USER" --name "VIPE:Sekai-${group_id}" --noheader 2>/dev/null | wc -l || echo "0")
            if [ "$running_jobs" -gt 0 ]; then
                should_submit=false
                reason="still running in SLURM"
                skipped_running=$((skipped_running + 1))
            fi
        fi
    fi

    if [ "$should_submit" = false ]; then
        echo "[$group_id] SKIP: $reason"
        continue
    fi

    # Generate job script from template
    sed -e "s/__GROUP_ID__/${group_id}/g" \
        "$TEMPLATE" > "$job_script"
    chmod +x "$job_script"

    if [ "$DRY_RUN" = true ]; then
        echo "[$group_id] DRY-RUN: Would submit $num_videos videos (indices $vid_start-$((vid_end-1))) - $reason"
    else
        # Submit job
        sbatch_output=$(sbatch "$job_script")
        job_id=$(echo "$sbatch_output" | grep -oP '\d+')
        echo "[$group_id] SUBMITTED: $num_videos videos, job_id=$job_id - $reason"
        submitted=$((submitted + 1))
    fi
done

echo ""
echo "=============================================="
echo "Submission Summary"
echo "=============================================="
echo "Total videos: $TOTAL_VIDEOS"
echo "Total job groups: $NUM_GROUPS ($VIDEOS_PER_GROUP videos/group)"
echo "Submitted: $submitted"
echo "Re-submitted (partial): $resubmitted"
echo "Skipped (completed): $skipped_completed"
echo "Skipped (running): $skipped_running"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a dry run. Use without --dry-run to actually submit jobs."
fi
