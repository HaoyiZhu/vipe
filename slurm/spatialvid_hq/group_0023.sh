#!/bin/bash
#SBATCH -A nvr_elm_llm
#SBATCH -p polar4,polar3,polar,grizzly
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node 8
#SBATCH --array=1-15%1
#SBATCH -J VIPE:SpatialVID-HQ-group_0023
#SBATCH -o logs/vipe_group_0023_%A_%a.out
#SBATCH -e logs/vipe_group_0023_%A_%a.err
#SBATCH --exclusive

# --- 1. Environment Setup ---
# Explicitly point to your Mamba install

export MYHOME="/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/haozhu"

MINIFORGE_ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/haozhu/miniforge3"
source "$MINIFORGE_ROOT/etc/profile.d/conda.sh"

conda activate vipe

# Fix for "Path too long" error: Use local /tmp on the node
export RAY_TMPDIR="/tmp/ray_vipe_$SLURM_JOB_ID"
mkdir -p $RAY_TMPDIR

# --- 2. Ray Cluster Setup ---
echo "Setting up Ray Cluster..."

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Head node IP: $ip_head"

# Start Head Node
echo "Starting Head on $head_node"
# Note: We use --block in background (&) so it stays alive but returns control
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir="$RAY_TMPDIR" --block & 

sleep 10 # Wait for head to initialize

# Start Worker Nodes (if -N > 1)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    echo "Starting $worker_num Worker nodes"
    for ((i=1; i<=worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Worker on $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
             ray start --address "$ip_head" \
             --temp-dir="$RAY_TMPDIR" --block &
    done
    sleep 10 # Wait for workers to connect
fi

# --- 3. Run Inference ---
echo "Cluster ready. Starting Inference Script..."

# Paths
BASE_VIDEO_PATH="$MYHOME/data/SpatialVID-HQ/videos/group_0023/"
OUTPUT_PATH="$MYHOME/data/SpatialVID-HQ/vipe_results/group_0023/"

cmd="python run_ray.py \
    ray=true \
    prefilter=true \
    pipeline=default \
    streams=raw_mp4_stream \
    streams.frame_end=-1 \
    streams.base_path=$BASE_VIDEO_PATH \
    pipeline.output.path=$OUTPUT_PATH"

echo "Executing: $cmd"
$cmd

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "Inference finished successfully."
else
    echo "Inference exited with code $exit_code."
fi