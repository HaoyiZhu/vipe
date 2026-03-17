import os
import sys
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# Import Ray
import ray
from ray.exceptions import RayTaskError

# Imports for the Ray status table
from rich.console import Console
from rich.table import Table


# --- Ray Helper Function ---
def initialize_ray(local_only: bool = False):
    """Initializes Ray and prints cluster info using Rich."""
    if ray.is_initialized():
        return

    console = Console()

    if local_only:
        console.print(
            "Starting a fresh local Ray instance.",
            style="bold yellow",
        )
        info = ray.init(ignore_reinit_error=True)
    else:
        try:
            # Try to connect to an existing cluster (if running on a Slurm node or similar)
            info = ray.init(address="auto", ignore_reinit_error=True)
        except ConnectionError:
            console.print(
                "Ray is not running. Starting a local Ray instance.",
                style="bold yellow",
            )
            info = ray.init(ignore_reinit_error=True)

    table = Table(title="Ray Cluster Info", style="bold green")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Node IP Address", info["node_ip_address"])
    table.add_row("Dashboard URL", str(info.dashboard_url))
    table.add_row("Python Version", info.python_version)
    table.add_row("Ray Version", info.ray_version)

    # Show available resources to confirm GPUs are detected
    resources = ray.available_resources()
    table.add_row("GPUs Available", str(resources.get("GPU", 0)))
    table.add_row("CPUs Available", str(resources.get("CPU", 0)))

    console.print(table)


# --- Remote Worker Function ---
# num_gpus=1 ensures each worker gets dedicated access to 1 GPU.
@ray.remote(num_gpus=1, num_cpus=8)
def run_video_stream(cwd, stream_list, stream_idx, pipeline_args):
    # 1. Set working directory to the hydra original cwd so relative paths work
    os.chdir(str(cwd))

    # 2. Imports must be inside the function for Ray serialization
    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # 3. Setup logging per worker
    configure_logging()

    # 4. Run Pipeline
    video_stream = stream_list[stream_idx]
    pipeline = make_pipeline(pipeline_args)
    pipeline.run(video_stream)

    return video_stream.name()


# --- Main Hydra Entry Point ---
@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.streams.base import StreamList

    # Record start time
    start_time = time.time()
    # 4 hours - 15 minutes buffer = 13500 seconds
    MAX_DURATION = 4 * 3600 - 15 * 60

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    # Optional: Prefilter streams to skip ones that are already done
    if args.get("prefilter", False):
        skip_exists = args.pipeline.output.get("skip_exists", False)
        if skip_exists:
            output_path = Path(args.pipeline.output.path)
            print(f"Prefiltering: Checking for existing files in {output_path}...")
            
            existing_names = set()
            vipe_dir = output_path / "vipe"
            if vipe_dir.exists():
                # Fast directory listing
                for p in vipe_dir.glob("*_info.pkl"):
                     # filename is {name}_info.pkl
                     # We assume standard naming: name + "_info.pkl"
                     if p.name.endswith("_info.pkl"):
                        name = p.name[:-9] 
                        existing_names.add(name)
            
            print(f"Found {len(existing_names)} existing files.")
            
            # We need to filter the stream_list. 
            # Since stream_list might not support deletion, we create a list of indices to process.
            # But the Ray loop iterates over indices.
            # We can just filter the indices here.
            
            indices_to_process = []
            for i in range(len(stream_list)):
                name = stream_list.stream_name(i)
                if name not in existing_names:
                    indices_to_process.append(i)
            
            print(f"Skipping {len(stream_list) - len(indices_to_process)} files. Processing {len(indices_to_process)} files.")
        else:
            indices_to_process = list(range(len(stream_list)))
    else:
        indices_to_process = list(range(len(stream_list)))

    stream_offset = int(args.get("stream_offset", 0) or 0)
    if len(indices_to_process) > 0 and stream_offset != 0:
        normalized_offset = stream_offset % len(indices_to_process)
        print(
            f"Rotating remaining stream order by offset {normalized_offset} "
            f"(requested {stream_offset})."
        )
        indices_to_process = (
            indices_to_process[normalized_offset:] + indices_to_process[:normalized_offset]
        )

    max_streams = args.get("max_streams", None)
    if max_streams is not None:
        max_streams = int(max_streams)
        if max_streams > 0 and len(indices_to_process) > max_streams:
            print(f"Limiting this run to {max_streams} streams (from {len(indices_to_process)} remaining).")
            indices_to_process = indices_to_process[:max_streams]

    # --- RAY EXECUTION BRANCH ---
    if args.get("ray", False):
        initialize_ray(local_only=bool(args.get("ray_local_only", False)))

        print(f"Submitting {len(indices_to_process)} jobs to Ray...")

        # Put large objects in shared memory to avoid serialization overhead
        stream_list_ref = ray.put(stream_list)
        pipeline_args_ref = ray.put(args.pipeline)
        cwd_ref = ray.put(Path.cwd().resolve())

        # Throttled Submission Logic
        # We limit the number of pending tasks to avoid submitting everything at once.
        # This allows us to check the time regularly and stop submitting new tasks
        # without needing to cancel already running ones.
        
        # Keep a small backlog per available GPU so we don't accumulate many idle/pending
        # Ray tasks when a single-GPU Slurm job is processing one video at a time.
        available_gpus = max(1, int(ray.available_resources().get("GPU", 1)))
        MAX_PENDING_TASKS = max(available_gpus * 4, available_gpus)
        
        futures = []
        next_idx_ptr = 0
        total_tasks = len(indices_to_process)
        success_count = 0
        failed_count = 0
        
        progress_bar = tqdm(total=total_tasks, desc="Processing Streams")
        
        stop_submitting = False
        
        while len(futures) > 0 or next_idx_ptr < total_tasks:
            # 1. Check time limit
            elapsed = time.time() - start_time
            if elapsed > MAX_DURATION and not stop_submitting:
                print(f"\nTime limit reached ({MAX_DURATION}s). Stopping submission of new tasks.")
                print(f"Waiting for {len(futures)} currently running/pending tasks to finish...")
                stop_submitting = True
                # We do NOT cancel existing futures here. We let them finish.
                # Since we have a 15 min buffer, they should finish in time.

            # 2. Submit new tasks if capacity allows and not stopped
            while not stop_submitting and len(futures) < MAX_PENDING_TASKS and next_idx_ptr < total_tasks:
                stream_idx = indices_to_process[next_idx_ptr]
                next_idx_ptr += 1
                
                futures.append(
                    run_video_stream.remote(
                        cwd_ref, stream_list_ref, stream_idx, pipeline_args_ref
                    )
                )

            # 3. Wait for tasks to finish
            # If we have nothing running and nothing to submit, we are done
            if not futures:
                break
                
            # Wait for at least one task to finish, or timeout to check time again
            done_ids, futures = ray.wait(futures, timeout=5.0)

            # 4. Update progress
            if done_ids:
                progress_bar.update(len(done_ids))
                for obj_ref in done_ids:
                    try:
                        ray.get(obj_ref)
                        success_count += 1
                    except (RayTaskError, Exception) as e:
                        failed_count += 1
                        print(f"\nRayExecutor: Exception in job: {e}")

        progress_bar.close()
        ray.shutdown()
        print(
            f"Ray summary: success={success_count}, failed={failed_count}, "
            f"submitted={total_tasks}"
        )
        
        if stop_submitting:
            print("Job finished early due to time limit.")
            sys.exit(10 if failed_count > 0 else 0)
        if failed_count > 0:
            sys.exit(10 if success_count > 0 else 11)

    # --- STANDARD EXECUTION BRANCH ---
    else:
        from vipe.pipeline import make_pipeline
        from vipe.utils.logging import configure_logging

        logger = configure_logging()
        # Use indices_to_process if prefilter is enabled
        loop_indices = indices_to_process if args.get("prefilter", False) else range(len(stream_list))
        
        for stream_idx in loop_indices:
            # Check time limit
            if time.time() - start_time > MAX_DURATION:
                 logger.info(f"Time limit reached ({MAX_DURATION}s). Stopping early.")
                 break

            video_stream = stream_list[stream_idx]
            logger.info(
                f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
            )
            pipeline = make_pipeline(args.pipeline)
            pipeline.run(video_stream)
            logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
