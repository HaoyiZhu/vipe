import os
import sys
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
def initialize_ray():
    """Initializes Ray and prints cluster info using Rich."""
    if ray.is_initialized():
        return

    console = Console()

    try:
        # Try to connect to an existing cluster (if running on a Slurm node or similar)
        info = ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        console.print(
            "Ray is not running. Starting a local Ray instance.",
            style="bold yellow",
        )
        # Start a local instance. This will auto-detect all 8 GPUs on your machine.
        info = ray.init()

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
# With 8 GPUs available, Ray will run 8 of these functions in parallel.
@ray.remote(num_gpus=1, num_cpus=4)
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

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    # Optional: Prefilter streams to skip ones that are already done
    if args.get("prefilter", False):
        raise NotImplementedError("Prefiltering is not implemented yet")
        # from vipe.streams.base import FilteredStreamList
        # from vipe.pipeline import make_pipeline

        # # We use a temporary pipeline just to check filter conditions
        # print("Prefiltering streams...")
        # stream_list = FilteredStreamList(
        #     stream_list, make_pipeline(args.pipeline).should_filter
        # )

    # --- RAY EXECUTION BRANCH ---
    if args.get("ray", False):
        initialize_ray()

        print(f"Submitting {len(stream_list)} jobs to Ray...")

        # Put large objects in shared memory to avoid serialization overhead
        stream_list_ref = ray.put(stream_list)
        pipeline_args_ref = ray.put(args.pipeline)
        cwd_ref = ray.put(Path.cwd().resolve())

        # Launch all tasks. Ray scheduler manages the queue based on available GPUs.
        futures = [
            run_video_stream.remote(
                cwd_ref, stream_list_ref, stream_idx, pipeline_args_ref
            )
            for stream_idx in range(len(stream_list))
        ]

        # Monitor progress
        progress_bar = tqdm(total=len(stream_list), desc="Processing Streams")
        while len(futures):
            # Wait for tasks to finish
            done_ids, futures = ray.wait(futures)

            # Update progress bar
            progress_bar.update(len(done_ids))

            # Check for errors in finished tasks
            for obj_ref in done_ids:
                try:
                    stream_name = ray.get(obj_ref)
                except (RayTaskError, Exception) as e:
                    print(f"\nRayExecutor: Exception in job: {e}")

        progress_bar.close()
        ray.shutdown()

    # --- STANDARD EXECUTION BRANCH ---
    else:
        from vipe.pipeline import make_pipeline
        from vipe.utils.logging import configure_logging

        logger = configure_logging()
        for stream_idx in range(len(stream_list)):
            video_stream = stream_list[stream_idx]
            logger.info(
                f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
            )
            pipeline = make_pipeline(args.pipeline)
            pipeline.run(video_stream)
            logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
