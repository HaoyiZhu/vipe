#!/usr/bin/env python3
"""
Large-scale OmniWorld annotation script.

Annotates OmniWorld scenes using sliding window chunks with:
- OmniWorldGTDepthPipeline (GT depth + MoGe2 metric scale)
- Per-frame intrinsics optimization
- Ray-based parallelism for multi-GPU execution

Usage:
    # Single scene test (interactive)
    python annotate_omniworld.py \
        --data-root /path/to/OmniWorld-Game \
        --output-dir /path/to/output \
        --scenes 007b24c8269d \
        --window-size 960 --overlap 480

    # Process chunk range (SLURM job)
    python annotate_omniworld.py \
        --data-root /path/to/OmniWorld-Game \
        --output-dir /path/to/output \
        --chunk-start 0 --chunk-end 100 \
        --use-ray
"""

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_all_chunks(
    data_root: Path,
    scenes: Optional[List[str]],
    window_size: int,
    overlap: int,
) -> List[Tuple[Path, int, int, str]]:
    """
    Compute all sliding window chunks across scenes.
    
    Returns list of (scene_dir, frame_start, frame_end, chunk_name).
    """
    from vipe.streams.omniworld_stream import (
        OmniWorldSlidingWindowStreamList,
        get_scene_total_frames,
    )
    
    # Use StreamList to compute chunks
    stream_list = OmniWorldSlidingWindowStreamList(
        base_path=str(data_root),
        scenes=scenes,
        window_size=window_size,
        overlap=overlap,
    )
    
    return list(stream_list.chunks)


def is_chunk_completed(output_dir: Path, chunk_name: str) -> bool:
    """Check if a chunk has already been processed."""
    from vipe.utils.io import ArtifactPath
    
    artifact_path = ArtifactPath(output_dir, chunk_name)
    return artifact_path.meta_info_path.exists()


def process_chunk(
    scene_dir: Path,
    frame_start: int,
    frame_end: int,
    chunk_name: str,
    output_dir: Path,
    pipeline_config: dict,
) -> bool:
    """
    Process a single chunk with the OmniWorldGTDepthPipeline.
    
    Returns True on success, False on failure.
    """
    from vipe.pipeline.omniworld_gt_depth import OmniWorldGTDepthPipeline
    
    logger.info(f"Processing chunk: {chunk_name}")
    logger.info(f"  Scene: {scene_dir.name}, frames [{frame_start}, {frame_end})")
    
    try:
        # Create pipeline
        config = OmegaConf.create(pipeline_config)
        config.output.path = str(output_dir)
        
        pipeline = OmniWorldGTDepthPipeline(
            init=config.init,
            slam=config.slam,
            post=config.post,
            output=config.output,
            scale_estimation=config.scale_estimation,
        )
        
        # Run pipeline with sliding window (uses global frame indices)
        pipeline.run_with_sliding_window(
            scene_dir=scene_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            chunk_name=chunk_name,
        )
        
        # Clean up
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"Completed chunk: {chunk_name}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to process chunk {chunk_name}: {e}")
        return False


def run_sequential(
    chunks: List[Tuple[Path, int, int, str]],
    output_dir: Path,
    pipeline_config: dict,
    skip_exists: bool = True,
    max_duration: float = 13500,  # 3h 45m
) -> Tuple[List[str], List[str]]:
    """
    Process chunks sequentially (single GPU).
    
    Returns (completed_chunks, failed_chunks).
    """
    start_time = time.time()
    completed = []
    failed = []
    
    for i, (scene_dir, frame_start, frame_end, chunk_name) in enumerate(chunks):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            logger.warning(f"Time limit reached ({max_duration}s). Stopping.")
            break
        
        # Skip if already completed
        if skip_exists and is_chunk_completed(output_dir, chunk_name):
            logger.info(f"Skipping completed chunk: {chunk_name}")
            completed.append(chunk_name)
            continue
        
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_name}")
        
        success = process_chunk(
            scene_dir=scene_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            chunk_name=chunk_name,
            output_dir=output_dir,
            pipeline_config=pipeline_config,
        )
        
        if success:
            completed.append(chunk_name)
        else:
            failed.append(chunk_name)
    
    return completed, failed


def run_with_ray(
    chunks: List[Tuple[Path, int, int, str]],
    output_dir: Path,
    pipeline_config: dict,
    skip_exists: bool = True,
    max_duration: float = 13500,  # 3h 45m
    max_pending_tasks: int = 64,
) -> Tuple[List[str], List[str]]:
    """
    Process chunks with Ray parallelism (multi-GPU).
    
    Returns (completed_chunks, failed_chunks).
    """
    import ray
    from ray.exceptions import RayTaskError
    from tqdm import tqdm
    
    # Initialize Ray
    try:
        info = ray.init(address="auto", ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster: {info}")
    except ConnectionError:
        logger.info("Starting local Ray instance")
        info = ray.init()
    
    resources = ray.available_resources()
    logger.info(f"Ray resources: GPUs={resources.get('GPU', 0)}, CPUs={resources.get('CPU', 0)}")
    
    # Define remote worker
    @ray.remote(num_gpus=1, num_cpus=8)
    def process_chunk_remote(
        scene_dir_str: str,
        frame_start: int,
        frame_end: int,
        chunk_name: str,
        output_dir_str: str,
        pipeline_config: dict,
    ) -> Tuple[str, bool]:
        """Remote worker for processing a single chunk."""
        # Set up environment
        os.chdir(str(Path(__file__).parent))
        
        from vipe.utils.logging import configure_logging
        configure_logging()
        
        scene_dir = Path(scene_dir_str)
        output_dir = Path(output_dir_str)
        
        success = process_chunk(
            scene_dir=scene_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            chunk_name=chunk_name,
            output_dir=output_dir,
            pipeline_config=pipeline_config,
        )
        
        return chunk_name, success
    
    # Put large objects in shared memory
    pipeline_config_ref = ray.put(pipeline_config)
    output_dir_str = str(output_dir)
    
    start_time = time.time()
    completed = []
    failed = []
    
    # Filter chunks that need processing
    chunks_to_process = []
    for scene_dir, frame_start, frame_end, chunk_name in chunks:
        if skip_exists and is_chunk_completed(output_dir, chunk_name):
            logger.info(f"Skipping completed chunk: {chunk_name}")
            completed.append(chunk_name)
        else:
            chunks_to_process.append((scene_dir, frame_start, frame_end, chunk_name))
    
    logger.info(f"Processing {len(chunks_to_process)} chunks ({len(completed)} already completed)")
    
    # Throttled submission
    futures = []
    next_idx = 0
    total_tasks = len(chunks_to_process)
    stop_submitting = False
    
    progress = tqdm(total=total_tasks, desc="Processing chunks")
    
    while len(futures) > 0 or next_idx < total_tasks:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_duration and not stop_submitting:
            logger.warning(f"Time limit reached ({max_duration}s). Stopping submission.")
            stop_submitting = True
        
        # Submit new tasks
        while not stop_submitting and len(futures) < max_pending_tasks and next_idx < total_tasks:
            scene_dir, frame_start, frame_end, chunk_name = chunks_to_process[next_idx]
            next_idx += 1
            
            futures.append(
                process_chunk_remote.remote(
                    str(scene_dir),
                    frame_start,
                    frame_end,
                    chunk_name,
                    output_dir_str,
                    pipeline_config_ref,
                )
            )
        
        # Wait for tasks
        if not futures:
            break
        
        done_ids, futures = ray.wait(futures, timeout=5.0)
        
        # Process completed tasks
        for obj_ref in done_ids:
            try:
                chunk_name, success = ray.get(obj_ref)
                if success:
                    completed.append(chunk_name)
                else:
                    failed.append(chunk_name)
            except (RayTaskError, Exception) as e:
                logger.error(f"Ray task failed: {e}")
                failed.append("unknown")
            
            progress.update(1)
    
    progress.close()
    ray.shutdown()
    
    return completed, failed


def load_pipeline_config(config_path: str) -> dict:
    """Load pipeline configuration."""
    config = OmegaConf.load(config_path)
    
    # Load and merge SLAM defaults
    slam_default = OmegaConf.load("configs/slam/default.yaml")
    config.slam = OmegaConf.merge(slam_default, config.slam)
    
    return OmegaConf.to_container(config, resolve=True)


def save_job_status(
    status_file: Path,
    state: str,
    completed_chunks: List[str],
    failed_chunks: List[str],
    total_chunks: int,
    slurm_job_id: Optional[str] = None,
    exit_code: Optional[int] = None,
):
    """Save job status atomically."""
    status = {
        "state": state,
        "completed_chunks": completed_chunks,
        "failed_chunks": failed_chunks,
        "total_chunks": total_chunks,
        "slurm_job_id": slurm_job_id,
        "exit_code": exit_code,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Write atomically
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = status_file.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(status, f, indent=2)
    tmp_file.rename(status_file)


def main():
    parser = argparse.ArgumentParser(description="Annotate OmniWorld dataset")
    
    # Data arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default="/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/haozhu/data/OmniWorld-Game",
        help="Path to OmniWorld-Game data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="Scene IDs to process (default: all scenes)",
    )
    
    # Sliding window arguments
    parser.add_argument(
        "--window-size",
        type=int,
        default=960,
        help="Sliding window size in frames",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=480,
        help="Overlap between consecutive windows",
    )
    
    # Chunk range arguments (for SLURM parallelism)
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=None,
        help="Start chunk index (inclusive)",
    )
    parser.add_argument(
        "--chunk-end",
        type=int,
        default=None,
        help="End chunk index (exclusive)",
    )
    
    # Pipeline arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline/omniworld_sliding_window.yaml",
        help="Pipeline config file",
    )
    parser.add_argument(
        "--per-frame-intrinsics",
        action="store_true",
        help="Enable per-frame intrinsics optimization",
    )
    parser.add_argument(
        "--skip-exists",
        action="store_true",
        default=True,
        help="Skip already processed chunks",
    )
    parser.add_argument(
        "--no-skip-exists",
        action="store_false",
        dest="skip_exists",
        help="Re-process all chunks",
    )
    
    # Execution arguments
    parser.add_argument(
        "--use-ray",
        action="store_true",
        help="Use Ray for multi-GPU parallelism",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=13500,  # 3h 45m
        help="Maximum duration in seconds",
    )
    
    # Status tracking arguments
    parser.add_argument(
        "--job-group",
        type=str,
        default=None,
        help="Job group ID for status tracking",
    )
    parser.add_argument(
        "--status-dir",
        type=str,
        default=None,
        help="Directory for status files",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get SLURM job ID if running in SLURM
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    
    # Load pipeline config
    logger.info(f"Loading config from {args.config}")
    pipeline_config = load_pipeline_config(args.config)
    
    # Override per-frame intrinsics if specified
    if args.per_frame_intrinsics:
        pipeline_config["slam"]["per_frame_intrinsics"] = True
        logger.info("Enabled per-frame intrinsics optimization")
    
    # Get all chunks
    logger.info("Computing sliding window chunks...")
    all_chunks = get_all_chunks(
        data_root=data_root,
        scenes=args.scenes,
        window_size=args.window_size,
        overlap=args.overlap,
    )
    logger.info(f"Total chunks: {len(all_chunks)}")
    
    # Select chunk range
    if args.chunk_start is not None or args.chunk_end is not None:
        chunk_start = args.chunk_start or 0
        chunk_end = args.chunk_end or len(all_chunks)
        chunks = all_chunks[chunk_start:chunk_end]
        logger.info(f"Processing chunk range [{chunk_start}, {chunk_end}): {len(chunks)} chunks")
    else:
        chunks = all_chunks
    
    # Status tracking
    status_file = None
    if args.job_group and args.status_dir:
        status_file = Path(args.status_dir) / f"{args.job_group}.json"
        save_job_status(
            status_file=status_file,
            state="running",
            completed_chunks=[],
            failed_chunks=[],
            total_chunks=len(chunks),
            slurm_job_id=slurm_job_id,
        )
    
    # Run processing
    try:
        if args.use_ray:
            completed, failed = run_with_ray(
                chunks=chunks,
                output_dir=output_dir,
                pipeline_config=pipeline_config,
                skip_exists=args.skip_exists,
                max_duration=args.max_duration,
            )
        else:
            completed, failed = run_sequential(
                chunks=chunks,
                output_dir=output_dir,
                pipeline_config=pipeline_config,
                skip_exists=args.skip_exists,
                max_duration=args.max_duration,
            )
        
        # Determine final state
        if len(failed) == 0 and len(completed) == len(chunks):
            state = "completed"
            exit_code = 0
        elif len(failed) > 0:
            state = "failed"
            exit_code = 1
        else:
            state = "partial"
            exit_code = 0
        
        logger.info(f"Completed: {len(completed)}, Failed: {len(failed)}")
        
    except Exception as e:
        logger.exception(f"Job failed with exception: {e}")
        state = "failed"
        exit_code = 1
        completed = []
        failed = []
    
    # Save final status
    if status_file:
        save_job_status(
            status_file=status_file,
            state=state,
            completed_chunks=completed,
            failed_chunks=failed,
            total_chunks=len(chunks),
            slurm_job_id=slurm_job_id,
            exit_code=exit_code,
        )
    
    logger.info(f"Job finished with state: {state}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
