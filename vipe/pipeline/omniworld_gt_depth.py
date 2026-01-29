# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OmniWorld GT Depth Pipeline.

Uses VIPE's standard pipeline (GeoCalib + SLAM) but with OmniWorld's GT depth.
The depth is scale-invariant, so we first compute metric scale via MoGe2 alignment.
"""

import gc
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from vipe.pipeline import AnnotationPipelineOutput, Pipeline
from vipe.pipeline.processors import GeoCalibIntrinsicsProcessor, TrackAnythingProcessor
from vipe.priors.depth.moge_v2 import MoGeV2Model, focal_length_to_fov_degrees
from vipe.priors.depth.pi3x_moge import mask_aware_nearest_resize_robust
from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.streams.omniworld_stream import (
    OmniWorldGTDepthStream,
    OmniWorldSlidingWindowStream,
    get_scene_total_frames,
    load_camera_data,
    load_omniworld_depth,
    load_split_info,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.visualization import save_projection_video

try:
    from moge.utils.alignment import align_points_scale_z_shift
except ImportError:
    align_points_scale_z_shift = None

logger = logging.getLogger(__name__)


def depthmap_to_camera_coordinates(depthmap: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
    """
    Project depth map to camera coordinate point cloud.
    
    Args:
        depthmap: (H, W) depth values
        camera_intrinsics: (3, 3) intrinsics matrix
    
    Returns:
        pointmap: (H, W, 3) XYZ coordinates in camera space
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    fu = camera_intrinsics[0, 0]
    fv = camera_intrinsics[1, 1]
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return X_cam


def compute_chunk_metric_scale(
    scene_dir: Path,
    split_idx: int,
    frame_indices: List[int],
    moge_model: MoGeV2Model,
    align_resolution: int = 64,
    focal_range: Tuple[float, float] = (20, 3000),
    moge_batch_size: int = 4,
    outlier_std: float = 2.0,
    max_frames_for_scale: int = 50,
) -> float:
    """
    Compute metric scale for a chunk of frames using MoGe2 alignment.
    
    Args:
        scene_dir: Path to scene directory
        split_idx: Split index
        frame_indices: List of local frame indices to process
        moge_model: Pre-loaded MoGe2 model
        align_resolution: Resolution for alignment (e.g., 64x64)
        focal_range: Valid focal length range [min, max]
        moge_batch_size: Batch size for MoGe2 inference
        outlier_std: Number of std deviations to filter outliers
        max_frames_for_scale: Maximum number of frames to use for scale estimation
    
    Returns:
        metric_scale: Scale to apply to GT depth
    """
    if align_points_scale_z_shift is None:
        logger.warning("align_points_scale_z_shift not available, using scale=1.0")
        return 1.0
    
    # Load camera data
    split_info = load_split_info(scene_dir)
    global_indices = split_info["split"][split_idx]
    intrinsics, _ = load_camera_data(scene_dir, split_idx)
    
    # Filter good frames (valid focal length range)
    good_frames = []
    for local_idx in frame_indices:
        if local_idx >= len(intrinsics):
            continue
        fx = intrinsics[local_idx, 0, 0]
        fy = intrinsics[local_idx, 1, 1]
        if focal_range[0] <= fx <= focal_range[1] and focal_range[0] <= fy <= focal_range[1]:
            good_frames.append(local_idx)
    
    if not good_frames:
        logger.warning(f"No frames with valid focal length in range {focal_range}, using scale=1.0")
        return 1.0
    
    # Subsample if too many frames
    if len(good_frames) > max_frames_for_scale:
        step = len(good_frames) // max_frames_for_scale
        good_frames = good_frames[::step][:max_frames_for_scale]
    
    logger.info(f"Computing metric scale from {len(good_frames)} good frames...")
    
    scales = []
    
    for local_idx in pbar(good_frames, desc="Metric scale estimation"):
        global_idx = global_indices[local_idx]
        
        # Load RGB
        rgb_path = scene_dir / "color" / f"{global_idx:06d}.png"
        if not rgb_path.exists():
            continue
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        
        # Load depth
        depth_path = scene_dir / "depth" / f"{global_idx:06d}.png"
        if not depth_path.exists():
            continue
        gt_depth, valid = load_omniworld_depth(depth_path)
        
        # Get intrinsics
        K = intrinsics[local_idx]
        
        # Project GT depth to point cloud
        gt_pts = depthmap_to_camera_coordinates(gt_depth, K)
        gt_pts = torch.from_numpy(gt_pts).cuda()
        valid_tensor = torch.from_numpy(valid).cuda()
        
        # Prepare RGB for MoGe2
        rgb_tensor = torch.from_numpy(rgb).float().cuda() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Run MoGe2
        fx = K[0, 0]
        fov_deg = focal_length_to_fov_degrees(fx, W)
        
        try:
            with torch.no_grad():
                moge_out = moge_model.forward(rgb_tensor, fov_x=fov_deg)
            moge_pts = moge_out['points'][0]  # (H, W, 3)
            moge_mask = moge_out.get('mask', torch.ones_like(moge_pts[..., 0]))
            if moge_mask.dim() == 3:
                moge_mask = moge_mask.squeeze(0)
            moge_mask = moge_mask.bool()
        except Exception as e:
            logger.warning(f"MoGe2 failed for frame {local_idx}: {e}")
            continue
        
        # Combine masks
        combined_mask = valid_tensor & moge_mask
        
        if combined_mask.sum() < 100:
            continue
        
        # Downsample for alignment
        indices, lr_mask = mask_aware_nearest_resize_robust(
            combined_mask, align_resolution, align_resolution
        )
        ni, nj = indices
        
        gt_pts_lr = gt_pts[ni, nj]
        moge_pts_lr = moge_pts[ni, nj]
        
        # Weight by inverse depth (closer points more important)
        weights = 1.0 / moge_pts_lr[..., 2].clamp(min=1e-3)
        
        if lr_mask.sum() >= 10:
            try:
                # align_points_scale_z_shift aligns src to tgt
                # We want: gt_scaled = gt * scale such that gt_scaled matches moge
                # So we align gt to moge, get scale s where gt * s ≈ moge
                scale, _ = align_points_scale_z_shift(
                    gt_pts_lr[lr_mask].unsqueeze(0),
                    moge_pts_lr[lr_mask].unsqueeze(0),
                    weights[lr_mask].unsqueeze(0),
                )
                scale_val = scale.item()
                
                # Sanity check: reject extremely abnormal scales
                if scale_val > 1e-6 and scale_val < 1e6 and torch.isfinite(scale):
                    scales.append(scale_val)
            except Exception as e:
                logger.warning(f"Alignment failed for frame {local_idx}: {e}")
                continue
    
    # Filter outliers and compute final scale
    if not scales:
        logger.warning("No valid scales computed, using scale=1.0")
        return 1.0
    
    scales = np.array(scales)
    median_scale = np.median(scales)
    std_scale = np.std(scales)
    
    if std_scale > 0:
        filtered_scales = scales[np.abs(scales - median_scale) < outlier_std * std_scale]
    else:
        filtered_scales = scales
    
    if len(filtered_scales) == 0:
        final_scale = median_scale
    else:
        final_scale = float(np.median(filtered_scales))
    
    logger.info(f"Computed metric scale: {final_scale:.4f} (from {len(filtered_scales)}/{len(scales)} frames)")
    
    return final_scale


class OmniWorldGTDepthPipeline(Pipeline):
    """
    Pipeline for OmniWorld dataset with GT depth.
    
    1. Pre-computes metric scale using MoGe2 alignment
    2. Creates stream with scaled GT depth
    3. Runs standard VIPE pipeline (GeoCalib + SLAM)
    """
    
    def __init__(
        self,
        init: DictConfig,
        slam: DictConfig,
        post: DictConfig,
        output: DictConfig,
        scale_estimation: DictConfig,
    ) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.scale_cfg = scale_estimation
        
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)
        
        # Pre-load MoGe2 model for scale estimation
        self.moge_model = None
    
    def _ensure_moge_loaded(self):
        if self.moge_model is None:
            logger.info("Loading MoGe2 model for scale estimation...")
            self.moge_model = MoGeV2Model()
    
    def should_filter(self, stream_name: str) -> bool:
        if self.out_cfg.get("skip_exists", False):
            artifact_path = io.ArtifactPath(self.out_path, stream_name)
            if artifact_path.meta_info_path.exists():
                return True
        return False
    
    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        """Add GeoCalib and other init processors (but skip metric_depth assertion)."""
        init_processors: list[StreamProcessor] = []
        
        # Note: We don't assert METRIC_DEPTH not in attributes because our stream provides it
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        
        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        
        if self.init_cfg.get("instance") is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        
        return ProcessedVideoStream(video_stream, init_processors)
    
    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        """Add post processors - for GT depth, we typically skip depth alignment."""
        if slam_output.per_frame_intrinsics:
            intrinsics_list = list(slam_output.intrinsics[:, view_idx])
        else:
            intrinsics_list = [slam_output.intrinsics[view_idx]] * len(video_stream)

        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),
                    FrameAttribute.INTRINSICS: intrinsics_list,
                }
            )
        ]
        
        # Skip depth alignment if depth_align_model is null
        depth_align_model = self.post_cfg.get("depth_align_model")
        if depth_align_model is not None:
            logger.warning(f"depth_align_model={depth_align_model} specified but GT depth is used. Skipping.")
        
        return ProcessedVideoStream(video_stream, post_processors)
    
    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        """
        Run the pipeline on an OmniWorldGTDepthStream.
        
        The stream should already have metric_scale applied (computed externally or via
        the run_with_scene method).
        """
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()
        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None
        
        annotate_output = AnnotationPipelineOutput()
        
        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been processed already, skip it!!")
            return annotate_output
        
        # Add init processors (GeoCalib, etc.)
        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True)
            for video_stream in video_streams
        ]
        
        # Run SLAM
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)
        
        # Save intermediate SLAM outputs
        if self.out_cfg.get("save_slam_intermediate", False):
            for view_idx, artifact_path in enumerate(artifact_paths):
                try:
                    io.save_slam_intermediate_artifacts(artifact_path, slam_output, view_idx=view_idx)
                except Exception:
                    logger.exception("Failed saving intermediate SLAM artifacts for view_idx=%d", view_idx)
        
        # Clean up SLAM system
        del slam_pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output
        
        # Add post processors
        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]
        
        # Save artifacts
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    slam_output.metrics["ba_residual"] = slam_output.ba_residual
                    pickle.dump(slam_output.metrics, f)
            
            if self.out_cfg.get("save_viz", False):
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.get("viz_downsample", 4),
                    self.out_cfg.get("viz_attributes", [["rgb", "depth"]]),
                )
            
            if self.out_cfg.get("save_slam_map", False) and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)
        
        if self.return_output_streams:
            annotate_output.output_streams = output_streams
        
        return annotate_output
    
    def run_with_scene(
        self,
        scene_dir: Path,
        split_idx: int,
        frame_start: int = 0,
        frame_end: int = -1,
        frame_skip: int = 1,
        precomputed_scale: Optional[float] = None,
    ) -> AnnotationPipelineOutput:
        """
        Convenience method to run on an OmniWorld scene.
        
        Computes metric scale automatically if not provided.
        """
        scene_dir = Path(scene_dir)
        
        # Determine frame range
        split_info = load_split_info(scene_dir)
        total_frames = len(split_info["split"][split_idx])
        actual_end = frame_end if frame_end != -1 else total_frames
        frame_indices = list(range(frame_start, actual_end, frame_skip))
        
        # Compute metric scale if not provided
        if precomputed_scale is not None:
            metric_scale = precomputed_scale
            logger.info(f"Using precomputed metric scale: {metric_scale:.4f}")
        else:
            self._ensure_moge_loaded()
            metric_scale = compute_chunk_metric_scale(
                scene_dir=scene_dir,
                split_idx=split_idx,
                frame_indices=frame_indices,
                moge_model=self.moge_model,
                align_resolution=self.scale_cfg.get("align_resolution", 64),
                focal_range=tuple(self.scale_cfg.get("focal_range", [20, 3000])),
                moge_batch_size=self.scale_cfg.get("moge_batch_size", 4),
                outlier_std=self.scale_cfg.get("outlier_std", 2.0),
            )
            
            # Free MoGe2 model to save GPU memory
            del self.moge_model
            self.moge_model = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # Create stream with computed scale
        stream = OmniWorldGTDepthStream(
            scene_dir=scene_dir,
            split_idx=split_idx,
            metric_scale=metric_scale,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
        )
        
        return self.run(stream)
    
    def run_with_sliding_window(
        self,
        scene_dir: Path,
        frame_start: int,
        frame_end: int,
        precomputed_scale: Optional[float] = None,
        chunk_name: Optional[str] = None,
    ) -> AnnotationPipelineOutput:
        """
        Run pipeline on a sliding window chunk (ignoring split boundaries).
        
        This method works with global frame indices directly, making it suitable
        for processing arbitrary frame ranges across a scene.
        
        Args:
            scene_dir: Path to scene directory
            frame_start: Start frame index (global)
            frame_end: End frame index (global, exclusive)
            precomputed_scale: Pre-computed metric scale (if None, will be computed)
            chunk_name: Optional chunk name for output naming
        """
        scene_dir = Path(scene_dir)
        
        # Get total frames
        total_frames = get_scene_total_frames(scene_dir)
        actual_end = min(frame_end, total_frames)
        frame_indices = list(range(frame_start, actual_end))
        
        if not frame_indices:
            raise ValueError(f"No frames in range [{frame_start}, {actual_end})")
        
        # Compute metric scale if not provided
        if precomputed_scale is not None:
            metric_scale = precomputed_scale
            logger.info(f"Using precomputed metric scale: {metric_scale:.4f}")
        else:
            self._ensure_moge_loaded()
            metric_scale = compute_sliding_window_metric_scale(
                scene_dir=scene_dir,
                frame_indices=frame_indices,
                moge_model=self.moge_model,
                align_resolution=self.scale_cfg.get("align_resolution", 64),
                focal_range=tuple(self.scale_cfg.get("focal_range", [20, 3000])),
                moge_batch_size=self.scale_cfg.get("moge_batch_size", 4),
                outlier_std=self.scale_cfg.get("outlier_std", 2.0),
            )
            
            # Free MoGe2 model to save GPU memory
            del self.moge_model
            self.moge_model = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # Create stream with computed scale
        stream = OmniWorldSlidingWindowStream(
            scene_dir=scene_dir,
            frame_start=frame_start,
            frame_end=actual_end,
            metric_scale=metric_scale,
            name=chunk_name,
        )
        
        return self.run(stream)


def compute_sliding_window_metric_scale(
    scene_dir: Path,
    frame_indices: List[int],
    moge_model: MoGeV2Model,
    align_resolution: int = 64,
    focal_range: Tuple[float, float] = (20, 3000),
    moge_batch_size: int = 4,
    outlier_std: float = 2.0,
    max_frames_for_scale: int = 50,
) -> float:
    """
    Compute metric scale for sliding window frames using MoGe2 alignment.
    
    Unlike compute_chunk_metric_scale, this works with global frame indices
    directly (not split-based).
    
    Args:
        scene_dir: Path to scene directory
        frame_indices: List of global frame indices to process
        moge_model: Pre-loaded MoGe2 model
        align_resolution: Resolution for alignment (e.g., 64x64)
        focal_range: Valid focal length range [min, max]
        moge_batch_size: Batch size for MoGe2 inference
        outlier_std: Number of std deviations to filter outliers
        max_frames_for_scale: Maximum number of frames to use for scale estimation
    
    Returns:
        metric_scale: Scale to apply to GT depth
    """
    if align_points_scale_z_shift is None:
        logger.warning("align_points_scale_z_shift not available, using scale=1.0")
        return 1.0
    
    # For sliding window, we need to find intrinsics for these frames
    # We'll check each split to find matching frames
    split_info = load_split_info(scene_dir)
    
    # Build a mapping from global frame index to (split_idx, local_idx)
    global_to_split = {}
    for split_idx, split_frames in enumerate(split_info["split"]):
        for local_idx, global_idx in enumerate(split_frames):
            global_to_split[global_idx] = (split_idx, local_idx)
    
    # Load all split camera data (cache to avoid reloading)
    split_camera_data = {}
    
    def get_intrinsics_for_frame(global_idx: int) -> Optional[np.ndarray]:
        """Get intrinsics for a global frame index."""
        if global_idx not in global_to_split:
            return None
        split_idx, local_idx = global_to_split[global_idx]
        if split_idx not in split_camera_data:
            try:
                intrinsics, _ = load_camera_data(scene_dir, split_idx)
                split_camera_data[split_idx] = intrinsics
            except Exception:
                return None
        intrinsics = split_camera_data[split_idx]
        if local_idx >= len(intrinsics):
            return None
        return intrinsics[local_idx]
    
    # Filter good frames (valid focal length range)
    good_frames = []
    for global_idx in frame_indices:
        K = get_intrinsics_for_frame(global_idx)
        if K is None:
            continue
        fx = K[0, 0]
        fy = K[1, 1]
        if focal_range[0] <= fx <= focal_range[1] and focal_range[0] <= fy <= focal_range[1]:
            good_frames.append((global_idx, K))
    
    if not good_frames:
        # No frames with valid intrinsics - use a default intrinsics estimate
        logger.warning(f"No frames with valid intrinsics in range {focal_range}")
        # Try to estimate from image size
        first_frame = frame_indices[0]
        rgb_path = scene_dir / "color" / f"{first_frame:06d}.png"
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            if rgb is not None:
                H, W = rgb.shape[:2]
                # Use a reasonable default focal length estimate
                fx = fy = max(H, W)
                cx, cy = W / 2, H / 2
                default_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                good_frames = [(global_idx, default_K) for global_idx in frame_indices[:max_frames_for_scale]]
                logger.warning(f"Using default intrinsics: fx={fx}, fy={fy}")
    
    if not good_frames:
        logger.warning("Could not determine intrinsics, using scale=1.0")
        return 1.0
    
    # Subsample if too many frames
    if len(good_frames) > max_frames_for_scale:
        step = len(good_frames) // max_frames_for_scale
        good_frames = good_frames[::step][:max_frames_for_scale]
    
    logger.info(f"Computing metric scale from {len(good_frames)} frames...")
    
    scales = []
    
    for global_idx, K in pbar(good_frames, desc="Metric scale estimation"):
        # Load RGB
        rgb_path = scene_dir / "color" / f"{global_idx:06d}.png"
        if not rgb_path.exists():
            continue
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        
        # Load depth
        depth_path = scene_dir / "depth" / f"{global_idx:06d}.png"
        if not depth_path.exists():
            continue
        gt_depth, valid = load_omniworld_depth(depth_path)
        
        # Project GT depth to point cloud
        gt_pts = depthmap_to_camera_coordinates(gt_depth, K)
        gt_pts = torch.from_numpy(gt_pts).cuda()
        valid_tensor = torch.from_numpy(valid).cuda()
        
        # Prepare RGB for MoGe2
        rgb_tensor = torch.from_numpy(rgb).float().cuda() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Run MoGe2
        fx = K[0, 0]
        fov_deg = focal_length_to_fov_degrees(fx, W)
        
        try:
            with torch.no_grad():
                moge_out = moge_model.forward(rgb_tensor, fov_x=fov_deg)
            moge_pts = moge_out['points'][0]  # (H, W, 3)
            moge_mask = moge_out.get('mask', torch.ones_like(moge_pts[..., 0]))
            if moge_mask.dim() == 3:
                moge_mask = moge_mask.squeeze(0)
            moge_mask = moge_mask.bool()
        except Exception as e:
            logger.warning(f"MoGe2 failed for frame {global_idx}: {e}")
            continue
        
        # Combine masks
        combined_mask = valid_tensor & moge_mask
        
        if combined_mask.sum() < 100:
            continue
        
        # Downsample for alignment
        indices, lr_mask = mask_aware_nearest_resize_robust(
            combined_mask, align_resolution, align_resolution
        )
        ni, nj = indices
        
        gt_pts_lr = gt_pts[ni, nj]
        moge_pts_lr = moge_pts[ni, nj]
        
        # Weight by inverse depth (closer points more important)
        weights = 1.0 / moge_pts_lr[..., 2].clamp(min=1e-3)
        
        if lr_mask.sum() >= 10:
            try:
                scale, _ = align_points_scale_z_shift(
                    gt_pts_lr[lr_mask].unsqueeze(0),
                    moge_pts_lr[lr_mask].unsqueeze(0),
                    weights[lr_mask].unsqueeze(0),
                )
                scale_val = scale.item()
                
                # Sanity check: reject extremely abnormal scales
                if scale_val > 1e-6 and scale_val < 1e6 and torch.isfinite(scale):
                    scales.append(scale_val)
            except Exception as e:
                logger.warning(f"Alignment failed for frame {global_idx}: {e}")
                continue
    
    # Filter outliers and compute final scale
    if not scales:
        logger.warning("No valid scales computed, using scale=1.0")
        return 1.0
    
    scales = np.array(scales)
    median_scale = np.median(scales)
    std_scale = np.std(scales)
    
    if std_scale > 0:
        filtered_scales = scales[np.abs(scales - median_scale) < outlier_std * std_scale]
    else:
        filtered_scales = scales
    
    if len(filtered_scales) == 0:
        final_scale = median_scale
    else:
        final_scale = float(np.median(filtered_scales))
    
    logger.info(f"Computed metric scale: {final_scale:.4f} (from {len(filtered_scales)}/{len(scales)} frames)")
    
    return final_scale
