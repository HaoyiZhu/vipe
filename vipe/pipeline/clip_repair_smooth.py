# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ClipRepairSmooth: Simple clip repair pipeline that smooths out pose jumps.

Instead of trying to align Pi3X to SLAM (which can fail), this approach:
1. Runs SLAM as usual
2. Detects large pose jumps (bad clips)
3. For each bad clip, uses Pi3X VO to get relative poses
4. Anchors Pi3X trajectory to SLAM at one end and smoothly blends

The key insight is that Pi3X VO gives good *relative* poses but unknown scale,
while SLAM gives good *absolute* scale and poses for good frames.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation, Slerp

from vipe.ext.lietorch import SE3
from vipe.pipeline import AnnotationPipelineOutput
from vipe.pipeline.chunked_robust import ChunkedRobustAnnotationPipeline
from vipe.pipeline.long_sequence import AssignInstancePhrasesProcessor, SlicedVideoStream
from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import save_projection_video

from .processors import Pi3XMetricDepthProcessor, Pi3XVOInitPoseProcessor

logger = logging.getLogger(__name__)


@dataclass
class ClipStats:
    index: int
    start: int
    end: int
    reason: str
    method: str  # "interpolate" or "pi3x_anchor"
    scale_used: float
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
            "method": self.method,
            "scale_used": self.scale_used,
        }


def pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> tuple[float, float]:
    """Compute translation and rotation distance between two poses."""
    trans_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1, 1)
    rot_dist = np.degrees(np.arccos(cos_theta))
    
    return trans_dist, rot_dist


def interpolate_poses(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """
    Interpolate n_frames poses between pose_start and pose_end (exclusive of endpoints).
    Uses SLERP for rotation and linear interpolation for translation.
    """
    if n_frames <= 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    
    # Rotation SLERP
    R_start = Rotation.from_matrix(pose_start[:3, :3])
    R_end = Rotation.from_matrix(pose_end[:3, :3])
    
    key_times = [0, 1]
    key_rots = Rotation.concatenate([R_start, R_end])
    slerp = Slerp(key_times, key_rots)
    
    # Alpha values (excluding endpoints)
    alphas = np.linspace(0, 1, n_frames + 2)[1:-1]
    
    interp_rots = slerp(alphas)
    
    # Translation linear interpolation
    t_start = pose_start[:3, 3]
    t_end = pose_end[:3, 3]
    
    result = np.zeros((n_frames, 4, 4), dtype=np.float32)
    for i, alpha in enumerate(alphas):
        result[i, :3, :3] = interp_rots[i].as_matrix()
        result[i, :3, 3] = (1 - alpha) * t_start + alpha * t_end
        result[i, 3, 3] = 1.0
    
    return result


def estimate_scale_from_overlap(
    slam_poses: np.ndarray,  # SLAM poses in world frame
    pi3x_poses: np.ndarray,  # Pi3X poses (relative, starting from identity)
    overlap_start: int,  # Start index in overlap region
    overlap_end: int,    # End index in overlap region (exclusive)
) -> float:
    """
    Estimate scale by comparing trajectory lengths in overlapping region.
    """
    if overlap_end - overlap_start < 2:
        return 1.0
    
    # Compute trajectory lengths
    slam_dists = []
    pi3x_dists = []
    
    for i in range(overlap_start, overlap_end - 1):
        slam_dist = np.linalg.norm(slam_poses[i + 1, :3, 3] - slam_poses[i, :3, 3])
        pi3x_dist = np.linalg.norm(pi3x_poses[i + 1, :3, 3] - pi3x_poses[i, :3, 3])
        slam_dists.append(slam_dist)
        pi3x_dists.append(pi3x_dist)
    
    slam_total = sum(slam_dists)
    pi3x_total = sum(pi3x_dists)
    
    if pi3x_total < 1e-6:
        return 1.0
    
    return slam_total / pi3x_total


def anchor_and_blend_trajectory(
    slam_poses: np.ndarray,     # (N, 4, 4) full SLAM trajectory
    pi3x_poses: np.ndarray,     # (M, 4, 4) Pi3X trajectory for clip
    clip_start: int,            # Where the clip starts in full trajectory
    clip_end: int,              # Where the clip ends in full trajectory
    bad_start: int,             # Where the bad region starts
    bad_end: int,               # Where the bad region ends
    align_overlap: int,         # Number of overlap frames for alignment
    blend_frames: int = 5,      # Number of frames to blend at boundaries
    max_scale: float = 10.0,    # Maximum acceptable scale
    min_scale: float = 0.1,     # Minimum acceptable scale
) -> tuple[np.ndarray, float, bool]:
    """
    Anchor Pi3X trajectory to SLAM trajectory and blend at boundaries.
    
    Returns the modified poses for the bad region and the scale used.
    """
    # The Pi3X clip spans [clip_start, clip_end)
    # The bad region is [bad_start, bad_end)
    # We have overlap before bad_start and after bad_end
    
    clip_len = clip_end - clip_start
    
    # Left overlap region (in clip coordinates)
    left_overlap_len = bad_start - clip_start
    # Right overlap region
    right_overlap_len = clip_end - bad_end
    
    # Estimate scale from left overlap if available
    if left_overlap_len >= 2:
        scale = estimate_scale_from_overlap(
            slam_poses[clip_start:bad_start],
            pi3x_poses[:left_overlap_len],
            0,
            left_overlap_len,
        )
    elif right_overlap_len >= 2:
        # Use right overlap for scale
        right_start = bad_end - clip_start
        right_end = clip_len
        scale = estimate_scale_from_overlap(
            slam_poses[bad_end:clip_end],
            pi3x_poses[right_start:right_end],
            0,
            right_end - right_start,
        )
    else:
        scale = 1.0
    
    # Check if scale is reasonable
    alignment_valid = True
    if scale < min_scale or scale > max_scale:
        logger.warning(f"  Scale {scale:.4f} out of range [{min_scale}, {max_scale}], alignment may be unreliable")
        alignment_valid = False
    
    # Clamp scale to reasonable range
    scale = np.clip(scale, min_scale, max_scale)
    
    # Transform Pi3X poses to world frame:
    # 1. Scale the translations
    # 2. Anchor to SLAM at the left boundary
    
    # The anchor point is the last SLAM pose before bad region
    anchor_idx = bad_start - 1
    if anchor_idx < 0:
        anchor_idx = 0
    anchor_pose = slam_poses[anchor_idx]
    
    # Pi3X pose at the corresponding position
    pi3x_anchor_idx = anchor_idx - clip_start
    if pi3x_anchor_idx < 0:
        pi3x_anchor_idx = 0
    pi3x_anchor = pi3x_poses[pi3x_anchor_idx]
    
    # Compute the transformation that maps pi3x_anchor to anchor_pose
    # anchor_pose = T @ scaled_pi3x_anchor
    # T = anchor_pose @ inv(scaled_pi3x_anchor)
    
    scaled_pi3x_anchor = pi3x_anchor.copy()
    scaled_pi3x_anchor[:3, 3] *= scale
    
    T = anchor_pose @ np.linalg.inv(scaled_pi3x_anchor)
    
    # Apply transformation to all Pi3X poses in the clip
    transformed_pi3x = np.zeros_like(pi3x_poses)
    for i in range(len(pi3x_poses)):
        scaled = pi3x_poses[i].copy()
        scaled[:3, 3] *= scale
        transformed_pi3x[i] = T @ scaled
    
    # Now copy the bad region poses
    bad_len = bad_end - bad_start
    result_poses = np.zeros((bad_len, 4, 4), dtype=np.float32)
    
    for i in range(bad_len):
        global_idx = bad_start + i
        pi3x_idx = global_idx - clip_start
        
        if pi3x_idx >= 0 and pi3x_idx < len(transformed_pi3x):
            result_poses[i] = transformed_pi3x[pi3x_idx]
        else:
            # Fallback to interpolation
            result_poses[i] = slam_poses[global_idx]
    
    # Blend at right boundary if needed
    if blend_frames > 0 and bad_end < len(slam_poses):
        right_anchor = slam_poses[bad_end]
        for i in range(min(blend_frames, bad_len)):
            idx = bad_len - 1 - i
            alpha = (i + 1) / (blend_frames + 1)
            
            # Blend position
            result_poses[idx, :3, 3] = (
                (1 - alpha) * right_anchor[:3, 3] + alpha * result_poses[idx, :3, 3]
            )
    
    return result_poses, scale, alignment_valid


class ClipRepairSmoothAnnotationPipeline(ChunkedRobustAnnotationPipeline):
    """
    Simplified clip repair that uses Pi3X VO with proper anchoring.
    """

    def __init__(
        self,
        init: DictConfig,
        slam: DictConfig,
        post: DictConfig,
        output: DictConfig,
        robust: DictConfig | None = None,
        clip: DictConfig | None = None,
    ) -> None:
        super().__init__(init, slam, post, output, robust, chunked=clip)
        self.clip_cfg = clip if clip is not None else OmegaConf.create({})

    @staticmethod
    def _rolling_mean(values: list[float], window: int) -> list[float]:
        if window <= 1:
            return values
        half = window // 2
        out = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            out.append(float(sum(values[lo:hi]) / max(1, hi - lo)))
        return out

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
        if not ranges:
            return []
        ranges = sorted(ranges, key=lambda x: x[0])
        merged = []
        cur_s, cur_e = ranges[0]
        for s, e in ranges[1:]:
            if s <= cur_e + merge_gap:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _detect_bad_clips(
        self,
        stream: VideoStream,
        trajectory: SE3,
        total_frames: int,
        slam_ba_residuals: list[float] | None = None,
    ) -> list[tuple[int, int]]:
        cfg = self.clip_cfg.get("detect", DictConfig({}))
        if not cfg or not cfg.get("enabled", True):
            return []

        window = int(cfg.get("window", 5))
        min_mask_ratio = float(cfg.get("min_mask_ratio", 0.05))
        min_mask_mean_ratio = float(cfg.get("min_mask_mean_ratio", 0.1))
        use_pose_jump = bool(cfg.get("use_pose_jump", True))
        pose_jump_translation = float(cfg.get("pose_jump_translation", 2.5))
        pose_jump_rotation_deg = float(cfg.get("pose_jump_rotation_deg", 25.0))
        use_slam_ba_residuals = bool(cfg.get("use_slam_ba_residuals", False))
        slam_ba_residual_thresh = float(cfg.get("slam_ba_residual_thresh", 0.005))
        slam_ba_residual_window = int(cfg.get("slam_ba_residual_window", window))
        min_len = int(cfg.get("min_len", 1))
        expand = int(cfg.get("expand", 10))
        merge_gap = int(cfg.get("merge_gap", 5))

        mask_ratios = []
        for frame in stream:
            if frame.mask is None:
                mask_ratio = 1.0
            else:
                mask = frame.mask
                if mask.dtype != torch.bool:
                    mask = mask > 0
                mask_ratio = float(mask.float().mean().item())
            mask_ratios.append(mask_ratio)

        mask_mean = self._rolling_mean(mask_ratios, window)
        ba_residual_mean = None
        if use_slam_ba_residuals and slam_ba_residuals is not None:
            if len(slam_ba_residuals) == total_frames:
                ba_residual_mean = self._rolling_mean(slam_ba_residuals, slam_ba_residual_window)

        bad_flags = []
        for idx in range(total_frames):
            bad = False
            if mask_ratios[idx] < min_mask_ratio or mask_mean[idx] < min_mask_mean_ratio:
                bad = True
            if use_pose_jump and idx > 0:
                pose_prev = trajectory[idx - 1]
                pose_curr = trajectory[idx]
                rel = (pose_prev.inv() * pose_curr).matrix().detach().cpu().numpy()
                translation = float(np.linalg.norm(rel[:3, 3]))
                trace = float(np.trace(rel[:3, :3]))
                cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
                rotation_deg = float(np.degrees(np.arccos(cos_theta)))
                if translation > pose_jump_translation or rotation_deg > pose_jump_rotation_deg:
                    bad = True
            if ba_residual_mean is not None and ba_residual_mean[idx] > slam_ba_residual_thresh:
                bad = True
            bad_flags.append(bad)

        ranges = []
        start = None
        for idx, bad in enumerate(bad_flags):
            if bad and start is None:
                start = idx
            elif not bad and start is not None:
                if idx - start >= min_len:
                    ranges.append((start, idx))
                start = None
        if start is not None and total_frames - start >= min_len:
            ranges.append((start, total_frames))

        expanded = []
        for start, end in ranges:
            s = max(0, start - expand)
            e = min(total_frames, end + expand)
            expanded.append((s, e))

        return self._merge_ranges(expanded, merge_gap)

    def _collect_arrays(self, stream: VideoStream) -> dict[str, Any]:
        extrinsics = []
        intrinsics = []
        depths = []
        masks = []
        instances = []
        instance_phrases: dict[int, str] = {}
        for frame in stream:
            pose = frame.pose.matrix() if frame.pose is not None else torch.eye(4, device=frame.rgb.device)
            extrinsics.append(pose.detach().cpu().numpy())
            intr = frame.intrinsics[:4] if frame.intrinsics is not None else torch.zeros(4, device=pose.device)
            intrinsics.append(intr.detach().cpu().numpy())
            depths.append(frame.metric_depth.detach().cpu().numpy() if frame.metric_depth is not None else None)
            masks.append(frame.mask.detach().cpu().numpy().astype(bool) if frame.mask is not None else None)
            instances.append(frame.instance.detach().cpu().numpy().astype(np.uint8) if frame.instance is not None else None)
            if frame.instance_phrases:
                instance_phrases.update(frame.instance_phrases)

        return {
            "extrinsic": np.stack(extrinsics),
            "intrinsic": np.stack(intrinsics),
            "depth": depths,
            "mask": masks,
            "instance": instances,
            "instance_phrases": instance_phrases if instance_phrases else None,
        }

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            if len(video_data) > 1:
                raise ValueError("ClipRepairSmooth pipeline supports single-view streams only.")
            video_stream = video_data[0]
            slam_rig = video_data.rig()
        else:
            assert isinstance(video_data, VideoStream)
            video_stream = video_data
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if self.should_filter(video_stream.name()):
            logger.info("%s has been processed already, skip it.", video_stream.name())
            return annotate_output

        # Step 1: Run init processors
        init_stream = self._add_init_processors(video_stream).cache("process", online=True)
        
        # Step 2: Run SLAM
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run([init_stream], rig=slam_rig, camera_type=self.camera_type)

        # Step 3: Add depth
        global_stream = self._add_post_processors(0, init_stream, slam_output).cache("depth", online=True)
        total_frames = len(global_stream)

        # Step 4: Detect bad clips
        detect_cfg = self.clip_cfg.get("detect", DictConfig({}))
        ba_residuals = None
        if detect_cfg.get("use_slam_ba_residuals", False):
            ba_residuals = slam_output.metrics.get("ba_residuals_per_frame")

        bad_ranges = self._detect_bad_clips(
            init_stream,
            slam_output.get_view_trajectory(0),
            total_frames,
            slam_ba_residuals=ba_residuals,
        )
        
        if not bad_ranges:
            logger.info("No bad clips detected, using full SLAM results.")

        # Step 5: Collect SLAM results
        global_data = self._collect_arrays(global_stream)
        clip_stats: list[ClipStats] = []

        align_overlap = int(self.clip_cfg.get("align_overlap", 15))
        pi3x_vo_cfg = self.init_cfg.get("pose_init", DictConfig({}))
        pi3x_vo_enabled = bool(self.clip_cfg.get("pi3x_vo_enabled", True))
        blend_frames = int(self.clip_cfg.get("boundary_blend", 5))
        use_interpolation_fallback = bool(self.clip_cfg.get("use_interpolation_fallback", True))

        # Step 6: Process each bad clip
        for idx, (start, end) in enumerate(bad_ranges):
            clip_start = max(0, start - align_overlap)
            clip_end = min(total_frames, end + align_overlap)
            
            logger.info(f"Processing bad clip {idx}: frames {start}-{end} (with overlap: {clip_start}-{clip_end})")
            
            method = "interpolate"
            scale_used = 1.0
            
            if pi3x_vo_enabled and pi3x_vo_cfg.get("enabled", True):
                # Run Pi3X VO on the clip
                clip_stream = SlicedVideoStream(init_stream, clip_start, clip_end)
                
                processors = [
                    Pi3XVOInitPoseProcessor(
                        clip_stream,
                        model=pi3x_vo_cfg.get("model", "yyfz233/Pi3X"),
                        chunk_size=pi3x_vo_cfg.get("chunk_size", 64),
                        overlap=pi3x_vo_cfg.get("overlap", 32),
                        conf_thre=pi3x_vo_cfg.get("conf_thre", 0.05),
                        dtype=pi3x_vo_cfg.get("dtype", "bf16"),
                        pose_convention=pi3x_vo_cfg.get("pose_convention", "c2w"),
                        return_depth=False,
                    ),
                    Pi3XMetricDepthProcessor(
                        model=self.clip_cfg.get("pi3x_model", "yyfz233/Pi3X"),
                        pixel_limit=int(self.clip_cfg.get("pixel_limit", 255000)),
                        batch_size=int(self.clip_cfg.get("depth_batch_size", 4)),
                        use_poses=True,
                    ),
                ]
                pi3x_stream = ProcessedVideoStream(clip_stream, processors).cache("pi3x_clip", online=True)
                pi3x_data = self._collect_arrays(pi3x_stream)
                
                try:
                    # Anchor Pi3X to SLAM and blend
                    repaired_poses, scale_used, alignment_valid = anchor_and_blend_trajectory(
                        global_data["extrinsic"],
                        pi3x_data["extrinsic"],
                        clip_start,
                        clip_end,
                        start,
                        end,
                        align_overlap,
                        blend_frames,
                    )
                    
                    if not alignment_valid:
                        logger.warning(f"  Alignment not reliable, falling back to interpolation")
                        method = "interpolate"
                    else:
                        # Update poses in global data
                        for i in range(end - start):
                            global_data["extrinsic"][start + i] = repaired_poses[i]
                            # Also update depth with scaled Pi3X depth
                            pi3x_idx = (start - clip_start) + i
                            if pi3x_idx < len(pi3x_data["depth"]) and pi3x_data["depth"][pi3x_idx] is not None:
                                global_data["depth"][start + i] = pi3x_data["depth"][pi3x_idx] * scale_used
                        
                        method = "pi3x_anchor"
                        logger.info(f"  Used Pi3X anchoring with scale={scale_used:.4f}")
                    
                except Exception as e:
                    logger.warning(f"  Pi3X anchoring failed: {e}, falling back to interpolation")
                    method = "interpolate"
            
            if method == "interpolate" and use_interpolation_fallback:
                # Simple interpolation between endpoints
                if start > 0 and end < total_frames:
                    pose_before = global_data["extrinsic"][start - 1]
                    pose_after = global_data["extrinsic"][end]
                    
                    interp_poses = interpolate_poses(pose_before, pose_after, end - start)
                    for i in range(end - start):
                        global_data["extrinsic"][start + i] = interp_poses[i]
                    
                    logger.info(f"  Used interpolation between frames {start-1} and {end}")
            
            clip_stats.append(
                ClipStats(
                    index=idx,
                    start=start,
                    end=end,
                    reason="pose_jump_or_mask",
                    method=method,
                    scale_used=scale_used,
                )
            )

        # Step 7: Build output stream
        merged_poses = [se3_matrix_to_se3(torch.from_numpy(p).float(), unbatch=True) for p in global_data["extrinsic"]]
        merged_intrinsics = [torch.from_numpy(i).float() for i in global_data["intrinsic"]]
        merged_depths = [torch.from_numpy(d).float() if d is not None else None for d in global_data["depth"]]
        merged_masks = [torch.from_numpy(m).bool() if m is not None else None for m in global_data["mask"]]
        merged_instances = [torch.from_numpy(inst).byte() if inst is not None else None for inst in global_data["instance"]]
        merged_camera_types = [self.camera_type] * total_frames

        stream_attributes: dict[FrameAttribute, list[Any]] = {
            FrameAttribute.POSE: merged_poses,
            FrameAttribute.INTRINSICS: merged_intrinsics,
            FrameAttribute.CAMERA_TYPE: merged_camera_types,
        }
        if any(d is not None for d in merged_depths):
            stream_attributes[FrameAttribute.METRIC_DEPTH] = merged_depths
        if any(m is not None for m in merged_masks):
            stream_attributes[FrameAttribute.MASK] = merged_masks
        if any(inst is not None for inst in merged_instances):
            stream_attributes[FrameAttribute.INSTANCE] = merged_instances

        instance_phrases_list = None
        if global_data.get("instance_phrases"):
            instance_phrases_list = [global_data["instance_phrases"]] * total_frames

        post_processors = [AssignAttributesProcessor(stream_attributes)]
        if instance_phrases_list is not None:
            post_processors.append(AssignInstancePhrasesProcessor(instance_phrases_list))

        output_stream = ProcessedVideoStream(video_stream, post_processors).cache("merged", online=True)

        # Save artifacts
        artifact_path = io.ArtifactPath(self.out_path, video_stream.name())
        if self.out_cfg.save_artifacts:
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info("Saving artifacts to %s", artifact_path)
            io.save_artifacts(artifact_path, output_stream)
            with artifact_path.meta_info_path.open("wb") as f:
                info = {
                    "pipeline": "clip_repair_smooth",
                    "bad_ranges": bad_ranges,
                    "clip_stats": [c.as_dict() for c in clip_stats],
                }
                pickle.dump(info, f)

        if self.out_cfg.save_viz:
            viz_attributes = self._sanitize_viz_attributes(self.out_cfg.viz_attributes, slam_output)
            save_projection_video(
                artifact_path.meta_vis_path,
                output_stream,
                slam_output,
                self.out_cfg.viz_downsample,
                viz_attributes,
            )

        if self.return_output_streams:
            annotate_output.output_streams = [output_stream]

        return annotate_output
