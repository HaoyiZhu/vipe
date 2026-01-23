# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ClipRepairV2: Improved clip repair pipeline with proper trajectory alignment.

Key fixes over the original clip_repair.py:
1. Align Pi3X trajectory to SLAM trajectory using Procrustes (camera position-based)
2. Use world-space points for depth alignment, not camera-space
3. Add smooth interpolation at boundaries
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

from vipe.ext.lietorch import SE3
from vipe.pipeline import AnnotationPipelineOutput
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.pipeline.long_sequence import AssignInstancePhrasesProcessor, SlicedVideoStream
from vipe.pipeline.chunked_robust import ChunkedRobustAnnotationPipeline
from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    VideoFrame,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import save_projection_video
from vipe.priors.depth.pi3x_moge import mask_aware_nearest_resize_robust

from .processors import Pi3XMetricDepthProcessor, Pi3XVOInitPoseProcessor

logger = logging.getLogger(__name__)


@dataclass
class ClipStats:
    index: int
    start: int
    end: int
    reason: str
    aligned: bool
    sparse_ba_applied: bool
    align_error: float = 0.0  # Add alignment quality metric

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
            "aligned": self.aligned,
            "sparse_ba_applied": self.sparse_ba_applied,
            "align_error": self.align_error,
        }


def procrustes_sim3(
    src_positions: torch.Tensor,  # (N, 3) camera positions in source (Pi3X) frame
    tgt_positions: torch.Tensor,  # (N, 3) camera positions in target (SLAM) frame
    weights: torch.Tensor | None = None,  # (N,) optional weights
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Procrustes analysis to find Sim3 (scale, rotation, translation) that aligns src to tgt.
    Returns (scale, R, t) such that: tgt ≈ scale * (R @ src) + t
    """
    device = src_positions.device
    n = src_positions.shape[0]
    
    if n < 3:
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    if weights is None:
        weights = torch.ones(n, device=device)
    
    # Normalize weights
    w_sum = weights.sum()
    if w_sum < 1e-9:
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    w_norm = weights / w_sum
    
    # Weighted centroids
    src_centroid = (w_norm[:, None] * src_positions).sum(dim=0)
    tgt_centroid = (w_norm[:, None] * tgt_positions).sum(dim=0)
    
    # Center the points
    src_centered = src_positions - src_centroid
    tgt_centered = tgt_positions - tgt_centroid
    
    # Weighted covariance for scale estimation
    src_var = (w_norm * (src_centered ** 2).sum(dim=1)).sum()
    tgt_var = (w_norm * (tgt_centered ** 2).sum(dim=1)).sum()
    
    if src_var < 1e-9:
        scale = 1.0
    else:
        scale = float(torch.sqrt(tgt_var / src_var).item())
    
    # Weighted SVD for rotation
    weighted_src = src_centered * torch.sqrt(w_norm)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(w_norm)[:, None]
    
    H = weighted_src.T @ weighted_tgt
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if torch.det(R) < 0:
        Vh_mod = Vh.clone()
        Vh_mod[-1] *= -1
        R = Vh_mod.T @ U.T
    
    # Compute translation
    t = tgt_centroid - scale * (R @ src_centroid)
    
    return scale, R, t


def apply_sim3_to_poses(
    poses: np.ndarray,  # (N, 4, 4) c2w poses
    scale: float,
    R: np.ndarray,  # (3, 3)
    t: np.ndarray,  # (3,)
) -> np.ndarray:
    """
    Apply Sim3 transformation to a set of c2w poses.
    For c2w poses: new_position = scale * R @ old_position + t
                   new_rotation = R @ old_rotation
    """
    new_poses = np.zeros_like(poses)
    for i in range(len(poses)):
        old_R = poses[i, :3, :3]
        old_t = poses[i, :3, 3]
        
        new_R = R @ old_R
        new_t = scale * (R @ old_t) + t
        
        new_poses[i, :3, :3] = new_R
        new_poses[i, :3, 3] = new_t
        new_poses[i, 3, 3] = 1.0
    
    return new_poses


def apply_sim3_to_depths(
    depths: list[np.ndarray | None],
    scale: float,
) -> list[np.ndarray | None]:
    """Apply scale to depths."""
    return [d * scale if d is not None else None for d in depths]


def interpolate_poses(
    pose_before: np.ndarray,  # (4, 4)
    pose_after: np.ndarray,  # (4, 4)
    n_interp: int,
) -> np.ndarray:
    """
    Linearly interpolate between two poses.
    Returns (n_interp, 4, 4) poses.
    """
    if n_interp <= 0:
        return np.zeros((0, 4, 4))
    
    from scipy.spatial.transform import Rotation, Slerp
    
    # Interpolate rotation via slerp
    R_before = Rotation.from_matrix(pose_before[:3, :3])
    R_after = Rotation.from_matrix(pose_after[:3, :3])
    
    key_times = [0, 1]
    key_rots = Rotation.concatenate([R_before, R_after])
    slerp = Slerp(key_times, key_rots)
    
    # Interpolation weights (excluding endpoints)
    alphas = np.linspace(0, 1, n_interp + 2)[1:-1]
    
    interp_rots = slerp(alphas)
    
    # Interpolate translation linearly
    t_before = pose_before[:3, 3]
    t_after = pose_after[:3, 3]
    
    result = np.zeros((n_interp, 4, 4))
    for i, alpha in enumerate(alphas):
        result[i, :3, :3] = interp_rots[i].as_matrix()
        result[i, :3, 3] = (1 - alpha) * t_before + alpha * t_after
        result[i, 3, 3] = 1.0
    
    return result


class ClipRepairV2AnnotationPipeline(ChunkedRobustAnnotationPipeline):
    """
    Improved clip repair pipeline:
    1) Run masks/intrinsics initialization + optional Pi3X VO init.
    2) Run a single global SLAM pass.
    3) Detect bad clips (mask coverage, pose jump, BA residual).
    4) For each bad clip:
       - Run Pi3X VO to get relative poses
       - Align Pi3X trajectory to SLAM trajectory using Procrustes on camera positions
       - Optional: refine with sparse BA
    5) Smoothly blend at boundaries.
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
        slam_depth_errors: list[float] | None = None,
        slam_ba_residual: float | None = None,
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
        use_slam_depth_error = bool(cfg.get("use_slam_depth_error", False))
        slam_depth_error_thresh = float(cfg.get("slam_depth_error_thresh", 0.2))
        slam_depth_error_window = int(cfg.get("slam_depth_error_window", window))
        use_slam_ba_residuals = bool(cfg.get("use_slam_ba_residuals", False))
        slam_ba_residual_thresh = float(cfg.get("slam_ba_residual_thresh", 0.005))
        slam_ba_residual_window = int(cfg.get("slam_ba_residual_window", window))
        use_global_ba_residual = bool(cfg.get("use_global_ba_residual", False))
        global_ba_residual_thresh = float(cfg.get("global_ba_residual_thresh", 0.005))
        min_len = int(cfg.get("min_len", 1))
        expand = int(cfg.get("expand", 10))
        merge_gap = int(cfg.get("merge_gap", 5))

        if use_global_ba_residual and slam_ba_residual is not None:
            if np.isfinite(slam_ba_residual) and slam_ba_residual > global_ba_residual_thresh:
                return [(0, total_frames)]

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
        depth_error_mean = None
        if use_slam_depth_error and slam_depth_errors is not None:
            depth_error_mean = self._rolling_mean(slam_depth_errors, slam_depth_error_window)
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
            if depth_error_mean is not None and depth_error_mean[idx] > slam_depth_error_thresh:
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

    def _align_trajectory(
        self,
        slam_poses: np.ndarray,  # (N, 4, 4) c2w from SLAM (good frames)
        pi3x_poses: np.ndarray,  # (M, 4, 4) c2w from Pi3X (clip frames)
        overlap_indices_slam: list[int],  # indices in slam_poses for overlap
        overlap_indices_pi3x: list[int],  # corresponding indices in pi3x_poses
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
        """
        Align Pi3X trajectory to SLAM trajectory using Procrustes on camera positions.
        Returns (scale, R, t, alignment_error).
        """
        if len(overlap_indices_slam) < 3 or len(overlap_indices_pi3x) < 3:
            logger.warning("Not enough overlap points for trajectory alignment")
            return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), float('inf')
        
        n = min(len(overlap_indices_slam), len(overlap_indices_pi3x))
        
        # Extract camera positions (c2w: position is the translation column)
        slam_positions = []
        pi3x_positions = []
        for i in range(n):
            slam_idx = overlap_indices_slam[i]
            pi3x_idx = overlap_indices_pi3x[i]
            slam_positions.append(slam_poses[slam_idx, :3, 3])
            pi3x_positions.append(pi3x_poses[pi3x_idx, :3, 3])
        
        slam_positions = torch.from_numpy(np.array(slam_positions)).float()
        pi3x_positions = torch.from_numpy(np.array(pi3x_positions)).float()
        
        # Procrustes alignment: find s, R, t such that slam ≈ s * R @ pi3x + t
        scale, R, t = procrustes_sim3(pi3x_positions, slam_positions)
        
        # Compute alignment error
        aligned_pi3x = scale * (pi3x_positions @ R.T) + t
        error = (aligned_pi3x - slam_positions).norm(dim=1).mean().item()
        
        return scale, R.numpy(), t.numpy(), error

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            if len(video_data) > 1:
                raise ValueError("Clip repair V2 pipeline supports single-view streams only.")
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

        # Step 1: Run init processors (masks, intrinsics, optional Pi3X VO init for poses)
        init_stream = self._add_init_processors(video_stream).cache("process", online=True)
        
        # Step 2: Run SLAM
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run([init_stream], rig=slam_rig, camera_type=self.camera_type)

        # Step 3: Add depth via post-processing
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
            slam_ba_residual=float(slam_output.ba_residual),
            slam_ba_residuals=ba_residuals,
        )
        
        if not bad_ranges:
            bad_ranges = []
            logger.info("No bad clips detected, using full SLAM results.")

        # Step 5: Collect global (SLAM) results
        global_data = self._collect_arrays(global_stream)
        clip_stats: list[ClipStats] = []

        align_overlap = int(self.clip_cfg.get("align_overlap", 10))
        pi3x_vo_cfg = self.init_cfg.get("pose_init", DictConfig({}))
        pi3x_vo_enabled = bool(self.clip_cfg.get("pi3x_vo_enabled", True))
        boundary_blend = int(self.clip_cfg.get("boundary_blend", 5))

        # Step 6: Process each bad clip
        for idx, (start, end) in enumerate(bad_ranges):
            clip_start = max(0, start - align_overlap)
            clip_end = min(total_frames, end + align_overlap)
            
            logger.info(f"Processing bad clip {idx}: frames {start}-{end} (with overlap: {clip_start}-{clip_end})")
            
            clip_stream = SlicedVideoStream(init_stream, clip_start, clip_end)

            # Run Pi3X VO to get poses for the clip
            processors = []
            if pi3x_vo_enabled and pi3x_vo_cfg.get("enabled", True):
                processors.append(
                    Pi3XVOInitPoseProcessor(
                        clip_stream,
                        model=pi3x_vo_cfg.get("model", "yyfz233/Pi3X"),
                        chunk_size=pi3x_vo_cfg.get("chunk_size", 64),
                        overlap=pi3x_vo_cfg.get("overlap", 32),
                        conf_thre=pi3x_vo_cfg.get("conf_thre", 0.05),
                        dtype=pi3x_vo_cfg.get("dtype", "bf16"),
                        pose_convention=pi3x_vo_cfg.get("pose_convention", "c2w"),
                        return_depth=False,
                    )
                )
            processors.append(
                Pi3XMetricDepthProcessor(
                    model=self.clip_cfg.get("pi3x_model", "yyfz233/Pi3X"),
                    pixel_limit=int(self.clip_cfg.get("pixel_limit", 255000)),
                    batch_size=int(self.clip_cfg.get("depth_batch_size", 4)),
                    use_poses=True,
                )
            )
            pi3x_stream = ProcessedVideoStream(clip_stream, processors).cache("pi3x_clip", online=True)
            pi3x_data = self._collect_arrays(pi3x_stream)

            # Optional: refine with sparse BA
            sparse_ba_applied = False
            if bool(self.sparse_ba_cfg.get("enabled", False)):
                refined = self._refine_chunk_sparse_ba(pi3x_data, clip_stream)
                sparse_ba_applied = bool(refined.get("sparse_ba_refined", False))
                pi3x_data = refined

            # Align Pi3X trajectory to SLAM trajectory
            overlap_left = min(align_overlap, start - clip_start)
            overlap_right = min(align_overlap, clip_end - end)
            
            align_result = None
            align_error = float('inf')
            aligned = False
            
            # Try left overlap first
            if overlap_left >= 3:
                # SLAM poses from [clip_start, start) = indices [0, overlap_left) in global_data starting at clip_start
                # Pi3X poses from [0, overlap_left)
                overlap_indices_slam = list(range(clip_start, start))
                overlap_indices_pi3x = list(range(overlap_left))
                
                s, R, t, err = self._align_trajectory(
                    global_data["extrinsic"],
                    pi3x_data["extrinsic"],
                    overlap_indices_slam,
                    overlap_indices_pi3x,
                )
                if err < 1.0:  # Reasonable alignment
                    align_result = (s, R, t)
                    align_error = err
                    aligned = True
                    logger.info(f"  Left alignment succeeded with error={err:.4f}, scale={s:.4f}")
            
            # Try right overlap if left failed
            if not aligned and overlap_right >= 3:
                # SLAM poses from [end, clip_end) = indices [end, clip_end) in global_data
                # Pi3X poses from the end of the clip
                clip_len = clip_end - clip_start
                overlap_indices_slam = list(range(end, clip_end))
                overlap_indices_pi3x = list(range(clip_len - overlap_right, clip_len))
                
                s, R, t, err = self._align_trajectory(
                    global_data["extrinsic"],
                    pi3x_data["extrinsic"],
                    overlap_indices_slam,
                    overlap_indices_pi3x,
                )
                if err < 1.0:
                    align_result = (s, R, t)
                    align_error = err
                    aligned = True
                    logger.info(f"  Right alignment succeeded with error={err:.4f}, scale={s:.4f}")
            
            # Fallback: identity alignment
            if align_result is None:
                logger.warning(f"  Alignment failed, using identity (no scale/rotation change)")
                align_result = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

            s, R, t = align_result

            # Apply Sim3 to Pi3X poses
            aligned_poses = apply_sim3_to_poses(pi3x_data["extrinsic"], s, R, t)
            aligned_depths = apply_sim3_to_depths(pi3x_data["depth"], s)

            # Replace the bad range (start to end) in global_data
            clip_offset = start - clip_start
            clip_len = end - start
            
            for i in range(clip_len):
                gi = start + i  # global index
                pi = clip_offset + i  # Pi3X index
                
                # Blend at boundaries
                if boundary_blend > 0 and i < boundary_blend and start > 0:
                    # Blend with previous SLAM pose at the start
                    alpha = (i + 1) / (boundary_blend + 1)
                    slam_pose = global_data["extrinsic"][gi]
                    pi3x_pose = aligned_poses[pi]
                    # Simple linear blend of positions (rotation blend would need slerp)
                    blended_pose = slam_pose.copy()
                    blended_pose[:3, 3] = (1 - alpha) * slam_pose[:3, 3] + alpha * pi3x_pose[:3, 3]
                    # For rotation, use pi3x as it's likely more reliable in bad regions
                    blended_pose[:3, :3] = pi3x_pose[:3, :3]
                    global_data["extrinsic"][gi] = blended_pose
                elif boundary_blend > 0 and (clip_len - i - 1) < boundary_blend and end < total_frames:
                    # Blend with next SLAM pose at the end
                    alpha = (clip_len - i) / (boundary_blend + 1)
                    slam_pose = global_data["extrinsic"][gi]
                    pi3x_pose = aligned_poses[pi]
                    blended_pose = slam_pose.copy()
                    blended_pose[:3, 3] = (1 - alpha) * slam_pose[:3, 3] + alpha * pi3x_pose[:3, 3]
                    blended_pose[:3, :3] = pi3x_pose[:3, :3]
                    global_data["extrinsic"][gi] = blended_pose
                else:
                    global_data["extrinsic"][gi] = aligned_poses[pi]
                
                global_data["depth"][gi] = aligned_depths[pi]

            clip_stats.append(
                ClipStats(
                    index=idx,
                    start=start,
                    end=end,
                    reason="mask_or_pose_jump",
                    aligned=aligned,
                    sparse_ba_applied=sparse_ba_applied,
                    align_error=align_error,
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
                    "pipeline": "clip_repair_v2",
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
