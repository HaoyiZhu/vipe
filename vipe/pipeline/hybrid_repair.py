# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HybridRepair: Conservative clip repair that only fixes truly broken segments.

Key principles:
1. Trust SLAM results by default - they work well for 84% of cases
2. Only detect clips that have SEVERE discontinuities (not just minor jumps)
3. Use trajectory-based alignment (Procrustes on camera positions)
4. Validate repair quality before applying - keep original if repair is worse
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
    method: str  # "kept_original", "pi3x_repair", "interpolate"
    scale_used: float
    repair_quality: str  # "good", "rejected", "fallback"
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
            "method": self.method,
            "scale_used": self.scale_used,
            "repair_quality": self.repair_quality,
        }


def compute_trajectory_smoothness(poses: np.ndarray, start: int, end: int) -> dict:
    """
    Compute smoothness metrics for a trajectory segment.
    Returns dict with translation jumps, rotation jumps, and overall score.
    """
    trans_jumps = []
    rot_jumps = []
    
    for i in range(max(1, start), min(end, len(poses))):
        # Translation jump
        pos_prev = poses[i-1, :3, 3]
        pos_curr = poses[i, :3, 3]
        trans_jump = np.linalg.norm(pos_curr - pos_prev)
        trans_jumps.append(trans_jump)
        
        # Rotation jump
        R_prev = poses[i-1, :3, :3]
        R_curr = poses[i, :3, :3]
        R_rel = R_prev.T @ R_curr
        trace = np.trace(R_rel)
        cos_theta = np.clip((trace - 1) / 2, -1, 1)
        rot_jump = np.degrees(np.arccos(cos_theta))
        rot_jumps.append(rot_jump)
    
    if not trans_jumps:
        return {"trans_max": 0, "rot_max": 0, "score": 0}
    
    trans_max = max(trans_jumps)
    rot_max = max(rot_jumps)
    trans_mean = np.mean(trans_jumps)
    rot_mean = np.mean(rot_jumps)
    
    # Smoothness score: lower is better
    # Penalize large jumps heavily
    score = trans_max + 0.1 * rot_max + 0.5 * trans_mean + 0.05 * rot_mean
    
    return {
        "trans_max": trans_max,
        "rot_max": rot_max,
        "trans_mean": trans_mean,
        "rot_mean": rot_mean,
        "score": score,
    }


def procrustes_sim3_align(
    src_positions: np.ndarray,  # (N, 3) camera positions in source frame
    tgt_positions: np.ndarray,  # (N, 3) camera positions in target frame
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Procrustes analysis to find Sim3 (scale, rotation, translation) that aligns src to tgt.
    Returns (scale, R, t) such that: tgt ≈ scale * (R @ src.T).T + t
    """
    assert len(src_positions) == len(tgt_positions)
    n = len(src_positions)
    
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)
    
    # Compute centroids
    src_centroid = src_positions.mean(axis=0)
    tgt_centroid = tgt_positions.mean(axis=0)
    
    # Center the points
    src_centered = src_positions - src_centroid
    tgt_centered = tgt_positions - tgt_centroid
    
    # Compute scale
    src_scale = np.sqrt((src_centered ** 2).sum())
    tgt_scale = np.sqrt((tgt_centered ** 2).sum())
    
    if src_scale < 1e-8:
        return 1.0, np.eye(3), tgt_centroid
    
    scale = tgt_scale / src_scale
    
    # Normalize
    src_normalized = src_centered / src_scale
    tgt_normalized = tgt_centered / tgt_scale
    
    # Compute rotation using SVD
    H = src_normalized.T @ tgt_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
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
        new_poses[i, :3, :3] = R @ poses[i, :3, :3]
        new_poses[i, :3, 3] = scale * (R @ poses[i, :3, 3]) + t
        new_poses[i, 3, 3] = 1.0
    return new_poses


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


class HybridRepairAnnotationPipeline(ChunkedRobustAnnotationPipeline):
    """
    Hybrid repair pipeline that conservatively repairs only truly broken segments.
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

    def _detect_severe_discontinuities(
        self,
        trajectory: SE3,
        total_frames: int,
        masks: list[torch.Tensor | None],
    ) -> list[tuple[int, int, str]]:
        """
        Detect only SEVERE discontinuities that indicate tracking failure.
        Much more conservative than the original detection.
        
        Returns list of (start, end, reason) tuples.
        """
        cfg = self.clip_cfg.get("detect", DictConfig({}))
        if not cfg or not cfg.get("enabled", True):
            return []

        # Thresholds - much more conservative
        severe_trans_jump = float(cfg.get("severe_trans_jump", 10.0))  # 10 meters
        severe_rot_jump = float(cfg.get("severe_rot_jump", 90.0))  # 90 degrees
        min_mask_ratio = float(cfg.get("min_mask_ratio", 0.01))  # Very low mask
        min_consecutive = int(cfg.get("min_consecutive", 3))  # At least 3 frames
        expand = int(cfg.get("expand", 5))
        merge_gap = int(cfg.get("merge_gap", 10))
        
        # Convert trajectory to numpy for analysis
        poses = trajectory.matrix().detach().cpu().numpy()
        
        bad_frames = []
        
        for i in range(1, total_frames):
            reasons = []
            
            # Check for severe pose jump
            pos_prev = poses[i-1, :3, 3]
            pos_curr = poses[i, :3, 3]
            trans_jump = np.linalg.norm(pos_curr - pos_prev)
            
            R_prev = poses[i-1, :3, :3]
            R_curr = poses[i, :3, :3]
            R_rel = R_prev.T @ R_curr
            trace = np.trace(R_rel)
            cos_theta = np.clip((trace - 1) / 2, -1, 1)
            rot_jump = np.degrees(np.arccos(cos_theta))
            
            if trans_jump > severe_trans_jump:
                reasons.append(f"trans_jump={trans_jump:.1f}m")
            if rot_jump > severe_rot_jump:
                reasons.append(f"rot_jump={rot_jump:.1f}deg")
            
            # Check for very low mask coverage
            if masks[i] is not None:
                mask = masks[i]
                if mask.dtype != torch.bool:
                    mask = mask > 0
                mask_ratio = float(mask.float().mean().item())
                if mask_ratio < min_mask_ratio:
                    reasons.append(f"mask={mask_ratio:.3f}")
            
            if reasons:
                bad_frames.append((i, ", ".join(reasons)))
        
        # Group consecutive bad frames
        ranges = []
        if bad_frames:
            start_idx = bad_frames[0][0]
            end_idx = bad_frames[0][0]
            reasons = [bad_frames[0][1]]
            
            for idx, reason in bad_frames[1:]:
                if idx <= end_idx + merge_gap:
                    end_idx = idx
                    reasons.append(reason)
                else:
                    if end_idx - start_idx + 1 >= min_consecutive:
                        ranges.append((start_idx, end_idx + 1, "; ".join(set(reasons))))
                    start_idx = idx
                    end_idx = idx
                    reasons = [reason]
            
            if end_idx - start_idx + 1 >= min_consecutive:
                ranges.append((start_idx, end_idx + 1, "; ".join(set(reasons))))
        
        # Expand and merge ranges
        expanded = []
        for start, end, reason in ranges:
            s = max(0, start - expand)
            e = min(total_frames, end + expand)
            expanded.append((s, e, reason))
        
        # Merge overlapping ranges
        if not expanded:
            return []
        
        expanded.sort(key=lambda x: x[0])
        merged = [expanded[0]]
        for s, e, reason in expanded[1:]:
            if s <= merged[-1][1] + merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e), merged[-1][2] + "; " + reason)
            else:
                merged.append((s, e, reason))
        
        return merged

    def _repair_clip_with_pi3x(
        self,
        init_stream: VideoStream,
        global_poses: np.ndarray,
        global_depths: list,
        clip_start: int,
        clip_end: int,
        bad_start: int,
        bad_end: int,
    ) -> tuple[np.ndarray, list, float, bool]:
        """
        Repair a bad clip using Pi3X VO.
        Returns (repaired_poses, repaired_depths, scale, success).
        """
        pi3x_cfg = self.init_cfg.get("pose_init", DictConfig({}))
        align_overlap = int(self.clip_cfg.get("align_overlap", 15))
        
        # Run Pi3X VO on the clip
        clip_stream = SlicedVideoStream(init_stream, clip_start, clip_end)
        
        processors = [
            Pi3XVOInitPoseProcessor(
                clip_stream,
                model=pi3x_cfg.get("model", "yyfz233/Pi3X"),
                chunk_size=pi3x_cfg.get("chunk_size", 64),
                overlap=pi3x_cfg.get("overlap", 32),
                conf_thre=pi3x_cfg.get("conf_thre", 0.05),
                dtype=pi3x_cfg.get("dtype", "bf16"),
                pose_convention=pi3x_cfg.get("pose_convention", "c2w"),
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
        
        # Collect Pi3X results
        pi3x_poses = []
        pi3x_depths = []
        for frame in pi3x_stream:
            pose = frame.pose.matrix() if frame.pose is not None else torch.eye(4)
            pi3x_poses.append(pose.detach().cpu().numpy())
            pi3x_depths.append(frame.metric_depth.detach().cpu().numpy() if frame.metric_depth is not None else None)
        pi3x_poses = np.stack(pi3x_poses)
        
        # Align Pi3X trajectory to SLAM trajectory using overlap regions
        # Use left overlap for alignment
        left_overlap_len = min(align_overlap, bad_start - clip_start)
        right_overlap_len = min(align_overlap, clip_end - bad_end)
        
        if left_overlap_len >= 3:
            # Use left overlap
            slam_positions = global_poses[clip_start:clip_start + left_overlap_len, :3, 3]
            pi3x_positions = pi3x_poses[:left_overlap_len, :3, 3]
        elif right_overlap_len >= 3:
            # Use right overlap
            right_start = bad_end - clip_start
            slam_positions = global_poses[bad_end:bad_end + right_overlap_len, :3, 3]
            pi3x_positions = pi3x_poses[right_start:right_start + right_overlap_len, :3, 3]
        else:
            logger.warning("  Not enough overlap for alignment")
            return global_poses[bad_start:bad_end], global_depths[bad_start:bad_end], 1.0, False
        
        # Compute Sim3 alignment
        scale, R, t = procrustes_sim3_align(pi3x_positions, slam_positions)
        
        # Validate scale
        min_scale = float(self.clip_cfg.get("min_scale", 0.1))
        max_scale = float(self.clip_cfg.get("max_scale", 10.0))
        
        if scale < min_scale or scale > max_scale:
            logger.warning(f"  Scale {scale:.4f} out of range [{min_scale}, {max_scale}]")
            return global_poses[bad_start:bad_end], global_depths[bad_start:bad_end], scale, False
        
        # Apply transformation to Pi3X poses
        aligned_pi3x_poses = apply_sim3_to_poses(pi3x_poses, scale, R, t)
        
        # Extract the repaired segment
        repair_start_in_clip = bad_start - clip_start
        repair_end_in_clip = bad_end - clip_start
        repaired_poses = aligned_pi3x_poses[repair_start_in_clip:repair_end_in_clip]
        repaired_depths = [d * scale if d is not None else None 
                          for d in pi3x_depths[repair_start_in_clip:repair_end_in_clip]]
        
        return repaired_poses, repaired_depths, scale, True

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
                raise ValueError("HybridRepair pipeline supports single-view streams only.")
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
        
        # Collect masks for detection
        masks = []
        for frame in init_stream:
            masks.append(frame.mask if frame.mask is not None else None)
        
        # Step 2: Run SLAM
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run([init_stream], rig=slam_rig, camera_type=self.camera_type)

        # Step 3: Add depth
        global_stream = self._add_post_processors(0, init_stream, slam_output).cache("depth", online=True)
        total_frames = len(global_stream)

        # Step 4: Detect SEVERE discontinuities only
        trajectory = slam_output.get_view_trajectory(0)
        bad_ranges = self._detect_severe_discontinuities(trajectory, total_frames, masks)
        
        if bad_ranges:
            logger.info(f"Detected {len(bad_ranges)} severe discontinuities:")
            for start, end, reason in bad_ranges:
                logger.info(f"  Frames {start}-{end}: {reason}")
        else:
            logger.info("No severe discontinuities detected, using SLAM results directly.")

        # Step 5: Collect SLAM results
        global_data = self._collect_arrays(global_stream)
        original_poses = global_data["extrinsic"].copy()
        clip_stats: list[ClipStats] = []

        align_overlap = int(self.clip_cfg.get("align_overlap", 15))
        validate_repair = bool(self.clip_cfg.get("validate_repair", True))

        # Step 6: Process each bad clip
        for idx, (start, end, reason) in enumerate(bad_ranges):
            clip_start = max(0, start - align_overlap)
            clip_end = min(total_frames, end + align_overlap)
            
            logger.info(f"Processing bad clip {idx}: frames {start}-{end}")
            logger.info(f"  Reason: {reason}")
            
            # Compute original smoothness
            orig_smoothness = compute_trajectory_smoothness(original_poses, start, end)
            logger.info(f"  Original smoothness: trans_max={orig_smoothness['trans_max']:.2f}m, rot_max={orig_smoothness['rot_max']:.1f}deg")
            
            # Try Pi3X repair
            try:
                repaired_poses, repaired_depths, scale, success = self._repair_clip_with_pi3x(
                    init_stream,
                    global_data["extrinsic"],
                    global_data["depth"],
                    clip_start,
                    clip_end,
                    start,
                    end,
                )
                
                if success and validate_repair:
                    # Create temp array with repaired segment
                    temp_poses = global_data["extrinsic"].copy()
                    temp_poses[start:end] = repaired_poses
                    
                    # Compute repaired smoothness
                    repair_smoothness = compute_trajectory_smoothness(temp_poses, start, end)
                    logger.info(f"  Repaired smoothness: trans_max={repair_smoothness['trans_max']:.2f}m, rot_max={repair_smoothness['rot_max']:.1f}deg")
                    
                    # Only apply if repair is significantly better
                    if repair_smoothness['score'] < orig_smoothness['score'] * 0.8:
                        logger.info(f"  Repair accepted (score {repair_smoothness['score']:.2f} < {orig_smoothness['score']*0.8:.2f})")
                        global_data["extrinsic"][start:end] = repaired_poses
                        for i, d in enumerate(repaired_depths):
                            if d is not None:
                                global_data["depth"][start + i] = d
                        
                        clip_stats.append(ClipStats(
                            index=idx,
                            start=start,
                            end=end,
                            reason=reason,
                            method="pi3x_repair",
                            scale_used=scale,
                            repair_quality="good",
                        ))
                    else:
                        logger.info(f"  Repair rejected (score {repair_smoothness['score']:.2f} >= {orig_smoothness['score']*0.8:.2f})")
                        clip_stats.append(ClipStats(
                            index=idx,
                            start=start,
                            end=end,
                            reason=reason,
                            method="kept_original",
                            scale_used=scale,
                            repair_quality="rejected",
                        ))
                elif success:
                    # No validation, just apply
                    global_data["extrinsic"][start:end] = repaired_poses
                    for i, d in enumerate(repaired_depths):
                        if d is not None:
                            global_data["depth"][start + i] = d
                    clip_stats.append(ClipStats(
                        index=idx,
                        start=start,
                        end=end,
                        reason=reason,
                        method="pi3x_repair",
                        scale_used=scale,
                        repair_quality="good",
                    ))
                else:
                    # Fall back to interpolation
                    if start > 0 and end < total_frames:
                        pose_before = global_data["extrinsic"][start - 1]
                        pose_after = global_data["extrinsic"][end]
                        interp_poses = interpolate_poses(pose_before, pose_after, end - start)
                        global_data["extrinsic"][start:end] = interp_poses
                        logger.info(f"  Using interpolation fallback")
                    
                    clip_stats.append(ClipStats(
                        index=idx,
                        start=start,
                        end=end,
                        reason=reason,
                        method="interpolate",
                        scale_used=1.0,
                        repair_quality="fallback",
                    ))
                    
            except Exception as e:
                logger.warning(f"  Repair failed: {e}")
                clip_stats.append(ClipStats(
                    index=idx,
                    start=start,
                    end=end,
                    reason=reason,
                    method="kept_original",
                    scale_used=1.0,
                    repair_quality="error",
                ))

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
                    "pipeline": "hybrid_repair",
                    "bad_ranges": [(s, e) for s, e, _ in bad_ranges],
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
