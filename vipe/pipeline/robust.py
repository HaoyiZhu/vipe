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

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vipe.ext.lietorch import SE3
from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import (
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoFrame,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput
from .default import DefaultAnnotationPipeline
from .processors import Pi3XMetricDepthProcessor

logger = logging.getLogger(__name__)


@dataclass
class PoseGuardStats:
    enabled: bool
    rejected: bool
    total_pairs: int
    outlier_pairs: int
    outlier_ratio: float
    max_translation: float
    max_rotation_deg: float
    max_consecutive_outliers: int
    missing_frames: int
    missing_ratio: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rejected": self.rejected,
            "total_pairs": self.total_pairs,
            "outlier_pairs": self.outlier_pairs,
            "outlier_ratio": self.outlier_ratio,
            "max_translation": self.max_translation,
            "max_rotation_deg": self.max_rotation_deg,
            "max_consecutive_outliers": self.max_consecutive_outliers,
            "missing_frames": self.missing_frames,
            "missing_ratio": self.missing_ratio,
            "reason": self.reason,
        }


@dataclass
class MaskCoverageStats:
    enabled: bool
    frames: int
    min_ratio: float
    mean_ratio: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frames": self.frames,
            "min_ratio": self.min_ratio,
            "mean_ratio": self.mean_ratio,
            "reason": self.reason,
        }


@dataclass
class SlamGuardStats:
    enabled: bool
    rejected: bool
    ba_residual: float
    max_ba_residual: float
    mask_min_ratio: float
    mask_mean_ratio: float
    min_valid_mask_ratio: float
    min_mean_valid_mask_ratio: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rejected": self.rejected,
            "ba_residual": self.ba_residual,
            "max_ba_residual": self.max_ba_residual,
            "mask_min_ratio": self.mask_min_ratio,
            "mask_mean_ratio": self.mask_mean_ratio,
            "min_valid_mask_ratio": self.min_valid_mask_ratio,
            "min_mean_valid_mask_ratio": self.min_mean_valid_mask_ratio,
            "reason": self.reason,
        }


class PoseDropProcessor(StreamProcessor):
    def __init__(self, drop_depth: bool = False) -> None:
        self.drop_depth = drop_depth

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        attributes = set(previous_attributes)
        attributes.discard(FrameAttribute.POSE)
        if self.drop_depth:
            attributes.discard(FrameAttribute.METRIC_DEPTH)
        return attributes

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.pose = None
        if self.drop_depth:
            frame.metric_depth = None
        return frame


class RobustAnnotationPipeline(DefaultAnnotationPipeline):
    def __init__(
        self,
        init: DictConfig,
        slam: DictConfig,
        post: DictConfig,
        output: DictConfig,
        robust: DictConfig | None = None,
    ) -> None:
        super().__init__(init, slam, post, output)
        self.robust_cfg = robust if robust is not None else OmegaConf.create({})

    @staticmethod
    def _rotation_angle_deg(rot: np.ndarray) -> float:
        trace = float(np.trace(rot))
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def _evaluate_pose_guard(self, stream: VideoStream, stream_name: str) -> PoseGuardStats:
        cfg = self.robust_cfg.get("pose_guard", None)
        if cfg is None or not cfg.get("enabled", False):
            return PoseGuardStats(
                enabled=False,
                rejected=False,
                total_pairs=0,
                outlier_pairs=0,
                outlier_ratio=0.0,
                max_translation=0.0,
                max_rotation_deg=0.0,
                max_consecutive_outliers=0,
                missing_frames=0,
                missing_ratio=0.0,
                reason="disabled",
            )

        if FrameAttribute.POSE not in stream.attributes():
            return PoseGuardStats(
                enabled=True,
                rejected=False,
                total_pairs=0,
                outlier_pairs=0,
                outlier_ratio=0.0,
                max_translation=0.0,
                max_rotation_deg=0.0,
                max_consecutive_outliers=0,
                missing_frames=0,
                missing_ratio=0.0,
                reason="no_pose_attribute",
            )

        max_translation = float(cfg.get("max_translation", 2.5))
        max_rotation_deg = float(cfg.get("max_rotation_deg", 25.0))
        max_outlier_ratio = float(cfg.get("max_outlier_ratio", 0.2))
        max_consecutive = int(cfg.get("max_consecutive_outliers", 5))
        max_missing_ratio = float(cfg.get("max_missing_ratio", 0.1))
        min_pairs = int(cfg.get("min_pairs", 5))

        prev_pose_mat = None
        total_pairs = 0
        outlier_pairs = 0
        max_seen_translation = 0.0
        max_seen_rotation = 0.0
        consecutive_outliers = 0
        max_consecutive_outliers = 0
        missing_frames = 0
        total_frames = 0

        for frame in stream:
            total_frames += 1
            if frame.pose is None:
                missing_frames += 1
                prev_pose_mat = None
                consecutive_outliers = 0
                continue

            pose_mat = frame.pose.matrix().detach().cpu().numpy()
            if prev_pose_mat is not None:
                total_pairs += 1
                rel_mat = np.linalg.inv(prev_pose_mat) @ pose_mat
                translation = float(np.linalg.norm(rel_mat[:3, 3]))
                rotation_deg = self._rotation_angle_deg(rel_mat[:3, :3])

                max_seen_translation = max(max_seen_translation, translation)
                max_seen_rotation = max(max_seen_rotation, rotation_deg)

                is_outlier = translation > max_translation or rotation_deg > max_rotation_deg
                if is_outlier:
                    outlier_pairs += 1
                    consecutive_outliers += 1
                    max_consecutive_outliers = max(max_consecutive_outliers, consecutive_outliers)
                else:
                    consecutive_outliers = 0

            prev_pose_mat = pose_mat

        if total_pairs < min_pairs:
            reason = "insufficient_pairs"
            outlier_ratio = 0.0
            rejected = False
        else:
            outlier_ratio = outlier_pairs / max(1, total_pairs)
            missing_ratio = missing_frames / max(1, total_frames)
            rejected = (
                outlier_ratio > max_outlier_ratio
                or max_consecutive_outliers > max_consecutive
                or missing_ratio > max_missing_ratio
            )
            reason = "rejected" if rejected else "accepted"

        if total_frames == 0:
            missing_ratio = 1.0
        else:
            missing_ratio = missing_frames / total_frames

        stats = PoseGuardStats(
            enabled=True,
            rejected=rejected,
            total_pairs=total_pairs,
            outlier_pairs=outlier_pairs,
            outlier_ratio=outlier_ratio,
            max_translation=max_seen_translation,
            max_rotation_deg=max_seen_rotation,
            max_consecutive_outliers=max_consecutive_outliers,
            missing_frames=missing_frames,
            missing_ratio=missing_ratio,
            reason=reason,
        )

        logger.info(
            "Pose guard (%s): pairs=%d, outliers=%d (%.3f), max_t=%.3f, max_r=%.2f, missing=%.3f, rejected=%s",
            stream_name,
            stats.total_pairs,
            stats.outlier_pairs,
            stats.outlier_ratio,
            stats.max_translation,
            stats.max_rotation_deg,
            stats.missing_ratio,
            stats.rejected,
        )
        return stats

    def _apply_pose_guard(
        self, stream: VideoStream, stream_name: str
    ) -> tuple[VideoStream, PoseGuardStats | None]:
        cfg = self.robust_cfg.get("pose_guard", None)
        if cfg is None or not cfg.get("enabled", False):
            return stream, None

        stats = self._evaluate_pose_guard(stream, stream_name)
        if stats.rejected:
            drop_depth = bool(cfg.get("drop_depth_on_reject", False))
            logger.warning("Pose guard rejected init poses for %s; dropping pose (depth=%s).", stream_name, drop_depth)
            return ProcessedVideoStream(stream, [PoseDropProcessor(drop_depth=drop_depth)]), stats

        return stream, stats

    def _compute_mask_coverage(self, stream: VideoStream, stream_name: str) -> MaskCoverageStats:
        ratios: list[float] = []
        frames = 0
        for frame in stream:
            frames += 1
            if frame.mask is None:
                continue
            mask = frame.mask
            if mask.dtype != torch.bool:
                mask = mask > 0
            ratios.append(mask.float().mean().item())

        if not ratios:
            stats = MaskCoverageStats(
                enabled=False,
                frames=frames,
                min_ratio=1.0,
                mean_ratio=1.0,
                reason="no_mask",
            )
        else:
            stats = MaskCoverageStats(
                enabled=True,
                frames=frames,
                min_ratio=float(min(ratios)),
                mean_ratio=float(sum(ratios) / len(ratios)),
                reason="ok",
            )

        logger.info(
            "Mask coverage (%s): frames=%d, min=%.3f, mean=%.3f, enabled=%s",
            stream_name,
            stats.frames,
            stats.min_ratio,
            stats.mean_ratio,
            stats.enabled,
        )
        return stats

    def _evaluate_slam_guard(
        self, slam_output: SLAMOutput | None, mask_stats: MaskCoverageStats | None
    ) -> SlamGuardStats:
        cfg = self.robust_cfg.get("slam_guard", None)
        if cfg is None or not cfg.get("enabled", False):
            return SlamGuardStats(
                enabled=False,
                rejected=False,
                ba_residual=float("nan"),
                max_ba_residual=float(cfg.get("max_ba_residual", float("inf"))) if cfg is not None else float("inf"),
                mask_min_ratio=1.0,
                mask_mean_ratio=1.0,
                min_valid_mask_ratio=float(cfg.get("min_valid_mask_ratio", 0.0)) if cfg is not None else 0.0,
                min_mean_valid_mask_ratio=float(cfg.get("min_mean_valid_mask_ratio", 0.0)) if cfg is not None else 0.0,
                reason="disabled",
            )

        max_ba_residual = float(cfg.get("max_ba_residual", 0.05))
        min_valid_mask_ratio = float(cfg.get("min_valid_mask_ratio", 0.0))
        min_mean_valid_mask_ratio = float(cfg.get("min_mean_valid_mask_ratio", 0.0))

        ba_residual = float("nan")
        if slam_output is not None:
            ba_residual = float(slam_output.ba_residual)

        mask_min_ratio = 1.0
        mask_mean_ratio = 1.0
        if mask_stats is not None:
            mask_min_ratio = mask_stats.min_ratio
            mask_mean_ratio = mask_stats.mean_ratio

        rejected = False
        reasons = []

        if not np.isfinite(ba_residual) or ba_residual > max_ba_residual:
            if cfg.get("fail_on_ba_residual", True):
                rejected = True
                reasons.append("ba_residual")

        if mask_min_ratio < min_valid_mask_ratio:
            if cfg.get("fail_on_mask_low", True):
                rejected = True
                reasons.append("mask_min")

        if mask_mean_ratio < min_mean_valid_mask_ratio:
            if cfg.get("fail_on_mask_low", True):
                rejected = True
                reasons.append("mask_mean")

        reason = "ok" if not rejected else ",".join(reasons)

        return SlamGuardStats(
            enabled=True,
            rejected=rejected,
            ba_residual=ba_residual,
            max_ba_residual=max_ba_residual,
            mask_min_ratio=mask_min_ratio,
            mask_mean_ratio=mask_mean_ratio,
            min_valid_mask_ratio=min_valid_mask_ratio,
            min_mean_valid_mask_ratio=min_mean_valid_mask_ratio,
            reason=reason,
        )

    @staticmethod
    def _sanitize_viz_attributes(
        attributes: list[list[str]], slam_output: SLAMOutput | None
    ) -> list[list[str]]:
        if slam_output is not None and slam_output.slam_map is not None:
            return attributes
        return [[("empty" if t == "pcd" else t) for t in row] for row in attributes]

    def _build_fallback_streams(self, init_streams: list[VideoStream]) -> list[VideoStream]:
        cfg = self.robust_cfg.get("fallback", None) or {}
        use_depth = bool(cfg.get("use_pi3x_depth", True))
        force_depth = bool(cfg.get("force_pi3x_depth", False))
        depth_model = cfg.get("pi3x_model", "yyfz233/Pi3X")
        pixel_limit = int(cfg.get("pixel_limit", 255000))
        batch_size = int(cfg.get("depth_batch_size", 1))
        use_poses = bool(cfg.get("use_poses", True))

        output_streams: list[VideoStream] = []
        for init_stream in init_streams:
            stream = init_stream
            if use_depth and (force_depth or FrameAttribute.METRIC_DEPTH not in stream.attributes()):
                stream = ProcessedVideoStream(
                    stream,
                    [
                        Pi3XMetricDepthProcessor(
                            model=depth_model,
                            pixel_limit=pixel_limit,
                            batch_size=batch_size,
                            use_poses=use_poses,
                        )
                    ],
                ).cache("pi3x_depth", online=True)
            output_streams.append(stream)
        return output_streams

    def _build_fallback_slam_output(
        self, streams: list[VideoStream], slam_rig: SE3 | None
    ) -> SLAMOutput:
        if not streams:
            trajectory = SE3.Identity(1)
            intrinsics = torch.zeros((1, 4))
            rig = SE3.Identity(1)
            return SLAMOutput(trajectory=trajectory, intrinsics=intrinsics, rig=rig, slam_map=None, ba_residual=float("nan"))

        ref_stream = streams[0]
        poses_data = []
        last_pose_data = None
        for frame in ref_stream:
            if frame.pose is None:
                if last_pose_data is None:
                    pose_data = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=frame.rgb.device)
                else:
                    pose_data = last_pose_data
            else:
                pose_data = frame.pose.data
                last_pose_data = pose_data
            poses_data.append(pose_data)

        if not poses_data:
            trajectory = SE3.Identity(1)
        else:
            trajectory = SE3(torch.stack(poses_data, dim=0))

        intrinsics_list = []
        for stream in streams:
            first_frame = stream[0]
            if first_frame.intrinsics is None:
                intr = torch.zeros(4)
            else:
                intr = first_frame.intrinsics[:4].detach().cpu()
            intrinsics_list.append(intr)
        intrinsics = torch.stack(intrinsics_list, dim=0)

        rig = slam_rig if slam_rig is not None else SE3.Identity(len(streams))

        return SLAMOutput(
            trajectory=trajectory,
            intrinsics=intrinsics,
            rig=rig,
            slam_map=None,
            ba_residual=float("nan"),
        )

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
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
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        init_streams: list[VideoStream] = []
        slam_streams: list[VideoStream] = []
        pose_guard_reports: list[dict[str, Any]] = []
        pose_guard_stats: list[PoseGuardStats] = []
        mask_stats_list: list[MaskCoverageStats] = []
        for video_stream in video_streams:
            cached_stream = self._add_init_processors(video_stream).cache("process", online=True)
            init_streams.append(cached_stream)
            mask_stats = self._compute_mask_coverage(cached_stream, video_stream.name())
            mask_stats_list.append(mask_stats)
            guarded_stream, stats = self._apply_pose_guard(cached_stream, video_stream.name())
            slam_streams.append(guarded_stream)
            if stats is not None:
                pose_guard_reports.append(stats.as_dict())
                pose_guard_stats.append(stats)

        mask_stats_agg = None
        if mask_stats_list:
            mask_stats_agg = MaskCoverageStats(
                enabled=any(stat.enabled for stat in mask_stats_list),
                frames=sum(stat.frames for stat in mask_stats_list),
                min_ratio=min(stat.min_ratio for stat in mask_stats_list),
                mean_ratio=float(np.mean([stat.mean_ratio for stat in mask_stats_list])),
                reason="ok",
            )

        fallback_cfg = self.robust_cfg.get("fallback", None) or {}
        fallback_enabled = bool(fallback_cfg.get("enabled", False))
        require_pose_guard = bool(fallback_cfg.get("require_pose_guard_accept", True))
        require_pose = bool(fallback_cfg.get("require_pose", True))

        if require_pose_guard and any(stat.rejected for stat in pose_guard_stats):
            fallback_enabled = False
            logger.warning("Fallback disabled due to pose guard rejection.")
        if require_pose and not all(FrameAttribute.POSE in stream.attributes() for stream in init_streams):
            fallback_enabled = False
            logger.warning("Fallback disabled due to missing poses in init stream.")

        slam_guard_cfg = self.robust_cfg.get("slam_guard", None) or {}
        skip_slam = False
        fallback_reason = None
        if fallback_enabled and bool(fallback_cfg.get("force_pi3x_only", False)):
            skip_slam = True
            fallback_reason = "force_pi3x_only"
        elif (
            fallback_enabled
            and bool(slam_guard_cfg.get("skip_slam_if_mask_low", False))
            and mask_stats_agg is not None
            and mask_stats_agg.min_ratio < float(slam_guard_cfg.get("min_valid_mask_ratio", 0.0))
        ):
            skip_slam = True
            fallback_reason = "mask_low"

        if skip_slam and not fallback_enabled:
            skip_slam = False
            logger.warning("skip_slam requested but fallback is disabled; running SLAM.")

        slam_output: SLAMOutput | None = None
        slam_guard_stats: SlamGuardStats | None = None
        use_fallback = False

        if not skip_slam:
            slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
            slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

            if pose_guard_reports:
                slam_output.metrics["pose_guard"] = pose_guard_reports
            if mask_stats_list:
                slam_output.metrics["mask_coverage"] = [stat.as_dict() for stat in mask_stats_list]
            if mask_stats_agg is not None:
                slam_output.metrics["mask_coverage_agg"] = mask_stats_agg.as_dict()

            slam_guard_stats = self._evaluate_slam_guard(slam_output, mask_stats_agg)
            if slam_guard_stats.enabled:
                slam_output.metrics["slam_guard"] = slam_guard_stats.as_dict()

            # Optional: save raw SLAM outputs (poses/intrinsics) before any post-processing,
            # to help debug trajectory differences across post depth processors / configs.
            if self.out_cfg.get("save_slam_intermediate", False):
                for view_idx, artifact_path in enumerate(artifact_paths):
                    try:
                        io.save_slam_intermediate_artifacts(artifact_path, slam_output, view_idx=view_idx)
                    except Exception:
                        logger.exception("Failed saving intermediate SLAM artifacts for view_idx=%d", view_idx)

            # Clean up SLAM system to free GPU memory for post-processing
            del slam_pipeline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            if (
                fallback_enabled
                and slam_guard_stats is not None
                and slam_guard_stats.rejected
                and bool(fallback_cfg.get("fallback_on_slam_guard", True))
            ):
                use_fallback = True
                fallback_reason = f"slam_guard:{slam_guard_stats.reason}"
                logger.warning("SLAM guard rejected result (residual=%.4f); falling back to Pi3X.", slam_guard_stats.ba_residual)
        else:
            use_fallback = fallback_enabled
            if use_fallback:
                logger.info("Using Pi3X fallback (force_pi3x_only=True or triggered by mask/pose guard).")

        if use_fallback:
            output_streams = self._build_fallback_streams(init_streams)
            slam_output = self._build_fallback_slam_output(output_streams, slam_rig)
            slam_output.metrics["fallback_used"] = True
            slam_output.metrics["fallback_reason"] = fallback_reason or "fallback"
            if pose_guard_reports:
                slam_output.metrics["pose_guard"] = pose_guard_reports
            if mask_stats_list:
                slam_output.metrics["mask_coverage"] = [stat.as_dict() for stat in mask_stats_list]
            if mask_stats_agg is not None:
                slam_output.metrics["mask_coverage_agg"] = mask_stats_agg.as_dict()
            if slam_guard_stats is not None:
                slam_output.metrics["slam_guard"] = slam_guard_stats.as_dict()
        else:
            assert slam_output is not None
            output_streams = [
                self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
                for view_idx, slam_stream in enumerate(slam_streams)
            ]

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    if slam_output is not None:
                        slam_output.metrics["ba_residual"] = slam_output.ba_residual
                        pickle.dump(slam_output.metrics, f)
                    else:
                        pickle.dump({}, f)

            if self.out_cfg.save_viz:
                viz_attributes = self._sanitize_viz_attributes(self.out_cfg.viz_attributes, slam_output)
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    viz_attributes,
                )

            if slam_output is not None and self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output

