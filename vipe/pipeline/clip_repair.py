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
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
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
from vipe.utils.sim3 import robust_weighted_align_point_maps
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
            "aligned": self.aligned,
            "sparse_ba_applied": self.sparse_ba_applied,
        }


class ClipRepairAnnotationPipeline(ChunkedRobustAnnotationPipeline):
    """
    Simplest repair pipeline:
    1) Run Pi3X VO init (optional) + masks/intrinsics.
    2) Run a single global SLAM pass.
    3) Detect bad clips (mask coverage, pose jump).
    4) Run Pi3X on those clips (+ optional sparse BA).
    5) Align Pi3X clips to good SLAM results and replace.
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
            else:
                logger.warning(
                    "Per-frame BA residuals length (%d) does not match total frames (%d); skipping.",
                    len(slam_ba_residuals),
                    total_frames,
                )

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

    def _compute_slam_depth_errors(
        self,
        stream: VideoStream,
        slam_output: SLAMOutput,
    ) -> list[float]:
        cfg = self.clip_cfg.get("detect", DictConfig({}))
        align_size = int(cfg.get("slam_depth_error_lr_size", 64))
        min_points = int(cfg.get("slam_depth_min_points", 30))
        errors: list[float] = []

        if slam_output.slam_map is None:
            return [0.0 for _ in range(len(stream))]

        for frame_idx, frame in enumerate(stream):
            if frame.metric_depth is None or frame.pose is None or frame.intrinsics is None:
                errors.append(0.0)
                continue

            slam_depth = slam_output.slam_map.project_map(
                frame_tstamp=frame_idx,
                view_idx=0,
                target_size=frame.size(),
                target_intrinsics=frame.intrinsics,
                target_pose=frame.pose,
                target_camera_type=frame.camera_type,
                infill=False,
            )
            slam_depth = slam_depth.to(frame.metric_depth.device)
            valid = (slam_depth > 0) & torch.isfinite(slam_depth) & torch.isfinite(frame.metric_depth)
            if frame.mask is not None:
                valid = valid & frame.mask
            if valid.sum().item() < min_points:
                errors.append(0.0)
                continue
            indices, lr_mask = mask_aware_nearest_resize_robust(valid, align_size, align_size)
            if lr_mask.sum().item() < min_points:
                errors.append(0.0)
                continue
            ni, nj = indices
            slam_sel = slam_depth[ni, nj][lr_mask]
            depth_sel = frame.metric_depth[ni, nj][lr_mask]
            rel_err = (depth_sel - slam_sel).abs() / slam_sel.clamp(min=1e-3)
            errors.append(float(torch.median(rel_err).item()))
        return errors

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

    def _depth_points_downsample(
        self,
        depth_list: list[np.ndarray],
        intr_list: list[np.ndarray],
        mask_list: list[np.ndarray | None],
        align_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pts = []
        conf = []
        mask_out = []
        for depth, intr, mask in zip(depth_list, intr_list, mask_list):
            depth_t = torch.from_numpy(depth).float()
            intr_t = torch.from_numpy(intr).float()
            pose = torch.eye(4)
            pts_world = self._backproject_depth(depth_t, intr_t, pose)
            conf_map = torch.ones_like(depth_t)
            if mask is None:
                mask_t = torch.ones_like(depth_t, dtype=torch.bool)
            else:
                mask_t = torch.from_numpy(mask).bool()
            indices, lr_mask = mask_aware_nearest_resize_robust(mask_t, align_size, align_size)
            ni, nj = indices
            pts.append(pts_world[ni, nj])
            conf.append(conf_map[ni, nj])
            mask_out.append(lr_mask)
        return torch.stack(pts), torch.stack(conf), torch.stack(mask_out)

    def _align_clip(
        self,
        slam_data: dict[str, Any],
        pi3x_data: dict[str, Any],
        overlap: int,
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        if overlap <= 0:
            return None
        n = min(overlap, len(pi3x_data["depth"]), len(slam_data["depth"]))
        if n <= 0:
            return None

        align_size = int(self.clip_cfg.get("align_lr_size", 64))
        min_points = int(self.clip_cfg.get("min_align_points", 50))

        pts_slam, conf_slam, mask_slam = self._depth_points_downsample(
            slam_data["depth"][:n], slam_data["intrinsic"][:n], slam_data["mask"][:n], align_size
        )
        pts_pi3x, conf_pi3x, mask_pi3x = self._depth_points_downsample(
            pi3x_data["depth"][:n], pi3x_data["intrinsic"][:n], pi3x_data["mask"][:n], align_size
        )
        combined_mask = mask_slam & mask_pi3x
        if combined_mask.sum().item() < min_points:
            return None

        s, R, t = robust_weighted_align_point_maps(
            pts_slam,
            conf_slam,
            pts_pi3x,
            conf_pi3x,
            combined_mask,
            conf_threshold=-1.0,
            delta=self.irls_delta,
            max_iters=self.irls_max_iters,
            tol=self.irls_tol,
            using_sim3=True,
        )
        return float(s), R.detach().cpu().numpy(), t.detach().cpu().numpy()

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            if len(video_data) > 1:
                raise ValueError("Clip repair pipeline supports single-view streams only.")
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

        init_stream = self._add_init_processors(video_stream).cache("process", online=True)
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run([init_stream], rig=slam_rig, camera_type=self.camera_type)

        global_stream = self._add_post_processors(0, init_stream, slam_output).cache("depth", online=True)
        total_frames = len(global_stream)

        depth_errors = None
        detect_cfg = self.clip_cfg.get("detect", DictConfig({}))
        if detect_cfg.get("use_slam_depth_error", False):
            depth_errors = self._compute_slam_depth_errors(global_stream, slam_output)
        ba_residuals = None
        if detect_cfg.get("use_slam_ba_residuals", False):
            ba_residuals = slam_output.metrics.get("ba_residuals_per_frame")
            if ba_residuals is None:
                logger.warning(
                    "Per-frame BA residuals not found. Enable slam.compute_ba_residuals_per_frame to use this signal."
                )

        bad_ranges = self._detect_bad_clips(
            init_stream,
            slam_output.get_view_trajectory(0),
            total_frames,
            slam_depth_errors=depth_errors,
            slam_ba_residual=float(slam_output.ba_residual),
            slam_ba_residuals=ba_residuals,
        )
        if not bad_ranges:
            bad_ranges = []

        global_data = self._collect_arrays(global_stream)
        clip_stats: list[ClipStats] = []

        align_overlap = int(self.clip_cfg.get("align_overlap", 10))
        pi3x_vo_cfg = self.init_cfg.get("pose_init", DictConfig({}))
        pi3x_vo_enabled = bool(self.clip_cfg.get("pi3x_vo_enabled", True))

        for idx, (start, end) in enumerate(bad_ranges):
            clip_start = max(0, start - align_overlap)
            clip_end = min(total_frames, end + align_overlap)
            clip_stream = SlicedVideoStream(init_stream, clip_start, clip_end)

            processors = []
            if pi3x_vo_enabled and pi3x_vo_cfg.get("enabled", True):
                processors.append(
                    Pi3XVOInitPoseProcessor(
                        clip_stream,
                        model=pi3x_vo_cfg.get("model", "yyfz233/Pi3X"),
                        chunk_size=pi3x_vo_cfg.get("chunk_size", 16),
                        overlap=pi3x_vo_cfg.get("overlap", 6),
                        conf_thre=pi3x_vo_cfg.get("conf_thre", 0.05),
                        dtype=pi3x_vo_cfg.get("dtype", "bf16"),
                        pose_convention=pi3x_vo_cfg.get("pose_convention", "c2w"),
                        return_depth=False,
                        depth_conf_thre=pi3x_vo_cfg.get("depth_conf_thre", None),
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
            if bool(self.sparse_ba_cfg.get("enabled", False)):
                refined = self._refine_chunk_sparse_ba(pi3x_data, clip_stream)
                pi3x_data = refined

            overlap_left = min(align_overlap, start - clip_start)
            overlap_right = min(align_overlap, clip_end - end)
            align_result = None
            aligned = False

            if overlap_left > 0:
                slam_left = {
                    "depth": global_data["depth"][clip_start: start],
                    "intrinsic": global_data["intrinsic"][clip_start: start],
                    "mask": global_data["mask"][clip_start: start],
                }
                pi3x_left = {
                    "depth": pi3x_data["depth"][:overlap_left],
                    "intrinsic": pi3x_data["intrinsic"][:overlap_left],
                    "mask": pi3x_data["mask"][:overlap_left],
                }
                align_result = self._align_clip(slam_left, pi3x_left, overlap_left)

            if align_result is None and overlap_right > 0:
                slam_right = {
                    "depth": global_data["depth"][end: clip_end],
                    "intrinsic": global_data["intrinsic"][end: clip_end],
                    "mask": global_data["mask"][end: clip_end],
                }
                pi3x_right = {
                    "depth": pi3x_data["depth"][-overlap_right:],
                    "intrinsic": pi3x_data["intrinsic"][-overlap_right:],
                    "mask": pi3x_data["mask"][-overlap_right:],
                }
                align_result = self._align_clip(slam_right, pi3x_right, overlap_right)

            if align_result is None:
                align_result = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            else:
                aligned = True

            s, R, t = align_result

            extrinsics = pi3x_data["extrinsic"]
            R_local = extrinsics[:, :3, :3]
            t_local = extrinsics[:, :3, 3]
            R_new = R[None, ...] @ R_local
            t_new = (s * (R[None, ...] @ t_local[..., None]).squeeze(-1)) + t[None, ...]

            poses_new = np.repeat(np.eye(4, dtype=np.float32)[None], len(extrinsics), axis=0)
            poses_new[:, :3, :3] = R_new
            poses_new[:, :3, 3] = t_new

            depths_new = []
            for d in pi3x_data["depth"]:
                if d is None:
                    depths_new.append(None)
                else:
                    depths_new.append(d * s)

            clip_offset = start - clip_start
            clip_len = end - start
            for i in range(clip_len):
                gi = start + i
                pi = clip_offset + i
                global_data["extrinsic"][gi] = poses_new[pi]
                global_data["depth"][gi] = depths_new[pi]

            clip_stats.append(
                ClipStats(
                    index=idx,
                    start=start,
                    end=end,
                    reason="mask_or_pose_jump",
                    aligned=aligned,
                    sparse_ba_applied=bool(pi3x_data.get("sparse_ba_refined", False)),
                )
            )

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

        artifact_path = io.ArtifactPath(self.out_path, video_stream.name())
        if self.out_cfg.save_artifacts:
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info("Saving artifacts to %s", artifact_path)
            io.save_artifacts(artifact_path, output_stream)
            with artifact_path.meta_info_path.open("wb") as f:
                info = {
                    "pipeline": "clip_repair",
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

