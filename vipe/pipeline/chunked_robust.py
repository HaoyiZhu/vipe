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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vipe.ext.lietorch import SE3
from vipe.pipeline import AnnotationPipelineOutput
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.pipeline.long_sequence import (
    AssignInstancePhrasesProcessor,
    SlicedVideoStream,
)
from vipe.slam.components.sparse_tracks import build_sparse_tracks
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
from vipe.utils.sim3 import Sim3LoopOptimizer, robust_weighted_align_point_maps
from vipe.utils.loop_detection import LoopDetector, process_loop_list
from vipe.utils.visualization import save_projection_video

from .processors import (
    AdaptiveDepthProcessor,
    MultiviewDepthProcessor,
    Pi3XAdaptiveDepthProcessor,
    Pi3XMetricDepthProcessor,
    Pi3XMoGePerFrameProcessor,
    Pi3XMoGeProcessor,
)
from .robust import RobustAnnotationPipeline

logger = logging.getLogger(__name__)


@dataclass
class ChunkStats:
    index: int
    start: int
    end: int
    kind: str
    guard_reason: str
    ba_residual: float | None
    mask_min_ratio: float
    mask_mean_ratio: float
    pose_guard_rejected: bool
    quality_weight: float
    sparse_ba_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "guard_reason": self.guard_reason,
            "ba_residual": self.ba_residual,
            "mask_min_ratio": self.mask_min_ratio,
            "mask_mean_ratio": self.mask_mean_ratio,
            "pose_guard_rejected": self.pose_guard_rejected,
            "quality_weight": self.quality_weight,
            "sparse_ba_applied": self.sparse_ba_applied,
        }


@dataclass
class ChunkResult:
    index: int
    start: int
    end: int
    kind: str
    data: dict[str, Any]
    quality_weight: float


class ChunkedRobustAnnotationPipeline(RobustAnnotationPipeline):
    def __init__(
        self,
        init: DictConfig,
        slam: DictConfig,
        post: DictConfig,
        output: DictConfig,
        robust: DictConfig | None = None,
        chunked: DictConfig | None = None,
    ) -> None:
        super().__init__(init, slam, post, output, robust)
        self.chunk_cfg = chunked if chunked is not None else OmegaConf.create({})

        self.chunk_size = int(self.chunk_cfg.get("chunk_size", 200))
        self.overlap = int(self.chunk_cfg.get("overlap", 50))
        self.min_chunk_size = int(self.chunk_cfg.get("min_chunk_size", 20))
        self.blend_len = int(self.chunk_cfg.get("blend_len", max(5, self.overlap // 3)))
        self.bad_chunk_weight = float(self.chunk_cfg.get("bad_chunk_weight", 0.5))
        self.anchor_first_good = bool(self.chunk_cfg.get("anchor_first_good", True))
        self.adaptive_cfg = self.chunk_cfg.get("adaptive", DictConfig({}))
        self.adaptive_chunking = bool(self.adaptive_cfg.get("enabled", False))
        self.loop_cfg = self.chunk_cfg.get("loop_edges", DictConfig({}))
        self.precompute_init = bool(self.chunk_cfg.get("precompute_init", self.adaptive_chunking))
        self.sparse_ba_cfg = self.chunk_cfg.get("sparse_ba", DictConfig({}))
        self.global_slam_cfg = self.chunk_cfg.get("global_slam", DictConfig({}))
        self.global_slam_enabled = bool(self.global_slam_cfg.get("enabled", False))

        self.world_points_source = self.chunk_cfg.get("world_points_source", "auto")
        self.align_conf_threshold_coef = float(self.chunk_cfg.get("align_conf_threshold_coef", 0.1))
        self.using_sim3 = bool(self.chunk_cfg.get("using_sim3", True))
        irls_cfg = self.chunk_cfg.get("irls", DictConfig({}))
        self.irls_delta = float(irls_cfg.get("delta", 0.1))
        self.irls_max_iters = int(irls_cfg.get("max_iters", 5))
        self.irls_tol = float(irls_cfg.get("tol", 1e-9))

        sim3_cfg = self.chunk_cfg.get("sim3_optimizer", DictConfig({}))
        self.opt_max_iterations = int(sim3_cfg.get("max_iterations", 30))
        self.opt_lambda_init = float(sim3_cfg.get("lambda_init", 1e-6))

        self.delete_temp_files = bool(self.chunk_cfg.get("delete_temp_files", True))
        self.result_unaligned_dir = self.out_path / "_tmp_chunked_robust"
        self.result_unaligned_dir.mkdir(exist_ok=True, parents=True)

        self.post_bad_cfg = self.chunk_cfg.get("post_bad", None)

    @staticmethod
    def _compose_sim3(
        s1: float, R1: np.ndarray, t1: np.ndarray, s2: float, R2: np.ndarray, t2: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        s = s1 * s2
        R = R1 @ R2
        t = t1 + s1 * (R1 @ t2)
        return s, R, t

    @staticmethod
    def _invert_sim3(s: float, R: np.ndarray, t: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s_inv = 1.0 / max(s, 1e-9)
        R_inv = R.T
        t_inv = -s_inv * (R_inv @ t)
        return s_inv, R_inv, t_inv

    def _make_chunk_indices(self, total_frames: int) -> list[tuple[int, int]]:
        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")

        if total_frames <= self.chunk_size:
            return [(0, total_frames)]

        step = self.chunk_size - self.overlap
        num_chunks = (total_frames - self.overlap + step - 1) // step
        chunk_indices = []
        for i in range(num_chunks):
            start_idx = i * step
            end_idx = min(start_idx + self.chunk_size, total_frames)
            if end_idx - start_idx < self.min_chunk_size:
                break
            chunk_indices.append((start_idx, end_idx))
        return chunk_indices

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

    def _make_chunk_indices_for_range(self, start: int, end: int) -> list[tuple[int, int]]:
        if end - start <= 0:
            return []
        length = end - start
        if length <= self.chunk_size:
            return [(start, end)]
        step = self.chunk_size - self.overlap
        num_chunks = (length - self.overlap + step - 1) // step
        indices = []
        for i in range(num_chunks):
            s = start + i * step
            e = min(s + self.chunk_size, end)
            if e - s < self.min_chunk_size:
                break
            indices.append((s, e))
        return indices

    def _rolling_mean(self, values: list[float], window: int) -> list[float]:
        if window <= 1:
            return values
        half = window // 2
        out = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            out.append(float(sum(values[lo:hi]) / max(1, hi - lo)))
        return out

    def _prepass_health(self, stream: VideoStream) -> dict[str, Any]:
        mask_ratios = []
        pose_valid = []
        pose_jump = []
        last_pose = None
        for frame in stream:
            if frame.mask is None:
                mask_ratio = 1.0
            else:
                mask = frame.mask
                if mask.dtype != torch.bool:
                    mask = mask > 0
                mask_ratio = float(mask.float().mean().item())
            mask_ratios.append(mask_ratio)

            valid_pose = frame.pose is not None
            pose_valid.append(valid_pose)
            if valid_pose and last_pose is not None:
                rel = (last_pose.inv() * frame.pose).matrix().detach().cpu().numpy()
                translation = float(np.linalg.norm(rel[:3, 3]))
                trace = float(np.trace(rel[:3, :3]))
                cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
                rotation_deg = float(np.degrees(np.arccos(cos_theta)))
                pose_jump.append((translation, rotation_deg))
            else:
                pose_jump.append((0.0, 0.0))
            if valid_pose:
                last_pose = frame.pose
        return {
            "mask_ratios": mask_ratios,
            "pose_valid": pose_valid,
            "pose_jump": pose_jump,
        }

    def _detect_bad_zones(self, health: dict[str, Any], total_frames: int) -> list[tuple[int, int]]:
        if total_frames == 0:
            return []

        min_mask_ratio = float(self.adaptive_cfg.get("min_mask_ratio", 0.05))
        min_mask_mean_ratio = float(self.adaptive_cfg.get("min_mask_mean_ratio", 0.1))
        bad_if_pose_missing = bool(self.adaptive_cfg.get("bad_if_pose_missing", True))
        bad_if_pose_jump = bool(self.adaptive_cfg.get("bad_if_pose_jump", True))
        pose_jump_translation = float(self.adaptive_cfg.get("pose_jump_translation", 2.5))
        pose_jump_rotation_deg = float(self.adaptive_cfg.get("pose_jump_rotation_deg", 25.0))
        window = int(self.adaptive_cfg.get("bad_zone_window", 5))
        min_len = int(self.adaptive_cfg.get("bad_zone_min_len", 1))
        expand = int(self.adaptive_cfg.get("bad_zone_expand", self.overlap))
        merge_gap = int(self.adaptive_cfg.get("bad_zone_merge_gap", 5))

        mask_ratios = health["mask_ratios"]
        mask_mean = self._rolling_mean(mask_ratios, window)
        pose_valid = health["pose_valid"]
        pose_jump = health["pose_jump"]

        bad_flags = []
        for idx in range(total_frames):
            bad = False
            if mask_ratios[idx] < min_mask_ratio or mask_mean[idx] < min_mask_mean_ratio:
                bad = True
            if bad_if_pose_missing and not pose_valid[idx]:
                bad = True
            if bad_if_pose_jump:
                translation, rotation_deg = pose_jump[idx]
                if translation > pose_jump_translation or rotation_deg > pose_jump_rotation_deg:
                    bad = True
            bad_flags.append(bad)

        zones = []
        start = None
        for idx, bad in enumerate(bad_flags):
            if bad and start is None:
                start = idx
            elif not bad and start is not None:
                if idx - start >= min_len:
                    zones.append((start, idx))
                start = None
        if start is not None and total_frames - start >= min_len:
            zones.append((start, total_frames))

        if not zones:
            return []

        expanded = []
        for start, end in zones:
            s = max(0, start - expand)
            e = min(total_frames, end + expand)
            expanded.append((s, e))

        expanded.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = expanded[0]
        for s, e in expanded[1:]:
            if s <= cur_e + merge_gap:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _build_adaptive_chunks(
        self,
        total_frames: int,
        bad_zones: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], set[int]]:
        if not bad_zones:
            return self._make_chunk_indices(total_frames), set()

        chunk_indices: list[tuple[int, int]] = []
        forced_bad: set[int] = set()

        cur = 0
        for start, end in bad_zones:
            if cur < start:
                good_segments = self._make_chunk_indices_for_range(cur, start)
                chunk_indices.extend(good_segments)
            if end - start > 0:
                chunk_indices.append((start, end))
                forced_bad.add(len(chunk_indices) - 1)
            cur = end
        if cur < total_frames:
            chunk_indices.extend(self._make_chunk_indices_for_range(cur, total_frames))

        return chunk_indices, forced_bad

    def _make_chunk_indices_with_params(
        self,
        total_frames: int,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> list[tuple[int, int]]:
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk size ({chunk_size})")

        if total_frames <= chunk_size:
            return [(0, total_frames)]

        step = chunk_size - overlap
        num_chunks = (total_frames - overlap + step - 1) // step
        chunk_indices = []
        for i in range(num_chunks):
            start_idx = i * step
            end_idx = min(start_idx + chunk_size, total_frames)
            if end_idx - start_idx < min_chunk_size:
                break
            chunk_indices.append((start_idx, end_idx))
        return chunk_indices

    def _make_chunk_indices_for_range_with_params(
        self,
        start: int,
        end: int,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> list[tuple[int, int]]:
        if end - start <= 0:
            return []
        length = end - start
        if length <= chunk_size:
            return [(start, end)]
        step = chunk_size - overlap
        num_chunks = (length - overlap + step - 1) // step
        indices = []
        for i in range(num_chunks):
            s = start + i * step
            e = min(s + chunk_size, end)
            if e - s < min_chunk_size:
                break
            indices.append((s, e))
        return indices

    def _build_adaptive_chunks_with_params(
        self,
        total_frames: int,
        bad_zones: list[tuple[int, int]],
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> tuple[list[tuple[int, int]], set[int]]:
        if not bad_zones:
            return self._make_chunk_indices_with_params(total_frames, chunk_size, overlap, min_chunk_size), set()

        chunk_indices: list[tuple[int, int]] = []
        forced_bad: set[int] = set()

        cur = 0
        for start, end in bad_zones:
            if cur < start:
                good_segments = self._make_chunk_indices_for_range_with_params(
                    cur, start, chunk_size, overlap, min_chunk_size
                )
                chunk_indices.extend(good_segments)
            if end - start > 0:
                chunk_indices.append((start, end))
                forced_bad.add(len(chunk_indices) - 1)
            cur = end
        if cur < total_frames:
            chunk_indices.extend(
                self._make_chunk_indices_for_range_with_params(cur, total_frames, chunk_size, overlap, min_chunk_size)
            )

        return chunk_indices, forced_bad

    def _bad_zones_from_chunk_stats(
        self,
        chunk_stats: list[ChunkStats],
        total_frames: int,
        max_ba_residual: float,
        bad_zone_expand: int,
        bad_zone_merge_gap: int,
        min_bad_len: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for stats in chunk_stats:
            if stats.ba_residual is None:
                continue
            if stats.ba_residual > max_ba_residual:
                start = stats.start
                end = stats.end
                if end - start >= min_bad_len:
                    ranges.append((start, end))

        if not ranges:
            return []

        expanded = []
        for start, end in ranges:
            s = max(0, start - bad_zone_expand)
            e = min(total_frames, end + bad_zone_expand)
            expanded.append((s, e))

        return self._merge_ranges(expanded, bad_zone_merge_gap)

    def _compute_conf_threshold(self, conf1: Any, conf2: Any) -> float:
        if conf1 is None or conf2 is None:
            return -1.0
        if isinstance(conf1, torch.Tensor):
            c1 = torch.median(conf1).item()
        else:
            c1 = float(np.median(conf1))
        if isinstance(conf2, torch.Tensor):
            c2 = torch.median(conf2).item()
        else:
            c2 = float(np.median(conf2))
        return min(c1, c2) * self.align_conf_threshold_coef

    def _maybe_build_align_mask(
        self,
        mask1: np.ndarray | None,
        mask2: np.ndarray | None,
        point_map1: np.ndarray,
    ) -> np.ndarray | None:
        if mask1 is None or mask2 is None:
            return None
        m1 = np.squeeze(mask1)
        m2 = np.squeeze(mask2)
        if m1.shape != m2.shape:
            return None
        if point_map1.ndim == 4 and m1.shape == point_map1.shape[:3]:
            return m1 & m2
        if point_map1.ndim == 3 and m1.shape == point_map1.shape[:2]:
            return m1 & m2
        return None

    def _backproject_depth(self, depth: torch.Tensor, intr: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        h, w = depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=depth.device),
            torch.arange(w, device=depth.device),
            indexing="ij",
        )
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        pts_cam = torch.stack([X, Y, Z], dim=-1)
        pts_flat = pts_cam.reshape(-1, 3)
        pts_world_flat = (pose[:3, :3] @ pts_flat.T).T + pose[:3, 3]
        return pts_world_flat.reshape(h, w, 3)

    def _point_maps_from_depth(
        self,
        data: dict[str, Any],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        depths = data.get("depth")
        if depths is None:
            raise ValueError("Depth data is not available for alignment.")

        depths = depths[start:end]
        intrinsics = data["intrinsic"][start:end]
        extrinsics = data["extrinsic"][start:end]
        masks = data.get("mask")
        masks = masks[start:end] if masks is not None and masks.shape[0] >= end else None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = []
        confs = []
        for i in range(len(depths)):
            depth = torch.from_numpy(depths[i]).float().to(device)
            intr = torch.from_numpy(intrinsics[i]).float().to(device)
            pose = torch.from_numpy(extrinsics[i]).float().to(device)
            pts_world = self._backproject_depth(depth, intr, pose)
            points.append(pts_world.detach().cpu().numpy())
            confs.append(np.ones_like(depths[i], dtype=np.float32))

        if points:
            return np.stack(points), np.stack(confs), masks
        return np.zeros((0, 0, 0, 3), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32), masks

    def _get_alignment_maps(
        self,
        data: dict[str, Any],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if data.get("depth") is not None:
            return self._point_maps_from_depth(data, start, end)

        points = data["world_points"]
        confs = data["world_points_conf"]
        masks = data.get("mask")

        if points.ndim == 4:
            pts = points[start:end]
            cfs = confs[start:end] if confs is not None else np.ones(points.shape[:3], dtype=np.float32)
            msk = masks[start:end] if masks is not None and masks.shape[0] >= end else None
            return pts, cfs, msk

        return points, confs, None

    def _build_post_processors(
        self,
        view_idx: int,
        video_stream: VideoStream,
        slam_output: SLAMOutput,
        post_cfg: DictConfig,
        assign_pose: bool = True,
    ) -> ProcessedVideoStream:
        post_processors = []
        if assign_pose:
            if slam_output.per_frame_intrinsics:
                intrinsics_list = list(slam_output.intrinsics[:, view_idx])
            else:
                intrinsics_list = [slam_output.intrinsics[view_idx]] * len(video_stream)

            post_processors.append(
                AssignAttributesProcessor(
                    {
                        FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                        FrameAttribute.INTRINSICS: intrinsics_list,
                    }
                )
            )
        if (depth_align_model := post_cfg.get("depth_align_model", None)) is not None:
            if depth_align_model == "pi3x_moge":
                post_processors.append(
                    Pi3XMoGeProcessor(
                        slam_output,
                        window_size=post_cfg.get("window_size", 100),
                        overlap_size=post_cfg.get("overlap_size", 20),
                        pixel_limit=post_cfg.get("pixel_limit", 255000),
                    )
                )
            elif depth_align_model == "pi3x_moge_perframe":
                post_processors.append(
                    Pi3XMoGePerFrameProcessor(
                        slam_output,
                        view_idx=view_idx,
                        window_size=post_cfg.get("window_size", 64),
                        overlap_size=post_cfg.get("overlap_size", 16),
                        pixel_limit=post_cfg.get("pixel_limit", 255000),
                        align_lr_size=post_cfg.get("align_lr_size", 64),
                        min_align_points=post_cfg.get("min_align_points", 200),
                        align_mode=post_cfg.get("align_mode", "per_frame_ema"),
                        align_momentum=post_cfg.get("align_momentum", 0.99),
                        scale_clamp=tuple(post_cfg.get("scale_clamp", (0.1, 10.0))),
                        shift_z_clamp=tuple(post_cfg.get("shift_z_clamp", (-1e3, 1e3))),
                        moge_bs=post_cfg.get("moge_bs", 4),
                        align_source=post_cfg.get("align_source", "moge2"),
                    )
                )
            elif depth_align_model == "adaptive_pi3x":
                post_processors.append(
                    Pi3XAdaptiveDepthProcessor(
                        slam_output,
                        view_idx=view_idx,
                        metric_model=post_cfg.get("metric_model", "unidepth-l"),
                        pixel_limit=post_cfg.get("pixel_limit", 255000),
                        batch_size=post_cfg.get("batch_size", 4),
                    )
                )
            elif depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            else:
                post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def _build_chunk_slam_output(self, stream: VideoStream, rig: SE3 | None) -> SLAMOutput:
        poses = []
        last_pose = None
        for frame in stream:
            if frame.pose is None:
                if last_pose is None:
                    pose = SE3.Identity(1, device=frame.rgb.device)
                else:
                    pose = last_pose
            else:
                pose = frame.pose
                last_pose = pose
            poses.append(pose)

        if not poses:
            trajectory = SE3.Identity(1)
        else:
            trajectory = SE3(torch.stack([p.data for p in poses], dim=0))

        intrinsics_list = []
        for frame in stream:
            if frame.intrinsics is None:
                intr = torch.zeros(4)
            else:
                intr = frame.intrinsics[:4].detach().cpu()
            intrinsics_list.append(intr)
        intrinsics = torch.stack(intrinsics_list, dim=0) if intrinsics_list else torch.zeros((1, 4))

        rig_out = rig if rig is not None else SE3.Identity(1)

        return SLAMOutput(
            trajectory=trajectory,
            intrinsics=intrinsics[:1],
            rig=rig_out,
            slam_map=None,
            ba_residual=float("nan"),
        )

    def _collect_chunk_data(
        self,
        output_stream: VideoStream,
        slam_output: SLAMOutput,
        world_points_source: str,
    ) -> dict[str, Any]:
        extrinsics = []
        intrinsics = []
        depths = []
        masks = []
        instances = []
        images = []
        world_points = []
        world_points_conf = []
        instance_phrases: dict[int, str] = {}

        has_depth = False
        has_mask = False
        has_instance = False

        use_slam_map = False
        if world_points_source in ("slam_map", "auto") and slam_output.slam_map is not None:
            use_slam_map = True

        for frame in output_stream:
            if frame.pose is None:
                pose = torch.eye(4, device=frame.rgb.device)
            else:
                pose = frame.pose.matrix()
            extrinsics.append(pose.detach().cpu().numpy())

            if frame.intrinsics is None:
                intr = torch.zeros(4, device=pose.device)
            else:
                intr = frame.intrinsics[:4]
            intrinsics.append(intr.detach().cpu().numpy())

            if frame.metric_depth is not None:
                depth = frame.metric_depth
                has_depth = True
                depths.append(depth.detach().cpu().numpy())
            else:
                depth = None
                depths.append(None)

            if frame.mask is not None:
                has_mask = True
                masks.append(frame.mask.detach().cpu().numpy().astype(bool))
            else:
                masks.append(None)

            if frame.instance is not None:
                has_instance = True
                instances.append(frame.instance.detach().cpu().numpy().astype(np.uint8))
            else:
                instances.append(None)

            if frame.instance_phrases:
                instance_phrases.update(frame.instance_phrases)

            images.append(frame.rgb.permute(2, 0, 1).detach().cpu().numpy())

            if not use_slam_map:
                if depth is None:
                    depth = torch.zeros(frame.size(), device=pose.device)
                conf = torch.ones_like(depth)
                pts_world = self._backproject_depth(depth, intr, pose)
                world_points.append(pts_world.detach().cpu().numpy())
                world_points_conf.append(conf.detach().cpu().numpy())

        extrinsics = np.stack(extrinsics) if extrinsics else np.zeros((0, 4, 4), dtype=np.float32)
        intrinsics = np.stack(intrinsics) if intrinsics else np.zeros((0, 4), dtype=np.float32)
        images = np.stack(images) if images else np.zeros((0, 3, 0, 0), dtype=np.float32)

        if has_depth:
            depth_template = next((d for d in depths if d is not None), None)
            for i, d in enumerate(depths):
                if d is None and depth_template is not None:
                    depths[i] = np.zeros_like(depth_template, dtype=np.float32)
            depths = np.stack(depths)
        else:
            depths = None

        if has_mask:
            mask_template = next((m for m in masks if m is not None), None)
            for i, m in enumerate(masks):
                if m is None and mask_template is not None:
                    masks[i] = np.ones_like(mask_template, dtype=bool)
            masks = np.stack(masks)
        else:
            masks = None

        if has_instance:
            inst_template = next((inst for inst in instances if inst is not None), None)
            for i, inst in enumerate(instances):
                if inst is None and inst_template is not None:
                    instances[i] = np.zeros_like(inst_template, dtype=np.uint8)
            instances = np.stack(instances)
        else:
            instances = None

        world_colors = None
        if use_slam_map:
            slam_map = slam_output.slam_map
            world_points = slam_map.dense_disp_xyz.detach().cpu().numpy()
            world_points = world_points[np.newaxis, ...]
            world_points_conf = np.ones((1, world_points.shape[1]), dtype=np.float32)
            world_colors = slam_map.dense_disp_rgb.detach().cpu().numpy()
        else:
            if world_points:
                world_points = np.stack(world_points)
                world_points_conf = np.stack(world_points_conf)
            else:
                world_points = np.zeros((1, 0, 3), dtype=np.float32)
                world_points_conf = np.zeros((1, 0), dtype=np.float32)

        chunk_data = {
            "world_points": world_points,
            "world_points_conf": world_points_conf,
            "world_colors": world_colors,
            "mask": masks,
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "depth": depths,
            "images": images,
            "instance": instances,
            "instance_phrases": instance_phrases if instance_phrases else None,
            "ba_residual": float(slam_output.ba_residual) if slam_output is not None else float("nan"),
        }
        return chunk_data

    def _guard_precheck(self, mask_stats, pose_stats) -> tuple[bool, str]:
        guard_cfg = self.chunk_cfg.get("guard", DictConfig({}))
        require_pose_guard = bool(guard_cfg.get("require_pose_guard_accept", True))
        min_valid_mask_ratio = float(guard_cfg.get("min_valid_mask_ratio", 0.0))
        min_mean_valid_mask_ratio = float(guard_cfg.get("min_mean_valid_mask_ratio", 0.0))

        if require_pose_guard and pose_stats is not None and getattr(pose_stats, "rejected", False):
            return True, "pose_guard"
        if mask_stats is not None and getattr(mask_stats, "enabled", False):
            if mask_stats.min_ratio < min_valid_mask_ratio:
                return True, "mask_min"
            if mask_stats.mean_ratio < min_mean_valid_mask_ratio:
                return True, "mask_mean"
        return False, "ok"

    def _guard_postcheck(self, ba_residual: float) -> tuple[bool, str]:
        guard_cfg = self.chunk_cfg.get("guard", DictConfig({}))
        max_ba_residual = float(guard_cfg.get("max_ba_residual", 0.05))
        if np.isfinite(ba_residual) and ba_residual > max_ba_residual:
            return True, "ba_residual"
        return False, "ok"

    def _compute_chunk_weight(self, kind: str) -> float:
        return 1.0 if kind == "slam" else self.bad_chunk_weight

    def _build_bad_chunk_stream(self, stream: VideoStream) -> VideoStream:
        fallback_cfg = self.chunk_cfg.get("fallback", DictConfig({}))
        use_pi3x_depth = bool(fallback_cfg.get("use_pi3x_depth", True))
        if not use_pi3x_depth:
            return stream
        depth_model = fallback_cfg.get("pi3x_model", "yyfz233/Pi3X")
        pixel_limit = int(fallback_cfg.get("pixel_limit", 255000))
        batch_size = int(fallback_cfg.get("depth_batch_size", 1))
        use_poses = bool(fallback_cfg.get("use_poses", True))
        return ProcessedVideoStream(
            stream,
            [
                Pi3XMetricDepthProcessor(
                    model=depth_model,
                    pixel_limit=pixel_limit,
                    batch_size=batch_size,
                    use_poses=use_poses,
                )
            ],
        )

    @staticmethod
    def _unproject_uv(
        uv: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if uv.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
        h, w = depth.shape
        u = np.rint(uv[:, 0]).astype(np.int64)
        v = np.rint(uv[:, 1]).astype(np.int64)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_bounds):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
        u = u[in_bounds]
        v = v[in_bounds]
        z = depth[v, u]
        valid = np.isfinite(z) & (z > 1e-6)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        fx, fy, cx, cy = intrinsics
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pts = np.stack([x, y, z], axis=-1).astype(np.float32)
        return pts, in_bounds

    def _compute_seq_transforms_from_extrinsics(
        self, extrinsics: np.ndarray
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        seq_transforms = []
        for i in range(extrinsics.shape[0] - 1):
            pose_i = extrinsics[i]
            pose_j = extrinsics[i + 1]
            rel = np.linalg.inv(pose_i) @ pose_j
            seq_transforms.append((1.0, rel[:3, :3], rel[:3, 3]))
        return seq_transforms

    def _refine_chunk_sparse_ba(
        self,
        chunk_data: dict[str, Any],
        init_stream: VideoStream,
    ) -> dict[str, Any]:
        if not bool(self.sparse_ba_cfg.get("enabled", False)):
            return chunk_data

        depths = chunk_data.get("depth")
        intrinsics = chunk_data.get("intrinsic")
        extrinsics = chunk_data.get("extrinsic")
        if depths is None or intrinsics is None or extrinsics is None:
            return chunk_data

        n_frames = extrinsics.shape[0]
        if n_frames < 2:
            return chunk_data

        tracks_cfg = self.sparse_ba_cfg.get("tracks", DictConfig({}))
        try:
            tracks = build_sparse_tracks(tracks_cfg, n_views=1)
        except Exception:
            logger.exception("Sparse BA: failed to build tracks.")
            return chunk_data

        try:
            tracks.precompute([init_stream])
        except Exception:
            logger.exception("Sparse BA: track precompute failed.")
            return chunk_data

        track_quality_cfg = self.sparse_ba_cfg.get("track_quality", DictConfig({}))
        if bool(track_quality_cfg.get("enabled", False)):
            stats = self._compute_track_quality(tracks, track_quality_cfg)
            chunk_data["sparse_ba_track_stats"] = stats
            if not stats.get("passed", False):
                chunk_data["sparse_ba_skipped"] = "track_quality"
                return chunk_data

        min_tracks = int(self.sparse_ba_cfg.get("min_tracks", 30))
        min_points = int(self.sparse_ba_cfg.get("min_points", 30))
        min_frame_gap = int(self.sparse_ba_cfg.get("min_frame_gap", 10))
        max_frame_gap = int(self.sparse_ba_cfg.get("max_frame_gap", 200))
        max_edges = int(self.sparse_ba_cfg.get("max_edges", 100))
        use_sim3 = bool(self.sparse_ba_cfg.get("use_sim3", False))
        max_iterations = int(self.sparse_ba_cfg.get("max_iterations", self.opt_max_iterations))
        lambda_init = float(self.sparse_ba_cfg.get("lambda_init", self.opt_lambda_init))

        try:
            ii, jj = tracks.get_overlapping_pairs(
                min_common=min_tracks,
                min_frame_gap=min_frame_gap,
                max_pairs=max_edges,
            )
        except Exception:
            logger.exception("Sparse BA: failed to find overlapping track pairs.")
            return chunk_data

        loop_constraints = []
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i < 0 or j < 0 or i >= n_frames or j >= n_frames:
                continue
            if j - i > max_frame_gap:
                continue
            kp_idx = tracks.get_correspondences(0, i, j)
            if len(kp_idx) < min_tracks:
                continue
            uv_i = tracks.get_observations(0, i, kp_idx).cpu().numpy()
            uv_j = tracks.get_observations(0, j, kp_idx).cpu().numpy()
            pts_i, valid_i = self._unproject_uv(uv_i, depths[i], intrinsics[i])
            pts_j, valid_j = self._unproject_uv(uv_j, depths[j], intrinsics[j])
            if pts_i.shape[0] == 0 or pts_j.shape[0] == 0:
                continue
            n_valid = min(pts_i.shape[0], pts_j.shape[0])
            if n_valid < min_points:
                continue
            pts_i = pts_i[:n_valid]
            pts_j = pts_j[:n_valid]
            conf = np.ones((n_valid,), dtype=np.float32)

            try:
                s, R, t = robust_weighted_align_point_maps(
                    torch.from_numpy(pts_i),
                    torch.from_numpy(conf),
                    torch.from_numpy(pts_j),
                    torch.from_numpy(conf),
                    None,
                    -1.0,
                    delta=self.irls_delta,
                    max_iters=self.irls_max_iters,
                    tol=self.irls_tol,
                    using_sim3=use_sim3,
                )
            except Exception:
                continue
            loop_constraints.append((i, j, (float(s), R, t)))

        if not loop_constraints:
            return chunk_data

        seq_transforms = self._compute_seq_transforms_from_extrinsics(extrinsics)
        optimizer = Sim3LoopOptimizer(max_iterations=max_iterations, lambda_init=lambda_init)
        seq_transforms = optimizer.optimize(seq_transforms, loop_constraints)
        abs_poses = optimizer.sequential_to_absolute_poses(seq_transforms)

        anchor_R = extrinsics[0][:3, :3]
        anchor_t = extrinsics[0][:3, 3]
        refined = np.zeros_like(extrinsics)
        for idx in range(n_frames):
            s_opt, R_opt, t_opt = optimizer.pypose_sim3_to_numpy(abs_poses[idx])
            s_new, R_new, t_new = self._compose_sim3(1.0, anchor_R, anchor_t, s_opt, R_opt, t_opt)
            refined[idx, :3, :3] = R_new
            refined[idx, :3, 3] = t_new
            refined[idx, 3, 3] = 1.0

        chunk_data["extrinsic"] = refined
        chunk_data["sparse_ba_refined"] = True
        return chunk_data

    def _compute_track_quality(self, tracks, cfg: DictConfig) -> dict[str, Any]:
        observations = tracks.observations[0] if tracks.observations else []
        frame_counts = [len(obs) for obs in observations]
        total_frames = len(frame_counts)
        total_tracks = 0
        unique_tracks = set()
        for obs in observations:
            unique_tracks.update(obs.keys())
        total_tracks = len(unique_tracks)

        mean_tracks = float(np.mean(frame_counts)) if frame_counts else 0.0
        median_tracks = float(np.median(frame_counts)) if frame_counts else 0.0
        min_tracks_per_frame = int(cfg.get("min_tracks_per_frame", 10))
        coverage = (
            float(sum(1 for c in frame_counts if c >= min_tracks_per_frame) / max(1, total_frames))
            if total_frames > 0
            else 0.0
        )

        min_total_tracks = int(cfg.get("min_total_tracks", 100))
        min_mean_tracks = float(cfg.get("min_mean_tracks", 10.0))
        min_median_tracks = float(cfg.get("min_median_tracks", 10.0))
        min_coverage = float(cfg.get("min_track_coverage", 0.5))

        passed = (
            total_tracks >= min_total_tracks
            and mean_tracks >= min_mean_tracks
            and median_tracks >= min_median_tracks
            and coverage >= min_coverage
        )

        return {
            "total_tracks": total_tracks,
            "mean_tracks": mean_tracks,
            "median_tracks": median_tracks,
            "coverage": coverage,
            "min_total_tracks": min_total_tracks,
            "min_mean_tracks": min_mean_tracks,
            "min_median_tracks": min_median_tracks,
            "min_track_coverage": min_coverage,
            "min_tracks_per_frame": min_tracks_per_frame,
            "passed": passed,
        }

    def _process_chunk(
        self,
        chunk_idx: int,
        chunk_stream: VideoStream,
        rig: SE3 | None,
        force_bad: bool = False,
        use_init_stream: bool = False,
    ) -> tuple[ChunkResult, ChunkStats]:
        chunk_name = f"{chunk_stream.name()}"
        if use_init_stream:
            init_stream = chunk_stream
        else:
            init_stream = self._add_init_processors(chunk_stream).cache("process", online=True)

        mask_stats = self._compute_mask_coverage(init_stream, chunk_name)
        pose_stats = self._evaluate_pose_guard(init_stream, chunk_name)
        skip_slam, guard_reason = self._guard_precheck(mask_stats, pose_stats)
        if force_bad:
            skip_slam = True
            guard_reason = "forced_bad"

        slam_output = None
        output_stream = None
        kind = "slam"
        ba_residual = None

        if not skip_slam:
            slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
            slam_output = slam_pipeline.run([init_stream], rig=rig, camera_type=self.camera_type)
            ba_residual = float(slam_output.ba_residual)
            skip_slam, guard_reason = self._guard_postcheck(ba_residual)
            if not skip_slam:
                output_stream = self._build_post_processors(0, init_stream, slam_output, self.post_cfg, assign_pose=True)
                output_stream = output_stream.cache("depth", online=True)
            del slam_pipeline
            torch.cuda.empty_cache()

        if skip_slam:
            kind = "pi3x"
            if self.post_bad_cfg is not None:
                dummy_output = self._build_chunk_slam_output(init_stream, rig)
                output_stream = self._build_post_processors(
                    0,
                    init_stream,
                    dummy_output,
                    self.post_bad_cfg,
                    assign_pose=False,
                ).cache("bad_depth", online=True)
                slam_output = self._build_chunk_slam_output(output_stream, rig)
            else:
                output_stream = self._build_bad_chunk_stream(init_stream).cache("pi3x_depth", online=True)
                slam_output = self._build_chunk_slam_output(output_stream, rig)
            ba_residual = None

        assert output_stream is not None
        assert slam_output is not None

        chunk_data = self._collect_chunk_data(
            output_stream,
            slam_output,
            world_points_source=self.world_points_source,
        )

        quality_weight = self._compute_chunk_weight(kind)
        stats = ChunkStats(
            index=chunk_idx,
            start=0,
            end=len(output_stream),
            kind=kind,
            guard_reason=guard_reason,
            ba_residual=ba_residual,
            mask_min_ratio=mask_stats.min_ratio if mask_stats is not None else 1.0,
            mask_mean_ratio=mask_stats.mean_ratio if mask_stats is not None else 1.0,
            pose_guard_rejected=getattr(pose_stats, "rejected", False),
            quality_weight=quality_weight,
        )
        sparse_ba_applied = False
        apply_sparse_ba = False
        if bool(self.sparse_ba_cfg.get("enabled", False)):
            apply_sparse_ba = kind == "pi3x" or bool(self.sparse_ba_cfg.get("apply_to_good_chunks", False))

        if apply_sparse_ba:
            refined = self._refine_chunk_sparse_ba(chunk_data, init_stream)
            chunk_data = refined
            sparse_ba_applied = bool(refined.get("sparse_ba_refined", False))
        stats.sparse_ba_applied = sparse_ba_applied
        result = ChunkResult(
            index=chunk_idx,
            start=0,
            end=len(output_stream),
            kind=kind,
            data=chunk_data,
            quality_weight=quality_weight,
        )
        return result, stats

    def _process_chunk_global(
        self,
        chunk_idx: int,
        chunk_stream: VideoStream,
        global_slam_output: SLAMOutput,
        rig: SE3 | None,
        force_bad: bool = False,
    ) -> tuple[ChunkResult, ChunkStats]:
        chunk_name = f"{chunk_stream.name()}"
        init_stream = chunk_stream

        mask_stats = self._compute_mask_coverage(init_stream, chunk_name)
        pose_stats = self._evaluate_pose_guard(init_stream, chunk_name)
        skip_slam, guard_reason = self._guard_precheck(mask_stats, pose_stats)
        if force_bad:
            skip_slam = True
            guard_reason = "forced_bad"

        kind = "slam"
        ba_residual = None

        if skip_slam:
            kind = "pi3x"
            if self.post_bad_cfg is not None:
                dummy_output = self._build_chunk_slam_output(init_stream, rig)
                output_stream = self._build_post_processors(
                    0,
                    init_stream,
                    dummy_output,
                    self.post_bad_cfg,
                    assign_pose=False,
                ).cache("bad_depth", online=True)
                slam_output = self._build_chunk_slam_output(output_stream, rig)
            else:
                output_stream = self._build_bad_chunk_stream(init_stream).cache("pi3x_depth", online=True)
                slam_output = self._build_chunk_slam_output(output_stream, rig)
        else:
            output_stream = self._build_post_processors(
                0,
                init_stream,
                global_slam_output,
                self.post_cfg,
                assign_pose=False,
            ).cache("global_depth", online=True)
            slam_output = global_slam_output

        chunk_data = self._collect_chunk_data(
            output_stream,
            slam_output,
            world_points_source=self.world_points_source,
        )

        quality_weight = self._compute_chunk_weight(kind)
        stats = ChunkStats(
            index=chunk_idx,
            start=0,
            end=len(output_stream),
            kind=kind,
            guard_reason=guard_reason,
            ba_residual=ba_residual,
            mask_min_ratio=mask_stats.min_ratio if mask_stats is not None else 1.0,
            mask_mean_ratio=mask_stats.mean_ratio if mask_stats is not None else 1.0,
            pose_guard_rejected=getattr(pose_stats, "rejected", False),
            quality_weight=quality_weight,
        )

        sparse_ba_applied = False
        apply_sparse_ba = False
        if bool(self.sparse_ba_cfg.get("enabled", False)):
            apply_sparse_ba = kind == "pi3x" or bool(self.sparse_ba_cfg.get("apply_to_good_chunks", False))
        if apply_sparse_ba:
            refined = self._refine_chunk_sparse_ba(chunk_data, init_stream)
            chunk_data = refined
            sparse_ba_applied = bool(refined.get("sparse_ba_refined", False))
        stats.sparse_ba_applied = sparse_ba_applied

        result = ChunkResult(
            index=chunk_idx,
            start=0,
            end=len(output_stream),
            kind=kind,
            data=chunk_data,
            quality_weight=quality_weight,
        )
        return result, stats

    def _run_chunk_pass(
        self,
        base_stream: VideoStream,
        chunk_indices: list[tuple[int, int]],
        forced_bad: set[int],
        slam_rig: SE3 | None,
        use_init_stream: bool,
    ) -> tuple[list[Path], list[ChunkResult], list[ChunkStats]]:
        chunk_paths: list[Path] = []
        chunk_results: list[ChunkResult] = []
        chunk_stats: list[ChunkStats] = []
        for chunk_idx, (start, end) in enumerate(chunk_indices):
            logger.info("Processing chunk %d: frames %d to %d", chunk_idx, start, end)
            chunk_stream = SlicedVideoStream(base_stream, start, end)
            result, stats = self._process_chunk(
                chunk_idx,
                chunk_stream,
                slam_rig,
                force_bad=chunk_idx in forced_bad,
                use_init_stream=use_init_stream,
            )
            stats.start = start
            stats.end = end
            chunk_results.append(result)
            chunk_stats.append(stats)
            chunk_paths.append(self._save_chunk_data(chunk_idx, result.data))
        return chunk_paths, chunk_results, chunk_stats

    def _run_chunk_pass_global(
        self,
        base_stream: VideoStream,
        chunk_indices: list[tuple[int, int]],
        forced_bad: set[int],
        slam_rig: SE3 | None,
        global_slam_output: SLAMOutput,
    ) -> tuple[list[Path], list[ChunkResult], list[ChunkStats]]:
        chunk_paths: list[Path] = []
        chunk_results: list[ChunkResult] = []
        chunk_stats: list[ChunkStats] = []
        for chunk_idx, (start, end) in enumerate(chunk_indices):
            logger.info("Processing chunk %d: frames %d to %d", chunk_idx, start, end)
            chunk_stream = SlicedVideoStream(base_stream, start, end)
            result, stats = self._process_chunk_global(
                chunk_idx,
                chunk_stream,
                global_slam_output,
                slam_rig,
                force_bad=chunk_idx in forced_bad,
            )
            stats.start = start
            stats.end = end
            chunk_results.append(result)
            chunk_stats.append(stats)
            chunk_paths.append(self._save_chunk_data(chunk_idx, result.data))
        return chunk_paths, chunk_results, chunk_stats

    def _save_chunk_data(self, chunk_idx: int, chunk_data: dict[str, Any]) -> Path:
        path = self.result_unaligned_dir / f"chunk_{chunk_idx}.npy"
        np.save(path, chunk_data)
        return path

    def _load_chunk_data(self, path: Path) -> dict[str, Any]:
        return np.load(path, allow_pickle=True).item()

    def _align_chunks(
        self,
        chunk_indices: list[tuple[int, int]],
        chunk_paths: list[Path],
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        sim3_list: list[tuple[float, np.ndarray, np.ndarray]] = []
        for chunk_idx in range(len(chunk_indices) - 1):
            data1 = self._load_chunk_data(chunk_paths[chunk_idx])
            data2 = self._load_chunk_data(chunk_paths[chunk_idx + 1])

            n1 = data1["extrinsic"].shape[0]
            n2 = data2["extrinsic"].shape[0]
            overlap = min(self.overlap, n1, n2)
            if overlap <= 0:
                sim3_list.append((1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
                continue

            pts1, conf1, mask1 = self._get_alignment_maps(data1, n1 - overlap, n1)
            pts2, conf2, mask2 = self._get_alignment_maps(data2, 0, overlap)

            mask = self._maybe_build_align_mask(mask1, mask2, pts1)
            conf_threshold = self._compute_conf_threshold(conf1, conf2)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            s, R, t = robust_weighted_align_point_maps(
                torch.from_numpy(pts1).float().to(device),
                torch.from_numpy(conf1).float().to(device),
                torch.from_numpy(pts2).float().to(device),
                torch.from_numpy(conf2).float().to(device),
                torch.from_numpy(mask).to(device) if mask is not None else None,
                conf_threshold,
                delta=self.irls_delta,
                max_iters=self.irls_max_iters,
                tol=self.irls_tol,
                using_sim3=self.using_sim3,
            )
            sim3_list.append((float(s), R.detach().cpu().numpy(), t.detach().cpu().numpy()))
        return sim3_list

    def _accumulate_sim3s(
        self,
        sim3_list: list[tuple[float, np.ndarray, np.ndarray]],
        anchor_idx: int,
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        accum = [(1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))]
        cur_s, cur_R, cur_t = accum[0]
        for s, R, t in sim3_list:
            cur_s, cur_R, cur_t = self._compose_sim3(cur_s, cur_R, cur_t, s, R, t)
            accum.append((cur_s, cur_R, cur_t))

        anchor_idx = max(0, min(anchor_idx, len(accum) - 1))
        s_a, R_a, t_a = accum[anchor_idx]
        inv_s, inv_R, inv_t = self._invert_sim3(s_a, R_a, t_a)
        reanchored = []
        for s, R, t in accum:
            s_new, R_new, t_new = self._compose_sim3(inv_s, inv_R, inv_t, s, R, t)
            reanchored.append((s_new, R_new, t_new))
        return reanchored

    def _blend_weight(self, idx: int, length: int) -> float:
        if self.blend_len <= 0 or length <= 1:
            return 1.0
        blend = min(self.blend_len, length)
        left = min(idx + 1, blend) / blend
        right = min(length - idx, blend) / blend
        return float(min(left, right))

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            if len(video_data) > 1:
                raise ValueError("Chunked robust pipeline currently supports single-view streams only.")
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

        precompute_init = self.precompute_init or self.adaptive_chunking or self.global_slam_enabled
        cached_init_stream: VideoStream | None = None
        if precompute_init:
            cached_init_stream = self._add_init_processors(video_stream).cache("process", online=True)
            total_frames = len(cached_init_stream)
        else:
            total_frames = len(video_stream)

        global_slam_output: SLAMOutput | None = None
        global_pose_stream: VideoStream | None = None
        if self.global_slam_enabled:
            assert cached_init_stream is not None
            slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
            global_slam_output = slam_pipeline.run([cached_init_stream], rig=slam_rig, camera_type=self.camera_type)
            if global_slam_output.per_frame_intrinsics:
                intrinsics_list = list(global_slam_output.intrinsics[:, 0])
            else:
                intrinsics_list = [global_slam_output.intrinsics[0]] * len(cached_init_stream)

            global_pose_stream = ProcessedVideoStream(
                cached_init_stream,
                [
                    AssignAttributesProcessor(
                        {
                            FrameAttribute.POSE: global_slam_output.get_view_trajectory(0),  # type: ignore
                            FrameAttribute.INTRINSICS: intrinsics_list,
                        }
                    )
                ],
            ).cache("global_pose", online=True)
            del slam_pipeline
            torch.cuda.empty_cache()

        prepass_bad_zones: list[tuple[int, int]] = []
        forced_bad: set[int] = set()
        if self.adaptive_chunking:
            assert cached_init_stream is not None
            health_stream = global_pose_stream if global_pose_stream is not None else cached_init_stream
            health = self._prepass_health(health_stream)
            prepass_bad_zones = self._detect_bad_zones(health, total_frames)
            chunk_indices, forced_bad = self._build_adaptive_chunks(total_frames, prepass_bad_zones)
        else:
            chunk_indices = self._make_chunk_indices(total_frames)

        logger.info("Processing %d frames in %d chunks.", total_frames, len(chunk_indices))

        base_stream = cached_init_stream if cached_init_stream is not None else video_stream
        use_init_stream = cached_init_stream is not None

        if self.global_slam_enabled:
            assert global_pose_stream is not None
            assert global_slam_output is not None
            chunk_paths, chunk_results, chunk_stats = self._run_chunk_pass_global(
                global_pose_stream,
                chunk_indices,
                forced_bad,
                slam_rig,
                global_slam_output,
            )
        else:
            chunk_paths, chunk_results, chunk_stats = self._run_chunk_pass(
                base_stream,
                chunk_indices,
                forced_bad,
                slam_rig,
                use_init_stream,
            )

        postpass_cfg = self.chunk_cfg.get("postpass", DictConfig({}))
        postpass_enabled = bool(postpass_cfg.get("enabled", False)) and not self.global_slam_enabled
        postpass_force = bool(postpass_cfg.get("force", False))
        postpass_bad_zones: list[tuple[int, int]] = []
        if postpass_enabled:
            postpass_bad_zones = self._bad_zones_from_chunk_stats(
                chunk_stats,
                total_frames,
                max_ba_residual=float(postpass_cfg.get("max_ba_residual", 0.05)),
                bad_zone_expand=int(postpass_cfg.get("bad_zone_expand", self.overlap)),
                bad_zone_merge_gap=int(postpass_cfg.get("bad_zone_merge_gap", 5)),
                min_bad_len=int(postpass_cfg.get("min_bad_len", 1)),
            )

        if postpass_enabled and (postpass_bad_zones or postpass_force):
            combined_bad_zones = self._merge_ranges(
                prepass_bad_zones + postpass_bad_zones,
                merge_gap=int(postpass_cfg.get("bad_zone_merge_gap", 5)),
            )
            post_chunk_size = int(postpass_cfg.get("chunk_size", self.chunk_size))
            post_overlap = int(postpass_cfg.get("overlap", self.overlap))
            post_min_chunk = int(postpass_cfg.get("min_chunk_size", self.min_chunk_size))
            chunk_indices, forced_bad = self._build_adaptive_chunks_with_params(
                total_frames,
                combined_bad_zones,
                post_chunk_size,
                post_overlap,
                post_min_chunk,
            )

            if self.result_unaligned_dir.exists():
                shutil.rmtree(self.result_unaligned_dir)
            self.result_unaligned_dir.mkdir(exist_ok=True, parents=True)

            logger.info(
                "Postpass enabled. Reprocessing %d frames in %d chunks.",
                total_frames,
                len(chunk_indices),
            )
            chunk_paths, chunk_results, chunk_stats = self._run_chunk_pass(
                base_stream,
                chunk_indices,
                forced_bad,
                slam_rig,
                use_init_stream,
            )

        sim3_list = self._align_chunks(chunk_indices, chunk_paths)

        loop_constraints = []
        if bool(self.loop_cfg.get("enabled", False)):
            loop_file = self.loop_cfg.get("loop_file", None)
            loop_detector = LoopDetector(
                similarity_threshold=self.loop_cfg.get("similarity_threshold", 0.85),
                loop_window=self.loop_cfg.get("loop_window", 200),
                nms_threshold=self.loop_cfg.get("nms_threshold", 25),
                ckpt_path=self.loop_cfg.get("ckpt_path", "./weights/dino_salad.ckpt"),
            )
            if loop_file is not None and Path(loop_file).exists():
                loop_detector.load_from_file(loop_file)
            else:
                loop_detector.detect(video_stream)
            loop_list = loop_detector.loop_list
            half_window = int(self.loop_cfg.get("half_window", 10))
            loop_results = process_loop_list(chunk_indices, loop_list, half_window=half_window)
            max_edges = int(self.loop_cfg.get("max_edges", 100))
            min_valid_points = int(self.loop_cfg.get("min_valid_points", 200))
            for item in loop_results[:max_edges]:
                chunk_idx_a, range_a, chunk_idx_b, range_b = item
                if chunk_idx_a == chunk_idx_b:
                    continue
                data_a = self._load_chunk_data(chunk_paths[chunk_idx_a])
                data_b = self._load_chunk_data(chunk_paths[chunk_idx_b])
                rel_start_a = range_a[0] - chunk_indices[chunk_idx_a][0]
                rel_end_a = range_a[1] - chunk_indices[chunk_idx_a][0]
                rel_start_b = range_b[0] - chunk_indices[chunk_idx_b][0]
                rel_end_b = range_b[1] - chunk_indices[chunk_idx_b][0]
                pts_a, conf_a, mask_a = self._get_alignment_maps(data_a, rel_start_a, rel_end_a)
                pts_b, conf_b, mask_b = self._get_alignment_maps(data_b, rel_start_b, rel_end_b)
                mask = self._maybe_build_align_mask(mask_a, mask_b, pts_a)
                conf_threshold = self._compute_conf_threshold(conf_a, conf_b)
                if pts_a.size == 0 or pts_b.size == 0:
                    continue
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                s, R, t = robust_weighted_align_point_maps(
                    torch.from_numpy(pts_a).float().to(device),
                    torch.from_numpy(conf_a).float().to(device),
                    torch.from_numpy(pts_b).float().to(device),
                    torch.from_numpy(conf_b).float().to(device),
                    torch.from_numpy(mask).to(device) if mask is not None else None,
                    conf_threshold,
                    delta=self.irls_delta,
                    max_iters=self.irls_max_iters,
                    tol=self.irls_tol,
                    using_sim3=self.using_sim3,
                )
                if not torch.isfinite(torch.tensor(s)):
                    continue
                valid_count = np.prod(pts_a.shape[:3]) if pts_a.ndim == 4 else pts_a.shape[0]
                if valid_count < min_valid_points:
                    continue
                loop_constraints.append((chunk_idx_a, chunk_idx_b, (float(s), R, t)))

        optimizer = Sim3LoopOptimizer(max_iterations=self.opt_max_iterations, lambda_init=self.opt_lambda_init)
        sim3_list = optimizer.optimize(sim3_list, loop_constraints)

        anchor_idx = 0
        if self.anchor_first_good:
            for idx, stats in enumerate(chunk_stats):
                if stats.kind == "slam":
                    anchor_idx = idx
                    break

        accum_sim3s = self._accumulate_sim3s(sim3_list, anchor_idx)

        full_traj: list[SE3 | None] = [None] * total_frames
        full_intrinsics: list[np.ndarray | None] = [None] * total_frames
        full_depths_sum: list[np.ndarray | None] = [None] * total_frames
        full_depths_weight: list[float] = [0.0] * total_frames
        full_masks: list[np.ndarray | None] = [None] * total_frames
        full_mask_weight: list[float] = [0.0] * total_frames
        full_instances: list[np.ndarray | None] = [None] * total_frames
        full_instance_weight: list[float] = [0.0] * total_frames
        full_instance_phrases: dict[int, str] = {}

        pose_ref: list[SE3 | None] = [None] * total_frames
        pose_log_sum: list[torch.Tensor | None] = [None] * total_frames
        pose_weight_sum: list[float] = [0.0] * total_frames

        pose_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for chunk_idx, (s, R, t) in enumerate(accum_sim3s):
            data = self._load_chunk_data(chunk_paths[chunk_idx])
            extrinsics = data["extrinsic"]
            intrinsics = data.get("intrinsic")
            depths = data.get("depth")
            masks = data.get("mask")
            instances = data.get("instance")

            R_local = extrinsics[:, :3, :3]
            t_local = extrinsics[:, :3, 3]
            R_new = R[None, ...] @ R_local
            t_new = (s * (R[None, ...] @ t_local[..., None]).squeeze(-1)) + t[None, ...]

            poses_new = np.repeat(np.eye(4, dtype=np.float32)[None], len(extrinsics), axis=0)
            poses_new[:, :3, :3] = R_new
            poses_new[:, :3, 3] = t_new

            if data.get("instance_phrases"):
                full_instance_phrases.update(data["instance_phrases"])

            chunk_start, chunk_end = chunk_indices[chunk_idx]
            chunk_len = chunk_end - chunk_start
            quality = chunk_results[chunk_idx].quality_weight

            for i in range(chunk_len):
                frame_idx = chunk_start + i
                w = self._blend_weight(i, chunk_len) * quality

                pose = se3_matrix_to_se3(
                    torch.from_numpy(poses_new[i]).float().to(pose_device),
                    unbatch=True,
                )
                if pose_ref[frame_idx] is None:
                    pose_ref[frame_idx] = pose
                    pose_log_sum[frame_idx] = torch.zeros(6, device=pose_device)
                rel = (pose_ref[frame_idx].inv() * pose).log()
                if rel.dim() > 1:
                    rel = rel.view(-1, 6)[0]
                pose_log_sum[frame_idx] += rel * w
                pose_weight_sum[frame_idx] += w

                if intrinsics is not None:
                    if full_intrinsics[frame_idx] is None:
                        full_intrinsics[frame_idx] = intrinsics[i]

                if depths is not None:
                    if full_depths_sum[frame_idx] is None:
                        full_depths_sum[frame_idx] = np.zeros_like(depths[i], dtype=np.float32)
                    full_depths_sum[frame_idx] += depths[i] * w
                    full_depths_weight[frame_idx] += w

                if masks is not None:
                    if w >= full_mask_weight[frame_idx]:
                        full_masks[frame_idx] = masks[i]
                        full_mask_weight[frame_idx] = w

                if instances is not None:
                    if w >= full_instance_weight[frame_idx]:
                        full_instances[frame_idx] = instances[i]
                        full_instance_weight[frame_idx] = w

        merged_poses: list[SE3] = []
        for idx in range(total_frames):
            if pose_ref[idx] is None or pose_weight_sum[idx] <= 0:
                merged_poses.append(SE3.Identity(1, device=pose_device))
                continue
            delta = pose_log_sum[idx] / pose_weight_sum[idx]
            merged_poses.append(pose_ref[idx] * SE3.exp(delta))

        merged_intrinsics = [torch.from_numpy(i).float() if i is not None else None for i in full_intrinsics]
        merged_depths = []
        for i in range(total_frames):
            if full_depths_sum[i] is None or full_depths_weight[i] <= 0:
                merged_depths.append(None)
            else:
                merged_depths.append(torch.from_numpy(full_depths_sum[i] / full_depths_weight[i]).float())
        merged_masks = [torch.from_numpy(m).bool() if m is not None else None for m in full_masks]
        merged_instances = [torch.from_numpy(inst).byte() if inst is not None else None for inst in full_instances]
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
        if full_instance_phrases:
            instance_phrases_list = [full_instance_phrases] * total_frames

        post_processors = [AssignAttributesProcessor(stream_attributes)]
        if instance_phrases_list is not None:
            post_processors.append(AssignInstancePhrasesProcessor(instance_phrases_list))

        output_stream = ProcessedVideoStream(video_stream, post_processors).cache("merged", online=True)

        slam_output = SLAMOutput(
            trajectory=SE3(torch.stack([p.data for p in merged_poses], dim=0)),
            intrinsics=torch.stack([i if i is not None else torch.zeros(4) for i in merged_intrinsics], dim=0)[:1],
            rig=slam_rig if slam_rig is not None else SE3.Identity(1),
            slam_map=None,
            ba_residual=float(
                np.mean([c.ba_residual for c in chunk_stats if c.ba_residual is not None])
            )
            if any(c.ba_residual is not None for c in chunk_stats)
            else float("nan"),
        )

        artifact_path = io.ArtifactPath(self.out_path, video_stream.name())
        if self.out_cfg.save_artifacts:
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info("Saving artifacts to %s", artifact_path)
            io.save_artifacts(artifact_path, output_stream)

            info = {
                "pipeline": "chunked_robust",
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "num_chunks": len(chunk_indices),
                "chunk_indices": chunk_indices,
                "chunk_stats": [c.as_dict() for c in chunk_stats],
                "anchor_chunk": anchor_idx,
                "world_points_source": self.world_points_source,
                "adaptive_chunking": self.adaptive_chunking,
                "prepass_bad_zones": prepass_bad_zones,
                "postpass_bad_zones": postpass_bad_zones,
                "postpass_enabled": postpass_enabled,
                "loop_edges": len(loop_constraints),
            }
            with artifact_path.meta_info_path.open("wb") as f:
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

        if self.delete_temp_files and self.result_unaligned_dir.exists():
            shutil.rmtree(self.result_unaligned_dir)

        return annotate_output

