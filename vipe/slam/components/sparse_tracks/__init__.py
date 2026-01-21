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

from abc import ABC, abstractmethod

import numpy as np
import torch

from omegaconf.dictconfig import DictConfig

from vipe.streams.base import VideoFrame, VideoStream
from vipe.utils.depth import bilinear_splatting_inplace


class SparseTracks(ABC):
    """Note that the current design only supports single-camera for now"""

    # view_idx -> frame_idx -> {keypoint_idx -> uv}
    observations: list[list[dict[int, np.ndarray]]]
    enabled: bool = True

    def __init__(self, n_views: int):
        self.observations = [[] for _ in range(n_views)]

    @abstractmethod
    def track_image(self, frame_data_list: list[VideoFrame]) -> None: ...

    # Optional hook: run an offline tracker after all frames are available.
    def finalize(self, video_streams: list[VideoStream]) -> None:
        return None

    # Optional hook: precompute tracks before SLAM starts.
    def precompute(self, video_streams: list[VideoStream]) -> None:
        return None

    def get_correspondences(self, view_idx: int, source_frame_idx: int, target_frame_idx: int) -> torch.Tensor:
        """
        Returns:
            - keypoint_indices: The indices of the keypoints that are observed in both frames.
        """
        source_kps = set(self.observations[view_idx][source_frame_idx].keys())
        target_kps = set(self.observations[view_idx][target_frame_idx].keys())
        keypoint_indices = list(source_kps.intersection(target_kps))
        return torch.tensor(keypoint_indices)

    def get_observations(self, view_idx: int, frame_idx: int, keypoint_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_idx: The index of the view.
            frame_idx: The index of the frame.
            keypoint_indices: The indices of the keypoints to get the observations for.
        Returns:
            The observations for the given keypoints.
        """
        if len(keypoint_indices) == 0:
            return torch.empty(0, 2, device=keypoint_indices.device)
        uvs = self.observations[view_idx][frame_idx]
        return (
            torch.tensor(np.stack([uvs[kp_idx] for kp_idx in keypoint_indices.cpu().numpy()], axis=0))
            .to(keypoint_indices.device)
            .float()
        )

    def get_overlapping_pairs(
        self, min_common: int = 15, min_frame_gap: int = 50, max_pairs: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find pairs of frames that share at least `min_common` tracks.
        Only returns pairs where the frame gap is at least `min_frame_gap`.
        Returns at most `max_pairs` pairs, prioritizing those with the most common tracks.
        Returns (ii, jj) tensors of frame indices.
        """
        pairs_list = []

        # Invert the observations: track_id -> set of frame_indices
        for view_idx in range(len(self.observations)):
            track_to_frames: dict[int, list[int]] = {}

            # self.observations[view_idx] is a list of dicts (frame -> {track_id: uv})
            for frame_idx, obs in enumerate(self.observations[view_idx]):
                for track_id in obs.keys():
                    if track_id not in track_to_frames:
                        track_to_frames[track_id] = []
                    track_to_frames[track_id].append(frame_idx)

            # Count shared tracks for each pair
            pair_counts: dict[tuple[int, int], int] = {}

            for frames in track_to_frames.values():
                if len(frames) < 2:
                    continue
                # All pairs in this list share this track
                for i in range(len(frames)):
                    for j in range(i + 1, len(frames)):
                        f1, f2 = frames[i], frames[j]
                        if f1 > f2:
                            f1, f2 = f2, f1
                        # Only consider pairs with sufficient frame gap
                        if f2 - f1 < min_frame_gap:
                            continue
                        pair = (f1, f2)
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1

            for pair, count in pair_counts.items():
                if count >= min_common:
                    pairs_list.append((count, pair[0], pair[1]))

        if not pairs_list:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        # Sort by count descending and take top max_pairs
        pairs_list.sort(key=lambda x: -x[0])
        pairs_list = pairs_list[:max_pairs]

        # Remove duplicates (in case multiple views found the same pair)
        seen = set()
        unique_pairs = []
        for _, i, j in pairs_list:
            if (i, j) not in seen:
                seen.add((i, j))
                unique_pairs.append((i, j))

        if not unique_pairs:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        ii = torch.tensor([p[0] for p in unique_pairs], dtype=torch.long)
        jj = torch.tensor([p[1] for p in unique_pairs], dtype=torch.long)
        return ii, jj

    def compute_dense_disp_target_weight(
        self,
        source_view_inds: torch.Tensor,
        source_frame_inds: torch.Tensor,
        target_view_inds: torch.Tensor,
        target_frame_inds: torch.Tensor,
        image_size: tuple[int, int],
        dense_disp_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_terms = len(source_view_inds)
        disp_h, disp_w = dense_disp_size
        uv_factor = torch.tensor(
            [disp_w / image_size[1], disp_h / image_size[0]],
            device=source_view_inds.device,
        )
        assert n_terms == len(target_view_inds) == len(source_frame_inds) == len(target_frame_inds), (
            "All indices must have the same length"
        )

        disp_value = torch.zeros(
            (n_terms, disp_h, disp_w, 2),
            dtype=torch.float32,
            device=source_view_inds.device,
        )
        disp_weight = torch.zeros(
            (n_terms, disp_h, disp_w),
            dtype=torch.float32,
            device=source_view_inds.device,
        )
        for term_idx, (s_vidx, s_fidx, t_vidx, t_fidx) in enumerate(
            zip(
                source_view_inds.cpu().numpy(),
                source_frame_inds.cpu().numpy(),
                target_view_inds.cpu().numpy(),
                target_frame_inds.cpu().numpy(),
            )
        ):
            assert s_vidx == t_vidx, "Only same view tracking is supported"
            kp_idx = self.get_correspondences(s_vidx, s_fidx, t_fidx)
            if len(kp_idx) == 0:
                continue
            uv_source = self.get_observations(s_vidx, s_fidx, kp_idx)
            uv_flow = self.get_observations(t_vidx, t_fidx, kp_idx) - uv_source

            bilinear_splatting_inplace(
                uv_flow.cuda() * uv_factor,
                uv_source.cuda() * uv_factor,
                disp_value[term_idx],
                disp_weight[term_idx],
            )

        disp_value /= disp_weight[..., None]
        disp_weight = disp_weight[..., None].repeat(1, 1, 1, 2)
        # If weight is 0, set to 0/whatever values since we don't care those positions.
        disp_value[torch.isnan(disp_value)] = 0.0
        # If weight is too small, then probably it's also not very reliable.
        disp_value[disp_weight < 0.1] = 0.0
        disp_weight[disp_weight < 0.1] = 0.0

        # We need to add original coordinates with the flow, so it becomes "target"
        y, x = torch.meshgrid(
            torch.arange(disp_h, device=disp_value.device),
            torch.arange(disp_w, device=disp_value.device),
            indexing="ij",
        )
        disp_value[..., 0] += x
        disp_value[..., 1] += y

        return disp_value, disp_weight


class DummySparseTracks(SparseTracks):
    enabled: bool = False

    def track_image(self, frame_data_list: list[VideoFrame]) -> None:
        for obs in self.observations:
            obs.append({})


def build_sparse_tracks(config: DictConfig, n_views: int) -> SparseTracks:
    if config.name == "dummy":
        return DummySparseTracks(n_views)

    if config.name == "cuvslam":
        from .cuvslam import CuVSLAMSparseTracks

        return CuVSLAMSparseTracks(n_views)

    if config.name == "cotracker3":
        from .cotracker import CoTrackerSparseTracks

        return CoTrackerSparseTracks(
            n_views,
            model_name=config.get("model_name", "cotracker3_offline"),
            grid_size=config.get("grid_size", 12),
            visibility_thre=config.get("visibility_thre", 0.5),
            device=config.get("device", "cuda"),
            online=config.get("online", False),
            step=config.get("step", 8),
            chunk_size=config.get("chunk_size", 256),
            overlap=config.get("overlap", 32),
            valid_mask_only=config.get("valid_mask_only", False),
            min_valid_ratio=config.get("min_valid_ratio", 0.05),
            min_valid_pixels=config.get("min_valid_pixels", 1000),
            save_vis=config.get("save_vis", False),
            vis_out_dir=config.get("vis_out_dir", "vipe_debug/cotracker"),
            vis_stride=config.get("vis_stride", 5),
            vis_query_frame=config.get("vis_query_frame", 0),
            vis_fps=config.get("vis_fps", 10),
            vis_max_points=config.get("vis_max_points", 2000),
            stitch_tracks=config.get("stitch_tracks", True),
            stitch_max_dist=config.get("stitch_max_dist", 5.0),
            stitch_min_frames=config.get("stitch_min_frames", 3),
            save_npz=config.get("save_npz", False),
            npz_out_dir=config.get("npz_out_dir", "vipe_debug/cotracker_npz"),
        )

    if config.name == "cotracker3_masked":
        from .cotracker_masked import CoTrackerSparseTracksMasked

        return CoTrackerSparseTracksMasked(
            n_views,
            model_name=config.get("model_name", "cotracker3_offline"),
            grid_size=config.get("grid_size", 12),
            visibility_thre=config.get("visibility_thre", 0.5),
            device=config.get("device", "cuda"),
            online=config.get("online", False),
            step=config.get("step", 8),
            chunk_size=config.get("chunk_size", 256),
            overlap=config.get("overlap", 32),
            valid_mask_only=config.get("valid_mask_only", False),
            min_valid_ratio=config.get("min_valid_ratio", 0.05),
            min_valid_pixels=config.get("min_valid_pixels", 1000),
            save_vis=config.get("save_vis", False),
            vis_out_dir=config.get("vis_out_dir", "vipe_debug/cotracker"),
            vis_stride=config.get("vis_stride", 5),
            vis_query_frame=config.get("vis_query_frame", 0),
            vis_fps=config.get("vis_fps", 10),
            vis_max_points=config.get("vis_max_points", 2000),
            stitch_tracks=config.get("stitch_tracks", True),
            stitch_max_dist=config.get("stitch_max_dist", 5.0),
            stitch_min_frames=config.get("stitch_min_frames", 3),
            save_npz=config.get("save_npz", False),
            npz_out_dir=config.get("npz_out_dir", "vipe_debug/cotracker_npz"),
        )

    raise ValueError(f"Unknown sparse tracks: {config.name}")
