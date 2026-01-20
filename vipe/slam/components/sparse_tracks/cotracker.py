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
from typing import Iterable
from pathlib import Path

import numpy as np
import torch

from vipe.streams.base import VideoStream, VideoFrame

from . import SparseTracks

logger = logging.getLogger(__name__)


class CoTrackerSparseTracks(SparseTracks):
    """
    Offline CoTracker3 integration (sparse tracks for BA).

    Notes:
      - This runs after the first pass (see SLAMSystem.run) and fills observations for all frames.
      - During the first pass, sparse tracks are not used for keyframe selection.
    """

    def __init__(
        self,
        n_views: int,
        model_name: str = "cotracker3_offline",
        grid_size: int = 12,
        visibility_thre: float = 0.5,
        device: str = "cuda",
        online: bool = False,
        step: int = 8,
        chunk_size: int = 256,
        overlap: int = 32,
        valid_mask_only: bool = False,
        min_valid_ratio: float = 0.05,
        min_valid_pixels: int = 1000,
        save_vis: bool = False,
        vis_out_dir: str = "vipe_debug/cotracker",
        vis_stride: int = 5,
        vis_query_frame: int = 0,
        vis_fps: int = 10,
        vis_max_points: int = 2000,
        stitch_tracks: bool = True,
        stitch_max_dist: float = 5.0,
        stitch_min_frames: int = 3,
        save_npz: bool = False,
        npz_out_dir: str = "vipe_debug/cotracker_npz",
    ):
        super().__init__(n_views)
        self.model_name = model_name
        self.grid_size = int(grid_size)
        self.visibility_thre = float(visibility_thre)
        self.device = torch.device(device)
        self.online = bool(online)
        self.step = int(step)
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.valid_mask_only = bool(valid_mask_only)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_valid_pixels = int(min_valid_pixels)
        self.save_vis = bool(save_vis)
        self.vis_out_dir = Path(vis_out_dir).expanduser().resolve()
        self.vis_stride = max(1, int(vis_stride))
        self.vis_query_frame = int(vis_query_frame)
        self.vis_fps = int(vis_fps)
        self.vis_max_points = int(vis_max_points)
        self.stitch_tracks = bool(stitch_tracks)
        self.stitch_max_dist = float(stitch_max_dist)
        self.stitch_min_frames = int(stitch_min_frames)
        self.save_npz = bool(save_npz)
        self.npz_out_dir = Path(npz_out_dir).expanduser().resolve()
        self._initialized = False
        self._precomputed = False
        self._frames: list[list[torch.Tensor]] = [[] for _ in range(n_views)]
        self._masks: list[list[torch.Tensor | None]] = [[] for _ in range(n_views)]
        
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return self.model
        
        # Prefer local import if available, otherwise use torch.hub.
        try:
            import cotracker  # noqa: F401
        except Exception:
            cotracker = None

        if cotracker is not None:
            try:
                from cotracker.models.core import CoTracker  # type: ignore

                model = CoTracker.from_pretrained(self.model_name)
                self.model = model.to(self.device).eval()
                return self.model
            except Exception:
                pass

        logger.info("Loading CoTracker via torch.hub: %s", self.model_name)
        model = torch.hub.load("facebookresearch/co-tracker", self.model_name)
        self.model = model.to(self.device).eval()
        return self.model

    def track_image(self, frame_data_list: list[VideoFrame]) -> None:
        if self._precomputed:
            return
        # Buffer frames for offline tracking. Still append empty observations.
        for view_idx, frame in enumerate(frame_data_list):
            self._frames[view_idx].append(frame.rgb.detach().cpu())
            if self.valid_mask_only and frame.mask is not None:
                # frame.mask: 0 valid, 1 invalid -> valid_mask = ~mask
                valid_mask = (~frame.mask.bool()).detach().cpu()
            else:
                valid_mask = None
            self._masks[view_idx].append(valid_mask)
            self.observations[view_idx].append({})

    def _run_offline_chunk(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # video: (1, T, 3, H, W) float, 0-255
        model = self._load_model()
        with torch.no_grad():
            tracks, visibility = model(video, grid_size=self.grid_size)
        return tracks, visibility

    def _run_online(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Basic online API (if supported): process in overlapping chunks.
        model = self._load_model()
        B, T, _, _, _ = video.shape
        all_tracks = []
        all_vis = []

        with torch.no_grad():
            model(video_chunk=video[:, : self.step * 2], is_first_step=True, grid_size=self.grid_size)
            for ind in range(0, T - self.step, self.step):
                chunk = video[:, ind : ind + self.step * 2]
                tracks, vis = model(video_chunk=chunk)
                all_tracks.append(tracks[:, : self.step])
                all_vis.append(vis[:, : self.step])

        if all_tracks:
            tracks = torch.cat(all_tracks, dim=1)
            vis = torch.cat(all_vis, dim=1)
        else:
            tracks = torch.empty((B, 0, 0, 2), device=self.device)
            vis = torch.empty((B, 0, 0, 1), device=self.device)
        return tracks, vis

    def _save_vis(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        view_idx: int,
        chunk_idx: int,
    ) -> None:
        if not self.save_vis:
            return
        try:
            from cotracker.utils.visualizer import Visualizer  # type: ignore
            self.vis_out_dir.mkdir(parents=True, exist_ok=True)
            vis = Visualizer(
                save_dir=str(self.vis_out_dir),
                fps=self.vis_fps,
                linewidth=2,
                mode="rainbow",
            )
            filename = f"view{view_idx}_chunk{chunk_idx:03d}"
            vis.visualize(
                video=video,
                tracks=tracks,
                visibility=visibility,
                query_frame=self.vis_query_frame,
                filename=filename,
                save_video=True,
            )
            logger.info("Saved CoTracker vis: %s", self.vis_out_dir / f"{filename}.mp4")
        except Exception:
            # Avoid failing precompute if matplotlib/cotracker visualizer is missing or broken.
            self.vis_out_dir.mkdir(parents=True, exist_ok=True)
            marker = self.vis_out_dir / "visualizer_failed.txt"
            marker.write_text("CoTracker Visualizer failed; check logs for details.\n")
            logger.exception("CoTracker Visualizer failed; skip visualization.")
            self._save_vis_fallback(video, tracks, visibility, view_idx, chunk_idx)
            return

    def _save_vis_fallback(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        view_idx: int,
        chunk_idx: int,
    ) -> None:
        """
        Lightweight renderer without matplotlib/CoTracker Visualizer.
        Draws small colored squares for visible tracks and writes an mp4.
        """
        try:
            import imageio.v3 as iio
        except Exception:
            import imageio as iio  # type: ignore

        self.vis_out_dir.mkdir(parents=True, exist_ok=True)
        filename = self.vis_out_dir / f"view{view_idx}_chunk{chunk_idx:03d}_fallback.mp4"

        vid = video[0].detach().cpu().numpy()  # (T, C, H, W)
        vid = np.transpose(vid, (0, 2, 3, 1)).astype(np.uint8)
        trk = tracks[0].detach().cpu().numpy()  # (T, N, 2)
        vis = visibility[0].detach().cpu().numpy()  # (T, N) or (T, N, 1)
        if vis.ndim == 3:
            vis = vis[..., 0]

        T, N, _ = trk.shape
        # Limit points for speed
        max_points = max(0, self.vis_max_points)
        if max_points and N > max_points:
            idx = np.linspace(0, N - 1, max_points, dtype=int)
            trk = trk[:, idx]
            vis = vis[:, idx]
            N = max_points

        def hsv_to_rgb(h, s=0.7, v=0.95):
            i = int(h * 6.0)
            f = h * 6.0 - i
            p = v * (1.0 - s)
            q = v * (1.0 - f * s)
            t = v * (1.0 - (1.0 - f) * s)
            i = i % 6
            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            return (int(r * 255), int(g * 255), int(b * 255))

        colors = [hsv_to_rgb((i * 0.61803398875) % 1.0) for i in range(N)]

        # writer = iio.get_writer(str(filename), fps=self.vis_fps)
        video = []
        radius = 2
        H, W = vid.shape[1:3]
        for t in range(T):
            frame = vid[t].copy()
            for n in range(N):
                if vis[t, n] < self.visibility_thre:
                    continue
                x, y = trk[t, n]
                xi, yi = int(round(x)), int(round(y))
                if xi < 0 or xi >= W or yi < 0 or yi >= H:
                    continue
                c = colors[n]
                x0, x1 = max(0, xi - radius), min(W - 1, xi + radius)
                y0, y1 = max(0, yi - radius), min(H - 1, yi + radius)
                frame[y0 : y1 + 1, x0 : x1 + 1] = c
            video.append(frame)
        iio.imwrite(filename, np.array(video), fps=self.vis_fps)
        logger.info("Saved fallback CoTracker vis: %s", filename)

    def _save_npz(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        view_idx: int,
        chunk_idx: int,
    ) -> None:
        if not self.save_npz:
            return
        self.npz_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.npz_out_dir / f"view{view_idx}_chunk{chunk_idx:03d}.npz"
        np.savez(
            out_path,
            video=video.detach().cpu().numpy(),
            tracks=tracks.detach().cpu().numpy(),
            visibility=visibility.detach().cpu().numpy(),
        )
        logger.info("Saved CoTracker npz: %s", out_path)

    def _stitch_chunk_ids(
        self,
        prev_tracks: np.ndarray,
        prev_vis: np.ndarray,
        prev_ids: np.ndarray,
        curr_tracks: np.ndarray,
        curr_vis: np.ndarray,
    ) -> dict[int, int]:
        """
        Match current chunk track indices to previous chunk global IDs using overlap frames.
        Returns mapping {curr_idx -> global_id} for matched tracks.
        """
        if self.overlap <= 0:
            return {}
        ov = min(self.overlap, prev_tracks.shape[0], curr_tracks.shape[0])
        if ov <= 0:
            return {}

        # Slices for overlap
        p_tracks = prev_tracks[-ov:]  # (ov, P, 2)
        c_tracks = curr_tracks[:ov]  # (ov, C, 2)

        p_vis = prev_vis[-ov:]
        c_vis = curr_vis[:ov]
        if p_vis.ndim == 3:
            p_vis = p_vis[..., 0]
        if c_vis.ndim == 3:
            c_vis = c_vis[..., 0]

        p_mask = p_vis >= self.visibility_thre  # (ov, P)
        c_mask = c_vis >= self.visibility_thre  # (ov, C)

        P = p_tracks.shape[1]
        C = c_tracks.shape[1]

        try:
            from scipy.spatial.distance import cdist
        except ImportError:
            logger.warning("scipy not found, skipping stitching optimization")
            return {}

        total_dist = np.zeros((P, C), dtype=np.float32)
        total_count = np.zeros((P, C), dtype=np.int32)

        for k in range(ov):
            idx_p = np.where(p_mask[k])[0]
            idx_c = np.where(c_mask[k])[0]

            if idx_p.size == 0 or idx_c.size == 0:
                continue

            pts_p = p_tracks[k, idx_p]
            pts_c = c_tracks[k, idx_c]
            dists = cdist(pts_p, pts_c)

            rows, cols = np.ix_(idx_p, idx_c)
            total_dist[rows, cols] += dists
            total_count[rows, cols] += 1

        valid_pairs = total_count >= self.stitch_min_frames
        if not valid_pairs.any():
            return {}

        avg_dist = np.full((P, C), np.inf, dtype=np.float32)
        np.divide(total_dist, total_count, out=avg_dist, where=valid_pairs)

        possible = (avg_dist <= self.stitch_max_dist) & valid_pairs
        rows, cols = np.where(possible)

        if rows.size == 0:
            return {}

        costs = avg_dist[rows, cols]
        sorted_idx = np.argsort(costs)
        rows, cols = rows[sorted_idx], cols[sorted_idx]

        mapping = {}
        used_p = set()
        used_c = set()

        for r, c in zip(rows, cols):
            if r in used_p or c in used_c:
                continue
            mapping[c] = int(prev_ids[r])
            used_p.add(r)
            used_c.add(c)

        return mapping

    def finalize(self, video_streams: list[VideoStream]) -> None:
        if self._initialized:
            return
        self._initialized = True

        for view_idx in range(len(video_streams)):
            frames = self._frames[view_idx]
            if not frames:
                continue
            # CoTracker expects 0-255 float
            rgb = torch.stack(frames, dim=0).permute(0, 3, 1, 2) * 255.0
            video = rgb.unsqueeze(0).to(self.device)

            if self.online:
                tracks, visibility = self._run_online(video)
                if self.save_vis:
                    vis_video = video[:, :: self.vis_stride].contiguous()
                    vis_tracks = tracks[:, :: self.vis_stride].contiguous()
                    vis_visibility = visibility[:, :: self.vis_stride].contiguous()
                    self._save_vis(vis_video, vis_tracks, vis_visibility, view_idx, 0)
                if self.save_npz:
                    self._save_npz(video, tracks, visibility, view_idx, 0)
                chunk_tracks = [(0, tracks, visibility)]
            else:
                if self.chunk_size <= 0 or video.shape[1] <= self.chunk_size:
                    tracks, visibility = self._run_offline_chunk(video)
                    if self.save_vis:
                        vis_video = video[:, :: self.vis_stride].contiguous()
                        vis_tracks = tracks[:, :: self.vis_stride].contiguous()
                        vis_visibility = visibility[:, :: self.vis_stride].contiguous()
                        self._save_vis(vis_video, vis_tracks, vis_visibility, view_idx, 0)
                    if self.save_npz:
                        self._save_npz(video, tracks, visibility, view_idx, 0)
                    chunk_tracks = [(0, tracks, visibility)]
                else:
                    stride = max(1, self.chunk_size - self.overlap)
                    chunk_tracks = []
                    chunk_idx = 0
                    for start in range(0, video.shape[1], stride):
                        end = min(start + self.chunk_size, video.shape[1])
                        chunk = video[:, start:end]
                        tracks, visibility = self._run_offline_chunk(chunk)
                        if self.save_vis:
                            vis_video = chunk[:, :: self.vis_stride].contiguous()
                            vis_tracks = tracks[:, :: self.vis_stride].contiguous()
                            vis_visibility = visibility[:, :: self.vis_stride].contiguous()
                            self._save_vis(vis_video, vis_tracks, vis_visibility, view_idx, chunk_idx)
                        if self.save_npz:
                            self._save_npz(chunk, tracks, visibility, view_idx, chunk_idx)
                        chunk_idx += 1
                        chunk_tracks.append((start, tracks, visibility))
                        # Free memory between chunks
                        del tracks, visibility, chunk
                        torch.cuda.empty_cache()

            next_track_id = 0
            prev_chunk_tracks = None
            prev_chunk_vis = None
            prev_chunk_ids = None
            for start, tracks, visibility in chunk_tracks:
                tracks = tracks[0].detach().cpu().numpy()  # (t, N, 2)
                visibility = visibility[0].detach().cpu().numpy()

                t_len, n_tracks, _ = tracks.shape
                if visibility.ndim == 3 and visibility.shape[-1] == 1:
                    visibility = visibility[..., 0]
                valid_init = None
                if self.valid_mask_only and self._masks[view_idx]:
                    start_valid = self._masks[view_idx][start]
                    if start_valid is not None:
                        valid_pixels = int(start_valid.sum().item())
                        total_pixels = int(start_valid.numel())
                        valid_ratio = valid_pixels / max(total_pixels, 1)
                        if valid_pixels < self.min_valid_pixels or valid_ratio < self.min_valid_ratio:
                            # If mask is too restrictive, skip filtering for this chunk.
                            start_valid = None
                        else:
                            logger.info(
                                "CoTracker valid mask for chunk %d: %d/%d (%.3f)",
                                start,
                                valid_pixels,
                                total_pixels,
                                valid_ratio,
                            )
                    if start_valid is not None:
                        # Compute validity for initial positions (t=0) using the chunk's first frame mask.
                        h0, w0 = start_valid.shape[-2:]
                        init = tracks[0]
                        init_x = np.rint(init[:, 0]).astype(np.int64)
                        init_y = np.rint(init[:, 1]).astype(np.int64)
                        in_bounds = (init_x >= 0) & (init_x < w0) & (init_y >= 0) & (init_y < h0)
                        valid_init = np.zeros((n_tracks,), dtype=bool)
                        valid_idx = np.where(in_bounds)[0]
                        if valid_idx.size > 0:
                            valid_init[valid_idx] = start_valid[init_y[valid_idx], init_x[valid_idx]].numpy()

                # stitch ids across chunks if enabled
                id_map = {}
                if (
                    self.stitch_tracks
                    and prev_chunk_tracks is not None
                    and prev_chunk_vis is not None
                    and prev_chunk_ids is not None
                ):
                    id_map = self._stitch_chunk_ids(
                        prev_chunk_tracks,
                        prev_chunk_vis,
                        prev_chunk_ids,
                        tracks,
                        visibility,
                    )

                curr_ids = np.full((n_tracks,), -1, dtype=np.int64)
                if id_map:
                    mapped_c = np.array(list(id_map.keys()))
                    mapped_ids = np.array(list(id_map.values()))
                    curr_ids[mapped_c] = mapped_ids

                unmapped_mask = curr_ids == -1
                n_new = np.sum(unmapped_mask)
                if n_new > 0:
                    curr_ids[unmapped_mask] = np.arange(next_track_id, next_track_id + n_new)
                    next_track_id += n_new

                # Vectorized observation update
                vis_mask = visibility >= self.visibility_thre
                if visibility.ndim == 3:
                    vis_mask = vis_mask[..., 0]

                if valid_init is not None:
                    vis_mask = vis_mask & valid_init[None, :]

                for t in range(t_len):
                    mask_t = vis_mask[t]
                    if not mask_t.any():
                        continue

                    # Bulk update the observation dictionary
                    frame_idx = start + t
                    self.observations[view_idx][frame_idx].update(
                        zip(curr_ids[mask_t], tracks[t, mask_t])
                    )
                prev_chunk_tracks = tracks
                prev_chunk_vis = visibility
                prev_chunk_ids = curr_ids

        logger.info("CoTracker sparse tracks computed.")

    def precompute(self, video_streams: list[VideoStream]) -> None:
        if self._precomputed:
            return
        # Populate frames via track_image, then finalize and lock observations.
        for frame_data_list in zip(*video_streams):
            self.track_image(list(frame_data_list))
        self.finalize(video_streams)
        self._precomputed = True

