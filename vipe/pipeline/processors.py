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


import logging
import math
from typing import Any, Iterable, Iterator, cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from vipe.ext.lietorch import SE3, SO3
from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_inv_depth_to_depth
from vipe.priors.depth.moge_v2 import focal_length_to_fov_degrees
from vipe.priors.depth.pi3x_moge import Pi3XMoGeV2Model, mask_aware_nearest_resize_robust
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.depth import get_camera_rays
from vipe.utils.geometry import project_points_to_panorama
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.model_cache import ModelCache
from vipe.utils.morph import erode

logger = logging.getLogger(__name__)

try:
    from moge.utils.alignment import align_points_scale_z_shift
except ModuleNotFoundError:
    align_points_scale_z_shift = None


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
        model_cache: ModelCache | None = None,
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        # GeoCalib is used purely for inference; when a cache is provided the
        # weights are loaded once and reused across streams instead of per video.
        def _build_geocalib():
            return GeoCalib(weights=weights).cuda()

        if model_cache is not None:
            model = model_cache.get(f"geocalib/{weights}", _build_geocalib)
        else:
            model = _build_geocalib()
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        camera_result = cast(Any, res["camera"])
        self.fov_y = camera_result.vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [camera_result.dist[0, 0].item()]


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
        model_cache: ModelCache | None = None,
    ) -> None:
        # Defensive copy: prevent mutation of caller's list
        self.mask_phrases = list(mask_phrases)
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(
            self.mask_phrases,
            sam_points_per_side=50,
            sam_run_gap=self.sam_run_gap,
            model_cache=model_cache,
        )
        self.mask_expand = mask_expand

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.instance, frame.instance_phrases = self.tracker.track(frame)
        self.last_track_frame = frame.raw_frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda"]
            self.video_depth_model: VideoDepthAnythingDepthModel | None = VideoDepthAnythingDepthModel(
                model="vits" if video_model == "svda" else "vitl"
            )

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        video_depth_model = unpack_optional(self.video_depth_model)
        video_depth_result: torch.Tensor = unpack_optional(
            video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias: tuple[torch.Tensor, torch.Tensor] | None = None
        min_uv_score: float = 1.0
        slam_map = unpack_optional(self.slam_output.slam_map)
        data_iterator: Iterable[VideoFrame]

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"

            if video_depth_result is not None:
                video_depth_inv_depth = video_depth_result[frame_idx]

                align_mask = video_depth_inv_depth > 1e-3
                if frame.mask is not None:
                    align_mask = align_mask & frame.mask & (~frame.sky_mask)

                try:
                    _, scale_tensor, bias_tensor = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    if self.cache_scale_bias is None:
                        raise
                    scale_tensor, bias_tensor = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale_tensor, bias_tensor)
                scale_tensor = self.cache_scale_bias[0] * self.update_momentum + scale_tensor * (
                    1 - self.update_momentum
                )
                bias_tensor = self.cache_scale_bias[1] * self.update_momentum + bias_tensor * (1 - self.update_momentum)
                self.cache_scale_bias = (scale_tensor, bias_tensor)

                video_inv_depth = video_depth_inv_depth * scale_tensor + bias_tensor
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame


class Pi3XMoGePerFrameProcessor(StreamProcessor):
    """Post depth using Pi3X video geometry aligned to MoGe2 or the SLAM map."""

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        window_size: int = 64,
        overlap_size: int = 16,
        pixel_limit: int = 255000,
        align_lr_size: int = 64,
        min_align_points: int = 200,
        align_mode: str = "window_shared_ema",
        align_momentum: float = 0.99,
        scale_clamp: tuple[float, float] = (0.1, 10.0),
        shift_z_clamp: tuple[float, float] = (-1e3, 1e3),
        moge_bs: int = 4,
        align_source: str = "slam_map",
        max_window_align_points: int = 2000,
        max_frame_align_points: int = 2000,
    ) -> None:
        super().__init__()
        if align_points_scale_z_shift is None:
            raise RuntimeError("MoGe alignment utilities are required for Pi3XMoGePerFrameProcessor.")
        if align_source not in {"moge2", "slam_map"}:
            raise ValueError(f"Unsupported Pi3X/MoGe2 alignment source: {align_source}")

        self.slam_output = slam_output
        self.view_idx = view_idx
        self.window_size = max(1, min(int(window_size), 180))
        self.overlap_size = max(0, min(int(overlap_size), self.window_size - 1))
        self.pixel_limit = pixel_limit
        self.align_lr_size = align_lr_size
        self.min_align_points = min_align_points
        self.align_mode = align_mode
        self.align_momentum = align_momentum
        self.scale_clamp = tuple(scale_clamp)
        self.shift_z_clamp = tuple(shift_z_clamp)
        self.moge_bs = max(1, int(moge_bs))
        self.align_source = align_source
        self.max_window_align_points = max(0, int(max_window_align_points))
        self.max_frame_align_points = max(0, int(max_frame_align_points))
        self._cache_scale: torch.Tensor | None = None

        self.model = Pi3XMoGeV2Model(pixel_limit=pixel_limit)
        self.pi3x = self.model.pi3x
        self.moge = self.model.moge
        self.n_passes_required = 1

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("Pi3XMoGePerFrameProcessor should not be called directly.")

    def _effective_align_size(self, base_size: int, max_points: int, multiplier: int = 1) -> int:
        if max_points <= 0:
            return base_size
        max_points = max(1, int(max_points // max(1, multiplier)))
        return min(base_size, max(4, int(math.sqrt(max_points))))

    def _prepare_inputs(
        self,
        frames: list[VideoFrame],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
        orig_h, orig_w = frames[0].size()
        target_h, target_w = self.model._get_resize_size(orig_h, orig_w)
        target_h = max(14, ((target_h + 13) // 14) * 14)
        target_w = max(14, ((target_w + 13) // 14) * 14)
        while target_h * target_w > self.pixel_limit and (target_h > 14 or target_w > 14):
            if target_h > target_w:
                target_h -= 14
            else:
                target_w -= 14
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        imgs: list[torch.Tensor] = []
        poses: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        for frame in frames:
            image = frame.rgb.cuda().permute(2, 0, 1).unsqueeze(0)
            image = F.interpolate(
                image,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            imgs.append(image)

            poses.append(unpack_optional(frame.pose).matrix().detach().cpu())
            fx, fy, cx, cy = unpack_optional(frame.intrinsics)[:4]
            K = torch.eye(3)
            K[0, 0] = fx.cpu() * scale_x
            K[1, 1] = fy.cpu() * scale_y
            K[0, 2] = cx.cpu() * scale_x
            K[1, 2] = cy.cpu() * scale_y
            intrinsics.append(K)

        return (
            torch.cat(imgs, dim=0).unsqueeze(0),
            torch.stack(poses).cuda().unsqueeze(0),
            torch.stack(intrinsics).cuda().unsqueeze(0),
            target_h,
            target_w,
            orig_h,
            orig_w,
        )

    def _depth_to_points(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = depth.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        z = depth
        x = (xx - cx) / fx * z
        y = (yy - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)

    def _resize_mask(
        self, frame: VideoFrame, target_size: tuple[int, int], device: torch.device
    ) -> torch.Tensor | None:
        if frame.mask is None:
            return None
        mask = frame.mask
        if mask.dim() != 2:
            mask = mask.squeeze()
        if mask.shape != target_size:
            mask = F.interpolate(mask.float()[None, None].to(device), size=target_size, mode="nearest")[0, 0].bool()
        else:
            mask = mask.to(device).bool()
        return mask

    def _slam_depth_for_frame(
        self,
        frame: VideoFrame,
        frame_idx: int,
        intrinsics: torch.Tensor,
        target_size: tuple[int, int],
        slam_device: torch.device,
    ) -> torch.Tensor:
        slam_map = unpack_optional(self.slam_output.slam_map)
        pose = unpack_optional(frame.pose).to(slam_device)
        K = intrinsics.to(slam_device)
        intr = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        return slam_map.project_map(
            frame_tstamp=frame_idx,
            view_idx=self.view_idx,
            target_size=target_size,
            target_intrinsics=intr,
            target_pose=pose,
            target_camera_type=unpack_optional(frame.camera_type),
            infill=False,
        )

    def _solve_scale(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        try:
            scale, shift = align_points_scale_z_shift(
                source_points.unsqueeze(0),
                target_points.unsqueeze(0),
                weights.unsqueeze(0),
            )
        except Exception:
            return None, None

        scale_i = scale[0].clamp(self.scale_clamp[0], self.scale_clamp[1])
        shiftz_i = shift[0, 2].clamp(self.shift_z_clamp[0], self.shift_z_clamp[1])
        if torch.isfinite(scale_i).all() and torch.isfinite(shiftz_i).all() and scale_i.item() > 0:
            return scale_i, shiftz_i
        return None, None

    @torch.no_grad()
    def estimate_depth(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        all_frames = [frame.cpu() for frame in previous_iterator]
        if not all_frames:
            return
        if self.align_source == "slam_map":
            if self.slam_output.slam_map is None:
                raise ValueError("align_source=slam_map requires slam_output.slam_map")
            for frame in all_frames:
                if frame.camera_type != CameraType.PINHOLE:
                    raise ValueError("align_source=slam_map currently supports only pinhole cameras")
            slam_device = self.slam_output.slam_map.dense_disp_xyz.device
        else:
            slam_device = torch.device("cuda")

        current_window: list[VideoFrame] = []
        current_indices: list[int] = []
        trailing_depth: torch.Tensor | None = None

        for frame_idx, frame in pbar(enumerate(all_frames), total=len(all_frames), desc="Pi3X+MoGe2 depth"):
            current_window.append(frame)
            current_indices.append(frame_idx)
            is_last_frame = frame_idx == len(all_frames) - 1
            if len(current_window) < self.window_size and not is_last_frame:
                continue

            imgs, _poses, intrinsics, target_h, target_w, orig_h, orig_w = self._prepare_inputs(current_window)

            pi3x_out = self.pi3x(imgs)
            pi3x_points = pi3x_out["local_points"][0][: len(current_window)]
            pi3x_conf = torch.sigmoid(pi3x_out["conf"][0, : len(current_window), ..., 0]) > 0.1

            moge_points = None
            moge_mask = None
            moge_depth: torch.Tensor | None = None
            if self.align_source == "moge2":
                imgs_moge = imgs[0]
                n_window = imgs_moge.shape[0]
                moge_points = torch.empty((n_window, target_h, target_w, 3), device=imgs_moge.device)
                moge_mask = torch.empty((n_window, target_h, target_w), device=imgs_moge.device, dtype=torch.bool)
                fov_x = focal_length_to_fov_degrees(unpack_optional(current_window[0].intrinsics)[0].item(), orig_w)
                for start in range(0, n_window, self.moge_bs):
                    out = self.moge.forward(imgs_moge[start : start + self.moge_bs], fov_x=fov_x)
                    points = out["points"]
                    moge_points[start : start + points.shape[0]] = points
                    mask = out.get("mask", torch.ones_like(points[..., 0])).bool()
                    if mask.dim() == 4:
                        mask = mask.squeeze(1)
                    moge_mask[start : start + mask.shape[0]] = mask
                    if "depth" in out:
                        if moge_depth is None:
                            moge_depth = torch.empty(
                                (n_window, target_h, target_w),
                                device=imgs_moge.device,
                                dtype=out["depth"].dtype,
                            )
                        moge_depth[start : start + out["depth"].shape[0]] = out["depth"]

            window_scale: torch.Tensor | None = None
            if self.align_mode in ("window_shared", "window_shared_ema"):
                align_size = self._effective_align_size(
                    self.align_lr_size, self.max_window_align_points, len(current_window)
                )
                src_parts: list[torch.Tensor] = []
                tgt_parts: list[torch.Tensor] = []
                weight_parts: list[torch.Tensor] = []
                for i, window_frame in enumerate(current_window):
                    if self.align_source == "slam_map":
                        target_depth = self._slam_depth_for_frame(
                            window_frame,
                            current_indices[i],
                            intrinsics[0, i],
                            (target_h, target_w),
                            slam_device,
                        ).to(pi3x_points.device)
                        target_mask = target_depth > 0
                        resized_mask = self._resize_mask(window_frame, (target_h, target_w), target_depth.device)
                        if resized_mask is not None:
                            target_mask = target_mask & resized_mask
                        combined_mask = pi3x_conf[i] & target_mask
                        if combined_mask.sum().item() < self.min_align_points:
                            continue
                        indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, align_size, align_size)
                        ni, nj = indices
                        if lr_mask.sum().item() < 10:
                            continue
                        src_parts.append(pi3x_points[i][ni, nj][lr_mask])
                        tgt_parts.append(self._depth_to_points(target_depth, intrinsics[0, i])[ni, nj][lr_mask])
                        weight_parts.append(1.0 / target_depth[ni, nj][lr_mask].clamp(min=1e-3))
                    else:
                        assert moge_points is not None and moge_mask is not None
                        combined_mask = pi3x_conf[i] & moge_mask[i]
                        if combined_mask.sum().item() < self.min_align_points:
                            continue
                        indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, align_size, align_size)
                        ni, nj = indices
                        if lr_mask.sum().item() < 10:
                            continue
                        target_points = moge_points[i][ni, nj]
                        src_parts.append(pi3x_points[i][ni, nj][lr_mask])
                        tgt_parts.append(target_points[lr_mask])
                        weight_parts.append(1.0 / target_points[..., 2][lr_mask].clamp(min=1e-3))

                if src_parts:
                    window_scale, _ = self._solve_scale(
                        torch.cat(src_parts, dim=0),
                        torch.cat(tgt_parts, dim=0),
                        torch.cat(weight_parts, dim=0),
                    )

            window_depth = torch.zeros(
                (len(current_window), orig_h, orig_w),
                device=pi3x_points.device,
                dtype=pi3x_points.dtype,
            )
            for i, window_frame in enumerate(current_window):
                if self.align_source == "slam_map":
                    target_depth = self._slam_depth_for_frame(
                        window_frame,
                        current_indices[i],
                        intrinsics[0, i],
                        (target_h, target_w),
                        slam_device,
                    ).to(pi3x_points.device)
                    target_mask = target_depth > 0
                    resized_mask = self._resize_mask(window_frame, (target_h, target_w), target_depth.device)
                    if resized_mask is not None:
                        target_mask = target_mask & resized_mask
                    combined_mask = pi3x_conf[i] & target_mask
                else:
                    assert moge_points is not None and moge_mask is not None
                    combined_mask = pi3x_conf[i] & moge_mask[i]

                scale_i = window_scale
                if scale_i is None and combined_mask.sum().item() >= self.min_align_points:
                    align_size = self._effective_align_size(self.align_lr_size, self.max_frame_align_points, 1)
                    indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, align_size, align_size)
                    ni, nj = indices
                    if lr_mask.sum().item() >= 10:
                        src_points = pi3x_points[i][ni, nj]
                        if self.align_source == "slam_map":
                            target_points = self._depth_to_points(target_depth, intrinsics[0, i])[ni, nj]
                            weights = 1.0 / target_depth[ni, nj].clamp(min=1e-3)
                        else:
                            assert moge_points is not None
                            target_points = moge_points[i][ni, nj]
                            weights = 1.0 / target_points[..., 2].clamp(min=1e-3)
                        scale_i, _ = self._solve_scale(src_points[lr_mask], target_points[lr_mask], weights[lr_mask])

                if self.align_mode in ("per_frame_ema", "window_shared_ema"):
                    if scale_i is None:
                        scale_i = self._cache_scale
                    elif self._cache_scale is None:
                        self._cache_scale = scale_i
                    else:
                        scale_i = self._cache_scale * self.align_momentum + scale_i * (1 - self.align_momentum)
                        self._cache_scale = scale_i

                if scale_i is None:
                    if self.align_source == "moge2" and moge_depth is not None:
                        depth = torch.nan_to_num(moge_depth[i], nan=0.0).clamp(min=0.0, max=1e4)
                        depth = depth * pi3x_conf[i].float()
                    else:
                        depth = pi3x_points[i, ..., 2].clamp(min=0.0) * pi3x_conf[i].float()
                else:
                    depth = (pi3x_points[i, ..., 2] * scale_i).clamp(min=0.0) * pi3x_conf[i].float()

                window_depth[i] = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="nearest",
                )[0, 0]

            n_yield = self.window_size - self.overlap_size if not is_last_frame else len(current_window)
            if trailing_depth is not None:
                n_interp = len(trailing_depth)
                alpha = torch.linspace(0, 1, n_interp + 2, device=window_depth.device)[1:-1, None, None]
                window_depth[:n_interp] = trailing_depth * (1 - alpha) + window_depth[:n_interp] * alpha

            for i in range(n_yield):
                current_window[i].metric_depth = window_depth[i].cpu()

            trailing_depth = window_depth[n_yield:].detach()
            current_window = current_window[n_yield:]
            current_indices = current_indices[n_yield:]

        for frame in all_frames:
            yield frame.cuda()

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx != 0:
            raise ValueError(f"Invalid pass index: {pass_idx}")
        yield from self.estimate_depth(previous_iterator)


class MultiviewDepthProcessor(StreamProcessor):
    """
    Use multi-view depth model (e.g. DAv3, MapAnything, CAPA) to estimate depth map for each frame.
    To ensure that the depth maps are consistent with the SLAM map/pose (metric), we condition the depth model either with
    (a) sparse points, or (b) camera poses & intrinsics.

    Depth is estimated in a sliding-window manner, and overlapped frames are linearly averaged to sharp transitions.
    To create enough parallex to improve estimation confidence, for each window we optionally also include
    neighboring keyframes, and their secondary neighboring keyframes.
    (Multi-view input video frames are currently not supported)
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        model: str = "mvd_dav3",
        window_size: int = 10,  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,  # This is found to cause jittering for some scenes due to abrupt context changes.
    ):
        super().__init__()
        self.slam_output = slam_output
        self.model = model
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.secondary_keyframe = secondary_keyframe

        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[VideoFrame] = []
        self.n_frames = 0

        # Need two passes for this iterator to work.
        self.n_passes_required = 2

        if self.model == "mvd_dav3":
            from vipe.priors.depth.dav3 import DepthAnything3
            from vipe.priors.depth.dav3.utils import logger as dav3_logger

            dav3_logger.level = 0  # Disable logging timing information
            self.dav3_api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT", model_name="da3-giant")
            self.dav3_api = self.dav3_api.cuda().eval()

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("MultiviewDepthProcessor should not be called directly.")

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        # Pick the farthest secondary keyframe from the left keyframe.
        if self.secondary_keyframe:
            slam_graph = unpack_optional(self.slam_output.slam_map).backend_graph
            if slam_graph is not None:
                matching_secondary_j = slam_graph[slam_graph[:, 0] == left_idx, 1].tolist()
                picked_sj_idx = np.argmax([abs(self.keyframes_inds[j] - frame_idx) for j in matching_secondary_j])
                inds.append(matching_secondary_j[picked_sj_idx])
        return inds

    def record_keyframes(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                self.keyframes_data.append(frame)
            yield frame

    def estimate_depth_sliding_window(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        current_sliding_window: list[VideoFrame] = []
        current_sliding_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        for frame_idx, frame in pbar(enumerate(previous_iterator), desc="Estimating multi-view depth"):
            current_sliding_window.append(frame)
            current_sliding_window_idx.append(frame_idx)
            is_last_frame = frame_idx == self.n_frames - 1

            if len(current_sliding_window) == self.window_size or is_last_frame:
                # Grab all neighboring keyframes to anchor the current sliding window.
                # Note that we remove redundant keyframes that already exist in the current sliding window.
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_sliding_window_idx], []))
                )
                sw_keyframe_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_sliding_window_idx
                ]

                sw_images, sw_exts, sw_ints = zip(*[frame.dav3_conditions() for frame in current_sliding_window])

                if len(sw_keyframe_inds) > 0:
                    kf_images, kf_exts, kf_ints = zip(
                        *[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds]
                    )
                else:
                    kf_images, kf_exts, kf_ints = tuple(), tuple(), tuple()

                # Perform inference
                dav3_inference_result = self.dav3_api.inference(
                    list(sw_images + kf_images),
                    extrinsics=np.stack(sw_exts + kf_exts, axis=0),
                    intrinsics=np.stack(sw_ints + kf_ints, axis=0),
                    process_res_method="lower_bound_resize",  # Keep aspect ratio
                )
                sw_depth = torch.from_numpy(dav3_inference_result.depth[: len(sw_images)]).float().cuda()
                sw_depth = torch.nn.functional.interpolate(sw_depth[:, None], frame.size(), mode="bilinear")[:, 0]

                n_frames_to_yield = (
                    self.window_size - self.overlap_size if not is_last_frame else len(current_sliding_window)
                )

                # Linearly interpolate the trailing depth with new depth
                if trailing_depth is not None:
                    n_interp_frames = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp_frames + 2)[1:-1].float().cuda()[:, None, None]
                    sw_depth[:n_interp_frames] = trailing_depth * (1 - alpha) + sw_depth[:n_interp_frames] * alpha

                for sw_idx, frame in enumerate(current_sliding_window[:n_frames_to_yield]):
                    frame.metric_depth = sw_depth[sw_idx]
                    yield frame

                trailing_depth = sw_depth[n_frames_to_yield:]
                current_sliding_window = current_sliding_window[n_frames_to_yield:]
                current_sliding_window_idx = current_sliding_window_idx[n_frames_to_yield:]

        assert len(current_sliding_window) == 0, "Current sliding window should be empty"

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth_sliding_window(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")


class EquirectProjectionProcessor(StreamProcessor):
    """
    Camera convention (with rotation = I, up of panorama is outward, Y is inward):
       -----
      (  Z  )
     (   |   )
    (    Y-X  )
     (       )
      (     )
       -<|>-
         |
    [boundary of image]
    """

    def __init__(self, rotation: SO3, frame_size: tuple[int, int], intrinsics: torch.Tensor) -> None:
        super().__init__()
        self.rotation = rotation.cuda()
        self.intrinsics = intrinsics.cuda()
        rays = get_camera_rays(frame_size[0], frame_size[1], self.intrinsics, normalize=True)
        rays = unpack_optional(self.rotation[None, None].act(rays))
        uv = project_points_to_panorama(rays, return_depth=False)
        self.uv = (uv * 2) - 1
        self.frame_size = frame_size

    @staticmethod
    def yaw_pitch_to_rotation(yaw: float, pitch: float) -> SO3:
        """
        First rotate around yaw, then pitch (positive is heads up, negative is down).
        """
        return SO3.InitFromVec(
            torch.from_numpy(R.from_euler("xyz", [pitch, yaw, 0], degrees=False).as_quat(canonical=True)).float()
        )

    def update_frame_size(self, previous_frame_size: tuple[int, int]):
        return self.frame_size

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert frame.metric_depth is None, "Metric depth is not supported for equirect projection"

        if (new_pose := frame.pose) is not None:
            rel_transform = SE3.InitFromVec(torch.cat((torch.zeros(3).cuda(), self.rotation.data)))
            new_pose = new_pose * rel_transform

        new_rgb = (
            torch.nn.functional.grid_sample(frame.rgb.moveaxis(-1, 0)[None], self.uv[None], align_corners=True)
            .squeeze()
            .moveaxis(0, -1)
        )

        if (new_instance := frame.instance) is not None:
            new_instance = torch.nn.functional.grid_sample(
                new_instance[None, None].float(), self.uv[None], align_corners=True, mode="nearest"
            )[0, 0]

        if (new_mask := frame.mask) is not None:
            new_mask = torch.nn.functional.grid_sample(
                new_mask[None, None].float(), self.uv[None], align_corners=True, mode="nearest"
            )[0, 0]

        return VideoFrame(
            raw_frame_idx=frame.raw_frame_idx,
            rgb=new_rgb,
            pose=new_pose,
            intrinsics=self.intrinsics.clone(),
            camera_type=CameraType.PINHOLE,
            instance=new_instance,
            mask=new_mask,
        )
