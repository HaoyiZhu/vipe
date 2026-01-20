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
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_inv_depth_to_depth
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.depth.videodepthanything.util import compute_scale_and_shift, get_interpolate_frames
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import (CachedVideoStream, FrameAttribute,
                               StreamProcessor, VideoFrame, VideoStream)
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.morph import erode

from vipe.priors.depth.pi3x_moge import Pi3XMoGeV2Model, mask_aware_nearest_resize_robust
from vipe.priors.depth.pi3x import Pi3XDepthModel
from vipe.priors.depth.moge_v2 import focal_length_to_fov_degrees

try:
    from moge.utils.alignment import align_points_scale_z_shift, align_points_z_shift
except ImportError:
    align_points_scale_z_shift = None
    align_points_z_shift = None

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).cuda()
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

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [res["camera"].dist[0, 0].item()]


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
    ) -> None:
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(self.mask_phrases, sam_points_per_side=50, sam_run_gap=self.sam_run_gap)
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


class Pi3XVOInitPoseProcessor(StreamProcessor):
    """
    Initialize per-frame poses using Pi3X VO predictions.
    This runs once over the full stream and sets FrameAttribute.POSE.
    """

    def __init__(
        self,
        video_stream: VideoStream,
        model: str = "yyfz233/Pi3X",
        chunk_size: int = 16,
        overlap: int = 6,
        conf_thre: float = 0.05,
        dtype: str = "bf16",
        pose_convention: str = "c2w",
        pixel_limit: int = 255000,
        return_depth: bool = False,
        depth_conf_thre: float | None = None,
    ) -> None:
        super().__init__()
        self.video_stream = video_stream
        self.chunk_size = max(2, int(chunk_size))
        self.overlap = max(0, int(overlap))
        self.conf_thre = float(conf_thre)
        self.dtype = dtype
        self.pose_convention = pose_convention
        self.pixel_limit = int(pixel_limit)
        self.return_depth = bool(return_depth)
        self.depth_conf_thre = conf_thre if depth_conf_thre is None else float(depth_conf_thre)

        try:
            from pi3.models.pi3x import Pi3X
            from pi3.pipe.pi3x_vo import Pi3XVO
        except ImportError as exc:
            raise ImportError(
                "Pi3X VO not available. Ensure thirdparty/Pi3 is on PYTHONPATH and Pi3X is installed."
            ) from exc

        if model.endswith(".safetensors") or model.endswith(".pt") or model.endswith(".pth"):
            # Use custom checkpoint
            self.pi3x_model = Pi3X().cuda().eval()
            if model.endswith(".safetensors"):
                from safetensors.torch import load_file
                weight = load_file(model)
            else:
                weight = torch.load(model, map_location="cpu", weights_only=False)
            self.pi3x_model.load_state_dict(weight, strict=False)
        else:
            # HuggingFace model id
            self.pi3x_model = Pi3X.from_pretrained(model).cuda().eval()

        self.vo = Pi3XVO(self.pi3x_model)
        self.require_cache = True

    def _get_resize_size(self, height: int, width: int) -> tuple[int, int]:
        # Match Pi3X constraints: resolution must be a multiple of 14 and within pixel_limit.
        if height * width > self.pixel_limit:
            scale = math.sqrt(self.pixel_limit / float(height * width))
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))

        # Round up to a multiple of 14
        target_h = ((height + 13) // 14) * 14
        target_w = ((width + 13) // 14) * 14

        # If rounding up exceeds pixel_limit, step down by 14s
        while target_h * target_w > self.pixel_limit and (target_h > 14 or target_w > 14):
            if target_h > target_w:
                target_h -= 14
            else:
                target_w -= 14

        return max(14, target_h), max(14, target_w)

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        attributes = {FrameAttribute.POSE}
        if self.return_depth:
            attributes.add(FrameAttribute.METRIC_DEPTH)
        return previous_attributes | attributes

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        frames = [frame.cpu() for frame in previous_iterator]
        if not frames:
            return iter([])

        rgb_list = [f.rgb.permute(2, 0, 1) for f in frames]  # (T, 3, H, W)
        imgs = torch.stack(rgb_list, dim=0)  # (T, 3, H, W)
        h0, w0 = imgs.shape[-2], imgs.shape[-1]
        h1, w1 = self._get_resize_size(h0, w0)
        if (h1, w1) != (h0, w0):
            imgs = F.interpolate(imgs, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)
        imgs = imgs.unsqueeze(0).cuda()  # (1, T, 3, H, W)

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        
        print(f"imgs.shape: {imgs.shape}")
        print(f"chunk_size: {self.chunk_size}")
        print(f"overlap: {self.overlap}")
        print(f"conf_thre: {self.conf_thre}")
        print(f"dtype: {torch_dtype}")
        print(f"pose_convention: {self.pose_convention}")

        result = self.vo(
            imgs,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            conf_thre=self.conf_thre,
            dtype=torch_dtype,
        )
        poses = result["camera_poses"][0].detach().cpu()  # (T, 4, 4)
        local_depth = None
        local_conf = None
        if self.return_depth:
            if "local_depth" not in result:
                raise KeyError("Pi3X VO did not return local_depth; please update Pi3XVO to expose it.")
            local_depth = result["local_depth"][0].detach().cpu()  # (T, H_t, W_t)
            local_conf = result.get("local_conf", None)
            if local_conf is not None:
                local_conf = local_conf[0].detach().cpu()  # (T, H_t, W_t)

        if self.pose_convention == "w2c":
            poses = torch.linalg.inv(poses)

        poses_se3 = se3_matrix_to_se3(poses, unbatch=False)
        n_frames = min(len(frames), poses_se3.shape[0])
        if n_frames != len(frames):
            logger.warning(
                "Pi3X VO returned %d poses for %d frames; truncating to %d.",
                poses_se3.shape[0],
                len(frames),
                n_frames,
            )

        for idx in range(n_frames):
            frame = frames[idx]
            frame.pose = poses_se3[idx].to(frame.rgb.device)
            if self.return_depth and local_depth is not None:
                depth = local_depth[idx][None, None].float()
                if (h1, w1) != (h0, w0):
                    depth = F.interpolate(depth, size=(h0, w0), mode="bilinear", align_corners=False)
                depth = depth[0, 0]
                if local_conf is not None and self.depth_conf_thre is not None:
                    conf = local_conf[idx][None, None].float()
                    if (h1, w1) != (h0, w0):
                        conf = F.interpolate(conf, size=(h0, w0), mode="bilinear", align_corners=False)
                    conf = conf[0, 0]
                    depth = torch.where(conf >= self.depth_conf_thre, depth, torch.zeros_like(depth))
                frame.metric_depth = depth.to(frame.rgb.device)
            yield frame


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
            self.video_depth_model = VideoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")

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

        video_depth_result: torch.Tensor = unpack_optional(
            self.video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
        )
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

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
                    depth_infilled = self.slam_output.slam_map.project_map(
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
                self.slam_output.metrics["confidence"] = min_uv_score

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = self.slam_output.slam_map.project_map(
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
                    _, scale, bias = align_inv_depth_to_depth(
                        unpack_optional(video_depth_inv_depth),
                        prompt_result,
                        align_mask,
                    )
                except RuntimeError:
                    scale, bias = self.cache_scale_bias

                # momentum update
                if self.cache_scale_bias is None:
                    self.cache_scale_bias = (scale, bias)
                scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                self.cache_scale_bias = (scale, bias)

                video_inv_depth = video_depth_inv_depth * scale + bias
                video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame


class Pi3XAdaptiveDepthProcessor(StreamProcessor):
    """
    Same alignment strategy as AdaptiveDepthProcessor, but replaces the video depth model with Pi3X.
    Pi3X is run on resized RGB images only (no intrinsics / poses).
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        metric_model: str = "unidepth-l",
        pixel_limit: int = 255000,
        batch_size: int = 4,
        # window_size: int = 32,
        # overlap_size: int = 10,
        # interp_len: int = 8,
        # window_align: bool = True,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.view_idx = view_idx
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Pi3X adaptive depth processor only supports view_idx=0"

        self.metric_model = metric_model
        self.pixel_limit = pixel_limit
        self.batch_size = max(1, int(batch_size))
        # self.window_size = window_size
        # self.overlap_size = overlap_size
        # self.interp_len = interp_len
        # self.window_align = window_align

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

        self.pi3x = Pi3XDepthModel(pixel_limit=pixel_limit)
        self.require_cache = True

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("Pi3XAdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_pi3x_inv_depth(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_data_list: list[VideoFrame] = []
        rgb_list: list[torch.Tensor] = []
        for frame in frame_iterator:
            frame_cpu = frame.cpu()
            frame_data_list.append(frame_cpu)
            rgb_list.append(frame_cpu.rgb)

        inv_depth_list: list[torch.Tensor] = []
        for i in range(0, len(rgb_list), self.batch_size):
            batch_rgb = torch.stack(rgb_list[i : i + self.batch_size], dim=0).float().cuda()
            # Pi3X is run unconditioned (no intrinsics / poses)
            depth = self.pi3x.estimate(DepthEstimationInput(rgb=batch_rgb)).metric_depth
            if depth is None:
                inv_depth = torch.zeros_like(batch_rgb[:, :, :, 0])
            else:
                inv_depth = torch.where(depth > 1e-3, depth.reciprocal(), torch.zeros_like(depth))
            inv_depth_list.append(inv_depth.cpu())

        
        return torch.cat(inv_depth_list, dim=0), frame_data_list
        # inv_depth = torch.cat(inv_depth_list, dim=0)

        # if self.window_align and inv_depth.shape[0] > self.window_size and self.overlap_size > 0:
        #     inv_depth = self._align_inv_depth_windows(inv_depth)

        # return inv_depth, frame_data_list

    # def _align_inv_depth_windows(self, inv_depth: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply a sliding-window scale+shift alignment on inverse depth (like VideoDepthAnything).
    #     """
    #     n_frames = inv_depth.shape[0]
    #     window = self.window_size
    #     overlap = min(self.overlap_size, window - 1)
    #     interp_len = min(self.interp_len, overlap)
    #     step = window - overlap
    #     if step <= 0:
    #         return inv_depth

    #     inv_np = inv_depth.cpu().numpy()
    #     depth_list = [inv_np[i] for i in range(n_frames)]

    #     aligned = []
    #     ref_align = []
    #     align_len = max(overlap - interp_len, 1)
    #     kf_align_list = list(range(align_len))

    #     for frame_id in range(0, len(depth_list), step):
    #         chunk = depth_list[frame_id : frame_id + window]
    #         if len(chunk) < window:
    #             chunk = chunk + [chunk[-1].copy()] * (window - len(chunk))

    #         if len(aligned) == 0:
    #             aligned += chunk[:window]
    #             ref_align = [chunk[i] for i in kf_align_list]
    #             continue

    #         curr_align = [chunk[i] for i in kf_align_list]
    #         scale, shift = compute_scale_and_shift(
    #             np.concatenate(curr_align),
    #             np.concatenate(ref_align),
    #             np.ones_like(np.concatenate(ref_align)),
    #         )

    #         pre_depth_list = aligned[-interp_len:] if interp_len > 0 else []
    #         post_depth_list = chunk[align_len : align_len + interp_len] if interp_len > 0 else []
    #         for i in range(len(post_depth_list)):
    #             post_depth_list[i] = post_depth_list[i] * scale + shift
    #             post_depth_list[i][post_depth_list[i] < 0] = 0
    #         if interp_len > 0:
    #             aligned[-interp_len:] = get_interpolate_frames(pre_depth_list, post_depth_list)

    #         for i in range(overlap, window):
    #             new_depth = chunk[i] * scale + shift
    #             new_depth[new_depth < 0] = 0
    #             aligned.append(new_depth)

    #         # Update reference frames
    #         ref_align = [chunk[i] * scale + shift for i in kf_align_list]

    #     aligned = aligned[:n_frames]
    #     return torch.from_numpy(np.stack(aligned, axis=0)).to(inv_depth.device)

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx != 0:
            raise ValueError(f"Invalid pass index: {pass_idx}")

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        video_depth_inv, data_iterator = self._compute_pi3x_inv_depth(previous_iterator)

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning Pi3X depth"):
            frame = frame.cuda()

            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = self.slam_output.slam_map.project_map(
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
                self.slam_output.metrics["confidence"] = min_uv_score

            if min_uv_score < 0.3:
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(), intrinsics=frame.intrinsics, camera_type=frame.camera_type
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
            else:
                depth_map = self.slam_output.slam_map.project_map(
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

            video_depth_inv_depth = video_depth_inv[frame_idx].cuda()
            align_mask = video_depth_inv_depth > 1e-3
            if frame.mask is not None:
                align_mask = align_mask & frame.mask & (~frame.sky_mask)

            try:
                _, scale, bias = align_inv_depth_to_depth(
                    unpack_optional(video_depth_inv_depth),
                    prompt_result,
                    align_mask,
                )
            except RuntimeError:
                if self.cache_scale_bias is None:
                    # First frame (or first failure): fall back to identity alignment.
                    scale, bias = 1.0, 0.0
                else:
                    scale, bias = self.cache_scale_bias
                if self.cache_scale_bias is None:
                    # First frame (or first failure): fall back to identity alignment.
                    scale, bias = 1.0, 0.0
                else:
                    scale, bias = self.cache_scale_bias

            if self.cache_scale_bias is None:
                self.cache_scale_bias = (scale, bias)
            scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
            bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
            self.cache_scale_bias = (scale, bias)

            video_inv_depth = video_depth_inv_depth * scale # + bias
            video_inv_depth[video_inv_depth < 1e-3] = 1e-3
            frame.metric_depth = video_inv_depth.reciprocal()

            yield frame

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
        window_size: int = 10,                  # Practically this should be as large as possible if memory permits.
        overlap_size: int = 3,
        secondary_keyframe: bool = False,       # This is found to cause jittering for some scenes due to abrupt context changes.
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
            try:
                from depth_anything_3.api import DepthAnything3
                from depth_anything_3.api import logger as dav3_logger
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "depth-anything-3 not found. Please reinstall vipe with `pip install --no-build-isolation -e .[dav3]`"
                )

            dav3_logger.level = 0  # Disable logging timing information
            self.dav3_api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
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
                    kf_images, kf_exts, kf_ints = zip(*[self.keyframes_data[t].dav3_conditions() for t in sw_keyframe_inds])
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


class Pi3XMoGeProcessor(StreamProcessor):
    """
    Use Pi3X for robust multi-view reconstruction and MoGe2 for metric alignment.
    Pi3X is conditioned on SLAM poses and intrinsics.
    The result is aligned to MoGe2's metric scale.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        window_size: int = 32,   # Reduced from 100 to prevent OOM. Pi3X processes the whole window jointly.
        overlap_size: int = 8,
        pixel_limit: int = 255000,
    ) -> None:
        super().__init__()
        self.slam_output = slam_output
        self.window_size = min(window_size, 120)
        self.overlap_size = overlap_size
        self.pixel_limit = pixel_limit

        self.model = Pi3XMoGeV2Model(pixel_limit=pixel_limit)
        self.pi3x = self.model.pi3x
        self.moge = self.model.moge
        
        # We need to count frames first
        self.n_frames = 0
        self.n_passes_required = 2  # Pass 0: record keyframes, Pass 1: Inference
        
        self.keyframes_inds = unpack_optional(self.slam_output.slam_map).dense_disp_frame_inds
        self.keyframes_data: list[VideoFrame] = []
        
        if align_points_scale_z_shift is None:
             raise ImportError("align_points_scale_z_shift not found. Check moge import.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("Pi3XMoGeProcessor should not be called directly.")

    def _probe_keyframe_indices(self, frame_idx: int) -> list[int]:
        inds: list[int] = []
        left_idx = np.searchsorted(self.keyframes_inds, frame_idx, side="right").item() - 1
        inds.append(left_idx)
        if frame_idx < self.keyframes_inds[-1]:
            inds.append(left_idx + 1)
        return inds

    def record_keyframes(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        for frame_idx, frame in enumerate(previous_iterator):
            self.n_frames += 1
            if frame_idx in self.keyframes_inds:
                self.keyframes_data.append(frame.cpu())
            yield frame

    def _compute_moge_keyframes(self):
        if not self.keyframes_data:
            return None, None
        
        kf_moge_pts_list = []
        kf_moge_masks_list = []
        bs = 4
        # Get H_orig, W_orig from first frame
        H_orig, W_orig = self.keyframes_data[0].size()
        H_t, W_t = self.model._get_resize_size(H_orig, W_orig)
        
        for i in range(0, len(self.keyframes_data), bs):
            batch = self.keyframes_data[i:i+bs]
            imgs_list = []
            fov_x_list = []
            for f in batch:
                input_tensor = f.rgb.cuda().permute(2, 0, 1).unsqueeze(0)
                img = F.interpolate(input_tensor, size=(H_t, W_t), mode='bilinear', align_corners=False, antialias=True)
                imgs_list.append(img)
                
                fx = unpack_optional(f.intrinsics)[0]
                fov_deg = focal_length_to_fov_degrees(fx.item(), W_orig)
                fov_x_list.append(fov_deg)
            
            imgs = torch.cat(imgs_list, dim=0)
            with torch.no_grad():
                res = self.moge.forward(imgs, fov_x=fov_x_list[0]) 
                kf_moge_pts_list.append(res['points'].cpu())
                kf_moge_masks_list.append(res['mask'].cpu().bool().squeeze(1))
        
        return torch.cat(kf_moge_pts_list, dim=0), torch.cat(kf_moge_masks_list, dim=0)

    def estimate_depth(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        # Move frames to CPU to save GPU memory during the whole video processing
        all_frames = [f.cpu() for f in previous_iterator]
        self.n_frames = len(all_frames)
        
        # 1. MoGe for keyframes
        logger.info(f"Computing MoGe2 for {len(self.keyframes_data)} keyframes...")
        moge_kf_pts, moge_kf_masks = self._compute_moge_keyframes() # Points: (N_kf, H_t, W_t, 3), Mask: (N_kf, H_t, W_t) on CPU
        
        # 2. Pi3X Sliding Window
        current_window: list[VideoFrame] = []
        current_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None
        
        all_scales = []
        
        for frame_idx, frame in pbar(enumerate(all_frames), total=self.n_frames, desc="Pi3X Sliding Window"):
            current_window.append(frame)
            current_window_idx.append(frame_idx)
            is_last_frame = frame_idx == self.n_frames - 1
            
            if len(current_window) == self.window_size or is_last_frame:
                sw_keyframe_inds = list(
                    set(sum([self._probe_keyframe_indices(i) for i in current_window_idx], []))
                )
                extra_kf_inds = [
                    t for t in sw_keyframe_inds if self.keyframes_inds[t] not in current_window_idx
                ]
                extra_kf = [self.keyframes_data[t] for t in extra_kf_inds]
                
                imgs, poses, intrinsics, _, H_t, W_t, H_orig, W_orig = self._prepare_inputs(current_window, extra_kf)
                
                with torch.no_grad():
                    pi3x_out = self.pi3x(imgs, intrinsics=intrinsics, poses=poses)
                    pi3x_local = pi3x_out['local_points'][0] # (N_all, H_t, W_t, 3)
                    pi3x_conf = torch.sigmoid(pi3x_out['conf'][0, ..., 0]) > 0.1 # (N_all, H_t, W_t)
                
                pi3x_window_pts = pi3x_local[:len(current_window)]
                pi3x_window_conf = pi3x_conf[:len(current_window)]
                
                # Alignment for scale collection
                window_kf_in_kf_data = [t for t in sw_keyframe_inds if self.keyframes_inds[t] in current_window_idx]
                
                if window_kf_in_kf_data and moge_kf_pts is not None:
                    for kf_idx in window_kf_in_kf_data:
                        global_frame_idx = self.keyframes_inds[kf_idx]
                        try:
                            win_idx = current_window_idx.index(global_frame_idx)
                            pts_pi3x = pi3x_window_pts[win_idx]
                            conf_pi3x = pi3x_window_conf[win_idx]
                            
                            pts_moge = moge_kf_pts[kf_idx].to(pts_pi3x.device)
                            mask_moge = moge_kf_masks[kf_idx].to(conf_pi3x.device)
                            
                            combined_mask = conf_pi3x & mask_moge
                            
                            # Downsample using mask_aware_nearest_resize_robust
                            indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, 64, 64)
                            nearest_i, nearest_j = indices
                            
                            pts_pi3x_lr = pts_pi3x[nearest_i, nearest_j]
                            pts_moge_lr = pts_moge[nearest_i, nearest_j]
                            
                            weights = 1.0 / pts_moge_lr[..., 2].clamp(min=1e-3)
                            
                            if lr_mask.sum() >= 10:
                                s, _ = align_points_scale_z_shift(
                                    pts_pi3x_lr[lr_mask].unsqueeze(0),
                                    pts_moge_lr[lr_mask].unsqueeze(0),
                                    weights[lr_mask].unsqueeze(0)
                                )
                                if s.item() > 0:
                                    all_scales.append(s.item())
                        except ValueError:
                            continue

                pi3x_window_depth = pi3x_window_pts[..., 2]
                sw_depth = F.interpolate(
                    pi3x_window_depth.unsqueeze(1),
                    size=(H_orig, W_orig),
                    mode='nearest',
                ).squeeze(1)
                
                n_yield = self.window_size - self.overlap_size if not is_last_frame else len(current_window)
                
                if trailing_depth is not None:
                    n_interp = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp + 2)[1:-1].to(sw_depth.device)[:, None, None]
                    sw_depth[:n_interp] = trailing_depth * (1 - alpha) + sw_depth[:n_interp] * alpha
                
                for i in range(n_yield):
                    current_window[i].metric_depth = sw_depth[i].cpu()
                
                trailing_depth = sw_depth[n_yield:]
                current_window = current_window[n_yield:]
                current_window_idx = current_window_idx[n_yield:]
        
        if all_scales:
            global_scale = np.median(all_scales)
            logger.info(f"Global median scale: {global_scale:.4f} (from {len(all_scales)} samples)")
        else:
            global_scale = 1.0
            logger.warning("No scales collected, using 1.0")
            
        for frame in all_frames:
            if frame.metric_depth is not None:
                frame.metric_depth *= global_scale
            yield frame.cuda()

    def _prepare_inputs(self, frames, key_frames=[]):
        all_frames = frames + key_frames
        
        H_orig, W_orig = frames[0].size()
        H_target, W_target = self.model._get_resize_size(H_orig, W_orig)
        
        imgs_list = []
        K_list = []
        poses_list = []
        fov_x_list = []
        
        scale_x = W_target / W_orig
        scale_y = H_target / H_orig
        
        for f in all_frames:
            input_tensor = f.rgb.cuda().permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(input_tensor, size=(H_target, W_target), mode='bilinear', align_corners=False, antialias=True)
            imgs_list.append(img)
            
            poses_list.append(torch.as_tensor(f.pose.matrix()).cpu())
            
            fx, fy, cx, cy = unpack_optional(f.intrinsics)[:4]
            K = torch.eye(3, device='cpu')
            K[0, 0] = fx * scale_x
            K[1, 1] = fy * scale_y
            K[0, 2] = cx * scale_x
            K[1, 2] = cy * scale_y
            K_list.append(K)
            
            fov_deg = focal_length_to_fov_degrees(fx.item(), W_orig)
            fov_x_list.append(fov_deg)

        imgs = torch.cat(imgs_list, dim=0).cuda().unsqueeze(0)
        poses = torch.stack(poses_list).cuda().unsqueeze(0)
        intrinsics = torch.stack(K_list).cuda().unsqueeze(0)
        
        return imgs, poses, intrinsics, fov_x_list, H_target, W_target, H_orig, W_orig

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.record_keyframes(previous_iterator)
        elif pass_idx == 1:
            yield from self.estimate_depth(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")


class Pi3XMoGePerFrameProcessor(StreamProcessor):
    """
    Post-processing depth for all frames:
    - Run Pi3X in a sliding window, conditioned on SLAM poses & intrinsics (multi-frame "video" depth).
    - For each frame, run MoGe2 and solve a robust L1 alignment between Pi3X point cloud and MoGe2 point cloud:
        z_tgt ≈ scale * z_src + shift_z   (xy shift fixed to 0, scale shared across xyz)
    - Apply (scale, shift_z) per-frame (no global median), producing metric depth.

    This is designed to be close in spirit to the default pipeline’s post depth stage, but uses Pi3X+MoGe2.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        window_size: int = 64,
        overlap_size: int = 16,
        pixel_limit: int = 255000,
        align_lr_size: int = 64,
        min_align_points: int = 200,
        align_mode: str = "per_frame_ema",  # per_frame | per_frame_ema | window_shared | window_shared_ema
        align_momentum: float = 0.99,
        scale_clamp: tuple[float, float] = (0.1, 10.0),
        shift_z_clamp: tuple[float, float] = (-1e3, 1e3),
        moge_bs: int = 4,
        align_source: str = "moge2",  # moge2 | slam_map
    ) -> None:
        super().__init__()
        self.slam_output = slam_output
        self.view_idx = view_idx
        self.window_size = min(window_size, 180)
        self.overlap_size = overlap_size
        self.pixel_limit = pixel_limit
        self.align_lr_size = align_lr_size
        self.min_align_points = min_align_points
        self.align_mode = align_mode
        self.align_momentum = align_momentum
        self.scale_clamp = scale_clamp
        self.shift_z_clamp = shift_z_clamp
        self.moge_bs = max(1, int(moge_bs))
        self.align_source = align_source
        # We only apply scale to Pi3X (Pi3X is scale-invariant); z-shift is used only to make the
        # scale estimation robust when MoGe2 has an additive depth offset.
        self._cache_scale: torch.Tensor | None = None

        self.model = Pi3XMoGeV2Model(pixel_limit=pixel_limit)
        self.pi3x = self.model.pi3x
        self.moge = self.model.moge

        # One-pass iterator: we materialize frames internally for sliding-window processing.
        self.n_frames = 0
        self.n_passes_required = 1

        if align_points_scale_z_shift is None:
            raise ImportError("align_points_scale_z_shift not found. Check moge import.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("Pi3XMoGePerFrameProcessor should not be called directly.")

    def _prepare_inputs(self, frames: list[VideoFrame]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, int, int, int]:
        """
        Returns:
          imgs: (1, N, 3, H_t, W_t)
          poses: (1, N, 4, 4)
          intrinsics: (1, N, 3, 3)
          fov_deg: float (horizontal fov in degrees)
          H_t, W_t, H_orig, W_orig
        """
        H_orig, W_orig = frames[0].size()
        H_t, W_t = self.model._get_resize_size(H_orig, W_orig)

        scale_x = W_t / W_orig
        scale_y = H_t / H_orig

        imgs_list: list[torch.Tensor] = []
        poses_list: list[torch.Tensor] = []
        K_list: list[torch.Tensor] = []

        # Assume intrinsics are consistent (assigned from SLAMOutput); compute per-frame anyway for safety.
        for f in frames:
            input_tensor = f.rgb.cuda().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            img = F.interpolate(input_tensor, size=(H_t, W_t), mode="bilinear", align_corners=False, antialias=True)
            imgs_list.append(img)

            poses_list.append(torch.as_tensor(unpack_optional(f.pose).matrix()).cpu())

            fx, fy, cx, cy = unpack_optional(f.intrinsics)[:4]
            K = torch.eye(3, device="cpu")
            K[0, 0] = fx * scale_x
            K[1, 1] = fy * scale_y
            K[0, 2] = cx * scale_x
            K[1, 2] = cy * scale_y
            K_list.append(K)

        # Horizontal fov from first frame fx and original width
        fx0 = unpack_optional(frames[0].intrinsics)[0].item()
        fov_deg = focal_length_to_fov_degrees(fx0, W_orig)

        imgs = torch.cat(imgs_list, dim=0).cuda().unsqueeze(0)
        poses = torch.stack(poses_list).cuda().unsqueeze(0)
        intrinsics = torch.stack(K_list).cuda().unsqueeze(0)

        return imgs, poses, intrinsics, fov_deg, H_t, W_t, H_orig, W_orig

    def _depth_to_points(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Convert depth map (H, W) to camera coordinates (H, W, 3) for pinhole cameras.
        K is a 3x3 intrinsics matrix.
        """
        assert depth.dim() == 2
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        z = depth
        x = (x - cx) / fx * z
        y = (y - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)

    def _resize_mask(self, frame: VideoFrame, target_size: tuple[int, int], device: torch.device) -> torch.Tensor | None:
        """
        Resize a frame mask to target size (H, W) using nearest interpolation.
        Returns a boolean mask on the requested device, or None if no mask exists.
        """
        if frame.mask is None:
            return None
        mask = frame.mask
        if mask.dim() != 2:
            mask = mask.squeeze()
        if mask.shape != target_size:
            mask = mask.float()[None, None].to(device)
            mask = F.interpolate(mask, size=target_size, mode="nearest")[0, 0].bool()
        else:
            mask = mask.to(device).bool()
        return mask

    @torch.no_grad()
    def estimate_depth(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        # Keep frames on CPU while iterating to reduce peak GPU memory.
        all_frames = [f.cpu() for f in previous_iterator]
        if not all_frames:
            return
        if self.align_source == "slam_map" and self.slam_output.slam_map is None:
            raise ValueError("align_source=slam_map requires slam_output.slam_map to be available")
        if self.align_source == "slam_map":
            for f in all_frames:
                if f.camera_type != CameraType.PINHOLE:
                    raise ValueError("align_source=slam_map currently supports only pinhole cameras")
        slam_device = None
        if self.align_source == "slam_map":
            slam_device = self.slam_output.slam_map.dense_disp_xyz.device

        current_window: list[VideoFrame] = []
        current_window_idx: list[int] = []
        trailing_depth: torch.Tensor | None = None

        for frame_idx, frame in pbar(enumerate(all_frames), total=len(all_frames), desc="Pi3X+MoGe2 Per-frame"):
            current_window.append(frame)
            current_window_idx.append(frame_idx)
            is_last_frame = frame_idx == len(all_frames) - 1

            if len(current_window) == self.window_size or is_last_frame:
                imgs, poses, intrinsics, fov_deg, H_t, W_t, H_orig, W_orig = self._prepare_inputs(current_window)

                # Pi3X inference (local points in camera coordinates)
                pi3x_out = self.pi3x(imgs)
                pi3x_pts = pi3x_out["local_points"][0][: len(current_window)]  # (N,H_t,W_t,3)
                pi3x_conf = torch.sigmoid(pi3x_out["conf"][0, : len(current_window), ..., 0]) > 0.1  # (N,H_t,W_t)

                moge_pts = None
                moge_mask = None
                moge_depth: torch.Tensor | None = None
                if self.align_source == "moge2":
                    # MoGe2 per-frame points (same resize)
                    imgs_moge = imgs[0]  # (N,3,H_t,W_t)
                    # IMPORTANT: run MoGe2 in small batches to avoid hitting 32-bit indexing limits
                    # in downstream conv kernels (error: "input tensor must fit into 32-bit index math").
                    Nw = imgs_moge.shape[0]
                    moge_pts = torch.empty((Nw, H_t, W_t, 3), device=imgs_moge.device, dtype=torch.float32)
                    moge_mask = torch.empty((Nw, H_t, W_t), device=imgs_moge.device, dtype=torch.bool)
                    for j in range(0, Nw, self.moge_bs):
                        out_j = self.moge.forward(imgs_moge[j : j + self.moge_bs])
                        pts_j = out_j["points"]
                        moge_pts[j : j + pts_j.shape[0]] = pts_j
                        mask_j = out_j.get("mask", torch.ones_like(pts_j[..., 0])).bool()
                        if mask_j.dim() == 4:
                            mask_j = mask_j.squeeze(1)
                        moge_mask[j : j + mask_j.shape[0]] = mask_j
                        if "depth" in out_j:
                            if moge_depth is None:
                                moge_depth = torch.empty(
                                    (Nw, H_t, W_t), device=imgs_moge.device, dtype=out_j["depth"].dtype
                                )
                            moge_depth[j : j + out_j["depth"].shape[0]] = out_j["depth"]

                # Per-frame alignment: scale + z-shift
                sw_depth = torch.zeros((len(current_window), H_orig, W_orig), device=pi3x_pts.device, dtype=pi3x_pts.dtype)
                # Optionally solve a single (scale, shift_z) shared across the window for better temporal consistency.
                window_shared_scale: torch.Tensor | None = None
                window_shared_shift_z: torch.Tensor | None = None
                if self.align_mode in ("window_shared", "window_shared_ema"):
                    pts_src_all = []
                    pts_tgt_all = []
                    w_all = []
                    for i in range(len(current_window)):
                        if self.align_source == "slam_map":
                            pose = unpack_optional(current_window[i].pose).to(slam_device)
                            K = intrinsics[0, i].to(slam_device)
                            intr = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                            slam_depth = self.slam_output.slam_map.project_map(
                                frame_tstamp=current_window_idx[i],
                                view_idx=self.view_idx,
                                target_size=(H_t, W_t),
                                target_intrinsics=intr,
                                target_pose=pose,
                                target_camera_type=unpack_optional(current_window[i].camera_type),
                                infill=False,
                            ).to(pi3x_pts.device)
                            slam_mask = slam_depth > 0
                            resized_mask = self._resize_mask(current_window[i], (H_t, W_t), slam_depth.device)
                            if resized_mask is not None:
                                slam_mask = slam_mask & resized_mask
                            combined_mask = pi3x_conf[i] & slam_mask
                            if combined_mask.sum().item() < self.min_align_points:
                                continue
                            indices, lr_mask = mask_aware_nearest_resize_robust(
                                combined_mask, self.align_lr_size, self.align_lr_size
                            )
                            ni, nj = indices
                            if lr_mask.sum().item() < 10:
                                continue
                            src_lr = pi3x_pts[i][ni, nj][lr_mask]
                            tgt_lr = self._depth_to_points(slam_depth, intrinsics[0, i])[ni, nj][lr_mask]
                            w_lr = (1.0 / slam_depth[ni, nj][lr_mask].clamp(min=1e-3))
                            pts_src_all.append(src_lr)
                            pts_tgt_all.append(tgt_lr)
                            w_all.append(w_lr)
                            continue

                        combined_mask = pi3x_conf[i] & moge_mask[i]
                        if combined_mask.sum().item() < self.min_align_points:
                            continue
                        indices, lr_mask = mask_aware_nearest_resize_robust(
                            combined_mask, self.align_lr_size, self.align_lr_size
                        )
                        ni, nj = indices
                        if lr_mask.sum().item() < 10:
                            continue
                        src_lr = pi3x_pts[i][ni, nj][lr_mask]
                        tgt_lr = moge_pts[i][ni, nj][lr_mask]
                        w_lr = (1.0 / tgt_lr[..., 2].clamp(min=1e-3))
                        pts_src_all.append(src_lr)
                        pts_tgt_all.append(tgt_lr)
                        w_all.append(w_lr)

                    if pts_src_all:
                        src_cat = torch.cat(pts_src_all, dim=0)[None]
                        tgt_cat = torch.cat(pts_tgt_all, dim=0)[None]
                        w_cat = torch.cat(w_all, dim=0)[None]
                        scale, shift = align_points_scale_z_shift(src_cat, tgt_cat, w_cat)
                        s = scale[0].clamp(self.scale_clamp[0], self.scale_clamp[1])
                        sz = shift[0, 2].clamp(self.shift_z_clamp[0], self.shift_z_clamp[1])
                        if torch.isfinite(s).all() and torch.isfinite(sz).all() and (s.item() > 0):
                            window_shared_scale = s
                            window_shared_shift_z = sz

                for i in range(len(current_window)):
                    if self.align_source == "slam_map":
                        pose = unpack_optional(current_window[i].pose).to(slam_device)
                        K = intrinsics[0, i].to(slam_device)
                        intr = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                        slam_depth = self.slam_output.slam_map.project_map(
                            frame_tstamp=current_window_idx[i],
                            view_idx=self.view_idx,
                            target_size=(H_t, W_t),
                            target_intrinsics=intr,
                            target_pose=pose,
                            target_camera_type=unpack_optional(current_window[i].camera_type),
                            infill=False,
                        ).to(pi3x_pts.device)
                        slam_mask = slam_depth > 0
                        resized_mask = self._resize_mask(current_window[i], (H_t, W_t), slam_depth.device)
                        if resized_mask is not None:
                            slam_mask = slam_mask & resized_mask
                        combined_mask = pi3x_conf[i] & slam_mask
                    else:
                        combined_mask = pi3x_conf[i] & moge_mask[i]

                    if combined_mask.sum().item() < self.min_align_points:
                        # Fallback: Pi3X (or MoGe2 if available)
                        if self.align_source == "moge2" and moge_depth is not None:
                            d = moge_depth[i]
                            d = torch.nan_to_num(d, nan=0.0).clamp(min=0.0, max=1e4)
                            d = d * pi3x_conf[i].float()
                        else:
                            d = (pi3x_pts[i, ..., 2].clamp(min=0.0) * pi3x_conf[i].float())
                        d_up = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=(H_orig, W_orig), mode="nearest")[0, 0]
                        sw_depth[i] = d_up
                        continue

                    indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, self.align_lr_size, self.align_lr_size)
                    ni, nj = indices
                    if lr_mask.sum().item() < 10:
                        d = (pi3x_pts[i, ..., 2].clamp(min=0.0) * pi3x_conf[i].float())
                        d_up = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=(H_orig, W_orig), mode="nearest")[0, 0]
                        sw_depth[i] = d_up
                        continue

                    pts_pi3x_lr = pi3x_pts[i][ni, nj]
                    if self.align_source == "slam_map":
                        pts_tgt_lr = self._depth_to_points(slam_depth, intrinsics[0, i])[ni, nj]
                        w = 1.0 / slam_depth[ni, nj].clamp(min=1e-3)
                    else:
                        pts_tgt_lr = moge_pts[i][ni, nj]
                        # L1 weighting: emphasize nearer (smaller z) points
                        w = 1.0 / pts_tgt_lr[..., 2].clamp(min=1e-3)

                    # Choose alignment parameters according to mode
                    scale_i: torch.Tensor | None = None
                    shiftz_i: torch.Tensor | None = None

                    if window_shared_scale is not None and window_shared_shift_z is not None:
                        scale_i, shiftz_i = window_shared_scale, window_shared_shift_z
                    else:
                        try:
                            scale, shift = align_points_scale_z_shift(
                                pts_pi3x_lr[lr_mask].unsqueeze(0),
                                pts_tgt_lr[lr_mask].unsqueeze(0),
                                w[lr_mask].unsqueeze(0),
                            )
                            scale_i = scale[0].clamp(self.scale_clamp[0], self.scale_clamp[1])
                            shiftz_i = shift[0, 2].clamp(self.shift_z_clamp[0], self.shift_z_clamp[1])
                            if not (torch.isfinite(scale_i).all() and torch.isfinite(shiftz_i).all() and (scale_i.item() > 0)):
                                scale_i = shiftz_i = None
                        except Exception:
                            scale_i = shiftz_i = None

                    # Temporal smoothing (like default pipeline): EMA on (scale, shiftz)
                    if self.align_mode in ("per_frame_ema", "window_shared_ema"):
                        if scale_i is None:
                            if self._cache_scale is not None:
                                scale_i = self._cache_scale
                        if scale_i is not None:
                            if self._cache_scale is None:
                                self._cache_scale = scale_i
                            else:
                                m = self.align_momentum
                                scale_i = self._cache_scale * m + scale_i * (1 - m)
                                self._cache_scale = scale_i

                    if scale_i is None:
                        d = (pi3x_pts[i, ..., 2].clamp(min=0.0) * pi3x_conf[i].float())
                    else:
                        # Apply ONLY scale to Pi3X (do not apply z-shift).
                        d = (pi3x_pts[i, ..., 2] * scale_i).clamp(min=0.0)
                        d = d * pi3x_conf[i].float()

                    d_up = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=(H_orig, W_orig), mode="nearest")[0, 0]
                    sw_depth[i] = d_up

                n_yield = self.window_size - self.overlap_size if not is_last_frame else len(current_window)

                if trailing_depth is not None:
                    n_interp = len(trailing_depth)
                    alpha = torch.linspace(0, 1, n_interp + 2, device=sw_depth.device)[1:-1][:, None, None]
                    sw_depth[:n_interp] = trailing_depth * (1 - alpha) + sw_depth[:n_interp] * alpha

                for i in range(n_yield):
                    current_window[i].metric_depth = sw_depth[i].cpu()

                trailing_depth = sw_depth[n_yield:].detach()
                current_window = current_window[n_yield:]
                current_window_idx = current_window_idx[n_yield:]

        for f in all_frames:
            yield f.cuda()

    def update_iterator(self, previous_iterator: Iterator[VideoFrame], pass_idx: int) -> Iterator[VideoFrame]:
        if pass_idx == 0:
            yield from self.estimate_depth(previous_iterator)
        else:
            raise ValueError(f"Invalid pass index: {pass_idx}")
