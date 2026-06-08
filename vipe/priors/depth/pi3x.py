# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from pi3.models.pi3x import Pi3X
except ModuleNotFoundError:
    Pi3X = None

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def _resize_multiple_of_14(height: int, width: int, pixel_limit: int) -> tuple[int, int]:
    if height * width <= pixel_limit:
        target_h, target_w = height, width
    else:
        scale = math.sqrt(pixel_limit / float(height * width))
        target_h, target_w = int(height * scale), int(width * scale)

    target_h = max(14, ((target_h + 13) // 14) * 14)
    target_w = max(14, ((target_w + 13) // 14) * 14)
    while target_h * target_w > pixel_limit and (target_h > 14 or target_w > 14):
        if target_h > target_w:
            target_h -= 14
        else:
            target_w -= 14
    return target_h, target_w


def _intrinsics_4_to_matrix(intrinsics: torch.Tensor, count: int) -> torch.Tensor:
    if intrinsics.dim() == 1:
        intrinsics = intrinsics[None].expand(count, -1)
    K = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype).repeat(intrinsics.shape[0], 1, 1)
    K[:, 0, 0] = intrinsics[:, 0]
    K[:, 1, 1] = intrinsics[:, 1]
    K[:, 0, 2] = intrinsics[:, 2]
    K[:, 1, 2] = intrinsics[:, 3]
    return K


def normalize_intrinsics(intrinsics: torch.Tensor | None, count: int) -> torch.Tensor | None:
    if intrinsics is None:
        return None
    if intrinsics.dim() == 1 or (intrinsics.dim() == 2 and intrinsics.shape[-1] == 4):
        return _intrinsics_4_to_matrix(intrinsics, count)
    if intrinsics.dim() == 2 and intrinsics.shape == (3, 3):
        return intrinsics[None].expand(count, -1, -1)
    if intrinsics.dim() == 3 and intrinsics.shape[-2:] == (3, 3):
        return intrinsics
    raise ValueError(f"Unsupported intrinsics shape for Pi3X: {tuple(intrinsics.shape)}")


class Pi3XModel(nn.Module):
    def __init__(self, pretrained: str = "yyfz233/Pi3X") -> None:
        super().__init__()
        if Pi3X is None:
            raise RuntimeError(
                "Pi3X is not available. Install Pi3X or add it to PYTHONPATH before using the 'pi3x' depth model."
            )
        self.model = Pi3X.from_pretrained(pretrained).cuda().eval()

    def forward(
        self,
        imgs: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        poses: torch.Tensor | None = None,
        depths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conditions: dict[str, Any] = {}
        if intrinsics is not None:
            conditions["intrinsics"] = intrinsics
        if poses is not None:
            conditions["poses"] = poses
        if depths is not None:
            conditions["depths"] = depths

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            return self.model(imgs=imgs, **conditions)


class Pi3XDepthModel(DepthEstimationModel):
    def __init__(self, pretrained: str = "yyfz233/Pi3X", pixel_limit: int = 255000) -> None:
        super().__init__()
        self.model = Pi3XModel(pretrained=pretrained)
        self.pixel_limit = pixel_limit

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    @torch.no_grad()
    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb = src.rgb
        if rgb is None:
            return DepthEstimationResult()

        single_frame = rgb.dim() == 3
        if single_frame:
            rgb_nhwc = rgb[None]
        elif rgb.dim() == 4:
            rgb_nhwc = rgb
        else:
            raise ValueError(f"Unsupported RGB shape for Pi3X: {tuple(rgb.shape)}")

        n, orig_h, orig_w, _ = rgb_nhwc.shape
        new_h, new_w = _resize_multiple_of_14(orig_h, orig_w, self.pixel_limit)

        imgs = rearrange(rgb_nhwc, "n h w c -> n c h w")
        if (new_h, new_w) != (orig_h, orig_w):
            imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        imgs = imgs.unsqueeze(0)

        intrinsics = normalize_intrinsics(src.intrinsics, n)
        if intrinsics is not None:
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, :] *= new_w / orig_w
            intrinsics[:, 1, :] *= new_h / orig_h
            intrinsics = intrinsics.unsqueeze(0)

        poses = src.poses
        if poses is not None:
            if poses.dim() == 2:
                poses = poses[None]
            poses = poses.unsqueeze(0)

        output = self.model(imgs=imgs, intrinsics=intrinsics, poses=poses)
        local_points = output.get("local_points")
        if local_points is None:
            return DepthEstimationResult()

        metric_depth = local_points[..., 2]
        if "conf" in output:
            metric_depth = metric_depth * (torch.sigmoid(output["conf"][..., 0]) > 0.1).float()

        if (new_h, new_w) != (orig_h, orig_w):
            metric_depth = F.interpolate(metric_depth, size=(orig_h, orig_w), mode="nearest")

        metric_depth = metric_depth.squeeze(0)
        if single_frame:
            metric_depth = metric_depth.squeeze(0)
        return DepthEstimationResult(metric_depth=metric_depth)
