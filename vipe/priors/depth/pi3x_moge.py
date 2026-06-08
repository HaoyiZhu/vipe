# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .moge_v2 import MoGeV2Model, focal_length_to_fov_degrees
from .pi3x import Pi3XModel, normalize_intrinsics

try:
    from moge.utils.alignment import align_points_scale_z_shift
except ModuleNotFoundError:
    align_points_scale_z_shift = None


def mask_aware_nearest_resize_robust(
    mask: torch.Tensor,
    target_width: int,
    target_height: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    height, width = mask.shape
    device = mask.device
    yy, xx = torch.meshgrid(
        torch.arange(target_height, device=device, dtype=torch.float32),
        torch.arange(target_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    src_y = ((yy + 0.5) * (height / target_height) - 0.5).round().long().clamp(0, height - 1)
    src_x = ((xx + 0.5) * (width / target_width) - 0.5).round().long().clamp(0, width - 1)
    return (src_y, src_x), mask[src_y, src_x]


class Pi3XMoGeV2Model(DepthEstimationModel):
    def __init__(
        self,
        pi3x: Pi3XModel | None = None,
        moge: MoGeV2Model | None = None,
        pixel_limit: int = 255000,
    ) -> None:
        super().__init__()
        if align_points_scale_z_shift is None:
            raise RuntimeError("MoGe alignment utilities are not available; install MoGe before using 'pi3x_moge'.")
        self.pi3x = pi3x if pi3x is not None else Pi3XModel()
        self.moge = moge if moge is not None else MoGeV2Model()
        self.pixel_limit = pixel_limit

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def _get_resize_size(self, height: int, width: int) -> tuple[int, int]:
        if height * width <= self.pixel_limit:
            return height, width
        scale = math.sqrt(self.pixel_limit / float(width * height))
        target_w, target_h = width * scale, height * scale
        k, m = round(target_w / 14), round(target_h / 14)
        while (k * 14) * (m * 14) > self.pixel_limit:
            if k / max(m, 1) > target_w / max(target_h, 1):
                k -= 1
            else:
                m -= 1
        return max(1, m) * 14, max(1, k) * 14

    @torch.no_grad()
    def estimate_with_scale(
        self,
        src: DepthEstimationInput,
        moge_indices: Sequence[int] | None = None,
    ) -> tuple[DepthEstimationResult, list[float]]:
        rgb = src.rgb
        if rgb is None:
            return DepthEstimationResult(), []
        if rgb.dim() == 3:
            rgb = rgb[None]

        n_frames, orig_h, orig_w, _ = rgb.shape
        target_h, target_w = self._get_resize_size(orig_h, orig_w)
        target_h = ((target_h + 13) // 14) * 14
        target_w = ((target_w + 13) // 14) * 14
        while target_h * target_w > self.pixel_limit and (target_h > 14 or target_w > 14):
            if target_h > target_w:
                target_h -= 14
            else:
                target_w -= 14

        imgs = rearrange(rgb, "n h w c -> n c h w")
        imgs = F.interpolate(
            imgs,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).unsqueeze(0)

        intrinsics_full = normalize_intrinsics(src.intrinsics, n_frames)
        if intrinsics_full is None:
            raise ValueError("Pi3XMoGeV2Model requires intrinsics")
        intrinsics = intrinsics_full.clone()
        intrinsics[:, 0, :] *= target_w / orig_w
        intrinsics[:, 1, :] *= target_h / orig_h
        intrinsics = intrinsics.unsqueeze(0)

        poses = src.poses
        if poses is not None:
            if poses.dim() == 2:
                poses = poses[None]
            poses = poses.unsqueeze(0)

        pi3x_out = self.pi3x(imgs, intrinsics=intrinsics, poses=poses)
        pi3x_points = pi3x_out["local_points"][0]
        pi3x_conf = torch.sigmoid(pi3x_out["conf"][0, ..., 0]) > 0.1

        if moge_indices is None:
            moge_indices = list(range(n_frames))

        fx = intrinsics_full[0, 0, 0].item()
        fov_deg = focal_length_to_fov_degrees(fx, orig_w)
        moge_points = torch.zeros((n_frames, target_h, target_w, 3), device=rgb.device)
        moge_masks = torch.zeros((n_frames, target_h, target_w), device=rgb.device, dtype=torch.bool)
        imgs_moge = imgs[0]
        for start in range(0, len(moge_indices), 4):
            batch_indices = list(moge_indices[start : start + 4])
            result = self.moge.forward(imgs_moge[batch_indices], fov_x=fov_deg)
            points = result["points"]
            moge_points[batch_indices] = points
            mask = result.get("mask", torch.ones_like(points[..., 0])).bool()
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            moge_masks[batch_indices] = mask

        scales: list[float] = []
        for frame_idx in moge_indices:
            combined_mask = pi3x_conf[frame_idx] & moge_masks[frame_idx]
            if combined_mask.sum().item() < 100:
                continue
            indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, 64, 64)
            ni, nj = indices
            if lr_mask.sum().item() < 10:
                continue
            pi3x_lr = pi3x_points[frame_idx][ni, nj]
            moge_lr = moge_points[frame_idx][ni, nj]
            weights = 1.0 / moge_lr[..., 2].clamp(min=1e-3)
            scale, _ = align_points_scale_z_shift(
                pi3x_lr[lr_mask].unsqueeze(0),
                moge_lr[lr_mask].unsqueeze(0),
                weights[lr_mask].unsqueeze(0),
            )
            scale_value = scale.item()
            if scale_value > 0 and np.isfinite(scale_value):
                scales.append(scale_value)

        global_scale = float(np.median(scales)) if scales else 1.0
        pi3x_depth = pi3x_points[..., 2] * global_scale
        pi3x_depth = pi3x_depth * pi3x_conf.float()
        depth = F.interpolate(pi3x_depth.unsqueeze(1), size=(orig_h, orig_w), mode="nearest").squeeze(1)
        return DepthEstimationResult(metric_depth=depth), scales

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        result, _ = self.estimate_with_scale(src)
        return result
