# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

try:
    from moge.model.v2 import MoGeModel
except ModuleNotFoundError:
    MoGeModel = None

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


def focal_length_to_fov_degrees(focal_length: float, image_width: float) -> float:
    fov_rad = 2 * torch.atan(torch.tensor(image_width / (2 * focal_length)))
    return torch.rad2deg(fov_rad).item()


class MoGeV2Model(DepthEstimationModel):
    """MoGe-2 metric depth wrapper."""

    def __init__(self, pretrained: str = "Ruicheng/moge-2-vitl") -> None:
        super().__init__()
        if MoGeModel is None:
            raise RuntimeError(
                "MoGe-2 is not available. Install MoGe or add it to PYTHONPATH before using the 'moge2' depth model."
            )
        self.model = MoGeModel.from_pretrained(pretrained).cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    @torch.no_grad()
    def forward(self, image: torch.Tensor, fov_x: float | None = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        dtype = torch.float16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        with torch.amp.autocast("cuda", dtype=dtype):
            return self.model.infer(image, fov_x=fov_x, use_fp16=(dtype == torch.float16), **kwargs)

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "MoGe-2 only supports pinhole cameras"

        focal_length = unpack_optional(src.intrinsics)[0].item()
        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        image = rgb.permute(0, 3, 1, 2) if rgb.shape[-1] == 3 else rgb
        result = self.forward(image, fov_x=focal_length_to_fov_degrees(focal_length, rgb.shape[2]))

        depth = torch.nan_to_num(result["depth"], nan=1e4).clamp(min=0, max=1e4)
        mask = result.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            depth = depth * mask.float()

        if not batch_dim:
            depth = depth.squeeze(0)
        return DepthEstimationResult(metric_depth=depth)
