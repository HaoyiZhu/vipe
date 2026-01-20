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

import torch
import torch.nn as nn
from typing import Optional, Dict
from einops import rearrange

try:
    from pi3.models.pi3x import Pi3X
except ImportError:
    Pi3X = None

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


class Pi3XModel(nn.Module):
    def __init__(self, pretrained: str = "yyfz233/Pi3X"):
        super().__init__()
        if Pi3X is None:
            raise ImportError("Pi3X not found. Ensure vipe/thirdparty/pi3 is in sys.path")
        self.model = Pi3X.from_pretrained(pretrained)
        self.model.eval()
        self.model.to("cuda")

    def forward(
        self, 
        imgs: torch.Tensor, 
        intrinsics: Optional[torch.Tensor] = None, 
        poses: Optional[torch.Tensor] = None, 
        depths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        imgs: (B, N, 3, H, W)
        intrinsics: (B, N, 3, 3)
        poses: (B, N, 4, 4)
        depths: (B, N, H, W)
        """
        conditions = {}
        if intrinsics is not None:
            conditions['intrinsics'] = intrinsics
        if poses is not None:
            conditions['poses'] = poses
        if depths is not None:
            conditions['depths'] = depths
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast('cuda', dtype=dtype):
             return self.model(imgs=imgs, **conditions)


class Pi3XDepthModel(DepthEstimationModel):
    def __init__(self, pretrained: str = "yyfz233/Pi3X", pixel_limit: int = 255000) -> None:
        super().__init__()
        self.model = Pi3XModel(pretrained=pretrained)
        self.pixel_limit = pixel_limit

    @property
    def depth_type(self) -> DepthType:
        # Pi3X depth depends on intrinsics in a non-trivial way (not a simple focal-proportional scaling).
        # Mark as MODEL_METRIC_DEPTH so SLAM will re-run depth when intrinsics are optimized, instead of
        # using the fast focal scaling shortcut in GraphBuffer.update_disps_sens().
        return DepthType.MODEL_METRIC_DEPTH

    def _get_resize_size(self, H, W):
        import math
        if H * W <= self.pixel_limit:
            return H, W
        scale = math.sqrt(self.pixel_limit / (W * H))
        W_t, H_t = W * scale, H * scale
        k, m = round(W_t / 14), round(H_t / 14)
        while (k * 14) * (m * 14) > self.pixel_limit:
             if k / m > W_t / H_t: k -= 1
             else: m -= 1
        return max(1, m) * 14, max(1, k) * 14

    @torch.no_grad()
    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        # imgs: (N, H, W, 3) or (H, W, 3)
        imgs = src.rgb
        if imgs is None:
            return DepthEstimationResult()

        # Extract original height and width
        if imgs.dim() == 3:
            orig_h, orig_w = imgs.shape[0], imgs.shape[1]
        else:
            orig_h, orig_w = imgs.shape[1], imgs.shape[2]

        new_h, new_w = self._get_resize_size(orig_h, orig_w)
        # Ensure it's multiple of 14 for Pi3X
        new_h = ((new_h + 13) // 14) * 14
        new_w = ((new_w + 13) // 14) * 14
        
        # Double check it doesn't exceed pixel limit after rounding up
        while new_h * new_w > self.pixel_limit and (new_h > 14 or new_w > 14):
            if new_h > new_w: new_h -= 14
            else: new_w -= 14

        if imgs.dim() == 3:
            imgs = imgs[None, None] # (1, 1, H, W, 3)
        elif imgs.dim() == 4:
            imgs = imgs[None] # (1, N, H, W, 3)
        
        imgs = rearrange(imgs, "b n h w c -> (b n) c h w")
        if new_h != orig_h or new_w != orig_w:
            imgs = torch.nn.functional.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        imgs = rearrange(imgs, "(b n) c h w -> b n c h w", b=1)

        intrinsics = src.intrinsics
        if intrinsics is not None:
            if intrinsics.dim() == 2:
                intrinsics = intrinsics[None, None] # (1, 1, 3, 3)
            elif intrinsics.dim() == 3:
                intrinsics = intrinsics[None] # (1, N, 3, 3)
            
            if new_h != orig_h or new_w != orig_w:
                intrinsics = intrinsics.clone()
                intrinsics[..., 0, 0] *= (new_w / orig_w)
                intrinsics[..., 1, 1] *= (new_h / orig_h)
                intrinsics[..., 0, 2] *= (new_w / orig_w)
                intrinsics[..., 1, 2] *= (new_h / orig_h)

        poses = src.poses
        if poses is not None:
            if poses.dim() == 2:
                poses = poses[None, None] # (1, 1, 4, 4)
            elif poses.dim() == 3:
                poses = poses[None] # (1, N, 4, 4)

        out = self.model(imgs=imgs, intrinsics=intrinsics, poses=poses)
        
        # Pi3X returns a dict with 'local_points' which is (B, N, H, W, 3)
        local_points = out.get("local_points")
        if local_points is not None:
             # Metric depth is the Z component of local points
             metric_depth = local_points[..., 2] # (B, N, H, W)
             
             # Calculate confidence mask
             if "conf" in out:
                 conf = torch.sigmoid(out["conf"][..., 0])
                 mask = conf > 0.1
                 metric_depth = metric_depth * mask.float()
        else:
             metric_depth = None
        
        if metric_depth is not None:
            if new_h != orig_h or new_w != orig_w:
                metric_depth = torch.nn.functional.interpolate(
                    metric_depth, size=(orig_h, orig_w), mode="nearest"
                )

            # Squeeze batch dimension
            metric_depth = metric_depth.squeeze(0) # (N, H, W)
            if src.rgb.dim() == 3:
                metric_depth = metric_depth.squeeze(0) # (H, W)

        return DepthEstimationResult(metric_depth=metric_depth)
