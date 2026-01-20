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
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from einops import rearrange
import math

from .base import DepthEstimationModel, DepthEstimationInput, DepthEstimationResult, DepthType
from .pi3x import Pi3XModel
from .moge_v2 import MoGeV2Model, focal_length_to_fov_degrees
from vipe.utils.misc import unpack_optional
from vipe.utils.cameras import CameraType

try:
    from moge.utils.alignment import align_points_scale_z_shift
except ImportError:
    try:
        from vipe.thirdparty.moge.utils.alignment import align_points_scale_z_shift
    except ImportError:
        align_points_scale_z_shift = None

def mask_aware_nearest_resize_robust(mask, target_width, target_height):
    H, W = mask.shape
    device = mask.device
    y_rng = torch.arange(target_height, device=device).float()
    x_rng = torch.arange(target_width, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_rng, x_rng, indexing="ij")
    y_src = (y_grid + 0.5) * (H / target_height) - 0.5
    x_src = (x_grid + 0.5) * (W / target_width) - 0.5
    y_src = y_src.round().long().clamp(0, H - 1)
    x_src = x_src.round().long().clamp(0, W - 1)
    nearest_i = y_src
    nearest_j = x_src
    target_mask = mask[nearest_i, nearest_j]
    return (nearest_i, nearest_j), target_mask

class Pi3XMoGeV2Model(DepthEstimationModel):
    def __init__(self, pi3x: Optional[Pi3XModel] = None, moge: Optional[MoGeV2Model] = None, pixel_limit: int = 255000):
        super().__init__()
        self.pi3x = pi3x if pi3x is not None else Pi3XModel()
        self.moge = moge if moge is not None else MoGeV2Model()
        self.pixel_limit = pixel_limit

    @property
    def depth_type(self) -> DepthType:
        # This model conditions Pi3X on intrinsics/poses then aligns to MoGe2.
        # The resulting depth is not guaranteed to be focal-proportional, so we
        # force re-estimation when SLAM optimizes intrinsics.
        return DepthType.MODEL_METRIC_DEPTH

    def _get_resize_size(self, H, W):
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
    def estimate_with_scale(self, src: DepthEstimationInput, moge_indices: Optional[List[int]] = None) -> Tuple[DepthEstimationResult, List[float]]:
        rgb = src.rgb.clone()
        if rgb is None:
            return DepthEstimationResult(), []
            
        if rgb.dim() == 3:
            rgb = rgb[None]
            
        N, H_orig, W_orig, _ = rgb.shape
        H_t, W_t = self._get_resize_size(H_orig, W_orig)
        
        # Ensure dimensions are multiples of 14 for Pi3X
        H_t = ((H_t + 13) // 14) * 14
        W_t = ((W_t + 13) // 14) * 14
        
        while H_t * W_t > self.pixel_limit and (H_t > 14 or W_t > 14):
             if H_t > W_t: H_t -= 14
             else: W_t -= 14

        # Prepare Pi3X inputs
        imgs_t = rearrange(rgb, "n h w c -> n c h w")
        
        imgs_t = F.interpolate(imgs_t, size=(H_t, W_t), mode='bilinear', align_corners=False, antialias=True).unsqueeze(0) # (1, N, 3, H_t, W_t)
        
        # Adjust intrinsics
        intrinsics = src.intrinsics # (N, 3, 3) or (4,) or (N, 4)
        if intrinsics is None:
             raise ValueError("Pi3XMoGe2 needs intrinsics")
        
        if intrinsics.dim() == 1:
             # (4,) -> (N, 3, 3)
             K = torch.eye(3, device=rgb.device).repeat(N, 1, 1)
             K[:, 0, 0] = intrinsics[0]
             K[:, 1, 1] = intrinsics[1]
             K[:, 0, 2] = intrinsics[2]
             K[:, 1, 2] = intrinsics[3]
             intrinsics = K
        elif intrinsics.dim() == 2 and intrinsics.shape[1] == 4:
             # (N, 4) -> (N, 3, 3)
             K = torch.eye(3, device=rgb.device).repeat(N, 1, 1)
             K[:, 0, 0] = intrinsics[:, 0]
             K[:, 1, 1] = intrinsics[:, 1]
             K[:, 0, 2] = intrinsics[:, 2]
             K[:, 1, 2] = intrinsics[:, 3]
             intrinsics = K

        intrinsics_t = intrinsics.clone()
        intrinsics_t[:, 0] *= (W_t / W_orig)
        intrinsics_t[:, 1] *= (H_t / H_orig)
        intrinsics_t = intrinsics_t.unsqueeze(0) # (1, N, 3, 3)
        
        poses = src.poses # (N, 4, 4)
        if poses is not None:
            poses = poses.unsqueeze(0) # (1, N, 4, 4)
        
        # 1. Run Pi3X
        pi3x_out = self.pi3x(imgs_t, intrinsics=intrinsics_t, poses=poses)
        # pi3x_out = self.pi3x(imgs_t)
        pi3x_pts = pi3x_out['local_points'][0] # (N, H_t, W_t, 3)
        pi3x_conf = torch.sigmoid(pi3x_out['conf'][0, ..., 0]) > 0.1 # (N, H_t, W_t)
        
        # 2. Run MoGe2
        # If moge_indices is None, run on all. Otherwise only on selected.
        if moge_indices is None:
             moge_indices = list(range(N))
             
        # MoGe2 expects fov_x. Let's use first frame fov.
        fx = src.intrinsics[0].item()
        fov_deg = focal_length_to_fov_degrees(fx, W_orig)
        
        moge_pts_full = torch.zeros((N, H_t, W_t, 3), device=rgb.device)
        moge_masks_full = torch.zeros((N, H_t, W_t), device=rgb.device, dtype=torch.bool)
        
        bs = 4
        imgs_moge = rearrange(rgb, "n h w c -> n c h w")
        imgs_moge = F.interpolate(imgs_moge, size=(H_t, W_t), mode='bilinear', align_corners=False, antialias=True)
        
        for i in range(0, len(moge_indices), bs):
             batch_indices = moge_indices[i:i+bs]
             moge_out = self.moge.forward(imgs_moge[batch_indices], fov_x=fov_deg)
             moge_pts_full[batch_indices] = moge_out['points']
             moge_masks_full[batch_indices] = moge_out.get('mask', torch.ones_like(moge_out['points'][..., 0])).bool()
        
        # 3. Alignment
        all_scales = []
        if align_points_scale_z_shift is not None:
            for i in moge_indices:
                combined_mask = pi3x_conf[i] & moge_masks_full[i]
                if combined_mask.sum() < 100: continue
                
                indices, lr_mask = mask_aware_nearest_resize_robust(combined_mask, 64, 64)
                ni, nj = indices
                pts_pi3x_lr = pi3x_pts[i][ni, nj]
                pts_moge_lr = moge_pts_full[i][ni, nj]
                weights = 1.0 / pts_moge_lr[..., 2].clamp(min=1e-3)
                
                if lr_mask.sum() >= 10:
                    s, _ = align_points_scale_z_shift(
                        pts_pi3x_lr[lr_mask].unsqueeze(0),
                        pts_moge_lr[lr_mask].unsqueeze(0),
                        weights[lr_mask].unsqueeze(0)
                    )
                    if s.item() > 0:
                        all_scales.append(s.item())
        
        global_scale = np.median(all_scales) if all_scales else 1.0
        
        # Final depth
        pi3x_depth = pi3x_pts[..., 2] * global_scale

        # Masking
        pi3x_depth = pi3x_depth * pi3x_conf.float()
        
        # Interpolate back to original resolution
        final_depth = F.interpolate(pi3x_depth.unsqueeze(1), size=(H_orig, W_orig), mode='nearest').squeeze(1)
        
        return DepthEstimationResult(metric_depth=final_depth), all_scales

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        res, _ = self.estimate_with_scale(src)
        return res

