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
# -------------------------------------------------------------------------------------------------
# This file includes code originally from the DROID-SLAM repository:
# https://github.com/cvg/DROID-SLAM
# Licensed under the MIT License. See THIRD_PARTY_LICENSES.md for details.
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass

import torch

from omegaconf import DictConfig

from vipe.ext import lietorch as lt
from vipe.ext.lietorch import SE3

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


@dataclass
class FilledReturn:
    poses: SE3  # Inverse of c2w
    dense_disps: torch.Tensor | None = None
    intrinsics: torch.Tensor | None = None  # (N, V, D) for per-frame intrinsics

    def scale(self, factor: float):
        self.poses.data[..., :3] *= factor
        if self.dense_disps is not None:
            self.dense_disps /= factor


class InnerFiller:
    """This class is used to fill in non-keyframe poses (and intrinsics when per_frame_intrinsics is enabled)"""

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.video = video
        self.net = net
        self.device = device
        self.start_idx = -1
        self.args = args

        self.filled_poses = []
        self.filled_dense_disps = []
        self.filled_intrinsics = []

    def set_start_idx(self, start_idx: int):
        self.start_idx = start_idx

    def check(self) -> bool:
        assert self.start_idx >= 0
        return self.video.n_frames - self.start_idx >= self.args.infill_chunk_size

    def compute(self):
        total_frames = self.video.n_frames

        # Setup initial value (for pose and disp)
        m_tstamp = self.video.tstamp[self.start_idx : total_frames]
        n_tstamp = self.video.tstamp[: self.start_idx]

        # Find left (inclusive) nearest keyframe
        t0 = torch.searchsorted(n_tstamp, m_tstamp, right=True) - 1
        t1 = torch.where(t0 < self.start_idx - 1, t0 + 1, t0)

        d_time = n_tstamp[t1] - n_tstamp[t0] + 1e-3  # Avoid if time is out of bound of kfs
        n_pose = SE3(self.video.poses[: self.start_idx])
        d_pose = n_pose[t1] * n_pose[t0].inv()
        vel = d_pose.log() / d_time.unsqueeze(-1)
        w = vel * (m_tstamp - n_tstamp[t0]).unsqueeze(-1)
        m_pose = SE3.exp(w) * n_pose[t0]

        self.video.poses[self.start_idx : total_frames] = m_pose.data

        if self.args.infill_dense_disp:
            self.video.disps[self.start_idx : total_frames] = self.video.disps[t0].mean(dim=[2, 3], keepdim=True)
            self.video.disps[self.start_idx : total_frames] = torch.where(
                self.video.disps_sens[self.start_idx : total_frames] > 0,
                self.video.disps_sens[self.start_idx : total_frames],
                self.video.disps[self.start_idx : total_frames],
            )

        # Interpolate intrinsics for per-frame intrinsics mode
        if self.video.per_frame_intrinsics:
            # Linear interpolation of intrinsics between keyframes
            n_intrinsics = self.video.intrinsics[: self.start_idx]  # (N_kf, V, D)
            interp_weight = ((m_tstamp - n_tstamp[t0]) / d_time).unsqueeze(-1).unsqueeze(-1)  # (M, 1, 1)
            m_intrinsics = (1 - interp_weight) * n_intrinsics[t0] + interp_weight * n_intrinsics[t1]
            self.video.intrinsics[self.start_idx : total_frames] = m_intrinsics

        # Build factor graph and optimize for the interpolated information.
        graph = FactorGraph(
            self.net,
            self.video,
            self.device,
            max_factors=-1,
            incremental=True,
            cross_view=False,  # No need for interpolation.
            debug=self.args.get("debug", None),
        )
        infill_inds = torch.arange(self.start_idx, total_frames).to(self.device)
        graph.add_factors(t0, infill_inds)
        graph.add_factors(t1, infill_inds)
        if self.args.infill_dense_disp:
            graph.add_factors(infill_inds, t0)
            graph.add_factors(infill_inds, t1)

        # Determine if we should optimize intrinsics during infill
        optimize_intrinsics_in_infill = (
            self.video.per_frame_intrinsics and 
            self.args.get("per_frame_intrinsics", False)
        )

        for _ in range(10):
            graph.update(
                self.start_idx,
                total_frames,
                motion_only=not self.args.infill_dense_disp,
                limited_disp=True,
                optimize_intrinsics=optimize_intrinsics_in_infill,
            )

        # (Optional) Metric computation of keyframe optimized disp and its original disp.
        # This will reflect the stability of the optimization.
        # m_kf_mask = n_tstamp[t0] == m_tstamp
        # m_kf_disps = self.video.disps[self.start_idx : total_frames][m_kf_mask]
        # n_kf_disps = self.video.disps[: self.start_idx][t0[m_kf_mask]]
        # print("Disparity diff", torch.mean(torch.abs(m_kf_disps - n_kf_disps)))

        current_poses = SE3(self.video.poses[self.start_idx : total_frames].clone())
        self.filled_poses.append(current_poses)

        if self.args.infill_dense_disp:
            current_dense_disps = self.video.disps[self.start_idx : total_frames].clone()
            self.filled_dense_disps.append(current_dense_disps)

        # Store filled intrinsics for per-frame mode
        if self.video.per_frame_intrinsics:
            current_intrinsics = self.video.intrinsics[self.start_idx : total_frames].clone()
            self.filled_intrinsics.append(current_intrinsics)

        self.video.n_frames = self.start_idx

    def get_result(self) -> FilledReturn:
        return FilledReturn(
            poses=lt.cat(self.filled_poses, dim=0),
            dense_disps=(torch.cat(self.filled_dense_disps, dim=0) if self.filled_dense_disps else None),
            intrinsics=(torch.cat(self.filled_intrinsics, dim=0) if self.filled_intrinsics else None),
        )
