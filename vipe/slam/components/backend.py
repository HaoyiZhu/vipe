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

import torch

from omegaconf import DictConfig

from vipe.priors.depth import DepthEstimationModel

from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph


class SLAMBackend:
    """
    Mainly used to run a pretty dense bundle adjustment for all the frames in the graph.
    """

    depth_model: DepthEstimationModel | None = None

    def __init__(self, net: DroidNet, video: GraphBuffer, args: DictConfig, device: torch.device):
        self.net = net
        self.video = video
        self.args = args
        self.device = device
        self.last_graph: torch.Tensor | None = None

    def _iterate_with_depth(self, graph: FactorGraph, steps: int, more_iters: bool):
        steps_preintr = steps // 2
        steps_postintr = steps - steps_preintr
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps_preintr,
            optimize_intrinsics=self.args.optimize_intrinsics,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )
        self.video.update_disps_sens(self.depth_model, frame_idx=None)
        # Don't update intrinsics again!
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps_postintr,
            optimize_intrinsics=False,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )

    def _iterate_without_depth(self, graph: FactorGraph, steps: int, more_iters: bool):
        graph.update_batch(
            itrs=16 if more_iters else 8,
            steps=steps,
            optimize_intrinsics=self.args.optimize_intrinsics,
            optimize_rig_rotation=self.args.optimize_rig_rotation,
            solver_verbose=True,
        )

    @torch.no_grad()
    def run(self, steps: int = 12, update_depth: bool = True, log: bool = False):
        """main update (reset GRU state)"""

        t = self.video.n_frames

        graph = FactorGraph(
            self.net,
            self.video,
            self.device,
            max_factors=16 * t,
            incremental=False,
            cross_view=self.args.cross_view,
            debug=self.args.get("debug", None),
        )

        graph.add_proximity_factors(
            rad=self.args.backend_radius,
            nms=self.args.backend_nms,
            thresh=self.args.backend_thresh,
            beta=self.args.beta,
        )

        if self.video.sparse_tracks.enabled:
            # Get long-range edges from sparse tracks (only for frames far apart)
            min_frame_gap = int(self.args.get("sparse_tracks_min_gap", 50))
            max_sparse_edges = int(self.args.get("sparse_tracks_max_edges", 100))
            ii_sparse, jj_sparse = self.video.sparse_tracks.get_overlapping_pairs(
                min_common=15, min_frame_gap=min_frame_gap, max_pairs=max_sparse_edges
            )
            if ii_sparse.numel() > 0:
                # Filter out pairs that are out of bounds of current video buffer
                # Backend might be called when only a subset of frames are in the buffer (during frontend run_if_necessary)
                valid_mask = (ii_sparse < t) & (jj_sparse < t)
                if valid_mask.any():
                    graph.add_factors(
                        ii_sparse[valid_mask].to(self.device),
                        jj_sparse[valid_mask].to(self.device),
                        remove=False
                    )

        if self.args.adaptive_cross_view:
            self.video.build_adaptive_cross_view_idx()

        if len(graph.ii) > 0:
            more_iters = self.args.optimize_intrinsics or self.args.optimize_rig_rotation
            if self.depth_model is not None:
                self._iterate_with_depth(graph, steps, more_iters)
            else:
                self._iterate_without_depth(graph, steps, more_iters)
        else:
            # Empty graph with only one keyframe, assign sensor depth
            self.video.disps[0] = torch.where(
                self.video.disps_sens[0] > 0,
                self.video.disps_sens[0],
                self.video.disps[0],
            )

        self.video.dirty[:t] = True
        self.last_graph = torch.stack([graph.ii, graph.jj], dim=-1)

        if log:
            self.video.log(self.args.map_filter_thresh)
            graph.log()

    @torch.no_grad()
    def run_if_necessary(self, steps: int = 12, log: bool = False):
        if self.args.optimize_intrinsics or self.args.optimize_rig_rotation:
            self.run(steps=steps, update_depth=True, log=log)
