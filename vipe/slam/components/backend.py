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

import logging

import numpy as np
import torch
from einops import rearrange

from omegaconf import DictConfig

from vipe.priors.depth import DepthEstimationModel

from vipe.ext.lietorch import SE3
from ..ba.kernel import HuberRobustKernel
from ..ba.terms import DenseDepthFlowTerm, DispSensRegularizationTerm
from ..networks.droid_net import DroidNet
from .buffer import GraphBuffer
from .factor_graph import FactorGraph

logger = logging.getLogger(__name__)


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
        self.last_ba_residuals_per_frame: list[float] | None = None

    @torch.no_grad()
    def _compute_ba_residuals_per_frame(self, graph: FactorGraph) -> list[float] | None:
        if graph.ii.numel() == 0:
            return None

        ht = self.video.height // 8
        wd = self.video.width // 8
        n_views = self.video.n_views

        ii = graph.ii
        jj = graph.jj
        if ii.numel() == 0:
            return None

        target = rearrange(graph.target, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)
        weight = rearrange(graph.weight, "1 k h w c -> k (h w) c", c=2, h=ht, w=wd)

        weight_dense_disp = float(getattr(self.video.ba_config, "weight_dense_disp", 0.001))
        weight_tracks = float(getattr(self.video.ba_config, "weight_tracks", 0.001))
        sparse_tracks_gap_thresh = int(getattr(self.video.ba_config, "sparse_tracks_gap_thresh", 10))

        frame_gaps = torch.abs(ii - jj).float()
        frame_gaps_exp = frame_gaps.unsqueeze(-1).repeat(1, n_views).reshape(-1)
        dense_scale = torch.where(
            frame_gaps_exp < sparse_tracks_gap_thresh,
            torch.ones_like(frame_gaps_exp),
            torch.full_like(frame_gaps_exp, 0.1),
        )
        sparse_scale = torch.where(
            frame_gaps_exp < sparse_tracks_gap_thresh,
            torch.full_like(frame_gaps_exp, 0.1),
            torch.ones_like(frame_gaps_exp),
        )

        scaled_dense_weight = weight_dense_disp * weight * dense_scale.view(-1, 1, 1)

        pi, qi, di, pj, qj, _ = self.video.expand_edge_multiview(ii, jj)
        disps_flattened = rearrange(self.video.flattened_disps, "nv h w -> nv (h w)")
        variables = {
            "pose": SE3(self.video.poses),
            "dense_disp": disps_flattened,
            "intrinsics": self.video.intrinsics,
            "rig": SE3(self.video.rig),
        }

        dense_term = DenseDepthFlowTerm(
            pose_i_inds=pi,
            pose_j_inds=pj,
            rig_i_inds=qi,
            rig_j_inds=qj,
            dense_disp_i_inds=di,
            target=target,
            weight=scaled_dense_weight,
            intrinsics=None,
            intrinsics_factor=8.0,
            rig=None,
            image_size=(ht, wd),
            camera_type=self.video.camera_type,
        )
        dense_ret = dense_term.forward(variables, jacobian=False)
        dense_energy = dense_ret.residual()

        n_frames = int(self.video.n_frames)
        energy = torch.zeros(n_frames, device=self.device)
        elements = torch.zeros(n_frames, device=self.device)
        dense_elements = float(ht * wd)
        half_energy = 0.5 * dense_energy
        half_elements = 0.5 * dense_elements

        energy.index_add_(0, pi, half_energy)
        energy.index_add_(0, pj, half_energy)
        elements_add = torch.full_like(pi, half_elements, dtype=energy.dtype)
        elements.index_add_(0, pi, elements_add)
        elements.index_add_(0, pj, elements_add)

        if self.video.sparse_tracks.enabled:
            sparse_target, sparse_weight = self.video.sparse_tracks.compute_dense_disp_target_weight(
                source_view_inds=qi,
                source_frame_inds=self.video.tstamp[pi],
                target_view_inds=qj,
                target_frame_inds=self.video.tstamp[pj],
                image_size=(self.video.height, self.video.width),
                dense_disp_size=(ht, wd),
            )
            sparse_target = sparse_target.flatten(1, 2)
            sparse_weight = sparse_weight.flatten(1, 2)
            scaled_sparse_weight = weight_tracks * sparse_weight * sparse_scale.view(-1, 1, 1)
            sparse_term = DenseDepthFlowTerm(
                pose_i_inds=pi,
                pose_j_inds=pj,
                rig_i_inds=qi,
                rig_j_inds=qj,
                dense_disp_i_inds=di,
                target=sparse_target,
                weight=scaled_sparse_weight,
                intrinsics=None,
                intrinsics_factor=8.0,
                rig=None,
                image_size=(ht, wd),
                camera_type=self.video.camera_type,
            )
            sparse_ret = sparse_term.forward(variables, jacobian=False)
            sparse_ret.apply_robust_kernel(HuberRobustKernel())
            sparse_energy = sparse_ret.residual()
            half_sparse_energy = 0.5 * sparse_energy
            energy.index_add_(0, pi, half_sparse_energy)
            energy.index_add_(0, pj, half_sparse_energy)

        disps_sens = rearrange(self.video.flattened_disps_sens, "nv h w -> nv (h w)")
        di_unique = torch.unique(di)
        sens_i_inds = di_unique[disps_sens[di_unique].sum(1) > 0.0]
        if len(sens_i_inds) > 0:
            reg_term = DispSensRegularizationTerm(
                i_inds=sens_i_inds,
                alpha=float(self.video.ba_config.dense_disp_alpha),
                disps_sens=disps_sens,
            )
            reg_ret = reg_term.forward({"dense_disp": disps_flattened}, jacobian=False)
            reg_energy = reg_ret.residual()
            frame_inds = sens_i_inds // n_views
            energy.index_add_(0, frame_inds, reg_energy)

        residuals = torch.zeros_like(energy)
        valid = elements > 0
        residuals[valid] = torch.sqrt(energy[valid] / (elements[valid] + 1e-6)) / 16.0

        tstamp = self.video.tstamp[: self.video.n_frames].detach().cpu().numpy().astype(np.int64)
        residuals_np = residuals.detach().cpu().numpy().astype(np.float32)
        if tstamp.size == 0:
            return None
        order = np.argsort(tstamp)
        tstamp = tstamp[order]
        residuals_np = residuals_np[order]
        tstamp_unique, unique_idx = np.unique(tstamp, return_index=True)
        residuals_np = residuals_np[unique_idx]
        total_frames = int(tstamp_unique[-1]) + 1
        frame_idx = np.arange(total_frames, dtype=np.int64)
        residuals_full = np.interp(frame_idx, tstamp_unique, residuals_np).astype(np.float32)
        return residuals_full.tolist()

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
        self.last_ba_residuals_per_frame = None
        if self.args.get("compute_ba_residuals_per_frame", False):
            try:
                self.last_ba_residuals_per_frame = self._compute_ba_residuals_per_frame(graph)
            except Exception:
                logger.exception("Failed to compute per-frame BA residuals; skipping.")

        if log:
            self.video.log(self.args.map_filter_thresh)
            graph.log()

    @torch.no_grad()
    def run_if_necessary(self, steps: int = 12, log: bool = False):
        if self.args.optimize_intrinsics or self.args.optimize_rig_rotation:
            self.run(steps=steps, update_depth=True, log=log)
