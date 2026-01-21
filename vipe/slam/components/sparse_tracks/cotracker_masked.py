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

from __future__ import annotations

from vipe.streams.base import VideoFrame

from .cotracker import CoTrackerSparseTracks


class CoTrackerSparseTracksMasked(CoTrackerSparseTracks):
    """
    CoTracker3 variant that interprets frame.mask as valid=1, invalid=0.
    This keeps masking semantics consistent with the rest of VIPE.
    """

    def track_image(self, frame_data_list: list[VideoFrame]) -> None:
        if self._precomputed:
            return
        # Buffer frames for offline tracking. Still append empty observations.
        for view_idx, frame in enumerate(frame_data_list):
            self._frames[view_idx].append(frame.rgb.detach().cpu())
            if self.valid_mask_only and frame.mask is not None:
                valid_mask = frame.mask.bool().detach().cpu()
            else:
                valid_mask = None
            self._masks[view_idx].append(valid_mask)
            self.observations[view_idx].append({})

