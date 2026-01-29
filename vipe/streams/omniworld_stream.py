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

"""
OmniWorld dataset stream with GT depth support.

OmniWorld data structure (after extraction):
    <scene_id>/
    ├─ color/              # RGB frames: 000000.png, 000001.png, ...
    ├─ depth/              # 16-bit depth: 000000.png, 000001.png, ...
    ├─ camera/             # split_0.json, split_1.json, ... (intrinsics + extrinsics)
    └─ split_info.json     # frame grouping: {"split": [[0,1,2,...], [316,317,...], ...]}
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imageio.v2
import numpy as np
import torch

from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream

logger = logging.getLogger(__name__)


def load_omniworld_depth(depthpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 16-bit depth PNG and convert to metric depth (scale-invariant).
    
    Returns
    -------
    depthmap : (H, W) float32, depth values (scale-invariant, needs metric_scale applied)
    valid : (H, W) bool, True for reliable pixels
    """
    if not depthpath.exists():
        raise FileNotFoundError(f"Depth file not found: {depthpath}")
    
    depthmap = imageio.v2.imread(str(depthpath)).astype(np.float32) / 65535.0
    near_mask = depthmap < 0.0015  # too close / invalid
    far_mask = depthmap > (65500.0 / 65535.0)  # sky / too far
    near, far = 1.0, 1000.0
    
    # Avoid division by zero
    denominator = far - depthmap * (far - near)
    denominator[denominator == 0] = 1e-6
    depthmap = depthmap / denominator / 0.004
    
    valid = ~(near_mask | far_mask)
    depthmap[~valid] = -1
    
    return depthmap, valid


def load_split_info(scene_dir: Path) -> dict:
    """Load split_info.json for a scene."""
    with open(scene_dir / "split_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_data(scene_dir: Path, split_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-frame intrinsics and extrinsics for a split.
    
    Returns
    -------
    intrinsics : (S, 3, 3) array, pixel-space K matrices
    extrinsics : (S, 4, 4) array, OpenCV world-to-camera matrices
    """
    from scipy.spatial.transform import Rotation as R
    
    split_info = load_split_info(scene_dir)
    frame_count = len(split_info["split"][split_idx])
    
    cam_file = scene_dir / "camera" / f"split_{split_idx}.json"
    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)
    
    # Intrinsics
    intrinsics = np.repeat(np.eye(3)[None, ...], frame_count, axis=0)
    
    focals = cam["focals"]
    if isinstance(focals, list):
        intrinsics[:, 0, 0] = np.array(focals)  # fx
        intrinsics[:, 1, 1] = np.array(focals)  # fy
    else:
        intrinsics[:, 0, 0] = focals
        intrinsics[:, 1, 1] = focals
    
    cxs, cys = cam["cx"], cam["cy"]
    intrinsics[:, 0, 2] = np.array(cxs) if isinstance(cxs, list) else cxs
    intrinsics[:, 1, 2] = np.array(cys) if isinstance(cys, list) else cys
    
    # Extrinsics
    extrinsics = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    
    if "quats" in cam and "trans" in cam:
        quat_wxyz = np.array(cam["quats"])  # (S, 4) (w, x, y, z)
        quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)
        rotations = R.from_quat(quat_xyzw).as_matrix()  # (S, 3, 3)
        translations = np.array(cam["trans"])  # (S, 3)
        extrinsics[:, :3, :3] = rotations
        extrinsics[:, :3, 3] = translations
    
    return intrinsics.astype(np.float32), extrinsics.astype(np.float32)


class OmniWorldGTDepthStream(VideoStream):
    """
    A video stream from OmniWorld dataset with GT depth.
    
    Loads RGB from color/ and depth from depth/, applies metric_scale to depth.
    """

    def __init__(
        self,
        scene_dir: Path,
        split_idx: int,
        metric_scale: float = 1.0,
        frame_start: int = 0,
        frame_end: int = -1,
        frame_skip: int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__()
        
        self.scene_dir = Path(scene_dir)
        self.split_idx = split_idx
        self.metric_scale = metric_scale
        
        # Load split info to get frame indices
        split_info = load_split_info(self.scene_dir)
        self.global_frame_indices = split_info["split"][split_idx]
        
        self._name = name if name is not None else f"{self.scene_dir.name}_split{split_idx}"
        
        # Determine frame range
        total_frames = len(self.global_frame_indices)
        self.start = frame_start
        self.end = frame_end if frame_end != -1 else total_frames
        self.end = min(self.end, total_frames)
        self.step = frame_skip
        
        # Build list of frames to process
        self.frame_list = list(range(self.start, self.end, self.step))
        
        if not self.frame_list:
            raise ValueError(f"No frames to process in range [{self.start}, {self.end}) with step {self.step}")
        
        # Read metadata from first frame
        first_global_idx = self.global_frame_indices[self.frame_list[0]]
        first_frame_path = self.scene_dir / "color" / f"{first_global_idx:06d}.png"
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
        
        self._height, self._width = first_frame.shape[:2]
        
        # Try to read FPS from fps.txt
        fps_file = self.scene_dir / "fps.txt"
        if fps_file.exists():
            with open(fps_file) as f:
                first_line = f.readline().strip()
                # Handle format "FPS: 24.0" or just "24.0"
                if first_line.startswith("FPS:"):
                    self._fps = float(first_line.split(":")[-1].strip())
                else:
                    self._fps = float(first_line)
        else:
            self._fps = 30.0
        
        self._fps = self._fps / self.step
        
        logger.info(f"OmniWorldGTDepthStream: {len(self)} frames, metric_scale={self.metric_scale:.4f}")

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self.frame_list)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self) -> VideoFrame:
        if self._iter_idx >= len(self.frame_list):
            raise StopIteration
        
        local_idx = self.frame_list[self._iter_idx]
        global_idx = self.global_frame_indices[local_idx]
        self._iter_idx += 1
        
        # Load RGB
        rgb_path = self.scene_dir / "color" / f"{global_idx:06d}.png"
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {rgb_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()
        
        # Load depth
        depth_path = self.scene_dir / "depth" / f"{global_idx:06d}.png"
        metric_depth = None
        if depth_path.exists():
            depth, valid = load_omniworld_depth(depth_path)
            # Apply metric scale
            depth = depth * self.metric_scale
            depth[~valid] = -1
            metric_depth = torch.as_tensor(depth).float().cuda()
        
        return VideoFrame(
            raw_frame_idx=local_idx,
            rgb=frame_rgb,
            metric_depth=metric_depth,
        )


class OmniWorldGTDepthStreamList(StreamList):
    """
    StreamList for OmniWorld scenes with GT depth.
    
    Can process a single scene/split or multiple scenes.
    """
    
    def __init__(
        self,
        base_path: str,
        scene_id: str,
        split_idx: int = 0,
        metric_scale: float = 1.0,
        frame_start: int = 0,
        frame_end: int = -1,
        frame_skip: int = 1,
        cached: bool = False,
    ) -> None:
        super().__init__()
        
        self.base_path = Path(base_path)
        self.scene_id = scene_id
        self.split_idx = split_idx
        self.metric_scale = metric_scale
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.frame_skip = frame_skip
        self.cached = cached
        
        # Scene directory
        self.scene_dir = self.base_path / scene_id
        if not self.scene_dir.exists():
            raise ValueError(f"Scene directory not found: {self.scene_dir}")

    def __len__(self) -> int:
        return 1  # Single scene/split

    def __getitem__(self, index: int) -> VideoStream:
        if index != 0:
            raise IndexError(f"Index {index} out of range for single-scene stream")
        
        stream: VideoStream = OmniWorldGTDepthStream(
            scene_dir=self.scene_dir,
            split_idx=self.split_idx,
            metric_scale=self.metric_scale,
            frame_start=self.frame_start,
            frame_end=self.frame_end,
            frame_skip=self.frame_skip,
        )
        
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading frames", online=False)
        
        return stream

    def stream_name(self, index: int) -> str:
        return f"{self.scene_id}_split{self.split_idx}"


def get_scene_total_frames(scene_dir: Path) -> int:
    """Get total number of frames in a scene by counting color images."""
    color_dir = scene_dir / "color"
    if not color_dir.exists():
        return 0
    return len(list(color_dir.glob("*.png")))


class OmniWorldSlidingWindowStream(VideoStream):
    """
    A video stream for a sliding window chunk of an OmniWorld scene.
    
    Unlike OmniWorldGTDepthStream, this stream ignores split boundaries and
    loads frames directly by global frame index. This is useful for annotating
    all frames in a scene with overlapping sliding windows.
    """

    def __init__(
        self,
        scene_dir: Path,
        frame_start: int,
        frame_end: int,
        metric_scale: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        
        self.scene_dir = Path(scene_dir)
        self.metric_scale = metric_scale
        self.frame_start = frame_start
        self.frame_end = frame_end
        
        # Get total frames in scene
        total_frames = get_scene_total_frames(self.scene_dir)
        if total_frames == 0:
            raise ValueError(f"No frames found in {self.scene_dir / 'color'}")
        
        # Clamp frame_end to actual total
        self.frame_end = min(self.frame_end, total_frames)
        
        # Build list of global frame indices to process
        self.frame_list = list(range(self.frame_start, self.frame_end))
        
        if not self.frame_list:
            raise ValueError(
                f"No frames to process in range [{self.frame_start}, {self.frame_end})"
            )
        
        # Default name: scene_id_framesSTART_END
        if name is not None:
            self._name = name
        else:
            self._name = f"{self.scene_dir.name}_frames{self.frame_start:04d}_{self.frame_end:04d}"
        
        # Read metadata from first frame
        first_frame_path = self.scene_dir / "color" / f"{self.frame_list[0]:06d}.png"
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
        
        self._height, self._width = first_frame.shape[:2]
        
        # Try to read FPS from fps.txt
        fps_file = self.scene_dir / "fps.txt"
        if fps_file.exists():
            with open(fps_file) as f:
                first_line = f.readline().strip()
                if first_line.startswith("FPS:"):
                    self._fps = float(first_line.split(":")[-1].strip())
                else:
                    self._fps = float(first_line)
        else:
            self._fps = 30.0
        
        logger.info(
            f"OmniWorldSlidingWindowStream: {len(self)} frames "
            f"[{self.frame_start}, {self.frame_end}), metric_scale={self.metric_scale:.4f}"
        )

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self.frame_list)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self) -> VideoFrame:
        if self._iter_idx >= len(self.frame_list):
            raise StopIteration
        
        global_idx = self.frame_list[self._iter_idx]
        self._iter_idx += 1
        
        # Load RGB
        rgb_path = self.scene_dir / "color" / f"{global_idx:06d}.png"
        frame = cv2.imread(str(rgb_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {rgb_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()
        
        # Load depth
        depth_path = self.scene_dir / "depth" / f"{global_idx:06d}.png"
        metric_depth = None
        if depth_path.exists():
            depth, valid = load_omniworld_depth(depth_path)
            # Apply metric scale
            depth = depth * self.metric_scale
            depth[~valid] = -1
            metric_depth = torch.as_tensor(depth).float().cuda()
        
        return VideoFrame(
            raw_frame_idx=global_idx,
            rgb=frame_rgb,
            metric_depth=metric_depth,
        )


class OmniWorldSlidingWindowStreamList(StreamList):
    """
    StreamList that generates sliding window chunks across OmniWorld scenes.
    
    For each scene, generates overlapping chunks of frames:
    - Chunk 0: [0, window_size)
    - Chunk 1: [stride, stride + window_size)
    - Chunk 2: [2*stride, 2*stride + window_size)
    - ...
    
    Where stride = window_size - overlap.
    
    The last chunk may be shorter than window_size if there aren't enough frames.
    """
    
    def __init__(
        self,
        base_path: str,
        scenes: List[str] | None = None,
        window_size: int = 960,
        overlap: int = 480,
        cached: bool = False,
    ) -> None:
        super().__init__()
        
        self.base_path = Path(base_path)
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.cached = cached
        
        # Discover scenes if not provided
        if scenes is None:
            scenes = self._discover_scenes()
        
        self.scenes = scenes
        
        # Build list of all chunks: (scene_dir, frame_start, frame_end, chunk_name)
        self.chunks: List[Tuple[Path, int, int, str]] = []
        self._build_chunks()
        
        logger.info(
            f"OmniWorldSlidingWindowStreamList: {len(self.scenes)} scenes, "
            f"{len(self.chunks)} chunks (window={window_size}, overlap={overlap})"
        )
    
    def _discover_scenes(self) -> List[str]:
        """Discover all valid scenes in base_path."""
        scenes = []
        for scene_dir in sorted(self.base_path.iterdir()):
            if not scene_dir.is_dir():
                continue
            # Check if scene has color directory with frames
            color_dir = scene_dir / "color"
            if color_dir.exists() and any(color_dir.glob("*.png")):
                scenes.append(scene_dir.name)
        return scenes
    
    def _build_chunks(self) -> None:
        """Build list of all chunks across all scenes.
        
        For scenes with multiple chunks, ensures all chunks have exactly window_size
        frames by adjusting the last chunk's start position backwards if needed.
        
        Example with window_size=960, overlap=480, total_frames=1240:
        - Naive: chunk0=[0,960), chunk1=[480,1240) <- chunk1 has only 760 frames
        - Fixed: chunk0=[0,960), chunk1=[280,1240) <- both have 960 frames
        """
        for scene_id in self.scenes:
            scene_dir = self.base_path / scene_id
            total_frames = get_scene_total_frames(scene_dir)
            
            if total_frames == 0:
                logger.warning(f"Scene {scene_id} has no frames, skipping")
                continue
            
            # Special case: scene has fewer frames than window_size
            if total_frames <= self.window_size:
                # Single chunk with all available frames
                if total_frames >= 10:  # Skip if too small
                    chunk_name = f"{scene_id}_frames{0:04d}_{total_frames:04d}"
                    self.chunks.append((scene_dir, 0, total_frames, chunk_name))
                continue
            
            # Generate sliding window chunks with fixed window_size
            # First, compute all chunk start positions using regular stride
            chunk_starts = []
            frame_start = 0
            while frame_start + self.window_size <= total_frames:
                chunk_starts.append(frame_start)
                frame_start += self.stride
            
            # Check if we need one more chunk to cover remaining frames
            if chunk_starts:
                last_chunk_end = chunk_starts[-1] + self.window_size
                if last_chunk_end < total_frames:
                    # Add a final chunk that ends at total_frames
                    # Adjust start backwards to ensure window_size frames
                    final_start = total_frames - self.window_size
                    # Only add if it doesn't completely overlap with previous chunk
                    if final_start > chunk_starts[-1]:
                        chunk_starts.append(final_start)
            else:
                # Edge case: stride > total_frames - window_size
                # Just add one chunk starting at 0
                chunk_starts.append(0)
            
            # Create chunks from computed start positions
            for frame_start in chunk_starts:
                frame_end = frame_start + self.window_size
                chunk_name = f"{scene_id}_frames{frame_start:04d}_{frame_end:04d}"
                self.chunks.append((scene_dir, frame_start, frame_end, chunk_name))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> VideoStream:
        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"Index {index} out of range for {len(self.chunks)} chunks")
        
        scene_dir, frame_start, frame_end, chunk_name = self.chunks[index]
        
        stream: VideoStream = OmniWorldSlidingWindowStream(
            scene_dir=scene_dir,
            frame_start=frame_start,
            frame_end=frame_end,
            metric_scale=1.0,  # Will be computed by pipeline
            name=chunk_name,
        )
        
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading frames", online=False)
        
        return stream

    def stream_name(self, index: int) -> str:
        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"Index {index} out of range for {len(self.chunks)} chunks")
        return self.chunks[index][3]
    
    def get_chunk_info(self, index: int) -> Tuple[Path, int, int, str]:
        """Get (scene_dir, frame_start, frame_end, chunk_name) for a chunk."""
        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"Index {index} out of range for {len(self.chunks)} chunks")
        return self.chunks[index]
