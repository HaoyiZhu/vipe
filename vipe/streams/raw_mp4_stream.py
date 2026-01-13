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

from pathlib import Path
import os
import shutil
import tempfile
import zipfile
import csv

import cv2
import torch

from vipe.streams.base import ProcessedVideoStream, StreamList, VideoFrame, VideoStream


class RawMp4Stream(VideoStream):
    """
    A video stream from a raw mp4 file, using opencv.
    This does not support nested iterations.
    """

    def __init__(self, path: Path, seek_range: range | None = None, name: str | None = None) -> None:
        super().__init__()
        if seek_range is None:
            seek_range = range(-1)

        self.path = path
        self._name = name if name is not None else path.stem

        # Read metadata
        vcap = cv2.VideoCapture(str(self.path))
        self._width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = vcap.get(cv2.CAP_PROP_FPS)
        _n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vcap.release()

        self.start = seek_range.start
        self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
        self.end = min(self.end, _n_frames)
        self.step = seek_range.step
        self._fps = _fps / self.step

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(range(self.start, self.end, self.step))

    def __iter__(self):
        self.vcap = cv2.VideoCapture(self.path)
        self.current_frame_idx = -1
        return self

    def __next__(self) -> VideoFrame:
        while True:
            ret, frame = self.vcap.read()
            self.current_frame_idx += 1

            if not ret:
                self.vcap.release()
                raise StopIteration

            if self.current_frame_idx >= self.end:
                self.vcap.release()
                raise StopIteration

            if self.current_frame_idx < self.start:
                continue

            if (self.current_frame_idx - self.start) % self.step == 0:
                break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        return VideoFrame(raw_frame_idx=self.current_frame_idx, rgb=frame_rgb)


class ZipMp4Stream(RawMp4Stream):
    """
    A video stream from a raw mp4 file inside a zip archive.
    Extracts the video to a temporary file on initialization.
    """

    def __init__(self, zip_path: Path, inner_path: str, seek_range: range | None = None, name: str | None = None) -> None:
        self.zip_path = zip_path
        self.inner_path = inner_path
        
        # Create a temporary directory for extraction
        self.temp_dir = tempfile.mkdtemp(prefix="vipe_zip_")
        self.temp_path = Path(self.temp_dir) / Path(inner_path).name
        
        # Extract the file
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                with zf.open(self.inner_path) as source, open(self.temp_path, "wb") as target:
                    shutil.copyfileobj(source, target)
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to extract {inner_path} from {zip_path}: {e}")

        # Initialize the parent RawMp4Stream with the temporary file path
        stream_name = name if name is not None else Path(inner_path).stem
        super().__init__(self.temp_path, seek_range=seek_range, name=stream_name)

    def _cleanup(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except OSError as e:
                print(f"Error cleaning up temp dir {self.temp_dir}: {e}")

    def __del__(self):
        self._cleanup()


class RawMP4StreamList(StreamList):
    def __init__(self, base_path: str, frame_start: int, frame_end: int, frame_skip: int, cached: bool = False) -> None:
        super().__init__()
        # Allow base_path to be a comma-separated list of paths
        base_paths = [Path(p.strip()) for p in str(base_path).split(",")]
        
        self.entries = []  # List of (path, inner_path_or_None)

        for bp in base_paths:
            if bp.is_file():
                if bp.suffix == ".mp4":
                    self.entries.append((bp, None))
                elif bp.suffix == ".zip":
                    self._add_zip_entries(bp)
                elif bp.suffix == ".csv":
                    self._add_csv_entries(bp)
            else:
                # Directory
                if bp.exists():
                    # Glob mp4s
                    for p in sorted(list(bp.glob("*.mp4"))):
                        self.entries.append((p, None))
                    # Glob zips
                    for p in sorted(list(bp.glob("*.zip"))):
                        self._add_zip_entries(p)
                else:
                    print(f"Warning: {bp} does not exist.")
        
        # Sort to ensure consistency
        self.entries.sort(key=lambda x: (x[0].name, x[1] if x[1] else ""))

        self.frame_range = range(frame_start, frame_end, frame_skip)
        self.cached = cached

    def _add_zip_entries(self, zip_path: Path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.lower().endswith('.mp4') and not name.startswith('__MACOSX') and not name.startswith('.'):
                        self.entries.append((zip_path, name))
        except zipfile.BadZipFile:
            print(f"Warning: {zip_path} is not a valid zip file.")

    def _add_csv_entries(self, csv_path: Path):
        """
        Reads a CSV file where each line contains:
        1. A single column with a path (mp4 or zip)
        2. OR Two columns: zip_path, inner_video_path
        """
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    
                    if len(row) >= 2:
                        # Assume format: zip_path, inner_path
                        zip_p = Path(row[0].strip())
                        inner_p = row[1].strip()
                        self.entries.append((zip_p, inner_p))
                    elif len(row) == 1:
                        # Assume format: path (mp4 or zip)
                        p = Path(row[0].strip())
                        if p.suffix == '.mp4':
                            self.entries.append((p, None))
                        elif p.suffix == '.zip':
                            self._add_zip_entries(p)
                        else:
                            print(f"Warning: Unknown file type in CSV: {p}")
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> VideoStream:
        path, inner_path = self.entries[index]
        if inner_path is None:
            stream = RawMp4Stream(path, seek_range=self.frame_range)
        else:
            stream = ZipMp4Stream(path, inner_path, seek_range=self.frame_range)
            
        if self.cached:
            stream = ProcessedVideoStream(stream, []).cache(desc="Loading video", online=False)
        return stream

    def stream_name(self, index: int) -> str:
        path, inner_path = self.entries[index]
        if inner_path is None:
            return path.stem
        else:
            return Path(inner_path).stem
