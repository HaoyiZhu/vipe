# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pi3X-only pipeline: Uses Pi3X for visual odometry and depth estimation.
No SLAM involved - directly uses feedforward model predictions.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    GeoCalibIntrinsicsProcessor,
    TrackAnythingProcessor,
    Pi3XVOInitPoseProcessor,
    Pi3XMetricDepthProcessor,
)

logger = logging.getLogger(__name__)


class Pi3XOnlyAnnotationPipeline(Pipeline):
    """
    Pipeline that uses only Pi3X for pose and depth estimation.
    No SLAM - directly uses feedforward model predictions.
    """

    def __init__(
        self,
        init: DictConfig,
        output: DictConfig,
        pi3x: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.init_cfg = init
        self.out_cfg = output
        self.pi3x_cfg = pi3x if pi3x is not None else OmegaConf.create({})
        self.out_path = Path(output.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(init.get("camera_type", "pinhole"))

    def should_filter(self, name: str) -> bool:
        """Check if this video should be skipped."""
        if not self.out_cfg.get("skip_exists", False):
            return False
        artifact_path = io.ArtifactPath(self.out_path, name)
        return artifact_path.meta_info_path.exists()

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        """Add initialization processors."""
        init_processors: list[StreamProcessor] = []
        
        # Intrinsics estimation
        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        
        # Instance segmentation
        if self.init_cfg.get("instance") is not None:
            instance_cfg = self.init_cfg.instance
            init_processors.append(
                TrackAnythingProcessor(
                    instance_cfg.get("phrases", []),
                    add_sky=instance_cfg.get("add_sky", True),
                    sam_run_gap=int(video_stream.fps() * instance_cfg.get("kf_gap_sec", 2.0)),
                )
            )
        
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_pi3x_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        """Add Pi3X VO and depth processors."""
        vo_cfg = self.pi3x_cfg.get("vo", DictConfig({}))
        depth_cfg = self.pi3x_cfg.get("depth", DictConfig({}))
        
        processors: list[StreamProcessor] = []
        
        # Pi3X VO for pose estimation
        processors.append(
            Pi3XVOInitPoseProcessor(
                video_stream,
                model=vo_cfg.get("model", "yyfz233/Pi3X"),
                chunk_size=vo_cfg.get("chunk_size", 64),
                overlap=vo_cfg.get("overlap", 32),
                conf_thre=vo_cfg.get("conf_thre", 0.05),
                dtype=vo_cfg.get("dtype", "bf16"),
                pose_convention=vo_cfg.get("pose_convention", "c2w"),
                return_depth=False,
            )
        )
        
        # Pi3X for metric depth
        processors.append(
            Pi3XMetricDepthProcessor(
                model=depth_cfg.get("model", "yyfz233/Pi3X"),
                pixel_limit=int(depth_cfg.get("pixel_limit", 255000)),
                batch_size=int(depth_cfg.get("batch_size", 8)),
                use_poses=True,
            )
        )
        
        return ProcessedVideoStream(video_stream, processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            if len(video_data) > 1:
                raise ValueError("Pi3XOnly pipeline supports single-view streams only.")
            video_stream = video_data[0]
        else:
            assert isinstance(video_data, VideoStream)
            video_stream = video_data

        annotate_output = AnnotationPipelineOutput()

        if self.should_filter(video_stream.name()):
            logger.info("%s has been processed already, skip it.", video_stream.name())
            return annotate_output

        logger.info("Processing %s with Pi3X-only pipeline", video_stream.name())

        # Step 1: Add init processors (intrinsics, instance segmentation)
        init_stream = self._add_init_processors(video_stream).cache("init", online=True)
        
        # Step 2: Add Pi3X processors (VO + depth)
        pi3x_stream = self._add_pi3x_processors(init_stream).cache("pi3x", online=True)
        
        # Step 3: Collect results
        total_frames = len(pi3x_stream)
        logger.info(f"Processing {total_frames} frames with Pi3X")
        
        # Iterate through stream to get results
        extrinsics = []
        intrinsics = []
        depths = []
        masks = []
        instances = []
        instance_phrases: dict[int, str] = {}
        
        for frame in pi3x_stream:
            # Pose
            if frame.pose is not None:
                pose = frame.pose.matrix().detach().cpu().numpy()
            else:
                pose = np.eye(4, dtype=np.float32)
            extrinsics.append(pose)
            
            # Intrinsics
            if frame.intrinsics is not None:
                intr = frame.intrinsics[:4].detach().cpu().numpy()
            else:
                intr = np.zeros(4, dtype=np.float32)
            intrinsics.append(intr)
            
            # Depth
            if frame.metric_depth is not None:
                depths.append(frame.metric_depth.detach().cpu().numpy())
            else:
                depths.append(None)
            
            # Mask
            if frame.mask is not None:
                masks.append(frame.mask.detach().cpu().numpy().astype(bool))
            else:
                masks.append(None)
            
            # Instance
            if frame.instance is not None:
                instances.append(frame.instance.detach().cpu().numpy().astype(np.uint8))
            else:
                instances.append(None)
            
            # Instance phrases
            if frame.instance_phrases:
                instance_phrases.update(frame.instance_phrases)
        
        # Step 4: Build output stream
        merged_poses = [se3_matrix_to_se3(torch.from_numpy(p).float(), unbatch=True) 
                        for p in extrinsics]
        merged_intrinsics = [torch.from_numpy(i).float() for i in intrinsics]
        merged_depths = [torch.from_numpy(d).float() if d is not None else None 
                         for d in depths]
        merged_masks = [torch.from_numpy(m).bool() if m is not None else None 
                        for m in masks]
        merged_instances = [torch.from_numpy(inst).byte() if inst is not None else None 
                            for inst in instances]
        merged_camera_types = [self.camera_type] * total_frames

        stream_attributes: dict[FrameAttribute, list[Any]] = {
            FrameAttribute.POSE: merged_poses,
            FrameAttribute.INTRINSICS: merged_intrinsics,
            FrameAttribute.CAMERA_TYPE: merged_camera_types,
        }
        if any(d is not None for d in merged_depths):
            stream_attributes[FrameAttribute.METRIC_DEPTH] = merged_depths
        if any(m is not None for m in merged_masks):
            stream_attributes[FrameAttribute.MASK] = merged_masks
        if any(inst is not None for inst in merged_instances):
            stream_attributes[FrameAttribute.INSTANCE] = merged_instances

        output_stream = ProcessedVideoStream(
            video_stream, 
            [AssignAttributesProcessor(stream_attributes)]
        ).cache("output", online=True)

        # Save artifacts
        artifact_path = io.ArtifactPath(self.out_path, video_stream.name())
        if self.out_cfg.get("save_artifacts", True):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info("Saving artifacts to %s", artifact_path)
            io.save_artifacts(artifact_path, output_stream)
            
            # Save pipeline info
            with artifact_path.meta_info_path.open("wb") as f:
                info = {
                    "pipeline": "pi3x_only",
                    "total_frames": total_frames,
                }
                pickle.dump(info, f)

        if self.return_output_streams:
            annotate_output.output_streams = [output_stream]

        logger.info("Finished processing %s", video_stream.name())
        return annotate_output
