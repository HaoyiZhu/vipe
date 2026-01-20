import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from vipe.ext.lietorch import SE3
from vipe.pipeline import AnnotationPipelineOutput
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.loop_detection import LoopDetector, process_loop_list
from vipe.utils.sim3 import Sim3LoopOptimizer, compute_sim3_ab, robust_weighted_align_point_maps

logger = logging.getLogger(__name__)

class SlicedVideoStream(VideoStream):
    def __init__(self, stream: VideoStream, start: int, end: int):
        self.stream = stream
        self.start = start
        self.end = end
        
    def frame_size(self):
        return self.stream.frame_size()
        
    def name(self):
        return f"{self.stream.name()}_{self.start}_{self.end}"
        
    def fps(self):
        return self.stream.fps()
        
    def __len__(self):
        return self.end - self.start
        
    def attributes(self):
        return self.stream.attributes()
        
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        # Check if stream supports getitem
        if hasattr(self.stream, "__getitem__"):
            return self.stream[self.start + index]
        else:
            # Fallback to iteration? Very slow for random access.
            # But we are processing sequentially inside the chunk.
            # We assume stream supports getitem or is cached.
            raise NotImplementedError("Underlying stream must support __getitem__")
        
    def __iter__(self):
        # Optimized iteration if stream supports it?
        # If stream is random access, use loop
        if hasattr(self.stream, "__getitem__"):
            for i in range(len(self)):
                yield self[i]
        else:
            # Skip until start
            it = iter(self.stream)
            for _ in range(self.start):
                next(it)
            for _ in range(len(self)):
                yield next(it)

class IndexedVideoStream(VideoStream):
    def __init__(self, stream: VideoStream, indices: list[int], name: str | None = None):
        self.stream = stream
        self.indices = list(indices)
        if not self.indices:
            raise ValueError("IndexedVideoStream requires at least one index.")
        self._name = name or f"{self.stream.name()}_{self.indices[0]}_{self.indices[-1]}"
        
    def frame_size(self):
        return self.stream.frame_size()
    
    def name(self):
        return self._name
    
    def fps(self):
        return self.stream.fps()
    
    def __len__(self):
        return len(self.indices)
    
    def attributes(self):
        return self.stream.attributes()
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        if hasattr(self.stream, "__getitem__"):
            return self.stream[self.indices[index]]
        raise NotImplementedError("Underlying stream must support __getitem__")
    
    def __iter__(self):
        if hasattr(self.stream, "__getitem__"):
            for idx in self.indices:
                yield self.stream[idx]
        else:
            it = iter(self.stream)
            curr = 0
            for target_idx in self.indices:
                while curr < target_idx:
                    next(it)
                    curr += 1
                yield next(it)
                curr += 1

class AssignInstancePhrasesProcessor(StreamProcessor):
    def __init__(self, instance_phrases_list: list[dict[int, str] | None] | None):
        self.instance_phrases_list = instance_phrases_list
    
    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes
    
    def __call__(self, frame_idx: int, frame):
        if self.instance_phrases_list is not None:
            frame.instance_phrases = self.instance_phrases_list[frame_idx]
        return frame

def remove_duplicates(data_list: list[tuple[int, tuple[int, int], int, tuple[int, int]]]):
    """
    Remove duplicated loop pairs by chunk indices. Matches VGGT-Long behavior.
    """
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result

class LongSequenceAnnotationPipeline(DefaultAnnotationPipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__(init, slam, post, output)
        
        # Long sequence specific configs (defaults aligned with vggt-long where possible)
        # Check for nested 'long_seq' config first
        long_seq_cfg = self.init_cfg.get("long_seq", DictConfig({}))
        if not long_seq_cfg:
             # Fallback to top-level init keys if long_seq not present (backward compat)
             pass
             
        # Model params
        self.chunk_size = self.init_cfg.get("chunk_size", 100) # vggt default: 100
        self.overlap = self.init_cfg.get("overlap", 50)        # vggt default: 50
        
        self.loop_chunk_size = long_seq_cfg.get("loop_chunk_size", 20)
        self.loop_half_window = max(1, int(self.loop_chunk_size / 2))
        self.loop_enable = long_seq_cfg.get("loop_enable", True)
        self.delete_temp_files = long_seq_cfg.get("delete_temp_files", True)
        self.using_sim3 = long_seq_cfg.get("using_sim3", True)
        self.world_points_source = long_seq_cfg.get("world_points_source", "slam_map")
        self.align_conf_threshold_coef = long_seq_cfg.get("alignment_conf_threshold_coef", 0.1)
        
        # Pointcloud Save / Alignment params
        pc_cfg = long_seq_cfg.get("pointcloud_save", DictConfig({}))
        self.sample_ratio = pc_cfg.get("sample_ratio", 0.015)
        self.conf_threshold_coef = pc_cfg.get("conf_threshold_coef", 0.75)
        self.use_confidence_filtering = pc_cfg.get("use_confidence_filtering", True)
        
        # Loop params
        loop_cfg = long_seq_cfg.get("loop", DictConfig({}))
        self.loop_similarity_threshold = loop_cfg.get("similarity_threshold", 0.85)
        self.loop_nms_threshold = loop_cfg.get("nms_threshold", 25)
        self.loop_window = loop_cfg.get("loop_window", 200) # Fallback if not in config
        
        # Weights
        weights_cfg = long_seq_cfg.get("weights", DictConfig({}))
        self.salad_ckpt_path = weights_cfg.get("salad", "./weights/dino_salad.ckpt")
        
        # Optimizer params
        opt_cfg = long_seq_cfg.get("sim3_optimizer", DictConfig({}))
        self.opt_max_iterations = opt_cfg.get("max_iterations", 30)
        self.opt_lambda_init = opt_cfg.get("lambda_init", 1e-6)
        
        # IRLS params
        irls_cfg = long_seq_cfg.get("irls", DictConfig({}))
        self.irls_delta = irls_cfg.get("delta", 0.1)
        self.irls_max_iters = irls_cfg.get("max_iters", 5)
        self.irls_tol = irls_cfg.get("tol", 1e-9)
        
        # Temp directories (align with VGGT-Long structure under output path)
        self.result_unaligned_dir = self.out_path / "_tmp_results_unaligned"
        self.result_aligned_dir = self.out_path / "_tmp_results_aligned"
        self.result_loop_dir = self.out_path / "_tmp_results_loop"
        self.result_unaligned_dir.mkdir(exist_ok=True, parents=True)
        self.result_aligned_dir.mkdir(exist_ok=True, parents=True)
        self.result_loop_dir.mkdir(exist_ok=True, parents=True)
        
        self.sim3_list = []
        self.loop_sim3_list = []
        self.chunk_indices = []
        self.loop_list = []
        self.chunk_residuals = []
        self._grid_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_grid(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, str(device))
        if key not in self._grid_cache:
            y, x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            self._grid_cache[key] = (y, x)
        return self._grid_cache[key]
    
    def _backproject_depth(self, depth: torch.Tensor, intr: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        h, w = depth.shape
        y, x = self._get_grid(h, w, depth.device)
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        pts_cam = torch.stack([X, Y, Z], dim=-1)
        pts_flat = pts_cam.reshape(-1, 3)
        pts_world_flat = (pose[:3, :3] @ pts_flat.T).T + pose[:3, 3]
        return pts_world_flat.reshape(h, w, 3)
    
    def _normalize_chunk_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if "world_points" in data:
            world_points = data["world_points"]
            if isinstance(world_points, np.ndarray) and world_points.ndim == 2 and world_points.shape[1] == 3:
                data["world_points"] = world_points[np.newaxis, ...]
        
        if "world_points_conf" not in data and "world_points" in data:
            world_points = data["world_points"]
            if world_points.ndim == 4:
                conf_shape = world_points.shape[:3]
            elif world_points.ndim == 3:
                conf_shape = world_points.shape[:2]
            else:
                conf_shape = world_points.shape[:1]
            data["world_points_conf"] = np.ones(conf_shape, dtype=np.float32)
        
        if "intrinsic" in data and data["intrinsic"] is not None:
            intr = data["intrinsic"]
            if isinstance(intr, np.ndarray) and intr.ndim == 3 and intr.shape[-2:] == (3, 3):
                fx = intr[:, 0, 0]
                fy = intr[:, 1, 1]
                cx = intr[:, 0, 2]
                cy = intr[:, 1, 2]
                data["intrinsic"] = np.stack([fx, fy, cx, cy], axis=-1)
        
        if "images" in data and data["images"] is not None:
            imgs = data["images"]
            if isinstance(imgs, np.ndarray) and imgs.ndim == 4 and imgs.shape[-1] == 3:
                data["images"] = imgs.transpose(0, 3, 1, 2)
        
        if "mask" in data and isinstance(data["mask"], np.ndarray) and data["mask"].ndim == 4 and data["mask"].shape[-1] == 1:
            data["mask"] = data["mask"].squeeze(-1)
        
        if "instance_phrases" not in data:
            data["instance_phrases"] = None
        
        return data
    
    def _load_chunk_data(self, path: Path) -> dict[str, Any]:
        data = np.load(path, allow_pickle=True).item()
        return self._normalize_chunk_data(data)
    
    def _collect_chunk_data(self, output_stream: VideoStream, slam_output) -> dict[str, Any]:
        extrinsics = []
        intrinsics = []
        depths = []
        masks = []
        instances = []
        images = []
        world_points = []
        world_points_conf = []
        instance_phrases: dict[int, str] = {}
        
        has_depth = False
        has_mask = False
        has_instance = False
        use_slam_map = self.world_points_source == "slam_map" and slam_output.slam_map is not None
        
        for frame in output_stream:
            if frame.pose is None:
                pose = torch.eye(4, device=frame.rgb.device)
            else:
                pose = frame.pose.matrix()
            extrinsics.append(pose.detach().cpu().numpy())
            
            if frame.intrinsics is None:
                intr = torch.zeros(4, device=pose.device)
            else:
                intr = frame.intrinsics[:4]
            intrinsics.append(intr.detach().cpu().numpy())
            
            if frame.metric_depth is not None:
                depth = frame.metric_depth
                has_depth = True
                depths.append(depth.detach().cpu().numpy())
            else:
                depth = None
                depths.append(None)
            
            if frame.mask is not None:
                has_mask = True
                masks.append(frame.mask.detach().cpu().numpy().astype(bool))
            else:
                masks.append(None)
            
            if frame.instance is not None:
                has_instance = True
                instances.append(frame.instance.detach().cpu().numpy().astype(np.uint8))
            else:
                instances.append(None)
            
            if frame.instance_phrases:
                instance_phrases.update(frame.instance_phrases)
            
            images.append(frame.rgb.permute(2, 0, 1).detach().cpu().numpy())
            
            if not use_slam_map:
                if depth is None:
                    depth = torch.zeros(frame.size(), device=pose.device)
                conf = torch.ones_like(depth)
                pts_world = self._backproject_depth(depth, intr, pose)
                world_points.append(pts_world.detach().cpu().numpy())
                world_points_conf.append(conf.detach().cpu().numpy())
        
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)
        images = np.stack(images)
        
        if has_depth:
            depth_template = next((d for d in depths if d is not None), None)
            for i, d in enumerate(depths):
                if d is None and depth_template is not None:
                    depths[i] = np.zeros_like(depth_template, dtype=np.float32)
            depths = np.stack(depths)
        else:
            depths = None
        
        if has_mask:
            mask_template = next((m for m in masks if m is not None), None)
            for i, m in enumerate(masks):
                if m is None and mask_template is not None:
                    masks[i] = np.ones_like(mask_template, dtype=bool)
            masks = np.stack(masks)
        else:
            masks = None
        
        if has_instance:
            inst_template = next((inst for inst in instances if inst is not None), None)
            for i, inst in enumerate(instances):
                if inst is None and inst_template is not None:
                    instances[i] = np.zeros_like(inst_template, dtype=np.uint8)
            instances = np.stack(instances)
        else:
            instances = None
        
        world_colors = None
        if use_slam_map:
            slam_map = slam_output.slam_map
            world_points = slam_map.dense_disp_xyz.detach().cpu().numpy()
            world_points = world_points[np.newaxis, ...]
            world_points_conf = np.ones((1, world_points.shape[1]), dtype=np.float32)
            world_colors = slam_map.dense_disp_rgb.detach().cpu().numpy()
        else:
            if world_points:
                world_points = np.stack(world_points)
                world_points_conf = np.stack(world_points_conf)
            else:
                world_points = np.zeros((1, 0, 3), dtype=np.float32)
                world_points_conf = np.zeros((1, 0), dtype=np.float32)
        
        chunk_data = {
            "world_points": world_points,
            "world_points_conf": world_points_conf,
            "world_colors": world_colors,
            "mask": masks,
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "depth": depths,
            "images": images,
            "instance": instances,
            "instance_phrases": instance_phrases if instance_phrases else None,
            "ba_residual": float(slam_output.ba_residual),
        }
        return chunk_data
    
    def _process_single_chunk(
        self,
        chunk_idx: int,
        chunk_stream: VideoStream,
        save_dir: Path | None = None,
        filename: str | None = None,
        record_residuals: bool = True,
    ) -> dict[str, Any]:
        save_dir = save_dir or self.result_unaligned_dir
        filename = filename or f"chunk_{chunk_idx}.npy"
        save_path = save_dir / filename
        
        if save_path.exists():
            data = self._load_chunk_data(save_path)
            if record_residuals and data.get("ba_residual") is not None:
                self.chunk_residuals.append(float(data["ba_residual"]))
            return data
        
        processed_stream = self._add_init_processors(chunk_stream).cache("process", online=True)
        
        from vipe.slam.system import SLAMSystem
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run([processed_stream], rig=None, camera_type=self.camera_type)
        
        output_stream = self._add_post_processors(0, processed_stream, slam_output).cache("depth", online=True)
        chunk_data = self._collect_chunk_data(output_stream, slam_output)
        chunk_data = self._normalize_chunk_data(chunk_data)
        
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(save_path, chunk_data)
        
        if record_residuals and chunk_data.get("ba_residual") is not None:
            self.chunk_residuals.append(float(chunk_data["ba_residual"]))
        
        del slam_pipeline
        del processed_stream
        del output_stream
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return chunk_data
    
    def _compute_conf_threshold(self, conf1: Any, conf2: Any) -> float:
        if not self.use_confidence_filtering:
            return -1.0
        if conf1 is None or conf2 is None:
            return -1.0
        if isinstance(conf1, torch.Tensor):
            c1 = torch.median(conf1).item()
        else:
            c1 = float(np.median(conf1))
        if isinstance(conf2, torch.Tensor):
            c2 = torch.median(conf2).item()
        else:
            c2 = float(np.median(conf2))
        return min(c1, c2) * self.align_conf_threshold_coef
    
    def _maybe_build_align_mask(
        self,
        mask1: np.ndarray | None,
        mask2: np.ndarray | None,
        point_map1: np.ndarray,
    ) -> np.ndarray | None:
        if mask1 is None or mask2 is None:
            return None
        m1 = np.squeeze(mask1)
        m2 = np.squeeze(mask2)
        if m1.shape != m2.shape:
            return None
        if point_map1.ndim == 4 and m1.shape == point_map1.shape[:3]:
            return m1 & m2
        if point_map1.ndim == 3 and m1.shape == point_map1.shape[:2]:
            return m1 & m2
        return None
    
    def _point_maps_from_depth(
        self,
        data: dict[str, Any],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        depths = data.get("depth")
        if depths is None:
            raise ValueError("Depth data is not available for alignment.")
        
        depths = depths[start:end]
        intrinsics = data["intrinsic"][start:end]
        extrinsics = data["extrinsic"][start:end]
        masks = data.get("mask")
        masks = masks[start:end] if masks is not None and masks.shape[0] >= end else None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = []
        confs = []
        for i in range(len(depths)):
            depth = torch.from_numpy(depths[i]).float().to(device)
            intr = torch.from_numpy(intrinsics[i]).float().to(device)
            pose = torch.from_numpy(extrinsics[i]).float().to(device)
            pts_world = self._backproject_depth(depth, intr, pose)
            points.append(pts_world.detach().cpu().numpy())
            confs.append(np.ones_like(depths[i], dtype=np.float32))
        
        if points:
            return np.stack(points), np.stack(confs), masks
        return np.zeros((0, 0, 0, 3), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32), masks
    
    def _get_alignment_maps(
        self,
        data: dict[str, Any],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if data.get("depth") is not None:
            return self._point_maps_from_depth(data, start, end)
        
        points = data["world_points"]
        confs = data["world_points_conf"]
        masks = data.get("mask")
        
        if points.ndim == 4:
            pts = points[start:end]
            cfs = confs[start:end] if confs is not None else np.ones(points.shape[:3], dtype=np.float32)
            msk = masks[start:end] if masks is not None and masks.shape[0] >= end else None
            return pts, cfs, msk
        
        return points, confs, None
    
    def _slice_point_maps(
        self,
        data: dict[str, Any],
        start: int,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        points = data["world_points"]
        confs = data["world_points_conf"]
        mask = data.get("mask")
        
        if points.shape[0] == 1:
            # For global point clouds (e.g., SLAM map), use the full set.
            if mask is not None and mask.shape[0] != 1:
                mask = None
            return points, confs, mask
        
        pts = points[start:end]
        cfs = confs[start:end]
        msk = mask[start:end] if mask is not None else None
        return pts, cfs, msk
    
    def _make_loop_stream(self, video_stream: VideoStream, range_a: tuple[int, int], range_b: tuple[int, int]) -> VideoStream:
        indices = list(range(range_a[0], range_a[1])) + list(range(range_b[0], range_b[1]))
        name = f"{video_stream.name()}_loop_{range_a[0]}_{range_a[1]}_{range_b[0]}_{range_b[1]}"
        return IndexedVideoStream(video_stream, indices, name=name)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            # For simplicity, handle monocular first or treat each view as separate?
            # VGGT-Long seems monocular focused. 
            # If stereo, we might need to adapt.
            logger.warning("LongSequenceAnnotationPipeline: Multiview input detected. Using first view for SLAM/Alignment.")
            video_stream = video_data[0]
        else:
            video_stream = video_data

        annotate_output = AnnotationPipelineOutput()
        
        if self.should_filter(video_stream.name()):
            logger.info(f"{video_stream.name()} has been processed already, skip it.")
            return annotate_output

        # Reset per-run state
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_list = []
        self.chunk_residuals = []

        total_frames = len(video_stream)
        
        # 1. Chunking
        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        
        if total_frames <= self.chunk_size:
            self.chunk_indices = [(0, total_frames)]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (total_frames - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, total_frames)
                self.chunk_indices.append((start_idx, end_idx))
        
        logger.info(f"Processing {total_frames} frames in {len(self.chunk_indices)} chunks.")
        
        # 2. Process chunks
        for chunk_idx, (start, end) in enumerate(self.chunk_indices):
            logger.info(f"Processing chunk {chunk_idx}: frames {start} to {end}")
            chunk_stream = SlicedVideoStream(video_stream, start, end)
            self._process_single_chunk(chunk_idx, chunk_stream)
        
        # 3. Align chunks (Sequential)
        self.sim3_list = []
        logger.info("Aligning chunks sequentially...")
        for chunk_idx in range(len(self.chunk_indices) - 1):
            logger.info(f"Aligning {chunk_idx} and {chunk_idx+1}")
            data1 = self._load_chunk_data(self.result_unaligned_dir / f"chunk_{chunk_idx}.npy")
            data2 = self._load_chunk_data(self.result_unaligned_dir / f"chunk_{chunk_idx+1}.npy")
            
            n1 = data1["extrinsic"].shape[0]
            n2 = data2["extrinsic"].shape[0]
            overlap = min(self.overlap, n1, n2)
            if overlap <= 0:
                logger.warning("No overlap between chunks %d and %d; skipping alignment.", chunk_idx, chunk_idx + 1)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.sim3_list.append((1.0, torch.eye(3, device=device), torch.zeros(3, device=device)))
                continue
            
            pts1, conf1, mask1 = self._get_alignment_maps(data1, n1 - overlap, n1)
            pts2, conf2, mask2 = self._get_alignment_maps(data2, 0, overlap)
            
            mask = self._maybe_build_align_mask(mask1, mask2, pts1)
            conf_threshold = self._compute_conf_threshold(conf1, conf2)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pts1_t = torch.from_numpy(pts1).float().to(device)
            conf1_t = torch.from_numpy(conf1).float().to(device)
            pts2_t = torch.from_numpy(pts2).float().to(device)
            conf2_t = torch.from_numpy(conf2).float().to(device)
            mask_t = torch.from_numpy(mask).to(device) if mask is not None else None
            
            s, R, t = robust_weighted_align_point_maps(
                pts1_t,
                conf1_t,
                pts2_t,
                conf2_t,
                mask_t,
                conf_threshold,
                delta=self.irls_delta,
                max_iters=self.irls_max_iters,
                tol=self.irls_tol,
                using_sim3=self.using_sim3,
            )
            self.sim3_list.append((float(s), R, t))
            logger.info(f"Sim3 {chunk_idx}->{chunk_idx+1}: s={float(s):.4f}")
        
        # 4. Loop Detection & Loop Constraints
        if self.loop_enable:
            logger.info("Loop detection enabled.")
            loop_detector = LoopDetector(
                similarity_threshold=self.loop_similarity_threshold,
                loop_window=self.loop_window,
                nms_threshold=self.loop_nms_threshold,
                ckpt_path=self.salad_ckpt_path,
            )
            loop_file = self.out_path / "loop_closures.txt"
            if loop_file.exists():
                logger.info(f"Loading loops from {loop_file}")
                loop_detector.load_from_file(loop_file)
            else:
                logger.info("Running loop detection...")
                loop_detector.detect(video_stream)
                if loop_detector.loop_list:
                    loop_file.parent.mkdir(exist_ok=True, parents=True)
                    with loop_file.open("w") as f:
                        for i, j in loop_detector.loop_list:
                            f.write(f"{i}, {j}\n")
            
            self.loop_list = loop_detector.loop_list
            logger.info(f"Detected {len(self.loop_list)} loops.")
            
            if self.loop_list:
                loop_results = process_loop_list(
                    self.chunk_indices,
                    self.loop_list,
                    half_window=self.loop_half_window,
                )
                loop_results = remove_duplicates(loop_results)
                logger.info(f"Processing {len(loop_results)} loop constraints...")
                
                for item in loop_results:
                    chunk_idx_a, range_a, chunk_idx_b, range_b = item
                    loop_filename = f"loop_{range_a[0]}_{range_a[1]}_{range_b[0]}_{range_b[1]}.npy"
                    loop_stream = self._make_loop_stream(video_stream, range_a, range_b)
                    loop_data = self._process_single_chunk(
                        chunk_idx=-1,
                        chunk_stream=loop_stream,
                        save_dir=self.result_loop_dir,
                        filename=loop_filename,
                        record_residuals=False,
                    )
                    
                    len_a = range_a[1] - range_a[0]
                    len_b = range_b[1] - range_b[0]
                    
                    loop_total = loop_data["extrinsic"].shape[0]
                    loop_end_a = min(len_a, loop_total)
                    loop_pts_a, loop_conf_a, loop_mask_a = self._get_alignment_maps(loop_data, 0, loop_end_a)
                    
                    chunk_a_data = self._load_chunk_data(self.result_unaligned_dir / f"chunk_{chunk_idx_a}.npy")
                    chunk_a_start = self.chunk_indices[chunk_idx_a][0]
                    rel_start_a = range_a[0] - chunk_a_start
                    rel_end_a = rel_start_a + len_a
                    pts_a, conf_a, mask_a = self._get_alignment_maps(chunk_a_data, rel_start_a, rel_end_a)
                    
                    mask_a_loop = self._maybe_build_align_mask(mask_a, loop_mask_a, pts_a)
                    conf_thresh_a = self._compute_conf_threshold(conf_a, loop_conf_a)
                    
                    s_a, R_a, t_a = robust_weighted_align_point_maps(
                        torch.from_numpy(pts_a).float().cuda(),
                        torch.from_numpy(conf_a).float().cuda(),
                        torch.from_numpy(loop_pts_a).float().cuda(),
                        torch.from_numpy(loop_conf_a).float().cuda(),
                        torch.from_numpy(mask_a_loop).cuda() if mask_a_loop is not None else None,
                        conf_thresh_a,
                        delta=self.irls_delta,
                        max_iters=self.irls_max_iters,
                        tol=self.irls_tol,
                        using_sim3=self.using_sim3,
                    )
                    
                    loop_b_start = min(loop_end_a, loop_total)
                    loop_b_end = min(loop_b_start + len_b, loop_total)
                    loop_pts_b, loop_conf_b, loop_mask_b = self._get_alignment_maps(loop_data, loop_b_start, loop_b_end)
                    
                    chunk_b_data = self._load_chunk_data(self.result_unaligned_dir / f"chunk_{chunk_idx_b}.npy")
                    chunk_b_start = self.chunk_indices[chunk_idx_b][0]
                    rel_start_b = range_b[0] - chunk_b_start
                    rel_end_b = rel_start_b + len_b
                    pts_b, conf_b, mask_b = self._get_alignment_maps(chunk_b_data, rel_start_b, rel_end_b)
                    
                    mask_b_loop = self._maybe_build_align_mask(mask_b, loop_mask_b, pts_b)
                    conf_thresh_b = self._compute_conf_threshold(conf_b, loop_conf_b)
                    
                    s_b, R_b, t_b = robust_weighted_align_point_maps(
                        torch.from_numpy(pts_b).float().cuda(),
                        torch.from_numpy(conf_b).float().cuda(),
                        torch.from_numpy(loop_pts_b).float().cuda(),
                        torch.from_numpy(loop_conf_b).float().cuda(),
                        torch.from_numpy(mask_b_loop).cuda() if mask_b_loop is not None else None,
                        conf_thresh_b,
                        delta=self.irls_delta,
                        max_iters=self.irls_max_iters,
                        tol=self.irls_tol,
                        using_sim3=self.using_sim3,
                    )
                    
                    s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                    self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))
                    logger.info(f"Loop {chunk_idx_a}->{chunk_idx_b}: s={float(s_ab):.4f}")
        
        if self.sim3_list:
            optimizer = Sim3LoopOptimizer(max_iterations=self.opt_max_iterations, lambda_init=self.opt_lambda_init)
            self.sim3_list = optimizer.optimize(self.sim3_list, self.loop_sim3_list)
        
        # 5. Apply Alignment and Merge
        logger.info("Merging results...")
        accum_sim3s = [(1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))]
        curr_s, curr_R, curr_t = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        for s, R, t in self.sim3_list:
            if isinstance(R, torch.Tensor):
                R_np = R.detach().cpu().numpy()
                t_np = t.detach().cpu().numpy()
                s_np = float(s)
            else:
                R_np = R
                t_np = t
                s_np = float(s)
            
            curr_t = curr_t + curr_s * (curr_R @ t_np)
            curr_R = curr_R @ R_np
            curr_s = curr_s * s_np
            accum_sim3s.append((curr_s, curr_R, curr_t))
        
        full_traj: list[np.ndarray | None] = [None] * total_frames
        full_intrinsics: list[np.ndarray | None] = [None] * total_frames
        full_depths: list[np.ndarray | None] = [None] * total_frames
        full_masks: list[np.ndarray | None] = [None] * total_frames
        full_instances: list[np.ndarray | None] = [None] * total_frames
        full_instance_phrases: dict[int, str] = {}
        
        for chunk_idx, (s, R, t) in enumerate(accum_sim3s):
            data = self._load_chunk_data(self.result_unaligned_dir / f"chunk_{chunk_idx}.npy")
            extrinsics = data["extrinsic"]
            
            R_local = extrinsics[:, :3, :3]
            t_local = extrinsics[:, :3, 3]
            
            R_new = R[None, ...] @ R_local
            t_new = (s * (R[None, ...] @ t_local[..., None]).squeeze(-1)) + t[None, ...]
            
            poses_new = np.repeat(np.eye(4, dtype=np.float32)[None], len(extrinsics), axis=0)
            poses_new[:, :3, :3] = R_new
            poses_new[:, :3, 3] = t_new
            
            chunk_start = self.chunk_indices[chunk_idx][0]
            for i in range(len(poses_new)):
                frame_idx = chunk_start + i
                full_traj[frame_idx] = poses_new[i]
                
                if data.get("intrinsic") is not None:
                    full_intrinsics[frame_idx] = data["intrinsic"][i]
                if data.get("depth") is not None:
                    full_depths[frame_idx] = data["depth"][i]
                if data.get("mask") is not None:
                    full_masks[frame_idx] = data["mask"][i]
                if data.get("instance") is not None:
                    full_instances[frame_idx] = data["instance"][i]
            
            if data.get("instance_phrases"):
                full_instance_phrases.update(data["instance_phrases"])
        
        pose_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged_poses = []
        for p in full_traj:
            if p is not None:
                merged_poses.append(SE3(torch.from_numpy(p).float().to(pose_device)))
            else:
                merged_poses.append(SE3.Identity(1, device=pose_device))
        
        merged_intrinsics = [torch.from_numpy(i).float() if i is not None else None for i in full_intrinsics]
        merged_depths = [torch.from_numpy(d).float() if d is not None else None for d in full_depths]
        merged_masks = [torch.from_numpy(m).bool() if m is not None else None for m in full_masks]
        merged_instances = [torch.from_numpy(inst).byte() if inst is not None else None for inst in full_instances]
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
        
        instance_phrases_list = None
        if full_instance_phrases:
            instance_phrases_list = [full_instance_phrases] * total_frames
        
        post_processors = [AssignAttributesProcessor(stream_attributes)]
        if instance_phrases_list is not None:
            post_processors.append(AssignInstancePhrasesProcessor(instance_phrases_list))
        
        output_stream = ProcessedVideoStream(video_stream, post_processors).cache("merged", online=True)
        
        artifact_path = io.ArtifactPath(self.out_path, video_stream.name())
        if self.out_cfg.save_artifacts:
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving artifacts to {artifact_path}")
            io.save_artifacts(artifact_path, output_stream)
            
            info = {
                "pipeline": "long_seq",
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "num_chunks": len(self.chunk_indices),
                "loop_enable": self.loop_enable,
                "loop_count": len(self.loop_list),
                "loop_constraints": len(self.loop_sim3_list),
                "chunk_indices": self.chunk_indices,
                "chunk_residuals": self.chunk_residuals,
                "world_points_source": self.world_points_source,
                "alignment_conf_threshold_coef": self.align_conf_threshold_coef,
            }
            if self.chunk_residuals:
                info["ba_residual"] = float(np.mean(self.chunk_residuals))
            with artifact_path.meta_info_path.open("wb") as f:
                pickle.dump(info, f)
        
        if self.return_output_streams:
            annotate_output.output_streams = [output_stream]
        
        if self.delete_temp_files:
            for temp_dir in [self.result_unaligned_dir, self.result_aligned_dir, self.result_loop_dir]:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        
        return annotate_output
