#!/usr/bin/env python3

import argparse
import logging
import struct
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from vipe.slam.interface import SLAMMap
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def write_ply(
    path: Path,
    points: List[Tuple[float, float, float, int, int, int]],
    edges: List[Tuple[int, int]] = None,
):
    """
    Write PLY file with points and optional edges.
    """
    with open(path, "wb") as f:
        # Write header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        
        if edges:
            f.write(f"element edge {len(edges)}\n".encode())
            f.write(b"property int vertex1\n")
            f.write(b"property int vertex2\n")
            
        f.write(b"end_header\n")

        # Write points
        # Using struct for faster packing
        # 'fffBBB' means 3 floats and 3 unsigned bytes
        s_point = struct.Struct("<fffBBB")
        for p in points:
            f.write(s_point.pack(*p))
            
        # Write edges
        if edges:
            s_edge = struct.Struct("<ii")
            for e in edges:
                f.write(s_edge.pack(*e))

    logger.info(f"Saved PLY to {path} with {len(points)} points and {len(edges) if edges else 0} edges")


def get_camera_frustum_geometry(
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: Tuple[int, int],
    frustum_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> Tuple[List[Tuple[float, float, float, int, int, int]], List[Tuple[int, int]]]:
    """
    Generate points and edges for camera frustum visualization.
    """
    H, W = image_shape
    fx, fy, cx, cy = intrinsics
    
    # Camera center
    center = c2w[:3, 3]
    
    # Image plane corners at z=1
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    corners_2d = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])
    
    corners_3d_cam = np.zeros((4, 3))
    corners_3d_cam[:, 2] = 1.0 # z=1
    corners_3d_cam[:, 0] = (corners_2d[:, 0] - cx) / fx
    corners_3d_cam[:, 1] = (corners_2d[:, 1] - cy) / fy
    
    # Scale frustum size
    corners_3d_cam *= frustum_scale
    
    # Transform to world
    # P_world = R * P_cam + t
    corners_3d_world = (c2w[:3, :3] @ corners_3d_cam.T).T + center
    
    points = []
    edges = []
    
    # Add center point
    # start_idx = 0 (relative to this camera group)
    points.append((center[0], center[1], center[2], color[0], color[1], color[2]))
    
    # Add corner points
    for i in range(4):
        p = corners_3d_world[i]
        points.append((p[0], p[1], p[2], color[0], color[1], color[2]))
        
        # Edge from center (0) to corner (i+1)
        edges.append((0, i + 1))
        
        # Edge between corners
        edges.append((i + 1, (i + 1) % 4 + 1 if i < 3 else 1))

    return points, edges


def process_sequence(
    artifact: ArtifactPath,
    output_path: Path,
    spatial_downsample: int,
    temporal_downsample: int,
):
    logger.info(f"Processing sequence: {artifact.artifact_name}")
    
    if not artifact.pose_path.exists() or not artifact.intrinsics_path.exists():
        logger.warning(f"Missing pose or intrinsics for {artifact.artifact_name}, skipping.")
        return

    # Load poses and intrinsics
    pose_inds, pose_se3 = read_pose_artifacts(artifact.pose_path)
    pose_data = pose_se3.matrix().numpy() # (N, 4, 4)
    # Map frame_idx to pose index
    frame_to_pose_idx = {idx: i for i, idx in enumerate(pose_inds)}
    
    intr_inds, intrinsics_data, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)
    frame_to_intr_idx = {idx: i for i, idx in enumerate(intr_inds)}
    
    # Prepare lists for PLY
    all_points = []
    all_edges = []
    
    # Iterate assuming synchronized streams (or handle unsync)
    rgb_iter = read_rgb_artifacts(artifact.rgb_path)
    depth_iter = read_depth_artifacts(artifact.depth_path)
    
    current_depth = next(depth_iter, None)
    
    processed_frames = 0
    
    for frame_idx, rgb in tqdm(rgb_iter, desc="Frames"):
        
        # Check if we should process this frame based on temporal downsample
        if frame_idx % temporal_downsample != 0:
            # Advance depth iter if it matches this frame (to keep sync)
            while current_depth is not None and current_depth[0] <= frame_idx:
                if current_depth[0] == frame_idx:
                    current_depth = next(depth_iter, None)
                    break
                current_depth = next(depth_iter, None)
            continue
            
        # We want to process this frame.
        # Find matching depth
        while current_depth is not None and current_depth[0] < frame_idx:
             current_depth = next(depth_iter, None)
             
        depth = None
        if current_depth is not None and current_depth[0] == frame_idx:
            depth = current_depth[1]
            current_depth = next(depth_iter, None)
        
        if depth is None:
            # No depth for this frame
            continue
            
        # Get Pose and Intrinsics
        if frame_idx not in frame_to_pose_idx or frame_idx not in frame_to_intr_idx:
            continue
            
        c2w = pose_data[frame_to_pose_idx[frame_idx]]
        intr = intrinsics_data[frame_to_intr_idx[frame_idx]].numpy()
        camera_type = camera_types[frame_to_intr_idx[frame_idx]]
        
        # --- Process Point Cloud ---
        H, W = rgb.shape[:2]
        
        # Downsample
        rgb_ds = rgb[::spatial_downsample, ::spatial_downsample]
        depth_ds = depth[::spatial_downsample, ::spatial_downsample]
        
        if camera_type == CameraType.PINHOLE:
             # Standard pinhole unprojection
            fx, fy, cx, cy = intr
            
            ys, xs = torch.meshgrid(
                torch.arange(0, H, spatial_downsample, dtype=torch.float32),
                torch.arange(0, W, spatial_downsample, dtype=torch.float32),
                indexing="ij"
            )
            
            # Directions in camera frame
            z_cam = torch.ones_like(xs)
            x_cam = (xs - cx) / fx
            y_cam = (ys - cy) / fy
            
            # (H_ds, W_ds, 3)
            rays_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
            
        elif camera_type == CameraType.PANORAMA:
            # Simplified Panorama (Equirectangular)
            ys, xs = torch.meshgrid(
                torch.arange(0, H, spatial_downsample, dtype=torch.float32),
                torch.arange(0, W, spatial_downsample, dtype=torch.float32),
                indexing="ij"
            )
            u = xs / (W - 1)
            v = ys / (H - 1)
            
            lon = (u - 0.5) * 2 * np.pi
            lat = -(v - 0.5) * np.pi
            
            x_cam = torch.cos(lat) * torch.sin(lon)
            y_cam = torch.sin(lat)
            z_cam = torch.cos(lat) * torch.cos(lon)
            
            rays_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

        else:
             continue

        # Filter valid depth
        # valid_mask = reliable_depth_mask_range(depth_ds) & (depth_ds > 0)
        valid_mask = depth_ds > 0
        
        rays_valid = rays_cam[valid_mask]
        depth_valid = depth_ds[valid_mask]
        rgb_valid = rgb_ds[valid_mask]
        
        # 3D points in camera frame
        points_cam = rays_valid * depth_valid.unsqueeze(-1)
        
        # Transform to world
        R = torch.from_numpy(c2w[:3, :3]).float()
        t = torch.from_numpy(c2w[:3, 3]).float()
        
        points_world = (points_cam @ R.T) + t
        
        # Add to list
        pts_np = points_world.numpy()
        rgb_np = (rgb_valid * 255).byte().numpy()
        
        for i in range(len(pts_np)):
            all_points.append((
                float(pts_np[i, 0]), float(pts_np[i, 1]), float(pts_np[i, 2]),
                int(rgb_np[i, 0]), int(rgb_np[i, 1]), int(rgb_np[i, 2])
            ))
            
        # --- Add Camera Frustum ---
        cam_points, cam_edges = get_camera_frustum_geometry(
            c2w, intr, (H, W), frustum_scale=0.2
        )
        
        # Adjust indices in edges by current vertex_count
        current_v_count = len(all_points)
        for p in cam_points:
            all_points.append(p)
            
        for e in cam_edges:
            # Note: we added cam_points after accumulated scene points
            # The indices in cam_edges are 0-based relative to cam_points
            # So we add current_v_count which is index of first cam point
            # Wait, current_v_count is length before adding cam points.
            all_edges.append((e[0] + current_v_count, e[1] + current_v_count))
            
        processed_frames += 1

    logger.info(f"Processed {processed_frames} frames. Total points: {len(all_points)}")
    
    # Write output
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
        
    write_ply(output_path, all_points, all_edges)


def main():
    parser = argparse.ArgumentParser(description="Convert ViPE results to a single PLY file.")
    parser.add_argument("vipe_path", type=Path, help="Path to ViPE results directory")
    parser.add_argument("--sequence", "-s", type=str, default=None, help="Specific sequence name")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output PLY file path")
    parser.add_argument("--spatial_downsample", type=int, default=20, help="Spatial downsample factor (default: 20)")
    parser.add_argument("--temporal_downsample", type=int, default=20, help="Temporal downsample factor (default: 20)")
    
    args = parser.parse_args()

    if not args.vipe_path.exists():
        logger.error(f"Path does not exist: {args.vipe_path}")
        return

    artifacts = list(ArtifactPath.glob_artifacts(args.vipe_path, use_video=True))
    if args.sequence:
        artifacts = [a for a in artifacts if a.artifact_name == args.sequence]
        
    if not artifacts:
        logger.error("No artifacts found.")
        return
        
    for artifact in artifacts:
        if args.output:
            out_file = args.output
        else:
            out_file = args.vipe_path / f"{artifact.artifact_name}.ply"
            
        process_sequence(artifact, out_file, args.spatial_downsample, args.temporal_downsample)


if __name__ == "__main__":
    main()

