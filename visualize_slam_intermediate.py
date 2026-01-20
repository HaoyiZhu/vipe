#!/usr/bin/env python3

import argparse
import logging
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def write_ply(
    path: Path,
    points: List[Tuple[float, float, float, int, int, int]],
    edges: List[Tuple[int, int]] | None = None,
) -> None:
    """
    Write a binary little-endian PLY with vertices (x,y,z,r,g,b) and optional edges.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
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

        s_point = struct.Struct("<fffBBB")
        for p in points:
            f.write(s_point.pack(*p))

        if edges:
            s_edge = struct.Struct("<ii")
            for e in edges:
                f.write(s_edge.pack(*e))


def get_camera_frustum_geometry(
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: Tuple[int, int],
    frustum_scale: float = 0.2,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Tuple[List[Tuple[float, float, float, int, int, int]], List[Tuple[int, int]]]:
    """
    Generate frustum vertices/edges for a pinhole camera.
    Returns indices relative to the returned points list (0..4).
    """
    H, W = image_shape
    fx, fy, cx, cy = (float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3]))

    center = c2w[:3, 3]

    # Image plane corners at z=1 in camera coords
    corners_2d = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    corners_3d_cam = np.zeros((4, 3), dtype=np.float32)
    corners_3d_cam[:, 2] = 1.0
    corners_3d_cam[:, 0] = (corners_2d[:, 0] - cx) / fx
    corners_3d_cam[:, 1] = (corners_2d[:, 1] - cy) / fy
    corners_3d_cam *= float(frustum_scale)

    corners_3d_world = (c2w[:3, :3] @ corners_3d_cam.T).T + center[None]

    points: List[Tuple[float, float, float, int, int, int]] = []
    edges: List[Tuple[int, int]] = []

    points.append((float(center[0]), float(center[1]), float(center[2]), color[0], color[1], color[2]))
    for i in range(4):
        p = corners_3d_world[i]
        points.append((float(p[0]), float(p[1]), float(p[2]), color[0], color[1], color[2]))
        edges.append((0, i + 1))
        edges.append((i + 1, (i + 1) % 4 + 1 if i < 3 else 1))

    return points, edges


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Visualize ViPE intermediate SLAM outputs (*_slam_intermediate.npz) by exporting a camera-frustum PLY."
    )
    parser.add_argument("slam_npz", type=Path, help="Path to *_slam_intermediate.npz")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output PLY path (default: alongside npz)")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride for visualization (default: 10)")
    parser.add_argument("--image-height", type=int, default=720, help="Image height for frustum geometry (default: 720)")
    parser.add_argument("--image-width", type=int, default=1280, help="Image width for frustum geometry (default: 1280)")
    parser.add_argument("--frustum-scale", type=float, default=0.2, help="Frustum scale (default: 0.2)")
    parser.add_argument("--traj", action="store_true", help="Add trajectory edges connecting camera centers")
    args = parser.parse_args()

    data = np.load(args.slam_npz)
    pose = data["pose"]  # (N,4,4)
    intr = data["intrinsics"]  # (4,)
    inds = data.get("inds", np.arange(pose.shape[0], dtype=np.int64))

    assert pose.ndim == 3 and pose.shape[1:] == (4, 4), f"Unexpected pose shape: {pose.shape}"
    assert intr.shape[0] >= 4, f"Unexpected intrinsics shape: {intr.shape}"

    out_ply = args.output
    if out_ply is None:
        out_ply = args.slam_npz.with_suffix("").with_suffix(".ply")

    points: List[Tuple[float, float, float, int, int, int]] = []
    edges: List[Tuple[int, int]] = []

    center_vertex_ids: List[int] = []
    last_center_vid: int | None = None

    stride = max(1, int(args.stride))
    for k in range(0, pose.shape[0], stride):
        c2w = pose[k]

        fr_pts, fr_edges = get_camera_frustum_geometry(
            c2w=c2w,
            intrinsics=intr[:4],
            image_shape=(int(args.image_height), int(args.image_width)),
            frustum_scale=float(args.frustum_scale),
            color=(255, 0, 0),
        )

        base_vid = len(points)
        points.extend(fr_pts)
        edges.extend([(a + base_vid, b + base_vid) for (a, b) in fr_edges])

        center_vid = base_vid  # first point is center
        center_vertex_ids.append(center_vid)
        if args.traj and last_center_vid is not None:
            edges.append((last_center_vid, center_vid))
        last_center_vid = center_vid

    write_ply(out_ply, points, edges if edges else None)
    logger.info(
        "Wrote %s (frames=%d, stride=%d, frustums=%d, vertices=%d, edges=%d)",
        str(out_ply),
        pose.shape[0],
        stride,
        len(center_vertex_ids),
        len(points),
        len(edges),
    )
    logger.info("Tip: open the PLY in MeshLab/CloudCompare and look for discontinuities in frustum positions/orientations.")


if __name__ == "__main__":
    main()


