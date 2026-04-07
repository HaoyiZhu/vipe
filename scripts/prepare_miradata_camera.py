"""Prepare MiraData camera poses for Sana CamCtrl training.

Reads VIPE pose-only outputs (slam_intermediate.npz + info.pkl) and produces
per-zip camera NPZ files that sit alongside the output zips:

  miradata_part0000.zip          (existing, untouched)
  miradata_part0000_camera.npz   (new: poses + intrinsics for valid clips)

Camera NPZ format (same as sekai pipeline):
  - pose:       (N_total, 4, 4) concatenated cam2world matrices
  - intrinsics: (N_total, 4)    concatenated [fx, fy, cx, cy]
  - ids:        (n_videos,)     clip IDs that passed filtering
  - ranges:     (n_videos, 2)   [start_idx, length] per video

Clips that fail quality checks are simply omitted from the NPZ. The training
dataloader detects missing camera entries and skips those videos.

Quality filters (kept from sekai pipeline):
  - BA residual must be finite and < threshold
  - Intrinsics: FOV in [min_fov, max_fov], fx/fy ratio within tolerance
  - Poses must be all finite
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_ba_residual(info_path: str) -> float | None:
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        val = info.get("ba_residual", None)
        return float(val) if val is not None else None
    except Exception:
        return None


def intrinsics_valid(
    intr: np.ndarray,
    width: int,
    height: int,
    min_fov_deg: float,
    max_fov_deg: float,
    max_focal_ratio_diff: float,
) -> bool:
    if intr.size == 0 or not np.isfinite(intr).all():
        return False
    fx, fy = intr[:, 0], intr[:, 1]
    if (fx <= 0).any() or (fy <= 0).any():
        return False

    min_fov_rad = math.radians(min_fov_deg)
    max_fov_rad = math.radians(max_fov_deg)
    fx_lo = (width / 2.0) / math.tan(max_fov_rad / 2.0)
    fy_lo = (height / 2.0) / math.tan(max_fov_rad / 2.0)
    fx_hi = (width / 2.0) / math.tan(min_fov_rad / 2.0)
    fy_hi = (height / 2.0) / math.tan(min_fov_rad / 2.0)

    if np.any(fx < fx_lo) or np.any(fx > fx_hi):
        return False
    if np.any(fy < fy_lo) or np.any(fy > fy_hi):
        return False

    ratio_diff = np.abs(fx - fy) / np.maximum(np.minimum(np.abs(fx), np.abs(fy)), 1e-6)
    if (ratio_diff > max_focal_ratio_diff).any():
        return False
    return True


def build_vipe_index(vipe_root: str) -> dict[str, str]:
    """Scan all group dirs and build clip_id -> group_dir mapping."""
    index = {}
    vipe_root = os.path.abspath(vipe_root)
    for entry in sorted(os.listdir(vipe_root)):
        group_dir = os.path.join(vipe_root, entry)
        if not os.path.isdir(group_dir) or not entry.startswith("group_"):
            continue
        vipe_dir = os.path.join(group_dir, "vipe")
        if not os.path.isdir(vipe_dir):
            continue
        for fname in os.listdir(vipe_dir):
            if fname.endswith("_slam_intermediate.npz"):
                clip_id = fname.replace("_slam_intermediate.npz", "")
                index[clip_id] = group_dir
    return index


def load_clip_meta(staging_dir: str, clip_id: str) -> dict | None:
    meta_path = os.path.join(staging_dir, clip_id + ".json")
    if not os.path.exists(meta_path):
        return None
    try:
        import json
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def process_one_zip(
    zip_idx: int,
    manifest_path: str,
    vipe_index: dict[str, str],
    output_dir: str,
    ba_residual_max: float,
    min_fov_deg: float,
    max_fov_deg: float,
    max_focal_ratio_diff: float,
) -> dict[str, int]:
    """Process one output zip's manifest and write its _camera.npz."""
    stats = {"total": 0, "success": 0, "no_vipe": 0, "bad_ba": 0,
             "bad_pose": 0, "bad_intrinsics": 0, "error": 0}

    with open(manifest_path) as f:
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            entries.append((parts[0], parts[1]))  # staging_dir, clip_id

    all_poses = []
    all_intrinsics = []
    video_ids = []
    video_ranges = []
    idx_counter = 0

    for staging_dir, clip_id in tqdm(entries, desc=f"zip_{zip_idx:04d}", leave=False):
        stats["total"] += 1

        group_dir = vipe_index.get(clip_id)
        if group_dir is None:
            stats["no_vipe"] += 1
            continue

        info_path = os.path.join(group_dir, "vipe", f"{clip_id}_info.pkl")
        npz_path = os.path.join(group_dir, "vipe", f"{clip_id}_slam_intermediate.npz")

        try:
            ba_res = load_ba_residual(info_path)
            if ba_res is None or not math.isfinite(ba_res) or ba_res >= ba_residual_max:
                stats["bad_ba"] += 1
                continue

            if not os.path.exists(npz_path):
                stats["no_vipe"] += 1
                continue

            with np.load(npz_path) as data:
                pose = data["pose"]
                intr = data["intrinsics"]

            if pose.ndim != 3 or pose.shape[1:] != (4, 4):
                stats["bad_pose"] += 1
                continue
            if not np.isfinite(pose).all():
                stats["bad_pose"] += 1
                continue

            if intr.ndim == 1:
                intr = np.tile(intr[:4], (pose.shape[0], 1))
            if intr.ndim != 2 or intr.shape[1] < 4:
                stats["bad_intrinsics"] += 1
                continue
            intr = intr[:, :4]

            if pose.shape[0] != intr.shape[0]:
                stats["bad_pose"] += 1
                continue

            meta = load_clip_meta(staging_dir, clip_id)
            if meta is None:
                stats["error"] += 1
                continue
            width = meta.get("width", 0)
            height = meta.get("height", 0)
            if width <= 0 or height <= 0:
                stats["error"] += 1
                continue

            if not intrinsics_valid(intr, width, height, min_fov_deg, max_fov_deg, max_focal_ratio_diff):
                stats["bad_intrinsics"] += 1
                continue

            n_frames = pose.shape[0]
            all_poses.append(pose.astype(np.float32))
            all_intrinsics.append(intr.astype(np.float32))
            video_ids.append(clip_id)
            video_ranges.append([idx_counter, n_frames])
            idx_counter += n_frames
            stats["success"] += 1

        except Exception as e:
            stats["error"] += 1
            continue

    zip_name = f"miradata_part{zip_idx:04d}"
    npz_out = os.path.join(output_dir, f"{zip_name}_camera.npz")

    if all_poses:
        huge_pose = np.concatenate(all_poses, axis=0)
        huge_intr = np.concatenate(all_intrinsics, axis=0)
        ids = np.array(video_ids)
        ranges = np.array(video_ranges, dtype=np.int32)
        np.savez(npz_out, pose=huge_pose, intrinsics=huge_intr, ids=ids, ranges=ranges)
        print(f"  {zip_name}_camera.npz: {len(video_ids)} videos, {huge_pose.shape[0]} frames")
    else:
        print(f"  {zip_name}_camera.npz: no valid videos")

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, help="Phase2 manifests dir")
    parser.add_argument("--vipe-root", required=True, help="VIPE results root (contains group_* dirs)")
    parser.add_argument("--output-dir", required=True, help="Output dir (same as zip output dir)")
    parser.add_argument("--ba-residual-max", type=float, default=5e-4)
    parser.add_argument("--min-fov-deg", type=float, default=25.0)
    parser.add_argument("--max-fov-deg", type=float, default=120.0)
    parser.add_argument("--max-focal-ratio-diff", type=float, default=0.15)
    parser.add_argument("--max-zips", type=int, default=0, help="Limit zips for testing (0=all)")
    args = parser.parse_args()

    manifest_dir = os.path.abspath(args.manifest_dir)
    vipe_root = os.path.abspath(args.vipe_root)
    output_dir = os.path.abspath(args.output_dir)

    print(f"[*] Building VIPE index from {vipe_root}...")
    vipe_index = build_vipe_index(vipe_root)
    print(f"[*] Found {len(vipe_index)} annotated clips")

    manifests = sorted(
        f for f in os.listdir(manifest_dir)
        if f.startswith("zip_") and f.endswith(".txt")
    )
    if args.max_zips > 0:
        manifests = manifests[:args.max_zips]

    print(f"[*] Processing {len(manifests)} zip manifests...")

    total_stats = {"total": 0, "success": 0, "no_vipe": 0, "bad_ba": 0,
                   "bad_pose": 0, "bad_intrinsics": 0, "error": 0}

    for manifest_file in manifests:
        zip_idx = int(manifest_file.replace("zip_", "").replace(".txt", ""))
        manifest_path = os.path.join(manifest_dir, manifest_file)

        stats = process_one_zip(
            zip_idx=zip_idx,
            manifest_path=manifest_path,
            vipe_index=vipe_index,
            output_dir=output_dir,
            ba_residual_max=args.ba_residual_max,
            min_fov_deg=args.min_fov_deg,
            max_fov_deg=args.max_fov_deg,
            max_focal_ratio_diff=args.max_focal_ratio_diff,
        )
        for k in total_stats:
            total_stats[k] += stats[k]

    print()
    print("[*] Summary:")
    for k, v in total_stats.items():
        print(f"    {k}: {v}")
    pct = total_stats["success"] * 100 / max(total_stats["total"], 1)
    print(f"    pass rate: {pct:.1f}%")
    print("[*] Done!")


if __name__ == "__main__":
    main()
