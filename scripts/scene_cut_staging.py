#!/usr/bin/env python3
"""TransNetV2-based scene cut detection for processed clips in staging dirs.

Adapted from Sana/dev/haozhu/tools/scene_cut/detect_scenes_transnetv2.py
to work on individual mp4 files in staging directories instead of zip archives.

Writes ``_scores_scene_cut.json`` into each staging dir with per-clip results.

Usage:
    # Single staging dir
    python scene_cut_staging.py --staging-dir miradata_processed/.staging/mira_00000002_part1

    # Slurm array (one staging dir per task)
    python scene_cut_staging.py --staging-root miradata_processed/.staging --job-id $SLURM_ARRAY_TASK_ID

    # All staging dirs sequentially
    python scene_cut_staging.py --staging-root miradata_processed/.staging
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from tqdm import tqdm

SANA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Sana"))
TRANSNETV2_DIR = os.path.join(SANA_ROOT, "local_libs", "TransNetV2", "inference-pytorch")
TRANSNETV2_WEIGHTS = os.path.join(TRANSNETV2_DIR, "transnetv2-pytorch-weights.pth")

TARGET_FPS = 16.0


# ---------------------------------------------------------------------------
# TransNetV2 detector (reused from the original script)
# ---------------------------------------------------------------------------


class TransNetV2Detector:
    FRAME_H, FRAME_W = 27, 48

    def __init__(self, weights_path: str = TRANSNETV2_WEIGHTS, device: str = "cuda"):
        sys.path.insert(0, TRANSNETV2_DIR)
        from transnetv2_pytorch import TransNetV2

        self.device = device
        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        self.model.eval()
        if device != "cpu":
            self.model = self.model.to(device)
        print(f"[TransNetV2] Model loaded on {device}")

    @staticmethod
    def _extract_frames_pyav(video_bytes: bytes) -> np.ndarray:
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        frames = []
        for frame in container.decode(video=0):
            small = frame.reformat(
                width=TransNetV2Detector.FRAME_W,
                height=TransNetV2Detector.FRAME_H,
                format="rgb24",
            )
            frames.append(small.to_ndarray())
        container.close()

        if not frames:
            raise ValueError("Decoded 0 frames from video")
        return np.stack(frames)

    @torch.no_grad()
    def _predict_raw(self, frames: np.ndarray) -> np.ndarray:
        pad_start = 25
        remainder = len(frames) % 50
        pad_end = 25 + (50 - remainder if remainder != 0 else 0)
        padded = np.concatenate(
            [np.repeat(frames[:1], pad_start, axis=0), frames, np.repeat(frames[-1:], pad_end, axis=0)]
        )

        preds: list[np.ndarray] = []
        for ptr in range(0, len(padded) - 100 + 1, 50):
            inp = torch.from_numpy(padded[ptr : ptr + 100][np.newaxis]).to(self.device)
            single_pred, _ = self.model(inp)
            preds.append(torch.sigmoid(single_pred)[0, 25:75, 0].cpu().numpy())

        return np.concatenate(preds)[: len(frames)]

    def detect_scenes(
        self,
        video_bytes: bytes,
        threshold: float = 0.5,
        fps: float = TARGET_FPS,
    ) -> list[dict[str, float]]:
        frames = self._extract_frames_pyav(video_bytes)
        scores = self._predict_raw(frames)
        transitions = (scores > threshold).astype(np.uint8)

        cut_frames = np.where(transitions[1:] & ~transitions[:-1])[0] + 1
        if len(cut_frames) == 0:
            return [{"start": 0.0, "end": round(len(frames) / fps, 3)}]

        boundaries = [0] + cut_frames.tolist() + [len(frames)]
        scenes = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            if e > s:
                scenes.append({"start": round(s / fps, 3), "end": round(e / fps, 3)})
        return scenes if scenes else [{"start": 0.0, "end": round(len(frames) / fps, 3)}]


# ---------------------------------------------------------------------------
# Staging-dir processing
# ---------------------------------------------------------------------------


def load_existing_results(json_path: str) -> dict[str, Any]:
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] Failed to load {json_path}: {e}")
    return {}


def save_results(json_path: str, results: dict[str, Any]) -> None:
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, json_path)


def process_staging_dir(
    stg_dir: str,
    detector: TransNetV2Detector,
    threshold: float = 0.5,
    save_interval: int = 50,
    fresh: bool = False,
) -> dict[str, Any]:
    """Process all mp4 clips in a staging directory."""
    output_json = os.path.join(stg_dir, "_scores_scene_cut.json")
    results = {} if fresh else load_existing_results(output_json)

    mp4_files = sorted(
        f for f in os.listdir(stg_dir)
        if f.endswith(".mp4")
    )
    total = len(mp4_files)

    if not total:
        print(f"[skip] {stg_dir}: no mp4 files")
        save_results(output_json, results)
        return results

    if len(results) >= total:
        print(f"[skip] {stg_dir}: all {total} clips already processed")
        return results

    pending = [f for f in mp4_files if Path(f).stem not in results]
    print(f"[info] {os.path.basename(stg_dir)}: {total} clips, {len(results)} cached, {len(pending)} to process")
    unsaved_count = 0
    t0 = time.time()

    pbar = tqdm(pending, desc=os.path.basename(stg_dir))
    for fname in pbar:
        clip_id = Path(fname).stem
        fpath = os.path.join(stg_dir, fname)

        try:
            with open(fpath, "rb") as f:
                video_bytes = f.read()

            scenes = detector.detect_scenes(video_bytes, threshold=threshold, fps=TARGET_FPS)

            results[clip_id] = {
                "num_scenes": len(scenes),
                "scenes": scenes,
                "threshold": threshold,
            }
            unsaved_count += 1
            pbar.set_postfix(scenes=len(scenes))

        except Exception as e:
            print(f"\n[error] {fname}: {e}")
            continue

        if unsaved_count >= save_interval:
            save_results(output_json, results)
            unsaved_count = 0

    save_results(output_json, results)
    elapsed = time.time() - t0
    print(f"[done] {os.path.basename(stg_dir)}: {len(results)}/{total} clips in {elapsed:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TransNetV2 scene detection for processed clips in staging dirs"
    )
    parser.add_argument("--staging-dir", type=str, help="Single staging directory to process")
    parser.add_argument("--staging-root", type=str, help="Root staging dir (contains mira_* subdirs)")
    parser.add_argument("--job-id", type=int, default=None, help="Slurm array task ID (selects one subdir)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing results and re-annotate all clips from scratch")
    parser.add_argument("--weights", type=str, default=TRANSNETV2_WEIGHTS)
    args = parser.parse_args()

    if args.staging_dir:
        dirs_to_process = [os.path.abspath(args.staging_dir)]
    elif args.staging_root:
        root = os.path.abspath(args.staging_root)
        all_dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.startswith("mira_")
        )
        # Only dirs that have at least one mp4
        all_dirs = [d for d in all_dirs if any(f.endswith(".mp4") for f in os.listdir(d))]
        if not all_dirs:
            print(f"No staging dirs with mp4 files found in {root}")
            return
        if args.job_id is not None:
            if args.job_id >= len(all_dirs):
                print(f"job_id {args.job_id} >= {len(all_dirs)} dirs, nothing to do")
                return
            dirs_to_process = [all_dirs[args.job_id]]
        else:
            dirs_to_process = all_dirs
    else:
        parser.error("Provide --staging-dir or --staging-root")
        return

    print(f"[info] Processing {len(dirs_to_process)} staging dir(s)")
    print(f"[info] Weights: {args.weights}")
    print(f"[info] Threshold: {args.threshold}")

    detector = TransNetV2Detector(weights_path=args.weights, device=args.device)

    for d in dirs_to_process:
        process_staging_dir(d, detector, threshold=args.threshold,
                            save_interval=args.save_interval, fresh=args.fresh)

    print("[info] All done!")


if __name__ == "__main__":
    main()
