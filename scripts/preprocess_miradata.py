#!/usr/bin/env python3
"""
Preprocess miradata: split videos by scene cuts, filter by quality/length,
normalize to 16fps, and package into output zips (max 10k clips each).

Two-phase architecture:
  Phase 1 (parallel): Workers process input zips into per-zip staging zips
  Phase 2 (sequential): Merge staging zips into final output zips with clip limit
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

logger = logging.getLogger("preprocess_miradata")

CAPTION_FIELDS = [
    "short_caption",
    "dense_caption",
    "background_caption",
    "main_object_caption",
    "style_caption",
    "camera_caption",
]
SCORE_TYPES = ["dover", "color", "unimatch", "vmafmotion", "scene_cut"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess miradata: scene-split, filter, normalize fps, repackage"
    )
    p.add_argument("--input-dir", required=True, help="Input miradata directory")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--num-workers", type=int, default=min(cpu_count(), 32))
    p.add_argument("--dover-threshold", type=float, default=0.3)
    p.add_argument("--min-frames", type=int, default=481, help="Min frames (16n+1)")
    p.add_argument("--max-frames", type=int, default=961, help="Window size (16n+1)")
    p.add_argument("--stride", type=int, default=480, help="Sliding window stride")
    p.add_argument("--target-fps", type=int, default=16)
    p.add_argument("--max-clips-per-zip", type=int, default=10000)
    p.add_argument("--resume", action="store_true", help="Skip already-staged zips")
    p.add_argument("--phase", choices=["1", "2", "all"], default="all",
                   help="Run phase 1 (process), 2 (package), or all")
    p.add_argument("--zip-list", type=str, default=None,
                   help="File listing zip basenames to process (one per line), "
                        "for distributed Phase 1")
    p.add_argument("--tmp-dir", type=str, default=None,
                   help="Directory for temp files (default: system temp). "
                        "Use local SSD like /tmp on cluster nodes.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def snap_to_16n_plus_1(n):
    """Round down to nearest 16k+1.  E.g. 500->497, 961->961, 481->481."""
    return ((n - 1) // 16) * 16 + 1


def load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def get_frame_count(video_path):
    """Return frame count via ffprobe, or None on failure."""
    # Method 1: container metadata (instant)
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            v = r.stdout.strip()
            if v and v != "N/A":
                return int(v)
    except Exception:
        pass
    # Method 2: count packets without decoding (fast)
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_packets",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            v = r.stdout.strip()
            if v and v != "N/A":
                return int(v)
    except Exception:
        pass
    # Method 3: full decode count (slow, last resort)
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip())
    except Exception:
        pass
    return None


def ffmpeg_cut(input_path, output_path, start_sec, target_frames, target_fps,
               source_fps):
    """
    Extract exactly *target_frames* frames starting at *start_sec*.

    Fast path (stream copy) is attempted when source fps matches target fps.
    Falls back to libx264 re-encode if the copy produces the wrong frame count.

    Returns the actual frame count of the output file.
    """
    same_fps = abs(source_fps - target_fps) < 0.5

    if same_fps:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start_sec:.6f}",
            "-i", input_path,
            "-frames:v", str(target_frames),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            actual = get_frame_count(output_path)
            if actual is not None and actual == target_frames:
                return actual
        if os.path.exists(output_path):
            os.remove(output_path)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}",
        "-i", input_path,
        "-frames:v", str(target_frames),
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr[:500]}")

    actual = get_frame_count(output_path)
    if actual is not None:
        return actual
    # ffmpeg succeeded with -frames:v N -- trust it
    return target_frames


def build_clip_metadata(orig_meta, actual_frames, fps):
    meta = {
        "width": orig_meta.get("width"),
        "height": orig_meta.get("height"),
        "fps": fps,
        "seconds": actual_frames / fps,
        "frame_number": actual_frames,
        "source": orig_meta.get("source", ""),
    }
    for field in CAPTION_FIELDS:
        meta[field] = ""
    return meta


# ---------------------------------------------------------------------------
# Phase 1 – parallel per-zip processing
# ---------------------------------------------------------------------------

def process_one_zip(args):
    """Process a single input zip → staging zip + manifest + score files."""
    zip_path, input_dir, staging_dir, config = args
    base = Path(zip_path).stem

    manifest_path = os.path.join(staging_dir, f"{base}_manifest.json")
    if config.get("resume") and os.path.isfile(manifest_path):
        stg = os.path.join(staging_dir, f"{base}.zip")
        if os.path.isfile(stg):
            logger.info(f"[{base}] Skipping (resume)")
            return base

    logger.info(f"[{base}] Starting")

    # ---- Load sidecar JSONs ----
    sc_data = load_json(os.path.join(input_dir, f"{base}_scene_cut.json"))
    if sc_data is None:
        logger.warning(f"[{base}] No scene_cut.json – skipping entire zip")
        return None

    dover_data = load_json(os.path.join(input_dir, f"{base}_dover.json")) or {}
    color_data = load_json(os.path.join(input_dir, f"{base}_color.json")) or {}
    uni_data = load_json(os.path.join(input_dir, f"{base}_unimatch.json")) or {}
    vmaf_data = load_json(os.path.join(input_dir, f"{base}_vmafmotion.json")) or {}

    target_fps = config["target_fps"]
    min_frames = config["min_frames"]
    max_frames = config["max_frames"]
    stride = config["stride"]
    dover_thresh = config["dover_threshold"]

    stg_zip_path = os.path.join(staging_dir, f"{base}.zip")
    manifest = []
    new_scores = {s: {} for s in SCORE_TYPES}
    stats = {
        "written": 0, "skip_dover": 0, "skip_scene": 0,
        "skip_short": 0, "errors": 0,
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zin:
            all_names = set(zin.namelist())
            video_names = sorted(n for n in all_names if n.endswith(".mp4"))
            logger.info(f"[{base}] {len(video_names)} videos")

            with tempfile.TemporaryDirectory(dir=config.get("tmp_dir")) as tmpdir:
                zout = zipfile.ZipFile(stg_zip_path, "w", zipfile.ZIP_STORED)
                try:
                    for vi, vname in enumerate(video_names):
                        clip_id = Path(vname).stem

                        if (vi + 1) % 100 == 0:
                            logger.info(
                                f"[{base}] {vi+1}/{len(video_names)} videos, "
                                f"{stats['written']} clips so far"
                            )

                        # -- Dover filter --
                        clip_dover = dover_data.get(clip_id)
                        if clip_dover is None or clip_dover.get("dover_score", 0) < dover_thresh:
                            stats["skip_dover"] += 1
                            continue

                        # -- Scene cuts --
                        clip_sc = sc_data.get(clip_id)
                        if clip_sc is None or not clip_sc.get("scenes"):
                            stats["skip_scene"] += 1
                            continue
                        scenes = clip_sc["scenes"]

                        # -- In-zip metadata --
                        json_name = f"{clip_id}.json"
                        if json_name not in all_names:
                            logger.warning(f"[{base}] No metadata JSON for {clip_id}")
                            continue
                        with zin.open(json_name) as jf:
                            orig_meta = json.load(jf)

                        source_fps = orig_meta.get("fps", target_fps)

                        # -- Extract video to disk --
                        vtmp = os.path.join(tmpdir, f"_src_{clip_id}.mp4")
                        with zin.open(vname) as src, open(vtmp, "wb") as dst:
                            shutil.copyfileobj(src, dst)

                        try:
                            _process_clip_scenes(
                                clip_id, vtmp, orig_meta, scenes,
                                source_fps, target_fps, min_frames,
                                max_frames, stride, tmpdir, zout,
                                manifest, new_scores, stats,
                                dover_data, color_data, uni_data, vmaf_data,
                                base,
                            )
                        except Exception as e:
                            logger.error(f"[{base}] Error on {clip_id}: {e}")
                            stats["errors"] += 1
                        finally:
                            if os.path.exists(vtmp):
                                os.remove(vtmp)

                finally:
                    zout.close()

    except Exception:
        logger.error(f"[{base}] Fatal error:\n{traceback.format_exc()}")
        if os.path.exists(stg_zip_path):
            os.remove(stg_zip_path)
        return None

    if not manifest:
        logger.info(f"[{base}] No output clips")
        if os.path.exists(stg_zip_path):
            os.remove(stg_zip_path)
        return None

    # Persist manifest + scores to staging dir (avoids large IPC payloads)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    for st in SCORE_TYPES:
        with open(os.path.join(staging_dir, f"{base}_scores_{st}.json"), "w") as f:
            json.dump(new_scores[st], f)

    logger.info(
        f"[{base}] Done: written={stats['written']}  "
        f"skip_dover={stats['skip_dover']}  skip_short={stats['skip_short']}  "
        f"skip_scene={stats['skip_scene']}  errors={stats['errors']}"
    )
    return base


def _process_clip_scenes(
    clip_id, vtmp, orig_meta, scenes,
    source_fps, target_fps, min_frames, max_frames, stride,
    tmpdir, zout, manifest, new_scores, stats,
    dover_data, color_data, uni_data, vmaf_data, base,
):
    """Split one clip's scenes into output sub-clips (single or sliding-window)."""
    for si, scene in enumerate(scenes):
        start_sec = scene["start"]
        end_sec = scene["end"]
        duration = end_sec - start_sec
        if duration <= 0:
            continue

        scene_frames = round(duration * target_fps)

        if scene_frames < min_frames:
            stats["skip_short"] += 1
            continue

        # Build list of (new_id, seek_sec, n_frames) to cut
        cuts = []
        if scene_frames <= max_frames:
            target_n = snap_to_16n_plus_1(scene_frames)
            if target_n < min_frames:
                stats["skip_short"] += 1
                continue
            cuts.append((f"{clip_id}_s{si}", start_sec, target_n))
        else:
            window_starts = list(range(0, scene_frames - max_frames + 1, stride))
            if not window_starts or window_starts[-1] + max_frames < scene_frames:
                window_starts.append(scene_frames - max_frames)
            window_starts = sorted(set(window_starts))
            for wi, sf in enumerate(window_starts):
                cuts.append((
                    f"{clip_id}_s{si}_w{wi}",
                    start_sec + sf / target_fps,
                    max_frames,
                ))

        for new_id, seek_sec, n_frames in cuts:
            out_path = os.path.join(tmpdir, f"{new_id}.mp4")
            try:
                actual_frames = ffmpeg_cut(
                    vtmp, out_path, seek_sec, n_frames, target_fps, source_fps,
                )
            except Exception as e:
                logger.error(f"[{base}] ffmpeg failed for {new_id}: {e}")
                stats["errors"] += 1
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            if actual_frames < min_frames:
                logger.warning(
                    f"[{base}] {new_id}: got {actual_frames} frames, skipping"
                )
                stats["errors"] += 1
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            new_meta = build_clip_metadata(orig_meta, actual_frames, target_fps)
            zout.write(out_path, f"{new_id}.mp4")
            zout.writestr(f"{new_id}.json", json.dumps(new_meta, indent=2))
            os.remove(out_path)

            # Accumulate scores
            manifest.append(new_id)
            if clip_id in dover_data:
                new_scores["dover"][new_id] = dover_data[clip_id]
            if clip_id in color_data:
                new_scores["color"][new_id] = color_data[clip_id]
            if clip_id in uni_data:
                new_scores["unimatch"][new_id] = uni_data[clip_id]
            if clip_id in vmaf_data:
                new_scores["vmafmotion"][new_id] = vmaf_data[clip_id]
            new_scores["scene_cut"][new_id] = {
                "num_scenes": 1,
                "scenes": [{"start": 0.0, "end": actual_frames / target_fps}],
                "threshold": 0.5,
            }
            stats["written"] += 1


# ---------------------------------------------------------------------------
# Phase 2 – sequential packaging
# ---------------------------------------------------------------------------

def package_outputs(staging_dir, output_dir, max_clips):
    """Merge staging zips into final output zips, capped at *max_clips* each."""
    os.makedirs(output_dir, exist_ok=True)

    manifest_files = sorted(glob.glob(os.path.join(staging_dir, "*_manifest.json")))
    if not manifest_files:
        logger.info("No staging outputs to package")
        return

    # Collect all clip references and scores
    all_clips = []  # (staging_zip_path, clip_id)
    all_scores = {s: {} for s in SCORE_TYPES}

    for mf in manifest_files:
        base = Path(mf).stem.replace("_manifest", "")
        sz = os.path.join(staging_dir, f"{base}.zip")
        if not os.path.isfile(sz):
            logger.warning(f"Staging zip missing for {base}, skipping")
            continue
        with open(mf) as f:
            clip_ids = json.load(f)
        all_clips.extend((sz, cid) for cid in clip_ids)
        for st in SCORE_TYPES:
            sp = os.path.join(staging_dir, f"{base}_scores_{st}.json")
            sd = load_json(sp)
            if sd:
                all_scores[st].update(sd)

    # Deduplicate (unlikely but be safe)
    seen = set()
    deduped = []
    for sz, cid in all_clips:
        if cid in seen:
            logger.warning(f"Duplicate clip_id '{cid}' – skipping")
            continue
        seen.add(cid)
        deduped.append((sz, cid))
    all_clips = deduped

    total = len(all_clips)
    logger.info(f"Phase 2: packaging {total} clips (max {max_clips}/zip)")

    zip_idx = 0
    clip_count = 0
    cur_zip = None
    cur_scores = {s: {} for s in SCORE_TYPES}
    open_staging = {}  # cache open staging ZipFile handles

    def _close_zip():
        nonlocal zip_idx, clip_count, cur_zip, cur_scores
        if cur_zip is None:
            return
        cur_zip.close()
        zip_name = f"miradata_part{zip_idx:04d}"
        for st in SCORE_TYPES:
            if cur_scores[st]:
                out_path = os.path.join(output_dir, f"{zip_name}_{st}.json")
                with open(out_path, "w") as f:
                    json.dump(cur_scores[st], f, indent=2)
        logger.info(f"  Written {zip_name}.zip ({clip_count} clips)")
        zip_idx += 1
        clip_count = 0
        cur_zip = None
        cur_scores = {s: {} for s in SCORE_TYPES}

    def _ensure_zip():
        nonlocal cur_zip
        if cur_zip is None:
            zip_name = f"miradata_part{zip_idx:04d}"
            zpath = os.path.join(output_dir, f"{zip_name}.zip")
            cur_zip = zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED)

    try:
        for i, (sz, cid) in enumerate(all_clips):
            if clip_count >= max_clips:
                _close_zip()

            _ensure_zip()

            if sz not in open_staging:
                open_staging[sz] = zipfile.ZipFile(sz, "r")
            szf = open_staging[sz]

            for ext in (".mp4", ".json"):
                entry = f"{cid}{ext}"
                if entry in szf.namelist():
                    cur_zip.writestr(entry, szf.read(entry))

            for st in SCORE_TYPES:
                if cid in all_scores[st]:
                    cur_scores[st][cid] = all_scores[st][cid]

            clip_count += 1

            if (i + 1) % 5000 == 0:
                logger.info(f"  Packaged {i+1}/{total}")

        _close_zip()

    finally:
        for szf in open_staging.values():
            szf.close()

    logger.info(f"Packaging complete: {zip_idx} output zip(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(processName)s  %(message)s",
    )

    # Dependency check
    for cmd in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([cmd, "-version"], capture_output=True, timeout=10)
        except FileNotFoundError:
            logger.error(f"'{cmd}' not found – please install ffmpeg")
            sys.exit(1)

    # Discover input zips
    all_zips = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".zip") and not f.startswith(".")
    )
    if not all_zips:
        logger.error(f"No zip files found in {input_dir}")
        sys.exit(1)

    # Filter to zip-list if provided (for distributed Phase 1)
    if args.zip_list:
        with open(args.zip_list) as f:
            wanted = {line.strip() for line in f if line.strip()}
        zip_files = [os.path.join(input_dir, z) for z in all_zips if z in wanted]
        logger.info(f"Zip-list: {len(zip_files)}/{len(all_zips)} zips selected")
    else:
        zip_files = [os.path.join(input_dir, z) for z in all_zips]

    logger.info(f"Input: {len(zip_files)} zips in {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(
        f"Config: fps={args.target_fps}  min_frames={args.min_frames}  "
        f"max_frames={args.max_frames}  stride={args.stride}  "
        f"dover>={args.dover_threshold}  max_clips/zip={args.max_clips_per_zip}  "
        f"workers={args.num_workers}  resume={args.resume}  phase={args.phase}"
    )

    staging_dir = os.path.join(output_dir, ".staging")
    os.makedirs(staging_dir, exist_ok=True)

    config = {
        "target_fps": args.target_fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "stride": args.stride,
        "dover_threshold": args.dover_threshold,
        "resume": args.resume,
        "tmp_dir": args.tmp_dir,
    }

    run_phase1 = args.phase in ("1", "all")
    run_phase2 = args.phase in ("2", "all")

    # ---- Phase 1 ----
    if run_phase1:
        tasks = [(zp, input_dir, staging_dir, config) for zp in zip_files]
        logger.info("=" * 60)
        logger.info(f"PHASE 1: Processing {len(tasks)} zips")
        logger.info("=" * 60)

        results = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one_zip, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
                    results.append(None)

        ok_count = sum(1 for r in results if r is not None)
        logger.info(f"Phase 1 complete: {ok_count}/{len(tasks)} zips produced clips")

    # ---- Phase 2 ----
    if run_phase2:
        logger.info("=" * 60)
        logger.info("PHASE 2: Packaging output zips")
        logger.info("=" * 60)

        package_outputs(staging_dir, output_dir, args.max_clips_per_zip)

        logger.info("Cleaning up staging area...")
        shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info("All done!")


if __name__ == "__main__":
    main()
