#!/usr/bin/env python3
"""
Preprocess miradata: split videos by scene cuts, filter by quality/length,
normalize to 16fps, and package into output zips (max 10k clips each).

Two-phase architecture:
  Phase 1 (parallel): Workers process input zips into per-zip staging directories
                       with per-video checkpointing for crash-safe resume.
  Phase 2 (sequential): Package staging dirs into final output zips with clip limit.
"""

import argparse
import json
import logging
import os
import re
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
                   help="File listing zip basenames to process (one per line)")
    p.add_argument("--tmp-dir", type=str, default=None,
                   help="Local temp dir for ffmpeg I/O (e.g. /tmp)")
    # Phase 2 quality filters
    p.add_argument("--p2-dover-min", type=float, default=0.35,
                   help="Phase 2: min dover_score")
    p.add_argument("--p2-saturation-min", type=float, default=0.0,
                   help="Phase 2: min mean_video_saturation")
    p.add_argument("--p2-saturation-max", type=float, default=180.0,
                   help="Phase 2: max mean_video_saturation")
    p.add_argument("--p2-vmafmotion-min", type=float, default=0.5,
                   help="Phase 2: min vmafmotion_score")
    p.add_argument("--p2-vmafmotion-max", type=float, default=50.0,
                   help="Phase 2: max vmafmotion_score")
    p.add_argument("--p2-unimatch-min", type=float, default=3.0,
                   help="Phase 2: min avg unimatch_flow_score")
    p.add_argument("--p2-unimatch-max", type=float, default=50.0,
                   help="Phase 2: max avg unimatch_flow_score")
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
    for method in (
        ["-show_entries", "stream=nb_frames"],
        ["-count_packets", "-show_entries", "stream=nb_read_packets"],
        ["-count_frames", "-show_entries", "stream=nb_read_frames"],
    ):
        timeout = 600 if "-count_frames" in method else 120
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0"]
                + method + ["-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode == 0:
                v = r.stdout.strip()
                if v and v != "N/A":
                    return int(v)
        except Exception:
            pass
    return None


def ffmpeg_cut(input_path, output_path, start_sec, target_frames, target_fps,
               source_fps):
    """
    Extract exactly *target_frames* frames starting at *start_sec*.
    Tries stream copy first for matching fps, falls back to re-encode.
    Returns the actual frame count.
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
    return actual if actual is not None else target_frames


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


def _original_clip_id(new_id):
    """Extract the original clip_id from an output clip name.
    '8720.0_s2_w0' -> '8720.0',  '1032.13_s0' -> '1032.13'"""
    return re.sub(r"_s\d+(_w\d+)?$", "", new_id)


# ---------------------------------------------------------------------------
# Phase 1 – parallel per-zip processing with per-video checkpointing
# ---------------------------------------------------------------------------

def process_one_zip(args):
    """
    Process a single input zip into a staging directory of individual files.
    Checkpoints after every video so interrupted jobs lose at most one video's work.
    """
    zip_path, input_dir, staging_dir, config = args
    base = Path(zip_path).stem

    stg_dir = os.path.join(staging_dir, base)
    os.makedirs(stg_dir, exist_ok=True)

    manifest_path = os.path.join(stg_dir, "_manifest.json")
    checkpoint_path = os.path.join(stg_dir, "_done_videos.txt")

    # Already fully done?
    if config.get("resume") and os.path.isfile(manifest_path):
        logger.info(f"[{base}] Skipping (fully done)")
        return base

    # Load checkpoint: set of original clip_ids already processed
    done_videos = set()
    if config.get("resume") and os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as f:
            done_videos = {line.strip() for line in f if line.strip()}
        if done_videos:
            logger.info(f"[{base}] Resuming: {len(done_videos)} videos already done")

    logger.info(f"[{base}] Starting")

    # ---- Load sidecar JSONs ----
    sc_data = load_json(os.path.join(input_dir, f"{base}_scene_cut.json"))
    if sc_data is None:
        logger.warning(f"[{base}] No scene_cut.json – skipping entire zip")
        with open(manifest_path, "w") as f:
            json.dump([], f)
        for st in SCORE_TYPES:
            with open(os.path.join(stg_dir, f"_scores_{st}.json"), "w") as f:
                json.dump({}, f)
        return base

    dover_data = load_json(os.path.join(input_dir, f"{base}_dover.json")) or {}
    color_data = load_json(os.path.join(input_dir, f"{base}_color.json")) or {}
    uni_data = load_json(os.path.join(input_dir, f"{base}_unimatch.json")) or {}
    vmaf_data = load_json(os.path.join(input_dir, f"{base}_vmafmotion.json")) or {}

    target_fps = config["target_fps"]
    min_frames = config["min_frames"]
    max_frames = config["max_frames"]
    stride = config["stride"]
    dover_thresh = config["dover_threshold"]

    stats = {
        "written": 0, "skip_dover": 0, "skip_scene": 0,
        "skip_short": 0, "errors": 0, "resumed": len(done_videos),
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zin:
            all_names = set(zin.namelist())
            video_names = sorted(n for n in all_names if n.endswith(".mp4"))
            logger.info(f"[{base}] {len(video_names)} videos")

            with tempfile.TemporaryDirectory(dir=config.get("tmp_dir")) as tmpdir:
                for vi, vname in enumerate(video_names):
                    clip_id = Path(vname).stem

                    if (vi + 1) % 20 == 0:
                        logger.info(
                            f"[{base}] {vi+1}/{len(video_names)} videos, "
                            f"{stats['written']} clips"
                        )

                    # -- Already checkpointed --
                    if clip_id in done_videos:
                        continue

                    # -- Dover filter --
                    clip_dover = dover_data.get(clip_id)
                    if clip_dover is None or clip_dover.get("dover_score", 0) < dover_thresh:
                        stats["skip_dover"] += 1
                        _checkpoint_video(checkpoint_path, clip_id)
                        done_videos.add(clip_id)
                        continue

                    # -- Scene cuts --
                    clip_sc = sc_data.get(clip_id)
                    if clip_sc is None or not clip_sc.get("scenes"):
                        stats["skip_scene"] += 1
                        _checkpoint_video(checkpoint_path, clip_id)
                        done_videos.add(clip_id)
                        continue
                    scenes = clip_sc["scenes"]

                    # -- In-zip metadata --
                    json_name = f"{clip_id}.json"
                    if json_name not in all_names:
                        logger.warning(f"[{base}] No metadata JSON for {clip_id}")
                        _checkpoint_video(checkpoint_path, clip_id)
                        done_videos.add(clip_id)
                        continue
                    with zin.open(json_name) as jf:
                        orig_meta = json.load(jf)

                    source_fps = orig_meta.get("fps", target_fps)

                    # -- Extract video to local temp --
                    vtmp = os.path.join(tmpdir, f"_src_{clip_id}.mp4")
                    with zin.open(vname) as src, open(vtmp, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    try:
                        _process_clip_scenes(
                            clip_id, vtmp, orig_meta, scenes,
                            source_fps, target_fps, min_frames,
                            max_frames, stride, tmpdir, stg_dir,
                            stats, dover_data, color_data, uni_data,
                            vmaf_data, base,
                        )
                    except Exception as e:
                        logger.error(f"[{base}] Error on {clip_id}: {e}")
                        stats["errors"] += 1
                    finally:
                        if os.path.exists(vtmp):
                            os.remove(vtmp)

                    # Checkpoint: this video is done
                    _checkpoint_video(checkpoint_path, clip_id)
                    done_videos.add(clip_id)

    except Exception:
        logger.error(f"[{base}] Fatal error:\n{traceback.format_exc()}")
        return None

    # ---- Build manifest and scores from staging dir ----
    manifest = sorted(
        Path(f).stem for f in os.listdir(stg_dir)
        if f.endswith(".mp4")
    )
    if not manifest:
        logger.info(f"[{base}] No output clips")
        with open(manifest_path, "w") as f:
            json.dump([], f)
        for st in SCORE_TYPES:
            with open(os.path.join(stg_dir, f"_scores_{st}.json"), "w") as f:
                json.dump({}, f)
        return base

    new_scores = _build_scores(manifest, dover_data, color_data, uni_data,
                               vmaf_data, target_fps, stg_dir)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    for st in SCORE_TYPES:
        with open(os.path.join(stg_dir, f"_scores_{st}.json"), "w") as f:
            json.dump(new_scores[st], f)

    logger.info(
        f"[{base}] Done: written={stats['written']}  "
        f"skip_dover={stats['skip_dover']}  skip_short={stats['skip_short']}  "
        f"skip_scene={stats['skip_scene']}  errors={stats['errors']}  "
        f"resumed={stats['resumed']}  total_clips={len(manifest)}"
    )
    return base


def _checkpoint_video(checkpoint_path, clip_id):
    """Append one clip_id to the checkpoint file (fast, append-only)."""
    with open(checkpoint_path, "a") as f:
        f.write(clip_id + "\n")


def _build_scores(manifest, dover_data, color_data, uni_data, vmaf_data,
                  target_fps, stg_dir):
    """Rebuild score dicts from sidecar JSONs for all clips in manifest."""
    scores = {s: {} for s in SCORE_TYPES}
    for new_id in manifest:
        orig_id = _original_clip_id(new_id)

        if orig_id in dover_data:
            scores["dover"][new_id] = dover_data[orig_id]
        if orig_id in color_data:
            scores["color"][new_id] = color_data[orig_id]
        if orig_id in uni_data:
            scores["unimatch"][new_id] = uni_data[orig_id]
        if orig_id in vmaf_data:
            scores["vmafmotion"][new_id] = vmaf_data[orig_id]

        meta_path = os.path.join(stg_dir, f"{new_id}.json")
        meta = load_json(meta_path)
        duration = meta["seconds"] if meta else 0
        scores["scene_cut"][new_id] = {
            "num_scenes": 1,
            "scenes": [{"start": 0.0, "end": duration}],
            "threshold": 0.5,
        }
    return scores


def _process_clip_scenes(
    clip_id, vtmp, orig_meta, scenes,
    source_fps, target_fps, min_frames, max_frames, stride,
    tmpdir, stg_dir, stats,
    dover_data, color_data, uni_data, vmaf_data, base,
):
    """Split one clip's scenes into output sub-clips, writing to staging dir."""
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

            # Write clip files to staging dir (individual files, not zip)
            shutil.move(out_path, os.path.join(stg_dir, f"{new_id}.mp4"))
            with open(os.path.join(stg_dir, f"{new_id}.json"), "w") as f:
                json.dump(new_meta, f, indent=2)

            stats["written"] += 1


# ---------------------------------------------------------------------------
# Phase 2 – sequential packaging
# ---------------------------------------------------------------------------

def _passes_quality_filter(cid, all_scores, filters):
    """Return True if clip passes all Phase 2 quality gates."""
    # Dover
    dover = all_scores["dover"].get(cid)
    if not dover or dover.get("dover_score", 0) < filters["dover_min"]:
        return False

    # Color saturation
    color = all_scores["color"].get(cid)
    if color:
        sat = color.get("mean_video_saturation", -1)
        if sat < filters["saturation_min"] or sat > filters["saturation_max"]:
            return False

    # VMAF motion
    vmaf = all_scores["vmafmotion"].get(cid)
    if vmaf:
        ms = vmaf.get("vmafmotion_score", -1)
        if ms < filters["vmafmotion_min"] or ms > filters["vmafmotion_max"]:
            return False

    # Unimatch (average of per-frame list)
    uni = all_scores["unimatch"].get(cid)
    if uni:
        flow = uni.get("unimatch_flow_score")
        if isinstance(flow, list) and flow:
            avg_flow = sum(flow) / len(flow)
            if avg_flow < filters["unimatch_min"] or avg_flow > filters["unimatch_max"]:
                return False

    return True


def package_outputs(staging_dir, output_dir, max_clips, quality_filters=None):
    """Read from staging dirs, quality-filter, package into final output zips."""
    os.makedirs(output_dir, exist_ok=True)

    # Find completed staging dirs (those with _manifest.json)
    completed = []
    for name in sorted(os.listdir(staging_dir)):
        dpath = os.path.join(staging_dir, name)
        if os.path.isdir(dpath) and os.path.isfile(os.path.join(dpath, "_manifest.json")):
            completed.append(dpath)

    if not completed:
        logger.info("No completed staging dirs to package")
        return

    all_clips = []  # (stg_dir_path, clip_id)
    all_scores = {s: {} for s in SCORE_TYPES}

    for dpath in completed:
        manifest = load_json(os.path.join(dpath, "_manifest.json"))
        if not manifest:
            continue
        all_clips.extend((dpath, cid) for cid in manifest)
        for st in SCORE_TYPES:
            sd = load_json(os.path.join(dpath, f"_scores_{st}.json"))
            if sd:
                all_scores[st].update(sd)

    # Deduplicate
    seen = set()
    deduped = []
    for dp, cid in all_clips:
        if cid not in seen:
            seen.add(cid)
            deduped.append((dp, cid))
        else:
            logger.warning(f"Duplicate clip_id '{cid}' – skipping")
    all_clips = deduped

    total_before = len(all_clips)

    # Quality filter
    if quality_filters:
        logger.info(
            f"Phase 2 quality filters: dover>={quality_filters['dover_min']}  "
            f"saturation=[{quality_filters['saturation_min']},{quality_filters['saturation_max']}]  "
            f"vmafmotion=[{quality_filters['vmafmotion_min']},{quality_filters['vmafmotion_max']}]  "
            f"unimatch_avg=[{quality_filters['unimatch_min']},{quality_filters['unimatch_max']}]"
        )
        filtered = [(dp, cid) for dp, cid in all_clips
                     if _passes_quality_filter(cid, all_scores, quality_filters)]
        n_dropped = total_before - len(filtered)
        logger.info(
            f"Quality filter: {total_before} -> {len(filtered)} clips "
            f"({n_dropped} dropped, {n_dropped*100/max(total_before,1):.1f}%)"
        )
        all_clips = filtered

    total = len(all_clips)
    logger.info(f"Phase 2: packaging {total} clips (max {max_clips}/zip)")

    zip_idx = 0
    clip_count = 0
    cur_zip = None
    cur_scores = {s: {} for s in SCORE_TYPES}

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
        for i, (dp, cid) in enumerate(all_clips):
            if clip_count >= max_clips:
                _close_zip()

            _ensure_zip()

            for ext in (".mp4", ".json"):
                fpath = os.path.join(dp, f"{cid}{ext}")
                if os.path.isfile(fpath):
                    cur_zip.write(fpath, f"{cid}{ext}")

            for st in SCORE_TYPES:
                if cid in all_scores[st]:
                    cur_scores[st][cid] = all_scores[st][cid]

            clip_count += 1

            if (i + 1) % 5000 == 0:
                logger.info(f"  Packaged {i+1}/{total}")

        _close_zip()

    except Exception:
        logger.error(f"Phase 2 error:\n{traceback.format_exc()}")
        if cur_zip is not None:
            cur_zip.close()

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

    for cmd in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([cmd, "-version"], capture_output=True, timeout=10)
        except FileNotFoundError:
            logger.error(f"'{cmd}' not found – please install ffmpeg")
            sys.exit(1)

    all_zips = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".zip") and not f.startswith(".")
    )
    if not all_zips:
        logger.error(f"No zip files found in {input_dir}")
        sys.exit(1)

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

    if run_phase2:
        logger.info("=" * 60)
        logger.info("PHASE 2: Packaging output zips")
        logger.info("=" * 60)

        quality_filters = {
            "dover_min": args.p2_dover_min,
            "saturation_min": args.p2_saturation_min,
            "saturation_max": args.p2_saturation_max,
            "vmafmotion_min": args.p2_vmafmotion_min,
            "vmafmotion_max": args.p2_vmafmotion_max,
            "unimatch_min": args.p2_unimatch_min,
            "unimatch_max": args.p2_unimatch_max,
        }
        package_outputs(staging_dir, output_dir, args.max_clips_per_zip,
                        quality_filters=quality_filters)

        logger.info("Cleaning up staging area...")
        shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info("All done!")


if __name__ == "__main__":
    main()
