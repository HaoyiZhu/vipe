#!/usr/bin/env python3
"""Phase 2 preparation: quality-filter clips and write per-zip manifests.

Reads all staging dirs, applies quality filters, assigns clips to output zips,
and writes:
  - manifest_dir/zip_NNNN.txt  (one line per clip: staging_dir_path<TAB>clip_id)
  - manifest_dir/zip_NNNN_scores.json  (merged scores for clips in that zip)
  - manifest_dir/summary.json  (total clips, num zips, filter stats)

The per-zip manifests are then consumed by parallel packing jobs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("phase2_prepare")

SCORE_TYPES = ["dover", "color", "unimatch", "vmafmotion", "scene_cut"]


def load_json(path):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def passes_quality_filter(cid, all_scores, filters):
    dover = all_scores["dover"].get(cid)
    if not dover or dover.get("dover_score", 0) < filters["dover_min"]:
        return False

    color = all_scores["color"].get(cid)
    if color:
        sat = color.get("mean_video_saturation", -1)
        if sat < filters["saturation_min"] or sat > filters["saturation_max"]:
            return False

    vmaf = all_scores["vmafmotion"].get(cid)
    if vmaf:
        ms = vmaf.get("vmafmotion_score", -1)
        if ms < filters["vmafmotion_min"] or ms > filters["vmafmotion_max"]:
            return False

    uni = all_scores["unimatch"].get(cid)
    if uni:
        flow = uni.get("unimatch_flow_score")
        if isinstance(flow, list) and flow:
            avg_flow = sum(flow) / len(flow)
            if avg_flow < filters["unimatch_min"] or avg_flow > filters["unimatch_max"]:
                return False

    sc = all_scores["scene_cut"].get(cid)
    if sc and sc.get("num_scenes", 1) > 1:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2 prep: filter and write per-zip manifests")
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Where manifests and summary go")
    parser.add_argument("--max-clips-per-zip", type=int, default=10000)
    parser.add_argument("--dover-min", type=float, default=0.35)
    parser.add_argument("--saturation-min", type=float, default=0.0)
    parser.add_argument("--saturation-max", type=float, default=180.0)
    parser.add_argument("--vmafmotion-min", type=float, default=0.5)
    parser.add_argument("--vmafmotion-max", type=float, default=50.0)
    parser.add_argument("--unimatch-min", type=float, default=3.0)
    parser.add_argument("--unimatch-max", type=float, default=50.0)
    parser.add_argument("--min-frames", type=int, default=0,
                        help="Min frame_number from clip metadata (0 = no filter)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    staging_dir = os.path.abspath(args.staging_dir)
    manifest_dir = os.path.abspath(args.output_dir)
    os.makedirs(manifest_dir, exist_ok=True)

    filters = {
        "dover_min": args.dover_min,
        "saturation_min": args.saturation_min,
        "saturation_max": args.saturation_max,
        "vmafmotion_min": args.vmafmotion_min,
        "vmafmotion_max": args.vmafmotion_max,
        "unimatch_min": args.unimatch_min,
        "unimatch_max": args.unimatch_max,
    }

    # Collect all clips and scores
    logger.info("Loading manifests and scores...")
    completed = sorted(
        os.path.join(staging_dir, d) for d in os.listdir(staging_dir)
        if os.path.isdir(os.path.join(staging_dir, d))
        and os.path.isfile(os.path.join(staging_dir, d, "_manifest.json"))
    )
    logger.info(f"Found {len(completed)} completed staging dirs")

    all_clips = []
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
    all_clips = deduped
    total_before = len(all_clips)
    logger.info(f"Total clips before filter: {total_before}")

    # Quality filter
    logger.info(
        f"Filters: dover>={filters['dover_min']}  "
        f"saturation=[{filters['saturation_min']},{filters['saturation_max']}]  "
        f"vmafmotion=[{filters['vmafmotion_min']},{filters['vmafmotion_max']}]  "
        f"unimatch_avg=[{filters['unimatch_min']},{filters['unimatch_max']}]  "
        f"scene_cut<=1"
    )
    filtered = [(dp, cid) for dp, cid in all_clips
                if passes_quality_filter(cid, all_scores, filters)]
    n_dropped = total_before - len(filtered)
    logger.info(f"After quality filter: {len(filtered)} clips ({n_dropped} dropped, {n_dropped*100/max(total_before,1):.1f}%)")

    # Frame count filter
    min_frames = args.min_frames
    if min_frames > 0:
        logger.info(f"Applying frame count filter: min_frames >= {min_frames}")
        before_frame_filter = len(filtered)
        kept = []
        for dp, cid in filtered:
            meta_path = os.path.join(dp, cid + ".json")
            meta = load_json(meta_path)
            if meta and meta.get("frame_number", 0) >= min_frames:
                kept.append((dp, cid))
        filtered = kept
        logger.info(f"After frame filter: {len(filtered)} clips ({before_frame_filter - len(filtered)} dropped)")

    # Assign to output zips
    max_clips = args.max_clips_per_zip
    n_zips = (len(filtered) + max_clips - 1) // max_clips
    logger.info(f"Will produce {n_zips} output zips (max {max_clips} clips each)")

    for zi in range(n_zips):
        start = zi * max_clips
        end = min(start + max_clips, len(filtered))
        chunk = filtered[start:end]

        zip_name = f"zip_{zi:04d}"
        manifest_path = os.path.join(manifest_dir, f"{zip_name}.txt")
        scores_path = os.path.join(manifest_dir, f"{zip_name}_scores.json")

        with open(manifest_path, "w") as f:
            for dp, cid in chunk:
                f.write(f"{dp}\t{cid}\n")

        chunk_scores = {s: {} for s in SCORE_TYPES}
        for _, cid in chunk:
            for st in SCORE_TYPES:
                if cid in all_scores[st]:
                    chunk_scores[st][cid] = all_scores[st][cid]

        with open(scores_path, "w") as f:
            json.dump(chunk_scores, f)

        logger.info(f"  {zip_name}: {len(chunk)} clips")

    summary = {
        "total_before_filter": total_before,
        "total_after_filter": len(filtered),
        "dropped": n_dropped,
        "n_zips": n_zips,
        "max_clips_per_zip": max_clips,
        "filters": filters,
    }
    with open(os.path.join(manifest_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done! Manifests written to {manifest_dir}")
    logger.info(f"Next: submit array job with --array=0-{n_zips-1}")


if __name__ == "__main__":
    main()
