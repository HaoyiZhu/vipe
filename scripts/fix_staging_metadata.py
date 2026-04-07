#!/usr/bin/env python3
"""Fix width/height in staging dir metadata JSONs by ffprobing actual mp4 files.

Only updates width and height fields. All other fields remain untouched.
Videos and clip IDs are unchanged -- this is a metadata-only fix.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_video_resolution(mp4_path):
    """Get (width, height) from ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", mp4_path],
            capture_output=True, text=True, timeout=30,
        )
        parts = r.stdout.strip().split(",")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None, None


def fix_one_json(json_path, mp4_path):
    """Fix width/height in one JSON file. Returns (status, old_wh, new_wh)."""
    w, h = get_video_resolution(mp4_path)
    if w is None:
        return "ffprobe_fail", None, None

    with open(json_path) as f:
        meta = json.load(f)

    old_w, old_h = meta.get("width"), meta.get("height")
    if old_w == w and old_h == h:
        return "already_correct", (old_w, old_h), (w, h)

    meta["width"] = w
    meta["height"] = h

    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, json_path)

    return "fixed", (old_w, old_h), (w, h)


def process_staging_dir(stg_dir):
    """Fix all JSONs in one staging dir. Returns stats dict."""
    stats = {"total": 0, "fixed": 0, "already_correct": 0, "ffprobe_fail": 0, "no_json": 0}

    mp4_files = sorted(f for f in os.listdir(stg_dir) if f.endswith(".mp4"))
    for mp4_name in mp4_files:
        clip_id = mp4_name[:-4]
        json_path = os.path.join(stg_dir, clip_id + ".json")
        mp4_path = os.path.join(stg_dir, mp4_name)

        stats["total"] += 1
        if not os.path.exists(json_path):
            stats["no_json"] += 1
            continue

        status, _, _ = fix_one_json(json_path, mp4_path)
        stats[status] = stats.get(status, 0) + 1

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-dirs", type=int, default=0, help="Limit dirs for testing")
    args = parser.parse_args()

    staging_root = os.path.abspath(args.staging_root)
    dirs = sorted(
        os.path.join(staging_root, d) for d in os.listdir(staging_root)
        if os.path.isdir(os.path.join(staging_root, d)) and d.startswith("mira_")
    )
    # Only dirs with mp4 files
    dirs = [d for d in dirs if any(f.endswith(".mp4") for f in os.listdir(d))]

    if args.max_dirs > 0:
        dirs = dirs[:args.max_dirs]

    print(f"[*] Fixing metadata in {len(dirs)} staging dirs ({args.workers} workers)")

    total_stats = {"total": 0, "fixed": 0, "already_correct": 0, "ffprobe_fail": 0, "no_json": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_staging_dir, d): d for d in dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Dirs"):
            stats = future.result()
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

    print()
    print("[*] Summary:")
    for k, v in total_stats.items():
        print(f"    {k}: {v}")
    print("[*] Done!")


if __name__ == "__main__":
    main()
