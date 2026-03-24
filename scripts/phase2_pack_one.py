#!/usr/bin/env python3
"""Pack one output zip from a pre-computed manifest.

Reads a manifest file (staging_dir<TAB>clip_id per line) and creates
a single output zip + sidecar score JSONs.

Usage:
    python phase2_pack_one.py --manifest-dir .../phase2_manifests \
        --output-dir miradata_processed --zip-index 0
"""

import json
import logging
import os
import sys
import time
import zipfile

logger = logging.getLogger("phase2_pack")

SCORE_TYPES = ["dover", "color", "unimatch", "vmafmotion", "scene_cut"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack one output zip from manifest")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip-index", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    zi = args.zip_index
    manifest_dir = os.path.abspath(args.manifest_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    manifest_file = os.path.join(manifest_dir, f"zip_{zi:04d}.txt")
    scores_file = os.path.join(manifest_dir, f"zip_{zi:04d}_scores.json")

    if not os.path.isfile(manifest_file):
        logger.error(f"Manifest not found: {manifest_file}")
        sys.exit(1)

    with open(manifest_file) as f:
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            dp, cid = line.split("\t", 1)
            entries.append((dp, cid))

    total = len(entries)
    zip_name = f"miradata_part{zi:04d}"
    zip_path = os.path.join(output_dir, f"{zip_name}.zip")

    logger.info(f"Packing {zip_name}.zip: {total} clips")
    logger.info(f"Output: {zip_path}")

    t0 = time.time()
    bytes_written = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, (dp, cid) in enumerate(entries):
            for ext in (".mp4", ".json"):
                fpath = os.path.join(dp, f"{cid}{ext}")
                if os.path.isfile(fpath):
                    zf.write(fpath, f"{cid}{ext}")
                    bytes_written += os.path.getsize(fpath)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = bytes_written / elapsed / 1024 / 1024
                logger.info(f"  {i+1}/{total} clips  {bytes_written/1024/1024/1024:.1f} GB  {rate:.0f} MB/s")

    elapsed = time.time() - t0
    rate = bytes_written / elapsed / 1024 / 1024 if elapsed > 0 else 0
    final_size = os.path.getsize(zip_path)

    logger.info(f"Done: {zip_name}.zip  {final_size/1024/1024/1024:.1f} GB  {elapsed/3600:.1f}h  {rate:.0f} MB/s avg")

    # Write sidecar score JSONs
    if os.path.isfile(scores_file):
        with open(scores_file) as f:
            all_scores = json.load(f)
        for st in SCORE_TYPES:
            if st in all_scores and all_scores[st]:
                out_path = os.path.join(output_dir, f"{zip_name}_{st}.json")
                with open(out_path, "w") as f:
                    json.dump(all_scores[st], f, indent=2)
        logger.info(f"Score sidecars written")

    logger.info("All done!")


if __name__ == "__main__":
    main()
