# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

os.environ["FFMPEG_LOGLEVEL"] = "error"
# os.environ["HF_TOKEN"] = "error"
import shutil
import tarfile
import tempfile
import zipfile
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import gdown
import pandas as pd

from huggingface_hub import HfApi
from tqdm import tqdm


def download_clips_wrapper(args):
    """
    Wrapper function to unpack arguments for the multiprocessing pool.
    """
    url, clips_timestamps, output_paths, cookie_file = args
    download_clips_from_url(url, clips_timestamps, output_paths, cookie_file)


def download_tar_wrapper(args):
    """Wrapper for Tar processing."""
    repo_id, remote_path, output_dir, marker_path, token = args
    download_and_extract_tar(repo_id, remote_path, output_dir, marker_path, token)


def download_and_extract_tar(repo_id, remote_path, output_dir, marker_path, token):
    """
    Downloads a TAR file from HF and extracts it.
    Features: Resume (via marker), thread-safe extraction.
    """
    # 1. Check Resume Marker
    if marker_path.exists():
        # print(f"[Skip] {remote_path} (Already extracted)")
        return

    api = HfApi(token=token)

    # 2. Check existence on HF (Check inside worker to parallelize metadata calls too)
    if not api.file_exists(repo_id=repo_id, repo_type="dataset", filename=remote_path):
        # If it doesn't exist, we just skip it silently or log a warning
        # print(f"[Warning] File not found: {remote_path}")
        return

    # 3. Download and Extract
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # print(f"[Start] Downloading {remote_path}...")
            tar_file = api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_path,
                local_dir=tmp_dir,
            )
            # print(f"[Extract] Extracting {remote_path}...")
            with tarfile.open(tar_file, "r") as tar:
                # Extract to the specific output directory
                tar.extractall(path=output_dir)

        # 4. Create Marker
        marker_path.touch()
        # print(f"[Done] {remote_path}")

    except Exception as e:
        print(f"[Error] Failed to process {remote_path}: {e}")


def download_clips_from_url(
    url: str,
    clips_timestamps: list[tuple[str, str]],
    output_paths: list[Path],
    cookie_file: str,
):
    """
    Downloads a YouTube video and extracts clips.
    """
    # 1. SMART CHECK: If all requested output paths exist, skip.
    if all(p.exists() for p in output_paths):
        return

    import datetime
    import ffmpeg
    import yt_dlp

    def _get_seconds(t: str) -> float:
        time_format = "%H:%M:%S.%f"
        t_obj = datetime.datetime.strptime(t, time_format).time()
        return (
            t_obj.second
            + t_obj.microsecond / 1e6
            + t_obj.minute * 60
            + t_obj.hour * 3600
        )

    # Use a unique temp dir for this specific process/video to avoid collisions
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / f"video_{os.getpid()}.mp4"

        ydl_opts = {
            "outtmpl": str(video_path),
            "format": "wv*[height>=720][ext=mp4]/w[height>=720][ext=mp4]/bv[ext=mp4]/b[ext=mp4]",
            "quiet": True,
            "no_warnings": True,
        }
        # if cookies_from_browser:
        #     ydl_opts["cookiesfrombrowser"] = ("chrome",)

        if cookie_file and os.path.exists(cookie_file):
            ydl_opts["cookiefile"] = cookie_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception:
            return  # Skip if download totally fails

        if not video_path.exists():
            return  # Skip if video file missing (e.g. private video)

        # Process clips
        for idx, (s, e) in enumerate(clips_timestamps):
            target_path = output_paths[idx]

            if target_path.exists():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            s_time = _get_seconds(s)
            e_time = _get_seconds(e)
            duration = e_time - s_time

            try:
                (
                    ffmpeg.input(str(video_path), ss=s_time, t=duration)
                    .output(
                        str(target_path), loglevel="error", threads=1
                    )  # Limit ffmpeg threads per process
                    .overwrite_output()
                    .run()
                )
            except ffmpeg.Error:
                pass  # Skip failed clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix of the dataset to be downloaded",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base directory to save the dataset",
    )
    parser.add_argument(
        "--nocam",
        action="store_true",
        help="No camera (intrinsics & poses) to be downloaded",
    )
    parser.add_argument(
        "--nointrinsics",
        action="store_true",
        help="No intrinsics to be downloaded",
    )
    parser.add_argument(
        "--rgb", action="store_true", help="Download RGB components of the videos"
    )
    parser.add_argument(
        "--depth", action="store_true", help="Download depth components of the videos"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel download processes"
    )

    args = parser.parse_args()
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Define a hidden directory to store "done" markers for tar/zip files
    marker_dir = output_base / ".download_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)

    COOKIE_FILE = "/lustre/fs12/portfolios/nvr/projects/nvr_elm_llm/users/haozhu/projects/vipe/cookies2.txt"

    attributes_to_download = ["intrinsics", "pose"]
    if args.nointrinsics:
        attributes_to_download.remove("intrinsics")
    if args.nocam:
        attributes_to_download = []
    if args.rgb:
        attributes_to_download.append("rgb")
    if args.depth:
        attributes_to_download.append("depth")

    if args.prefix.startswith("dpsp"):
        repo_id = "nvidia/vipe-dynpose-100kpp"
    elif args.prefix.startswith("wsdg"):
        repo_id = "nvidia/vipe-wild-sdg-1m"
    elif args.prefix.startswith("w360"):
        repo_id = "nvidia/vipe-web360"
    else:
        raise ValueError(f"Invalid prefix: {args.prefix}")

    api = HfApi(token=os.getenv("HF_TOKEN"))

    # Grab Metadata
    print("Fetching metadata...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="meta.parquet",
            local_dir=tmp_dir,
        )
        df = pd.read_parquet(meta_file)

    related_videos = df.loc[df["tar_name"].str.startswith(args.prefix)]
    related_tar_names = list(set(related_videos["tar_name"].tolist()))

    print(
        f"Found {len(related_videos)} videos across {len(related_tar_names)} tar files."
    )

    for attribute in attributes_to_download:
        # --- 1. SPECIAL CASE: YouTube RGB (DPSP) ---
        if attribute == "rgb" and args.prefix.startswith("dpsp"):
            print(f"Preparing Parallel YouTube Downloads for {attribute}...")

            merged_videos = {}
            for video_info in related_videos.iterrows():
                link = video_info[1]["youtube_link"]
                if link not in merged_videos:
                    merged_videos[link] = []
                merged_videos[link].append(video_info)

            tasks = []
            for link, merged_video in merged_videos.items():
                time_slices = [
                    t[1]["youtube_timestamp"].split("-") for t in merged_video
                ]
                output_links = [t[1]["sequence"] for t in merged_video]
                download_links = [
                    output_base / attribute / f"{ft}.mp4" for ft in output_links
                ]
                tasks.append((link, time_slices, download_links, COOKIE_FILE))

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(download_clips_wrapper, task) for task in tasks
                ]
                for _ in tqdm(
                    as_completed(futures), total=len(futures), desc="YouTube Clips"
                ):
                    pass

        # --- 2. SPECIAL CASE: Web360 RGB (Zip) ---
        elif attribute == "rgb" and args.prefix.startswith("w360"):
            marker_w360 = marker_dir / "web360_raw_zip.done"
            if marker_w360.exists():
                print("[Skip] Web360 RGB (Already done)")
            else:
                print("Downloading Web360 Zip...")
                zip_output_path = output_base / "web360_raw.zip"
                if not zip_output_path.exists():
                    gdown.download(
                        "https://drive.google.com/file/d/1W1eLmaP16GZOeisAR1q-y9JYP9gT1CRs/view",
                        output=str(zip_output_path),
                        fuzzy=True,
                        use_cookies=False,
                    )
                print("Extracting Web360 Zip...")
                with zipfile.ZipFile(zip_output_path, "r") as zip_ref:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        zip_ref.extractall(tmp_dir)
                        target_dir = output_base / attribute
                        target_dir.mkdir(parents=True, exist_ok=True)
                        for video_file in Path(tmp_dir).glob("**/*.mp4"):
                            dest_file = target_dir / video_file.name
                            if not dest_file.exists():
                                shutil.copy(video_file, dest_file)
                marker_w360.touch()

        # --- 3. GENERAL CASE: Tar Files (WSDG, Intrinsics, Pose) ---
        # This now handles WSDG parallel downloading!
        else:
            print(f"Preparing Parallel Tar Downloads for {attribute}...")

            tar_tasks = []
            output_dir = output_base / attribute
            output_dir.mkdir(parents=True, exist_ok=True)
            hf_token = os.getenv("HF_TOKEN")

            for tar_name in related_tar_names:
                remote_path = f"payload/{tar_name}/{attribute}.tar"
                marker_path = marker_dir / f"{tar_name}_{attribute}.done"

                # Check marker here to avoid adding finished tasks to the queue
                if marker_path.exists():
                    # print(f"Skipping {tar_name} (Done)")
                    continue

                tar_tasks.append(
                    (repo_id, remote_path, output_dir, marker_path, hf_token)
                )

            print(f"Queueing {len(tar_tasks)} tar files to download...")

            if tar_tasks:
                # Note: For large tar files, you might want fewer workers than for small YouTube clips
                # to avoid saturating disk I/O. 8-16 is usually fine.
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(download_tar_wrapper, task)
                        for task in tar_tasks
                    ]
                    for _ in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Extracting {attribute}",
                    ):
                        pass
            else:
                print(f"All {attribute} tars are already downloaded!")


if __name__ == "__main__":
    main()
