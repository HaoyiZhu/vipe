#!/usr/bin/env bash
set -e

mamba env create -f envs/base.yml -y
eval "$(mamba shell hook --shell bash)"
mamba activate vipe
mamba install -c nvidia cuda-toolkit=12.8 -y
mamba install ffmpeg -y
pip install ninja "numpy<2"

cd ~/projects/vipe
pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation -e .
mamba install ffmpeg aria2 -y

pip install debugpy trimesh viser OpenEXR plyfile decord datasets
pip install yt_dlp ffmpeg-python pandas "imageio[ffmpeg]" "ray[default]" tqdm rich
