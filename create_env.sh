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

pip install git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
pip install git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6
pip install trimesh click gradio scipy plyfile safetensors opencv-python "huggingface-hub<1.0,>=0.16.4"

# for loop closure detection
pip install xformers==0.0.30 "torch<=2.7.0" --index-url https://download.pytorch.org/whl/cu128
pip install faiss-gpu
pip install pandas prettytable pytorch-lightning pytorch-metric-learning torchmetrics pypose trimesh matplotlib plyfile 