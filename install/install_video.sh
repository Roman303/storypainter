#!/bin/bash
set -e

echo "=============================="
echo " UPDATE SYSTEM"
echo "=============================="
apt update -y
apt upgrade -y

echo "=============================="
echo " INSTALL BUILD TOOLS"
echo "=============================="
apt install -y build-essential git pkg-config wget curl yasm nasm

echo "=============================="
echo " INSTALL FREETYPE + DRAW TEXT SUPPORT"
echo "=============================="
apt install -y libfreetype6 libfreetype6-dev libharfbuzz-dev libfribidi-dev

echo "=============================="
echo " INSTALL VIDEO LIBS"
echo "=============================="
apt install -y libass-dev libvdpau-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
               libva-dev libdrm-dev libx264-dev libx265-dev libnuma-dev zlib1g-dev \
               libfontconfig1-dev libxml2-dev

echo "=============================="
echo " FIX OPENCV DEPENDENCIES"
echo "=============================="
apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

echo "=============================="
echo " INSTALL PYTHON + PIP"
echo "=============================="
apt install -y python3 python3-pip python3-venv

echo "=============================="
echo " INSTALL PYTORCH + CUDA"
echo "=============================="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=============================="
echo " INSTALL PYTHON LIBS"
echo "=============================="
pip install numpy opencv-python pillow tqdm

echo "=============================="
echo " INSTALL NVIDIA NVENC / NPP HEADERS"
echo "=============================="
apt install -y nvidia-cuda-toolkit

echo "=============================="
echo " BUILD FFMPEG WITH NVENC + DRAWTEXT"
echo "=============================="

FFMPEG_VERSION=n6.1

cd /usr/local/src
rm -rf ffmpeg
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout $FFMPEG_VERSION

./configure \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-libnpp \
  --extra-cflags="-I/usr/local/cuda/include" \
  --extra-ldflags="-L/usr/local/cuda/lib64" \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libass \
  --enable-libfreetype \
  --enable-libfontconfig \
  --enable-nvenc \
  --enable-libdrm \
  --enable-libvdpau \
  --enable-libxcb \
  --enable-libxcb-shm \
  --enable-libxcb-xfixes \
  --enable-libxml2 \
  --enable-openssl \
  --enable-libharfbuzz \
  --enable-libfribidi \
  --enable-ffnvcodec \
  --enable-filter=drawtext \
  --enable-filter=zoompan \
  --enable-filter=gblur \
  --enable-filter=fade \
  --enable-filter=xfade

make -j$(nproc)
make install

hash -r

echo "=============================="
echo " VERIFY FFMPEG BUILD"
echo "=============================="
ffmpeg -hide_banner -filters | grep drawtext || echo "❌ drawtext missing!"
ffmpeg -hide_banner -encoders | grep nvenc || echo "❌ NVENC missing!"

echo "=============================="
echo " INSTALL DONE!"
echo "=============================="
echo "FFmpeg, NVENC, drawtext, CUDA, PyTorch, OpenCV → SUCCESS"
