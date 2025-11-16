#!/bin/bash
set -e

echo "=== FFmpeg-Build mit NVENC (CUDA 12.1.1 Devel + nvcc) ==="

# Abhängigkeiten (Ubuntu 22.04)
apt update -qq
apt install -y \
    build-essential git yasm nasm pkg-config \
    libx264-dev libx265-dev libvpx-dev \
    libfdk-aac-dev libmp3lame-dev libopus-dev \
    libass-dev libfreetype6-dev zlib1g-dev libnuma-dev \
    || { echo "Abhängigkeiten fehlgeschlagen!"; exit 1; }

# CUDA-Pfad (Devel-Image hat nvcc)
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# nvcc testen (Devel-Image)
nvcc --version || { echo "nvcc fehlt! Image-Probleme?"; exit 1; }
echo "nvcc: $(nvcc --version | head -n1)"

# FFmpeg Source
cd /tmp
rm -rf ffmpeg
git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg

# Configure (sauber: NVENC für RTX 40, compute_89)
NVCCFLAGS="-gencode=arch=compute_89,code=sm_89 -O2"

./configure \
    --enable-gpl --enable-nonfree \
    --enable-nvenc --enable-cuda-nvcc --enable-libnpp --enable-cuvid \
    --enable-libx264 --enable-libx265 --enable-libvpx \
    --enable-libfdk-aac --enable-libmp3lame --enable-libopus \
    --enable-libass --enable-libfreetype \
    --extra-cflags="-I$CUDA_HOME/include" \
    --extra-ldflags="-L$CUDA_HOME/lib64" \
    --nvccflags="$NVCCFLAGS" \
    || { echo "Configure fehlgeschlagen! Prüfe config.log"; exit 1; }

# Build & Install
make -j$(nproc)
make install

# Cleanup
cd / && rm -rf /tmp/ffmpeg

# Test
echo "FFmpeg Version: $(ffmpeg -version | head -n1)"
if ffmpeg -encoders | grep -q h264_nvenc; then
    echo "=== NVENC AKTIV! ==="
    ffmpeg -encoders | grep nvenc | head -3
else
    echo "=== NVENC FEHLT – prüfe: nvidia-smi -q -d SUPPORTED_CLOCKS | grep Encoder ==="
fi

echo "=== Build FERTIG! ==="