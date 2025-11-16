#!/bin/bash
set -e

apt update && apt install -y \
    build-essential pkg-config yasm nasm cmake git \
    libfreetype6-dev libfontconfig1-dev libass-dev \
    libx264-dev libx265-dev libvpx-dev libopus-dev \
    libvorbis-dev libnuma-dev libmp3lame-dev wget

echo "📦 CUDA Pfade setzen…"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:$CPLUS_INCLUDE_PATH

echo "🧹 Entferne alte nv-codec-headers…"
rm -rf /tmp/nv-codec-headers

echo "📥 Installiere nv-codec-headers (für NVENC)…"
cd /tmp
git clone https://github.com/FFmpeg/nv-codec-headers.git
cd nv-codec-headers
make
make install

echo "🧹 Entferne alte FFmpeg Quelle…"
rm -rf /tmp/ffmpeg

echo "📥 Lade FFmpeg…"
cd /tmp
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg
cd ffmpeg

echo "⚙️ Konfiguriere FFmpeg (NVENC ohne NPP)…"
./configure \
  --enable-nonfree \
  --enable-gpl \
  --enable-cuda \
  --enable-cuvid \
  --enable-nvenc \
  --disable-libnpp \
  --extra-cflags="-I/usr/local/cuda/include" \
  --extra-ldflags="-L/usr/local/cuda/lib64" \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libvpx \
  --enable-libfreetype \
  --enable-libfontconfig \
  --enable-libass \
  --enable-libvorbis \
  --enable-libopus \
  --enable-libmp3lame \
  --prefix=/usr/local

echo "🔨 Baue FFmpeg…"
make -j$(nproc)
make install

hash -r

echo "🎉 Fertig! Prüfe NVENC:"
ffmpeg -hide_banner -encoders | grep nvenc || echo '❌ NVENC fehlt!'
