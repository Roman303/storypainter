#!/bin/bash
set -e

echo "=== FFmpeg NVENC Auto-Compatible Build ==="

apt update -qq
apt install -y \
    build-essential git yasm nasm pkg-config \
    libfreetype6-dev libfontconfig1-dev libfribidi-dev \
    libass-dev zlib1g-dev libnuma-dev

echo "→ NVIDIA Driver Version:"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d'.' -f1)
echo "Gefundener Treiber: $DRV"

# -------------------------------------------
# 1. WÄHLE DIE PASSENDE NV-CODEC-HEADERS VERSION
# -------------------------------------------

rm -rf /tmp/nv-codec-headers
mkdir -p /tmp/nv-codec-headers

if [ "$DRV" -ge 570 ]; then
    echo "→ Nutze nv-codec-headers (NEU) für Treiber 570+"
    git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers

elif [ "$DRV" -ge 550 ]; then
    echo "→ Nutze nv-codec-headers 12.x für Treiber 550–569"
    git clone -b sdk/12.1.14 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers

else
    echo "→ Nutze nv-codec-headers 11.x für ältere Treiber (<550)"
    git clone -b sdk/11.1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers
fi

echo "→ Installiere nv-codec-headers"
make -C /tmp/nv-codec-headers install PREFIX=/usr/local

# -------------------------------------------
# 2. FFmpeg BUILDEN
# -------------------------------------------

echo "→ Lade FFmpeg (git trunk)"
rm -rf /tmp/ffmpeg
git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg
cd /tmp/ffmpeg

NVCCFLAGS="-gencode=arch=compute_89,code=sm_89 -O2"

echo "→ Konfiguriere FFmpeg"
./configure \
  --prefix=/usr/local \
  --disable-shared --enable-static \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda \
  --enable-cuda-nvcc \
  --enable-nvenc \
  --enable-libnpp \
  --extra-cflags="-I/usr/local/cuda/include" \
  --extra-ldflags="-L/usr/local/cuda/lib64" \
  --nvccflags="$NVCCFLAGS" \
  --enable-libfontconfig \
  --enable-libfreetype \
  --enable-libfribidi \
  --enable-libass \
  --disable-doc

echo "→ Baue FFmpeg"
make -j$(nproc)

echo "→ Installiere FFmpeg"
make install

# -------------------------------------------
# 3. PYTORCH AUTO-INSTALL
# -------------------------------------------

echo "=== Prüfe PyTorch ==="
if python3 -c "import torch" 2>/dev/null; then
    echo "✔ PyTorch bereits installiert."
else
    echo "→ Installiere PyTorch CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# -------------------------------------------
# 4. TESTS
# -------------------------------------------

echo "=== Test NVENC ==="
/usr/local/bin/ffmpeg -encoders | grep nvenc || echo "❌ NVENC fehlt!"

echo "=== Test drawtext ==="
/usr/local/bin/ffmpeg -filters | grep drawtext || echo "❌ drawtext fehlt!"

echo "=== INSTALLATION FERTIG ==="
echo "FFmpeg unter: /usr/local/bin/ffmpeg"
