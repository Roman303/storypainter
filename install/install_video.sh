#!/bin/bash
# 🎬 GPU Video Rendering Environment Installer
# Baut FFmpeg mit voller CUDA-Unterstützung (NVENC, NPP, fade_cuda, gblur_cuda)
# und installiert Python-Umgebung für story_renderer_v8_fullgpu.py
set -e

echo "🎞️  GPU Video Rendering Environment Setup startet..."

# ---------------------------------------------------------------------------
# 🧩 Systempakete & Build-Tools
# ---------------------------------------------------------------------------
echo "📦 Installiere System- und Build-Tools..."
apt update && apt install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential pkg-config yasm nasm \
    git wget curl \
    libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \
    nvidia-cuda-toolkit

# ---------------------------------------------------------------------------
# 🗑️ Alte Umgebung entfernen
# ---------------------------------------------------------------------------
echo "🧹 Bereinige alte Video-Umgebung..."
rm -rf /workspace/video_env

pip install pydub
sudo apt update
sudo apt install ffmpeg

# ---------------------------------------------------------------------------
# 🐍 Neue Virtual Environment
# ---------------------------------------------------------------------------
echo "🐍 Erstelle Virtual Environment..."
python3 -m venv /workspace/video_env
source /workspace/video_env/bin/activate

# ---------------------------------------------------------------------------
# ⬆️ Upgrade pip
# ---------------------------------------------------------------------------
echo "⬆️ Upgrade pip..."
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 🎨 Python-Pakete für Story Rendering
# ---------------------------------------------------------------------------
echo "📦 Installiere Python-Pakete..."
pip install numpy==1.26.4 pillow==10.4.0 tqdm==4.66.3 moviepy==1.0.3 opencv-python==4.10.0.84

# ---------------------------------------------------------------------------
# 🎬 FFmpeg mit CUDA / NVENC / NPP bauen
# ---------------------------------------------------------------------------
echo "🎥 Baue FFmpeg mit voller CUDA-Unterstützung (kann 5–15 Minuten dauern)..."
cd /tmp
rm -rf ffmpeg-src
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg-src
cd ffmpeg-src

./configure \
  --prefix=/usr/local \
  --enable-cuda \
  --enable-cuvid \
  --enable-nvenc \
  --enable-libnpp \
  --enable-gpl \
  --enable-nonfree \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libvpx \
  --enable-libfdk-aac \
  --enable-shared \
  --disable-static \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64

make -j$(nproc)
make install
hash -r
cd /workspace

# ---------------------------------------------------------------------------
# ✅ Verifikation
# ---------------------------------------------------------------------------
echo ""
echo "🧪 Teste CUDA & FFmpeg..."
ffmpeg -hide_banner -filters | grep cuda || echo "⚠️ WARNUNG: Keine CUDA-Filter gefunden!"

python - <<'PY'
import subprocess, sys
print("✅ Python:", sys.version.split()[0])
try:
    import numpy, moviepy, cv2, PIL
    print("✅ Numpy:", numpy.__version__)
    print("✅ MoviePy:", moviepy.__version__)
    print("✅ OpenCV:", cv2.__version__)
    print("✅ Pillow:", PIL.__version__)
except Exception as e:
    print("❌ Python-Paket-Fehler:", e)
PY

echo ""
echo "🎉 GPU Video Rendering Installation abgeschlossen!"
echo "💡 Aktivieren mit: source /workspace/video_env/bin/activate"
echo "💡 Teste CUDA-Filter mit: ffmpeg -hide_banner -filters | grep cuda"
