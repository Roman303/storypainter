#!/bin/bash
# 🎬 GPU Video Rendering Environment Installer (optimized for CUDA 11.8)
# Installiert FFmpeg mit NVENC/NPP-Unterstützung & Python-Umgebung für story_renderer.py
# Basis: nvidia/cuda:11.8.0-devel-ubuntu22.04
set -e

echo "🎞️  GPU Video Rendering Environment Setup startet..."

# ---------------------------------------------------------------------------
# 🧩 Systempakete
# ---------------------------------------------------------------------------
echo "📦 Installiere System- und Build-Tools..."
apt update && apt install -y \
    python3 python3-pip python3-venv python3-dev \
    git wget curl \
    libsm6 libxext6 libgl1 \
    nvidia-cuda-toolkit

# ---------------------------------------------------------------------------
# 🎬 FFmpeg mit CUDA/NVENC prüfen/inst.
# ---------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "⚙️ Installiere FFmpeg (Basisversion)..."
    apt install -y ffmpeg
fi

if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q nvenc; then
    echo "⚙️ Installiere NVIDIA FFmpeg (mit CUDA/NVENC)…"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ffmpeg4.4-nvidia_4.4.2-1_amd64.deb
    apt install -y ./ffmpeg4.4-nvidia_4.4.2-1_amd64.deb
    rm -f ffmpeg4.4-nvidia_4.4.2-1_amd64.deb
else
    echo "✅ FFmpeg mit NVENC bereits vorhanden."
fi

echo ""
echo "🎥 Verfügbare NVENC-Encoder:"
ffmpeg -hide_banner -encoders | grep nvenc || echo "⚠️ Keine NVENC-Encoder gefunden!"
echo ""
echo "🎥 Verfügbare CUDA-Filter:"
ffmpeg -hide_banner -filters | grep -E "cuda|npp" || echo "⚠️ Keine CUDA-Filter gefunden!"

# ---------------------------------------------------------------------------
# 🧹 Alte Umgebung bereinigen
# ---------------------------------------------------------------------------
echo "🧹 Bereinige alte Virtual Environment..."
rm -rf /workspace/video_env

# ---------------------------------------------------------------------------
# 🐍 Neue Virtual Environment
# ---------------------------------------------------------------------------
echo "🐍 Erstelle Python-Venv..."
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
pip install \
    numpy==1.26.4 \
    pillow==10.4.0 \
    tqdm==4.66.3 \
    moviepy==1.0.3 \
    opencv-python==4.10.0.84 \
    pydub

# ---------------------------------------------------------------------------
# ✅ Verifikation
# ---------------------------------------------------------------------------
echo ""
echo "🧪 Teste GPU-Setup..."
ffmpeg -hide_banner -hwaccels | grep -E "cuda|nvdec" || echo "⚠️ Keine CUDA-HWACCELs gefunden!"

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
echo "💡 Teste NVENC mit: ffmpeg -hide_banner -encoders | grep nvenc"
echo "💡 Teste CUDA-Filter mit: ffmpeg -hide_banner -filters | grep cuda"
