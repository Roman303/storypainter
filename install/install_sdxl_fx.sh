#!/bin/bash
# ======================================================
# 🎨 SDXL / RealisticVision Installation (GPU-optimiert)
# Kompatibel mit FFmpeg + Python 3.10+
# ======================================================
set -e

echo "🎨 Starte SDXL Installation ..."

# ------------------------------------------------------
# Python-Umgebung
# ------------------------------------------------------
if ! dpkg -l | grep -q python3-venv; then
    apt update && apt install -y python3-venv
fi

rm -rf /workspace/sdxl_env
python3 -m venv /workspace/sdxl_env
source /workspace/sdxl_env/bin/activate

echo "⚙️ Upgrade pip..."
pip install --upgrade pip setuptools wheel

# ------------------------------------------------------
# PyTorch + CUDA (A4000 → CUDA 11.8)
# ------------------------------------------------------
echo "🔥 Installiere PyTorch..."
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------
# Diffusion + Tools
# ------------------------------------------------------
echo "📦 Installiere Hauptpakete..."
pip install \
    "numpy<2.0.0" \
    "huggingface_hub==0.20.3" \
    "diffusers==0.27.2" \
    "transformers==4.37.2" \
    "accelerate==0.27.2" \
    "safetensors" \
    "pillow" \
    "opencv-python-headless" \
    "moviepy" \
    "tqdm" \
    "requests" \
    "psutil"

# Invisible Watermark (optional, SDXL nutzt es manchmal)
pip install invisible-watermark

# xformers → optional (nur wenn du mehr VRAM hast)
# pip install xformers==0.0.25

# ------------------------------------------------------
# Systemabhängigkeiten
# ------------------------------------------------------
apt update && apt install -y libgl1-mesa-glx libglib2.0-0 ffmpeg

# ------------------------------------------------------
# Quick Test
# ------------------------------------------------------
echo ""
echo "🧪 Teste Installation..."
python - <<'PYTEST'
import torch, diffusers, numpy, PIL, subprocess
print(f"✅ Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"✅ Diffusers: {diffusers.__version__}")
print(f"✅ NumPy: {numpy.__version__}")
print(f"✅ Pillow: {PIL.__version__}")
subprocess.run(["ffmpeg", "-version"], check=False)
PYTEST

echo ""
echo "🎉 SDXL-Umgebung bereit!"
echo "👉 Aktiviere sie mit: source /workspace/sdxl_env/bin/activate"
