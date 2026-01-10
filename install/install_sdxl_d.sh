#!/bin/bash
set -e

echo "🚀 DreamShaper XL 1.0 Installation für Vast.ai (optimiert für CUDA 11.8)"

# System-Abhängigkeiten (minimal)
apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsndfile1

# Symlink für python3
ln -sf /usr/bin/python3.10 /usr/bin/python3

# Alte Environment löschen
rm -rf /workspace/dreamshaper_env

# Neue Environment mit Python 3.10
python3.10 -m venv /workspace/dreamshaper_env --system-site-packages
source /workspace/dreamshaper_env/bin/activate

# Upgrade pip und setuptools
pip install --upgrade pip setuptools wheel

# ✅ PYTORCH FÜR CUDA 11.8 (stabilste Version)
echo "🔥 Installiere PyTorch 2.1.0 mit CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ✅ DREAMSHAPER XL 1.0 SPEZIFISCHE VERSIONEN
echo "📦 Installiere DreamShaper XL 1.0 Abhängigkeiten..."
pip install \
    "diffusers==0.23.1" \
    "transformers==4.34.1" \
    "accelerate==0.24.1" \
    "numpy==1.24.4" \
    "huggingface-hub==0.19.4" \
    "safetensors==0.4.1" \
    "pillow==10.0.1" \
    "invisible-watermark==0.2.0" \
    "omegaconf==2.3.0" \
    "einops==0.7.0" \
    "open-clip-torch==2.23.0" \
    "scipy==1.11.3" \
    "ftfy==6.1.1" \
    "regex==2023.10.3" \
    "tqdm==4.66.1" \
    "albumentations==1.3.0" \
    "opencv-python-headless==4.8.1.78" \
    "pyyaml==6.0.1" \
    "scikit-image==0.22.0"

# ✅ XFORMERS FÜR CUDA 11.8 (OPTIONAL - dein Script nutzt SDPA)
echo "⚡ Installiere xformers für CUDA 11.8..."
pip install xformers==0.0.22 --no-deps

# ✅ CACHE EINRICHTEN
mkdir -p /workspace/.cache/huggingface
export HF_HOME="/workspace/.cache/huggingface"
export HF_ENDPOINT="https://huggingface.co"

# ✅ MODELL VORHERUNTERLADEN (empfohlen!)
echo "📥 Lade DreamShaper XL 1.0 vorab herunter..."
python3 -c "
from diffusers import DiffusionPipeline
print('Downloading DreamShaper XL...')
DiffusionPipeline.from_pretrained(
    'Lykon/dreamshaper-xl-1-0',
    use_safetensors=True,
    variant='fp16'
)
print('✅ Download abgeschlossen!')
"

echo "✅ Installation abgeschlossen!"
echo "🚀 Aktiviere Environment mit: source /workspace/dreamshaper_env/bin/activate"