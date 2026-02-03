#!/bin/bash
set -e

echo "🚀 DreamShaper XL 1.0 Installation für Vast.ai (RTX 4090 optimiert)"

# System-Abhängigkeiten
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

# Neue Environment
python3.10 -m venv /workspace/dreamshaper_env --system-site-packages
source /workspace/dreamshaper_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ✅ PYTORCH FÜR CUDA 11.8
echo "🔥 Installiere PyTorch 2.1.2 mit CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ✅ DREAMSHAPER XL - KOMPATIBLE VERSIONEN (alle auf einmal!)
echo "📦 Installiere DreamShaper XL Abhängigkeiten..."
pip install \
    "diffusers==0.25.0" \
    "transformers==4.36.2" \
    "accelerate==0.25.0" \
    "tokenizers==0.15.0" \
    "huggingface-hub==0.20.3" \
    "safetensors==0.4.1" \
    "pillow==10.2.0" \
    "numpy==1.24.4" \
    "invisible-watermark==0.2.0" \
    "omegaconf==2.3.0" \
    "einops==0.7.0" \
    "open-clip-torch==2.23.0" \
    "scipy==1.11.4" \
    "ftfy==6.1.3" \
    "regex==2023.12.25" \
    "tqdm==4.66.1" \
    "albumentations==1.3.1" \
    "opencv-python-headless==4.9.0.80" \
    "pyyaml==6.0.1" \
    "scikit-image==0.22.0" \
    "peft==0.7.1"

# ✅ XFORMERS FÜR CUDA 11.8 (optional, da wir SDPA nutzen)
echo "⚡ Installiere xformers für CUDA 11.8..."
pip install xformers==0.0.23.post1 --no-deps

# ✅ CACHE EINRICHTEN
mkdir -p /workspace/.cache/huggingface
export HF_HOME="/workspace/.cache/huggingface"
export HF_ENDPOINT="https://huggingface.co"

# ✅ ENV-VARS DAUERHAFT SETZEN
echo 'export HF_HOME="/workspace/.cache/huggingface"' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> ~/.bashrc
echo 'source /workspace/dreamshaper_env/bin/activate' >> ~/.bashrc

# ✅ MODELL VORHERUNTERLADEN
echo "📥 Lade DreamShaper XL 1.0 vorab herunter..."
python3 << 'PYEOF'
import torch
from diffusers import DiffusionPipeline

print("Downloading DreamShaper XL 1.0...")
try:
    pipe = DiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-xl-1-0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    print("✅ DreamShaper XL erfolgreich heruntergeladen!")
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA verfügbar: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA nicht verfügbar!")
        
except Exception as e:
    print(f"❌ Fehler beim Download: {e}")
PYEOF

echo ""
echo "✅ Installation abgeschlossen!"
echo ""
echo "🚀 Zum Aktivieren der Environment:"
echo "   source /workspace/dreamshaper_env/bin/activate"
echo ""
echo "🧪 Zum Testen:"
echo "   python3 -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\")''"
echo ""