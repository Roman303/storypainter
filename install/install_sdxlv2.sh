#!/bin/bash
set -e

echo "🚀 Installiere SDXL Ultra Setup für RTX 4090..."

apt update
apt install -y python3-venv git libgl1-mesa-glx libglib2.0-0

# CREATE ENV
rm -rf /workspace/sdxl_env
python3 -m venv /workspace/sdxl_env
source /workspace/sdxl_env/bin/activate

pip install --upgrade pip

echo "🔥 Installiere PyTorch (CUDA 12.1 kompatibel)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installiere SDXL Dependencies..."
pip install \
    "diffusers==0.27.2" \
    "transformers==4.37.2" \
    "accelerate==0.27.2" \
    safetensors \
    pillow \
    numpy==1.26.4 \
    huggingface_hub \
    invisible-watermark

echo "⚡ Installiere xFormers für 4090..."
pip install xformers==0.0.23.post1

echo "🧪 TEST..."
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
import diffusers
print("Diffusers:", diffusers.__version__)
from diffusers import StableDiffusionXLRefinerPipeline
print("Refiner ok")
EOF

echo "🎉 FERTIG! Aktivieren mit:"
echo "source /workspace/sdxl_env/bin/activate"
