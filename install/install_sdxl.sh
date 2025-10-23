#!/bin/bash
# SDXL Installation - PERFEKT kompatible Versionen
set -e

echo "🎨 SDXL Installation mit perfekten Versionen..."

# python3-venv sicherstellen
if ! dpkg -l | grep -q python3-venv; then
    apt update && apt install -y python3-venv
fi

# Alte Environment
rm -rf /workspace/sdxl_env

# Neue Environment
python3 -m venv /workspace/sdxl_env
source /workspace/sdxl_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# PyTorch
echo "🔥 Installiere PyTorch..."
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PERFEKT kompatible Versionen:
echo "📦 Installiere kompatible Pakete..."
pip install \
    "numpy<2.0.0" \
    "huggingface_hub==0.20.3" \
    "diffusers==0.27.2" \
    "transformers==4.37.2" \
    "accelerate==0.27.2" \
    safetensors \
    pillow \
    invisible-watermark

# xformers weglassen (zu viele Konflikte)
echo "⚡ Überspringe xformers (stabiler ohne)"

apt update && apt install -y libgl1-mesa-glx libglib2.0-0

# Test
echo ""
echo "🧪 Teste Installation..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')"
python -c "from diffusers import DiffusionPipeline; print('✅ Diffusers: OK')"

echo ""
echo "🎉 SDXL PERFEKT installiert! source /workspace/sdxl_env/bin/activate"