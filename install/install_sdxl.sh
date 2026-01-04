#!/bin/bash
set -e

echo "🚀 SDXL Installation mit Refiner Support (downgrade auf 0.21.4)..."

# python3-venv sicherstellen
if ! dpkg -l | grep -q python3-venv; then
    apt update && apt install -y python3-venv libgl1-mesa-glx libglib2.0-0
fi

# Alte Environment löschen
rm -rf /workspace/sdxl_env

# Neue Environment erstellen
python3 -m venv /workspace/sdxl_env
source /workspace/sdxl_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# PyTorch für CUDA 12.1
echo "🔥 Installiere PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# WICHTIG: diffusers 0.21.4 hat funktionierende RefinerPipeline!
echo "📦 Installiere diffusers 0.21.4 (mit Refiner Support)..."
pip install \
    "diffusers==0.21.4" \
    "transformers==4.35.2" \
    "accelerate==0.25.0" \
    "numpy<2.0.0" \
    "huggingface_hub==0.19.4" \
    safetensors \
    pillow \
    invisible-watermark

# xformers für Performance
echo "⚡ Installiere xformers..."
pip install xformers==0.0.23.post1 --no-deps

# Test
echo ""
echo "🧪 Teste Installation..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')"
python -c "from diffusers import StableDiffusionXLPipeline; print('✅ SDXL Pipeline: OK')"
python -c "from diffusers import StableDiffusionXLRefinerPipeline; print('✅ SDXL Refiner: OK')"

echo ""
echo "🎉 SDXL mit Refiner installiert!"
echo "Aktivieren mit: source /workspace/sdxl_env/bin/activate"