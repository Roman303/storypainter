#!/bin/bash
set -e

echo "🚀 SDXL Installation für DreamShaper XL"

# python3-venv sicherstellen
if ! dpkg -l | grep -q python3-venv; then
    apt install -y python3 python3-venv python3-dev \
               libgl1-mesa-glx libglib2.0-0 git

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

# DreamShaper XL benötigt spezifische diffusers Version
echo "📦 Installiere diffusers (optimiert für DreamShaper XL)..."
pip install \
    "diffusers==0.23.1 " \
    "transformers==4.35.2" \
    "accelerate==0.25.0" \
    "numpy<2.0.0" \
    "huggingface_hub==0.19.4" \
    safetensors \
    pillow \
    invisible-watermark \
    "omegaconf>=2.3.0" \
    "einops>=0.6.1"

# xformers für Performance
echo "⚡ Installiere xformers..."
pip install xformers==0.0.25.post1


# Zusätzliche Abhängigkeiten für DreamShaper
echo "🎨 Installiere zusätzliche Abhängigkeiten für DreamShaper..."
pip install \
    "open_clip_torch" \
    "scipy" \
    "ftfy" \
    "regex"

# Test der Installation mit DreamShaper-spezifischen Komponenten
echo ""
echo "🧪 Teste DreamShaper XL Installation..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')"
python -c "from diffusers import DiffusionPipeline; print('✅ DiffusionPipeline: OK')"
python -c "from diffusers import StableDiffusionXLPipeline; print('✅ SDXL Pipeline: OK')"

# DreamShaper XL spezifischer Test
echo "🧪 Teste DreamShaper XL Model-Loading..."
python << 'EOF'
import torch
from diffusers import DiffusionPipeline

# Test ob DreamShaper geladen werden kann
try:
    pipeline = DiffusionPipeline.from_pretrained(
        "Lykon/DreamShaper-XL",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    print("✅ DreamShaper XL kann erfolgreich geladen werden")
    del pipeline
    torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ Fehler beim Laden von DreamShaper XL: {e}")
EOF

echo ""
echo "🎉 DreamShaper XL Installation abgeschlossen!"
echo "Deine image_generator_d.py ist bereits für DreamShaper XL konfiguriert"
echo "Aktivieren mit: source /workspace/sdxl_env/bin/activate"