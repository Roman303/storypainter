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
    "diffusers==0.23.1" \  # Stabil mit DreamShaper
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

# ✅ XFORMERS FÜR CUDA 11.8
echo "⚡ Installiere xformers für CUDA 11.8..."
pip install xformers==0.0.22 --no-deps

# ✅ TENSORRT/CUDNN CHECK
if [ -f "/usr/lib/x86_64-linux-gnu/libcudnn.so" ]; then
    echo "✅ cuDNN gefunden"
    pip install "nvidia-cudnn-cu11==8.9.4.25"
fi

# ✅ CACHE EINRICHTEN
mkdir -p /workspace/.cache/huggingface
export HF_HOME="/workspace/.cache/huggingface"
export HF_ENDPOINT="https://huggingface.co"

# ✅ GRUNDLEGENDE TESTS
echo ""
echo "🧪 Führe grundlegende Tests aus..."

# Test 1: PyTorch CUDA
python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA verfügbar: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA Version: {torch.version.cuda}')
    # Memory test
    tensor = torch.randn(1000, 1000).cuda()
    print(f'✅ CUDA Tensor Operation erfolgreich')
    del tensor
"

# Test 2: Diffusers Import
python3 -c "
try:
    from diffusers import DiffusionPipeline
    print('✅ Diffusers import erfolgreich')
except Exception as e:
    print(f'❌ Diffusers Import fehlgeschlagen: {e}')
"

# Test 3: DreamShaper Download Test
echo "⬇️  Teste DreamShaper XL 1.0 Download..."
python3 << 'EOF'
from huggingface_hub import try_to_load_from_cache, snapshot_download
import os

model_id = "Lykon/dreamshaper-xl-1-0"
cache_dir = "/workspace/.cache/huggingface"

try:
    # Prüfe ob schon gecached
    cached_path = try_to_load_from_cache(
        repo_id=model_id,
        filename="model_index.json"
    )
    
    if cached_path is not None:
        print(f"✅ Model bereits gecached: {cached_path}")
    else:
        print("ℹ️  Model nicht im Cache. Wird beim ersten Aufruf automatisch geladen.")
        # Lade nur minimal für Test
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=["model_index.json", "*.json"],
            local_files_only=False
        )
        print("✅ Model-Metadaten erfolgreich geladen")
        
except Exception as e:
    print(f"⚠️  Warnung beim Cache-Check: {e}")
EOF

echo ""
echo "🎉 INSTALLATION ABGESCHLOSSEN!"
echo "="*70
echo "📋 NÄCHSTE SCHRITTE FÜR VAST.AI:"
echo ""
echo "1. IMAGE KONFIGURATION IM DASHBOARD:"
echo "   • Docker Image: nvidia/cuda:11.8.0-devel-ubuntu22.04"
echo "   • Python Version: 3.10"
echo ""
echo "2. STARTUP COMMAND (im Vast.ai Dashboard):"
echo "   cd /workspace/storypainter/install && bash install_sdxl_d.sh"
echo ""
echo "3. NACH INSTALLATION:"
echo "   source /workspace/dreamshaper_env/bin/activate"
echo ""
echo "4. DEINEN GENERATOR ANPASSEN:"
echo "   Ändere in image_generator_d.py:"
echo "   model_base = 'Lykon/dreamshaper-xl-1-0'"
echo ""
echo "5. OPTIMIERTE EINSTELLUNGEN FÜR DREAMSHAPER:"
echo "   --steps 35 --guidance 7.5 --width 2048 --height 1152"
echo ""
echo "⚠️  WICHTIGE NOTIZEN FÜR VAST.AI:"
echo "   • Nutze 'Spot Instances' für 40-60% Ersparnis"
echo "   • RTX 4090 (24GB) ist Preis/Leistungs-Sieger"
echo "   • A100 (40GB) für Batch-Rendering"
echo "   • Setze 'Idle shutdown' auf 15-30 Minuten"
echo "="*70

# ✅ FINAL CHECK SCRIPT
cat > /workspace/check_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess

print("🔍 Installation Check für DreamShaper XL 1.0")
print("="*60)

# Check Python version
python_version = sys.version.split()[0]
print(f"Python: {python_version}")

# Check packages
packages = [
    ("torch", "2.1.0"),
    ("diffusers", "0.23.1"),
    ("transformers", "4.34.1"),
    ("accelerate", "0.24.1"),
]

for pkg, expected in packages:
    try:
        exec(f"import {pkg.split('-')[0]}")
        version = eval(f"{pkg.split('-')[0]}.__version__")
        status = "✅" if version.startswith(expected) else "⚠️"
        print(f"{status} {pkg}: {version} (erwartet: {expected})")
    except ImportError:
        print(f"❌ {pkg}: Nicht installiert")

# CUDA check
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA nicht verfügbar!")
except:
    print("❌ PyTorch CUDA Check fehlgeschlagen")

print("="*60)
print("💡 Tipp: Führe 'source /workspace/dreamshaper_env/bin/activate' aus")
print("       bevor du deinen Generator startest")
EOF

chmod +x /workspace/check_installation.py

echo ""
echo "💡 Führe nach der Installation aus:"
echo "   python /workspace/check_installation.py"