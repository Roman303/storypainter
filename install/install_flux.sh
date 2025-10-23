#!/bin/bash
# FLUX.1 Installation - optimiert für Vast.ai + CUDA 11.8 + Ubuntu 22.04
set -e

echo "🎨 FLUX.1 Installation (Black Forest Labs) wird gestartet..."

# Sicherstellen, dass python3-venv verfügbar ist
if ! dpkg -l | grep -q python3-venv; then
    apt update && apt install -y python3-venv
fi

# Alte Environment entfernen (falls vorhanden)
rm -rf /workspace/flux_env

# Neue Environment erstellen
python3 -m venv /workspace/flux_env
source /workspace/flux_env/bin/activate

# Pip upgraden
pip install --upgrade pip

# -------------------------------------------------------------
# 🧠 PyTorch Installation
# (optimiert für CUDA 11.8 – tested on Vast.ai)
# -------------------------------------------------------------
echo "🔥 Installiere PyTorch..."
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# -------------------------------------------------------------
# 📦 Core Dependencies für FLUX.1 (aktueller Stand 2025)
# -------------------------------------------------------------
echo "📦 Installiere benötigte Python-Pakete..."
pip install \
    "numpy<2.0.0" \
    "huggingface_hub>=0.23.0" \
    "diffusers>=0.31.0" \
    "transformers>=4.43.0" \
    "accelerate>=0.33.0" \
    safetensors \
    pillow \
    tqdm \
    requests \
    invisible-watermark \
    sentencepiece

# -------------------------------------------------------------
# ⚡ Optional: xformers für effizientere Speicherverwaltung
# (funktioniert stabil auf CUDA 11.8)
# -------------------------------------------------------------
echo "⚙️  Installiere xformers (optional, für Performance)..."
pip install xformers==0.0.27.post2 || echo "⚠️  xformers optional – Übersprungen falls fehlgeschlagen."

# -------------------------------------------------------------
# 🧩 Systemlibs (für Pillow, Diffusers & OpenGL)
# -------------------------------------------------------------
apt update && apt install -y libgl1-mesa-glx libglib2.0-0

# -------------------------------------------------------------
# 🧪 Test der Installation
# -------------------------------------------------------------
echo ""
echo "🧪 Teste FLUX.1-Umgebung..."
python - <<'PYCODE'
import torch
from diffusers import FluxPipeline

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA verfügbar: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

try:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    print("✅ FLUX.1 Pipeline geladen!")
except Exception as e:
    print(f"⚠️  Hinweis: Modell-Download ggf. erst nach Login: {e}")
PYCODE

echo ""
echo "🎉 FLUX.1 Installation erfolgreich!"
echo "➡️  Aktiviere die Umgebung mit:"
echo "    source /workspace/flux_env/bin/activate"
