#!/bin/bash
# ==============================================================
# FLUX.1 Install Script – Automatische GPU/CUDA-Erkennung
# Kompatibel mit RTX A4000 (CUDA 11.8) und RTX 4090 (CUDA 12.1)
# ==============================================================

set -e
echo "🚀 Starte FLUX.1 Installation (automatisch)..."

# --------------------------------------------------------------
# Sicherstellen, dass python3-venv installiert ist
# --------------------------------------------------------------
if ! dpkg -l | grep -q python3-venv; then
    echo "📦 Installiere python3-venv..."
    apt update && apt install -y python3-venv
fi

# --------------------------------------------------------------
# 3️⃣ Virtuelle Umgebung erstellen
# --------------------------------------------------------------
echo "🐍 Erstelle virtuelle Umgebung..."
rm -rf /workspace/flux_env
python3 -m venv /workspace/flux_env
source /workspace/flux_env/bin/activate

pip install --upgrade pip setuptools wheel

# --------------------------------------------------------------
# 4️⃣ Hugging Face Login (nach Aktivierung der venv)
# --------------------------------------------------------------
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "⚠️  Kein HUGGINGFACE_HUB_TOKEN gefunden!"
    echo "Bitte exportiere deinen Token vor dem Start z.B.:"
    echo "export HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxx"
    exit 1
fi

echo "📦 Installiere huggingface_hub..."
pip install --upgrade huggingface_hub

echo "🔐 Logge dich bei Hugging Face ein..."
/workspace/flux_env/bin/huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN --add-to-git-credential

# --------------------------------------------------------------
# 5️⃣ CUDA-Version automatisch erkennen
# --------------------------------------------------------------
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
echo "🔍 Erkannte CUDA Version: $CUDA_VER"

# --------------------------------------------------------------
# 6️⃣ Passende Torch-Version auswählen
# --------------------------------------------------------------
if [[ "$CUDA_VER" == "11.8" ]]; then
    echo "💾 Verwende Torch-Build für CUDA 11.8 (z. B. A4000)..."
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    pip install xformers==0.0.26.post1
elif [[ "$CUDA_VER" == "12.1" ]]; then
    echo "💾 Verwende Torch-Build für CUDA 12.1 (z. B. 4090)..."
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install xformers==0.0.27.post2
else
    echo "⚠️  Unbekannte CUDA-Version ($CUDA_VER). Standard: CUDA 11.8."
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
fi

# --------------------------------------------------------------
# 7️⃣ Diffusers / Transformers / Extras
# --------------------------------------------------------------
echo "📦 Installiere FLUX-Abhängigkeiten..."
pip install \
    diffusers==0.31.0 \
    transformers==4.46.1 \
    accelerate==0.33.0 \
    safetensors \
    sentencepiece \
    timm \
    openai-clip \
    "protobuf<5"

# --------------------------------------------------------------
# 8️⃣ Systemlibs
# --------------------------------------------------------------
apt update && apt install -y libgl1-mesa-glx libglib2.0-0

# --------------------------------------------------------------
# 9️⃣ Test: FLUX laden
# --------------------------------------------------------------
echo "🧪 Teste FLUX.1 Pipeline..."
python - <<'PY'
from diffusers import FluxPipeline
import torch

try:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16
    ).to("cuda")
    print("✅ FLUX.1 erfolgreich geladen – bereit für HD-Generierungen!")
except Exception as e:
    print("❌ Fehler beim Laden:", e)
PY

echo ""
echo "🎉 Installation abgeschlossen!"
echo "🔹 Aktiviere Umgebung mit: source /workspace/flux_env/bin/activate"
echo "🔹 Danach kannst du image_generator_flux.py starten"
echo "=============================================================="
