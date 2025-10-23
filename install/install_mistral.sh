#!/bin/bash
# Mistral LLM - STANDALONE Installation (ohne XTTS)
set -e

echo "🤖 Mistral Standalone Installation startet..."

# Stelle sicher dass python3-venv installiert ist
if ! dpkg -l | grep -q python3-venv; then
    echo "📦 Installiere python3-venv..."
    apt update && apt install -y python3-venv
fi

# Alte Environment entfernen
if [ -d "/workspace/mistral_env" ]; then
    echo "🗑️ Entferne alte mistral_env..."
    rm -rf /workspace/mistral_env
fi

# Neue Environment
echo "🐍 Erstelle Mistral Environment..."
python3 -m venv /workspace/mistral_env
source /workspace/mistral_env/bin/activate

# Upgrade pip
pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch mit CUDA
echo "🔥 Installiere PyTorch..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Mistral Essentials
echo "📦 Installiere Mistral-Pakete..."
pip install --no-cache-dir \
    transformers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf

# Test
echo ""
echo "✅ Mistral Standalone Installation abgeschlossen!"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print('✅ Transformers: OK')"

echo ""
echo "🎉 Fertig! source /workspace/mistral_env/bin/activate"