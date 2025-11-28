#!/bin/bash
# XTTS COMPLETE Installation mit allen Build-Tools
set -e

echo "🎙️ XTTS Complete Installation..."

# System-Pakete inklusive Build-Tools
echo "📦 Installiere System-Pakete..."
apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    git \
    portaudio19-dev

# Cleanup
echo "🗑️ Bereinige alte Environment..."
rm -rf /workspace/xtts_env

# Neue Environment
echo "🐍 Erstelle Virtual Environment..."
python3 -m venv /workspace/xtts_env
source /workspace/xtts_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrade pip..."
pip install --upgrade pip setuptools wheel

# KERN-PAKETE mit getesteten Versionen
echo "📦 Installiere Kern-Pakete..."
pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.39.3
pip install numpy==1.24.3

# AUDIO-PAKETE
echo "🎵 Installiere Audio-Pakete..."
pip install librosa==0.10.1 soundfile==0.12.1 numba==0.58.1 resampy==0.4.2

# TTS DEPENDENCIES
echo "🔧 Installiere TTS Dependencies..."
pip install aiohttp==3.9.3 scipy==1.13.0 inflect==7.0.0 unidecode==1.3.8

# TTS VON GITHUB (JETZT MIT BUILD-TOOLS)
echo "🎤 Installiere TTS von GitHub..."
pip install "git+https://github.com/coqui-ai/TTS.git@v0.22.0"

pip install pydub

# TEST
echo ""
echo "🧪 Teste Installation..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')

import transformers
print(f'✅ Transformers: {transformers.__version__}')

try:
    from transformers import BeamSearchScorer
    print('✅ BeamSearchScorer: OK')
except ImportError as e:
    print('❌ BeamSearchScorer:', e)

try:
    from TTS.api import TTS
    print('✅ TTS Import: OK')
except ImportError as e:
    print('❌ TTS Import:', e)

try:
    from faster_whisper import WhisperModel
    print('✅ faster-whisper Import: OK')
except ImportError as e:
    print('❌ faster-whisper Import:', e)
"

echo "🎉 XTTS Complete Installation fertig!"
echo "💡 Aktivieren mit: source /workspace/xtts_env/bin/activate"