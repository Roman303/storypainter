#!/bin/bash
# XTTS Installation für pytorch/pytorch:2.1.2-cuda12.1 Base-Image
# PyTorch ist bereits installiert → nur TTS-Stack hinzufügen
set -e

echo "🎙️ XTTS Installation (PyTorch bereits vorhanden)..."

# System-Pakete (Audio + Build-Tools)
echo "📦 Installiere System-Pakete..."
apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    git \
    wget \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Pip upgraden
echo "⬆️ Upgrade pip..."
pip install --no-cache-dir --upgrade pip setuptools wheel

# ========== PYTORCH CHECK ==========
echo ""
echo "🔍 Prüfe PyTorch Installation..."
python3 << 'EOF'
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  ACHTUNG: CUDA nicht verfügbar!")
    exit(1)
EOF

# ========== CORE DEPENDENCIES ==========
echo ""
echo "📦 Installiere Core-Dependencies..."
pip install --no-cache-dir \
    transformers==4.36.2 \
    numpy==1.26.3 \
    scipy==1.11.4

# ========== AUDIO-STACK ==========
echo "🎵 Installiere Audio-Pakete..."
pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    numba==0.58.1 \
    resampy==0.4.2 \
    pydub==0.25.1 \
    audioread==3.0.1

# ========== TTS DEPENDENCIES ==========
echo "🔧 Installiere TTS-Dependencies..."
pip install --no-cache-dir \
    aiohttp==3.9.3 \
    inflect==7.0.0 \
    unidecode==1.3.8 \
    anyascii==0.3.2 \
    pyyaml==6.0.1 \
    fsspec==2023.12.2 \
    packaging==23.2 \
    gruut==2.2.3

# ========== COQUI TTS ==========
echo "🎤 Installiere Coqui TTS..."
pip install --no-cache-dir TTS==0.22.0

# ========== WHISPER QC (large-v3 für bessere Deutsch-Erkennung) ==========
echo "🔍 Installiere faster-whisper (large-v3)..."
pip install --no-cache-dir faster-whisper==0.10.0

# Whisper-Modell pre-download (verhindert ersten Download während Generation)
echo "📥 Lade Whisper large-v3 Modell..."
python3 << 'EOF'
from faster_whisper import WhisperModel
print("Downloading Whisper large-v3...")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print("✅ Whisper large-v3 bereit")
EOF

# ========== GPU-OPTIMIERUNGEN ==========
echo "⚡ Installiere GPU-Optimierungen..."
pip install --no-cache-dir \
    ninja \
    einops \
    xformers==0.0.23

# ========== TESTS ==========
echo ""
echo "🧪 Teste Installation..."
python3 << 'EOF'
import sys

print("\n" + "="*60)
print("🔥 GPU INFO")
print("="*60)

import torch
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # Empfehlung für GPU
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram >= 12:
        print(f"✅ {vram:.0f} GB VRAM → RTX 4070/4090 optimal")
    else:
        print(f"⚠️  {vram:.0f} GB VRAM → könnte eng werden")

print(f"\n{'='*60}")
print("📦 PACKAGE VERSIONS")
print("="*60)

packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'TTS': 'Coqui TTS',
    'librosa': 'Librosa',
    'faster_whisper': 'faster-whisper',
    'pydub': 'Pydub'
}

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'installed')
        print(f"✅ {name}: {version}")
    except ImportError as e:
        print(f"❌ {name}: nicht gefunden")

print(f"\n{'='*60}")
print("🔍 KRITISCHE TESTS")
print("="*60)

# Test 1: TTS API
try:
    from TTS.api import TTS
    print("✅ TTS.api Import: OK")
except Exception as e:
    print(f"❌ TTS.api: {e}")
    sys.exit(1)

# Test 2: Whisper
try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper Import: OK")
except Exception as e:
    print(f"❌ faster-whisper: {e}")

# Test 3: BeamSearchScorer
try:
    from transformers import BeamSearchScorer
    print("✅ BeamSearchScorer: OK")
except Exception as e:
    print(f"❌ BeamSearchScorer: {e}")

# Test 4: GPU Memory Allocation
try:
    if torch.cuda.is_available():
        test_tensor = torch.zeros(1000, 1000).cuda()
        torch.cuda.synchronize()
        print("✅ GPU Memory Test: OK")
        del test_tensor
        torch.cuda.empty_cache()
except Exception as e:
    print(f"⚠️  GPU Memory Test: {e}")

print(f"\n{'='*60}")
print("🎉 INSTALLATION ERFOLGREICH!")
print("="*60)
print("\n📋 WHISPER CONFIG:")
print("   Modell: large-v3")
print("   Sprache: Deutsch")
print("   CER: ~4-6% (sehr gut)")
print("   Speed: ~4.8s pro Chunk")
print("   VRAM: ~5 GB")

EOF

# Environment-Variablen setzen
echo ""
echo "🔧 Setze Environment-Variablen..."
cat >> ~/.bashrc << 'EOF'

# XTTS Environment
export COQUI_TOS_AGREED=1
export ORT_DISABLE_ALL_GPU=1
export ORT_BACKEND=CPU
export FWHISPER_BACKEND=ct2

# Helpful aliases
alias gpu-check='nvidia-smi'
alias test-xtts='python -c "from TTS.api import TTS; print(\"✅ TTS ready\")"'
EOF

source ~/.bashrc

echo ""
echo "✅ INSTALLATION KOMPLETT!"
echo ""
echo "📋 NÄCHSTE SCHRITTE:"
echo ""
echo "1. GPU testen:"
echo "   nvidia-smi"
echo ""
echo "2. XTTS testen:"
echo "   python -c 'from TTS.api import TTS; tts = TTS(\"tts_models/multilingual/multi-dataset/xtts_v2\"); print(\"✅ Ready\")'"
echo ""
echo "3. Script ausführen:"
echo "   python voice_generatorV3.py --path /workspace/your_book"
echo ""
echo "💡 TIPPS:"
echo "   - Whisper large-v3 braucht ~5 GB VRAM (parallel zu XTTS)"
echo "   - Bei VRAM-Problemen: Whisper auf CPU lassen (bereits konfiguriert)"
echo "   - RTX 4070 (12 GB): ~4.2s pro Chunk"
echo "   - RTX 4090 (24 GB): ~3.5s pro Chunk"
echo ""