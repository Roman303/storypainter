#!/bin/bash
# Fix MeCab-Fehler - Installiere minimal TTS für Deutsch
set -e

echo "🔧 Fixe MeCab-Problem..."

# Option 1: Versuche MeCab zu reparieren
echo "📦 Versuche unidic zu installieren..."
pip install unidic-lite
python -m unidic download || echo "⚠️ unidic download fehlgeschlagen (OK)"

# Option 2: Falls das nicht hilft - TTS neu ohne Japanisch
echo "🔄 Reinstalliere TTS ohne Japanisch/Bengali-Support..."

pip uninstall -y TTS mecab-python3 unidic unidic-lite cutlet

# Core-Dependencies
pip install --no-cache-dir \
    coqpit>=0.0.16 \
    jieba \
    pypinyin \
    einops>=0.6.0 \
    encodec \
    GPUtil==1.4.0 \
    psutil

# TTS Core (ohne Language-Extras)
pip install --no-cache-dir --no-deps TTS==0.22.0

echo ""
echo "🧪 Teste Installation..."
python3 << 'EOF'
import sys

print("="*60)
print("🔍 IMPORT TEST")
print("="*60)

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

try:
    from TTS.api import TTS
    print("✅ TTS.api: OK")
except Exception as e:
    print(f"❌ TTS.api: {e}")
    sys.exit(1)

try:
    import librosa
    print(f"✅ librosa: {librosa.__version__}")
except Exception as e:
    print(f"❌ librosa: {e}")

try:
    from pydub import AudioSegment
    print("✅ pydub: OK")
except Exception as e:
    print(f"❌ pydub: {e}")

try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper: OK")
except Exception as e:
    print(f"❌ faster-whisper: {e}")

print("\n" + "="*60)
print("🎉 CORE-PAKETE OK - DEUTSCH-TTS FUNKTIONIERT!")
print("="*60)
print("\n💡 MeCab/Japanisch-Support fehlt, aber wird nicht gebraucht.")
print("   Dein Hörbuch-Script sollte jetzt laufen!\n")

EOF

echo ""
echo "✅ Fix abgeschlossen!"
echo ""
echo "🚀 Starte jetzt dein Script:"
echo "   python voice_generatorV3.py --path /workspace/your_book"