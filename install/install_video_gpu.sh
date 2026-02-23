#!/bin/bash
set -e

# ============================================================================
# VIDEO GENERATOR INSTALL
# Für: vastai/base-image:cuda-12.1.1-auto auf Ubuntu 22.04
# GPU: NVIDIA A4000 (oder kompatibel)
#
# KEIN FFmpeg-Build nötig! Ubuntu 22.04 FFmpeg-Package hat bereits:
#   ✅ h264_nvenc, h264_cuvid, overlay_cuda, scale_cuda, hwupload_cuda
# ============================================================================

echo "=============================================="
echo " 🚀 VIDEO GENERATOR SETUP"
echo " vastai cuda-12.1.1-auto / Ubuntu 22.04"
echo "=============================================="

# ============================================================================
# SYSTEM PAKETE
# ============================================================================
echo ""
echo "📦 System packages..."
apt update -y -q
apt install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    ffmpeg \
    ffprobe \
    libcuda1 \
    fonts-dejavu-core \
    fonts-liberation

# ============================================================================
# PYTHON VENV
# ============================================================================
echo ""
echo "🐍 Python venv setup..."
VENV_PATH="/workspace/video_env"
python3 -m venv "$VENV_PATH" --system-site-packages
source "$VENV_PATH/bin/activate"
pip install --no-cache-dir --upgrade pip -q

# ============================================================================
# PYTORCH (CUDA 12.1)
# ============================================================================
echo ""
echo "🔥 PyTorch (cu121)..."
pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121 \
    -q

# ============================================================================
# PYTHON PAKETE
# ============================================================================
echo ""
echo "📚 Python packages..."
pip install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    tqdm \
    -q

# ============================================================================
# VERIFICATION
# ============================================================================
echo ""
echo "🔍 Verification..."
echo "----------------------------------------------"

# PyTorch + CUDA
python3 -c "
import torch
ok = torch.cuda.is_available()
print('✅ PyTorch CUDA' if ok else '❌ PyTorch CUDA FEHLT')
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f'   GPU:  {p.name}')
    print(f'   VRAM: {p.total_memory // 1024**3} GB')
    print(f'   CUDA: {torch.version.cuda}')
"

# FFmpeg Version
echo ""
FFV=$(ffmpeg -version 2>/dev/null | head -n1)
echo "✅ FFmpeg: $FFV"

# Encoder/Decoder
for enc in h264_nvenc hevc_nvenc; do
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "$enc"; then
        echo "✅ Encoder: $enc"
    else
        echo "⚠️  Encoder $enc nicht verfügbar"
    fi
done

for dec in h264_cuvid hevc_cuvid; do
    if ffmpeg -hide_banner -decoders 2>/dev/null | grep -q "$dec"; then
        echo "✅ Decoder: $dec"
    else
        echo "⚠️  Decoder $dec nicht verfügbar"
    fi
done

# GPU-Filter
echo ""
echo "🎛️  GPU-Filter:"
for f in overlay_cuda scale_cuda hwupload_cuda thumbnail_cuda yadif_cuda; do
    if ffmpeg -filters 2>/dev/null | grep -q "$f"; then
        echo "   ✅ $f"
    else
        echo "   ⚠️  $f fehlt"
    fi
done

# Basis-Filter
echo ""
echo "🎛️  Basis-Filter:"
MISSING_FILTER=0
for f in drawtext fade scale overlay gblur xfade; do
    if ffmpeg -hide_banner -filters 2>/dev/null | grep -qE "\s${f}\s"; then
        echo "   ✅ $f"
    else
        echo "   ❌ $f FEHLT"
        MISSING_FILTER=1
    fi
done

if [ "$MISSING_FILTER" = "1" ]; then
    echo ""
    echo "❌ Kritische Filter fehlen! Überprüfe FFmpeg-Installation."
    exit 1
fi

# ffprobe
if command -v ffprobe &>/dev/null; then
    echo ""
    echo "✅ ffprobe verfügbar"
else
    echo "⚠️  ffprobe fehlt – Overlay-Längenermittlung nutzt Fallback"
fi

# ============================================================================
# ACTIVATION SCRIPT
# ============================================================================
cat > /workspace/activate_video.sh << 'ACTIVATE_EOF'
#!/bin/bash
source /workspace/video_env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

echo "🎬 Video Generator Ready"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "VRAM: $(python3 -c 'import torch; print(str(torch.cuda.get_device_properties(0).total_memory // 1024**3) + " GB" if torch.cuda.is_available() else "N/A")')"
echo ""
echo "GPU-Filter:"
ffmpeg -filters 2>/dev/null | grep -E "overlay_cuda|scale_cuda|hwupload_cuda" | awk '{print "  "$0}'
echo ""
echo "Usage:"
echo "  python video_generator_2.py --path /workspace/storypainter/input/PROJEKT"
ACTIVATE_EOF
chmod +x /workspace/activate_video.sh

# ============================================================================
# CLEANUP
# ============================================================================
echo ""
echo "🧹 Cleanup..."
apt clean -q
rm -rf /var/lib/apt/lists/*
pip cache purge 2>/dev/null || true

# ============================================================================
# FERTIG
# ============================================================================
echo ""
echo "=============================================="
echo " ✅ SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "📊 Disk:"
du -sh /workspace/video_env 2>/dev/null | awk '{print "   venv:   "$1}'
du -sh /usr/bin/ffmpeg      2>/dev/null | awk '{print "   ffmpeg: "$1}'
echo ""
echo "🚀 Start:"
echo "   source /workspace/activate_video.sh"
echo "   python video_generator_2.py --path /workspace/storypainter/input/PROJEKT"
echo ""
echo "=============================================="

deactivate
