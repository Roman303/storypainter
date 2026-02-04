#!/bin/bash
set -e

# ============================================================================
# ULTRA-MINIMAL INSTALL FÜR video_generator_2.py
# Für: vastai/base-image:cuda-12.1.1-auto
# Vermeidet Konflikte mit held packages!
# ============================================================================

echo "=============================================="
echo " 🚀 VIDEO GENERATOR - ULTRA-MINIMAL SETUP"
echo "=============================================="

# ============================================================================
# PYTHON VENV (klein und sauber)
# ============================================================================
echo ""
echo "🐍 Python setup..."
apt update -y
apt install -y --no-install-recommends python3-venv python3-pip

VENV_PATH="/workspace/video_env"
python3 -m venv "$VENV_PATH" --system-site-packages
source "$VENV_PATH/bin/activate"

pip install --no-cache-dir --upgrade pip

# ============================================================================
# PYTORCH + MINIMAL PACKAGES
# ============================================================================
echo ""
echo "🔥 Installing PyTorch..."
pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "📚 Installing Python packages..."
pip install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    tqdm

# ============================================================================
# FFMPEG - VERWENDE VORHANDENES AUS BASE IMAGE
# ============================================================================
echo ""
echo "🎥 Checking FFmpeg..."

# Base image hat bereits FFmpeg, wir prüfen nur ob h264_nvenc verfügbar ist
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg gefunden: $(ffmpeg -version | head -n1)"
    
    # Prüfe NVENC
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
        echo "✅ h264_nvenc verfügbar"
    else
        echo "⚠️  h264_nvenc nicht verfügbar - installiere FFmpeg neu..."
        
        # Minimal build tools nur für FFmpeg
        apt install -y --no-install-recommends \
            build-essential \
            git \
            pkg-config \
            yasm \
            nasm
        
        # FFmpeg dependencies
        apt install -y --no-install-recommends \
            libfreetype6-dev \
            libfontconfig1-dev \
            libx264-dev \
            libx265-dev \
            libharfbuzz-dev \
            libfribidi-dev \
            libass-dev \
            libssl-dev \
            zlib1g-dev
        
        # NVENC headers
        cd /tmp
        git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
        cd nv-codec-headers
        git checkout n12.1.14.0
        make install
        cd /tmp
        rm -rf nv-codec-headers
        
        # CUDA paths
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        export CUDA_HOME=/usr/local/cuda
        
        # Build FFmpeg
        echo "🔨 Building FFmpeg..."
        cd /usr/local/src
        rm -rf ffmpeg
        git clone --depth 1 --branch n6.1 https://git.ffmpeg.org/ffmpeg.git
        cd ffmpeg
        
        ./configure \
          --prefix=/usr/local \
          --enable-gpl \
          --enable-nonfree \
          --disable-doc \
          --disable-debug \
          --enable-cuda-nvcc \
          --enable-libnpp \
          --nvccflags="-gencode arch=compute_86,code=sm_86" \
          --extra-cflags="-I/usr/local/cuda/include" \
          --extra-ldflags="-L/usr/local/cuda/lib64" \
          --enable-libx264 \
          --enable-libx265 \
          --enable-libass \
          --enable-libfreetype \
          --enable-libfontconfig \
          --enable-nvenc \
          --enable-openssl \
          --enable-libharfbuzz \
          --enable-libfribidi \
          --enable-ffnvcodec \
          --disable-ffplay \
          --disable-ffprobe
        
        make -j$(nproc)
        make install
        ldconfig
        hash -r
        
        # Cleanup build deps - OHNE --purge und OHNE *-dev wildcard
        apt remove -y build-essential git pkg-config yasm nasm
        apt autoremove -y
        rm -rf /usr/local/src/ffmpeg
    fi
else
    echo "❌ FFmpeg nicht gefunden!"
    exit 1
fi

# ============================================================================
# VERIFICATION
# ============================================================================
echo ""
echo "✅ Verifying installation..."

# PyTorch GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✅ GPU:', torch.cuda.get_device_name(0))"

# FFmpeg + NVENC
if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    echo "❌ h264_nvenc still not available!"
    exit 1
fi

# Wichtige Filter für dein Script
for filter in drawtext fade scale overlay; do
    if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q "^.*${filter}"; then
        echo "❌ Filter $filter missing!"
        exit 1
    fi
done

echo "✅ All checks passed!"

# ============================================================================
# ACTIVATION SCRIPT
# ============================================================================
cat > /workspace/activate_video.sh << 'EOF'
#!/bin/bash
source /workspace/video_env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "🎬 Video Generator Ready"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""
echo "Usage: python video_generator_2.py --path /project"
EOF
chmod +x /workspace/activate_video.sh

# ============================================================================
# CLEANUP (vorsichtig!)
# ============================================================================
echo ""
echo "🧹 Cleanup..."
apt clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
rm -rf /root/.cache/pip

echo ""
echo "=============================================="
echo " ✅ SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "📊 Disk usage:"
du -sh /workspace/video_env 2>/dev/null || echo "  venv: N/A"
du -sh /usr/local/bin/ffmpeg 2>/dev/null || echo "  ffmpeg: N/A"
echo ""
echo "🚀 Next steps:"
echo "  1. source /workspace/activate_video.sh"
echo "  2. python video_generator_2.py --path /project"
echo ""
echo "=============================================="

deactivate
echo "🎬 Done!"