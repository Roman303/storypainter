#!/bin/bash
set -e

# ============================================================================
# MINIMAL INSTALL - NUR WAS WIRKLICH GEBRAUCHT WIRD!
# Für: vastai/base-image:cuda-12.1.1-auto
# Reduziert von 2GB auf ~1GB Installation
# ============================================================================

echo "=============================================="
echo " 🚀 VIDEO GENERATOR - MINIMAL SETUP"
echo "=============================================="

# ============================================================================
# MINIMAL BUILD TOOLS (nur für FFmpeg compile)
# ============================================================================
echo ""
echo "📦 Installing minimal build tools..."
apt update -y
apt install -y --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    yasm \
    nasm \
    ca-certificates

# ============================================================================
# MINIMAL FFMPEG DEPENDENCIES (nur was dein Script braucht)
# ============================================================================
echo ""
echo "📦 Installing FFmpeg dependencies..."
apt install -y --no-install-recommends \
    libfreetype6-dev \
    libfontconfig1-dev \
    libx264-dev \
    libx265-dev \
    zlib1g-dev \
    libssl-dev \
    libgl1

# Text rendering (für drawtext - WICHTIG!)
apt install -y --no-install-recommends \
    libharfbuzz-dev \
    libfribidi-dev \
    libass-dev

echo "✅ Minimal dependencies installed"

# ============================================================================
# PYTHON VENV (klein und sauber)
# ============================================================================
echo ""
echo "🐍 Python setup..."
apt install -y --no-install-recommends python3-venv python3-pip python3-dev

VENV_PATH="/workspace/video_env"
python3 -m venv "$VENV_PATH" --system-site-packages
source "$VENV_PATH/bin/activate"

pip install --no-cache-dir --upgrade pip

# ============================================================================
# PYTORCH + MINIMAL PACKAGES
# ============================================================================
echo ""
echo "🔥 Installing PyTorch (this is the big one - 800MB)..."
pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "📚 Installing minimal Python packages..."
pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    tqdm

# ============================================================================
# NVENC HEADERS (klein aber wichtig für GPU encoding)
# ============================================================================
echo ""
echo "🎮 Installing NVENC headers..."
cd /tmp
git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make install
cd /tmp
rm -rf nv-codec-headers

# ============================================================================
# CUDA SETUP
# ============================================================================
echo ""
echo "🔧 CUDA setup..."
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found!"
    exit 1
fi

echo "✅ nvcc found: $(nvcc --version | grep release | awk '{print $5}')"

# ============================================================================
# BUILD FFMPEG - MINIMAL CONFIG
# ============================================================================
echo ""
echo "🎥 Building FFmpeg (minimal - only what you need)..."

FFMPEG_VERSION="n6.1"
cd /usr/local/src
rm -rf ffmpeg
git clone --depth 1 --branch $FFMPEG_VERSION https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

# MINIMAL FFmpeg - nur deine Filter + NVENC
./configure \
  --prefix=/usr/local \
  --enable-gpl \
  --enable-nonfree \
  --disable-doc \
  --disable-htmlpages \
  --disable-manpages \
  --disable-podpages \
  --disable-txtpages \
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
  --enable-filter=drawtext \
  --enable-filter=fade \
  --enable-filter=xfade \
  --enable-filter=gblur \
  --enable-filter=scale \
  --enable-filter=pad \
  --enable-filter=format \
  --enable-filter=setsar \
  --disable-everything \
  --enable-encoder=h264_nvenc \
  --enable-encoder=aac \
  --enable-decoder=h264 \
  --enable-decoder=png \
  --enable-decoder=mjpeg \
  --enable-muxer=mp4 \
  --enable-muxer=rawvideo \
  --enable-demuxer=image2 \
  --enable-demuxer=rawvideo \
  --enable-protocol=file \
  --enable-protocol=pipe

if [ $? -ne 0 ]; then
    echo "❌ Configure failed!"
    tail -50 ffbuild/config.log
    exit 1
fi

echo "🔨 Compiling (10-15 min)..."
make -j$(nproc)
make install
ldconfig
hash -r

# ============================================================================
# VERIFICATION
# ============================================================================
echo ""
echo "✅ Verifying installation..."

# FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg not installed!"
    exit 1
fi

# h264_nvenc
if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
    echo "❌ h264_nvenc not found!"
    exit 1
fi

# Filters
for filter in drawtext fade xfade gblur scale pad; do
    if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q "${filter}"; then
        echo "❌ Filter $filter missing!"
        exit 1
    fi
done

# PyTorch GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✅ GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "✅ All checks passed!"

# ============================================================================
# ACTIVATION SCRIPT
# ============================================================================
cat > /workspace/activate_video.sh << 'EOF'
#!/bin/bash
source /workspace/video_env/bin/activate
echo "🎬 Video Generator Ready"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
EOF
chmod +x /workspace/activate_video.sh

# ============================================================================
# AGGRESSIVE CLEANUP
# ============================================================================
echo ""
echo "🧹 Aggressive cleanup..."

# Remove build deps (nicht mehr gebraucht)
apt remove -y --purge \
    build-essential \
    git \
    pkg-config \
    yasm \
    nasm \
    *-dev

apt autoremove -y --purge
apt clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
rm -rf /root/.cache/pip
rm -rf /usr/local/src/ffmpeg

echo ""
echo "=============================================="
echo " ✅ MINIMAL SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "📊 Disk usage:"
du -sh /workspace/video_env 2>/dev/null || echo "  venv: N/A"
du -sh /usr/local/bin/ffmpeg 2>/dev/null || echo "  ffmpeg: N/A"
echo ""
echo "🚀 Usage:"
echo "  source /workspace/activate_video.sh"
echo "  python video_generator_2.py --path /project"
echo ""
echo "=============================================="

deactivate
echo "🎬 Done!"