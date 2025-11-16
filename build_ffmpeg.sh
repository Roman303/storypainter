#!/bin/bash
# ^^^ WICHTIG: Shebang ist bash!

# === Farben ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }

log "Starte FFmpeg-Build mit NVENC für RTX 4060 Ti..."

# === 1. Abhängigkeiten ===
log "Installiere Build-Tools..."
apt update -qq && \
apt install -y \
    build-essential git yasm nasm pkg-config \
    libx264-dev libx265-dev libvpx-dev \
    libfdk-aac-dev libmp3lame-dev libopus-dev \
    libass-dev libfreetype6-dev zlib1g-dev libnuma-dev \
    || error "Abhängigkeiten fehlgeschlagen!"

# === 2. CUDA ===
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
log "CUDA: $CUDA_HOME"

# === 3. FFmpeg Source ===
log "Lade FFmpeg..."
cd /tmp || error "Kann nicht in /tmp wechseln!"
rm -rf ffmpeg
git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git || error "Git-Clone fehlgeschlagen!"
cd ffmpeg || error "Kann nicht in ffmpeg-Ordner!"

# === 4. Configure ===
log "Konfiguriere FFmpeg mit NVENC (RTX 4060 Ti)..."
NVCCFLAGS="-gencode=arch=compute_89,code=sm_89 -O2"

./configure \
    --enable-gpl --enable-nonfree \
    --enable-nvenc --enable-cuda-nvcc --enable-libnpp --enable-cuvid \
    --enable-libx264 --enable-libx265 --enable-libvpx \
    --enable-libfdk-aac --enable-libmp3lame --enable-libopus \
    --enable-libass --enable-libfreetype \
    --extra-cflags="-I${CUDA_HOME}/include" \
    --extra-ldflags="-L${CUDA_HOME}/lib64" \
    --nvccflags="${NVCCFLAGS}" \
    || error "Configure fehlgeschlagen! Prüfe config.log"

# === 5. Build & Install ===
log "Kompiliere FFmpeg..."
make -j$(nproc) || error "Make fehlgeschlagen!"
make install || error "Install fehlgeschlagen!"

# === 6. Aufräumen ===
cd / && rm -rf /tmp/ffmpeg
success "FFmpeg mit NVENC installiert!"

# === 7. Test ===
log "Teste NVENC..."
if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
    success "NVENC AKTIV!"
    ffmpeg -encoders 2>/dev/null | grep nvenc | head -3
else
    warn "NVENC fehlt – aber FFmpeg ist neu gebaut!"
fi

echo ""
success "FFMPEG MIT NVENC FERTIG!"
echo "Starte:"
echo "  cd /workspace/storypainter"
echo "  python3 story_renderer.py --path /workspace/data --quality sd"