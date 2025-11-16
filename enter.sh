#!/bin/bash
set -euo pipefail  # Strenge Fehlerbehandlung

# === Farben ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }

# === 1. Workspace ===
log "Prüfe Workspace..."
mkdir -p /workspace && cd /workspace || { error "Workspace-Fehler!"; exit 1; }
success "Workspace: /workspace"

# === 2. Git prüfen & Repo ===
if ! command -v git &> /dev/null; then
    log "Installiere git..."
    apt-get update -qq && apt-get install -y git || { error "git-Install fehlgeschlagen!"; exit 1; }
fi

REPO_DIR="/workspace/storypainter"
if [ ! -d "$REPO_DIR" ]; then
    log "Klone storypainter..."
    git clone https://github.com/Roman303/storypainter.git "$REPO_DIR" || { error "Clone fehlgeschlagen!"; exit 1; }
    cd "$REPO_DIR"
    success "Repo geklont"
else
    log "Aktualisiere Repo..."
    cd "$REPO_DIR"
    git pull --ff-only || warn "Pull fehlgeschlagen – nutze lokalen Stand"
fi

# === 3. Python prüfen ===
command -v python3 &> /dev/null || { error "python3 fehlt!"; exit 1; }
success "Python: $(python3 --version 2>&1 | cut -d' ' -f2)"

# === 4. FFmpeg installieren (neu: automatisiert) ===
if ! command -v ffmpeg &> /dev/null; then
    log "FFmpeg fehlt – installiere via apt (mit NVENC-Support)..."
    apt-get update -qq
    apt-get install -y \
        ffmpeg \
        libx264-dev \
        libx265-dev \
        libvpx-dev \
        libfdk-aac-dev \
        libmp3lame-dev \
        libopus-dev \
        libass-dev \
        libfreetype6-dev \
        || { error "FFmpeg-Install fehlgeschlagen! Prüfe apt-Quellen."; exit 1; }
    success "FFmpeg installiert: $(ffmpeg -version | head -n1)"
else
    success "FFmpeg bereits da: $(ffmpeg -version | head -n1)"
fi

# === 5. NVENC testen ===
log "Teste NVENC..."
if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
    success "NVENC AKTIV! (h264_nvenc verfügbar)"
    ffmpeg -encoders 2>/dev/null | grep nvenc | head -2  # Zeige 1–2 Encoder
else
    warn "NVENC NICHT gefunden – prüfe GPU/CUDA!"
    if command -v nvidia-smi &> /dev/null; then
        log "GPU-Info: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
    fi
    log "Tipp: RTX 3080+ mit CUDA 12.1 nutzen"
fi

# === 6. Datenordner ===
mkdir -p /workspace/data/{images,audiobook} || warn "Datenordner-Fehler"
success "Datenordner: /workspace/data"

# === 7. Abschluss ===
echo ""
success "ON-START ERFOLGREICH!"
echo "Nächste Schritte:"
echo "  cd $REPO_DIR"
echo "  python3 story_renderer.py --path /workspace/data --quality sd"
echo "  (Überwache: watch -n1 'nvidia-smi --query-gpu=encoders_utilization --format=csv')"