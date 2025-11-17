#!/bin/bash

set -e

echo "==============================================="
echo "              SYSTEM CHECK START"
echo "==============================================="

# -----------------------------------------------
echo
echo "▶ FFmpeg Pfad:"
which ffmpeg || echo "❌ ffmpeg nicht im PATH"

# -----------------------------------------------
echo
echo "▶ FFmpeg Version:"
if ffmpeg -version &>/dev/null; then
    ffmpeg -version | head -n 3
else
    echo "❌ FFmpeg startet nicht"
fi

# -----------------------------------------------
echo
echo "▶ NVENC Encoder:"
if ffmpeg -encoders 2>/dev/null | grep -q nvenc; then
    ffmpeg -encoders 2>/dev/null | grep nvenc
    echo "✔ NVENC vorhanden"
else
    echo "❌ Kein NVENC gefunden"
fi

# -----------------------------------------------
echo
echo "▶ NVDEC / CUDA HWACCEL:"
if ffmpeg -hwaccels 2>/dev/null | grep -q cuda; then
    ffmpeg -hwaccels | grep cuda
    echo "✔ CUDA-HWACCEL aktiv"
else
    echo "❌ Kein CUDA-HWACCEL"
fi

# -----------------------------------------------
echo
echo "▶ drawtext Filter:"
if ffmpeg -filters 2>/dev/null | grep -q drawtext; then
    echo "✔ drawtext aktiv"
else
    echo "❌ drawtext fehlt!"
fi

# -----------------------------------------------
echo
echo "▶ NVIDIA GPU:"
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv
else
    echo "❌ NVIDIA GPU nicht sichtbar"
fi

# -----------------------------------------------
echo
echo "▶ NVENC API Kompatibilität:"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d'.' -f1)

if [ "$DRV" -ge 570 ]; then
    echo "✔ Treiber modern genug für neueste NVENC API (570+)"
elif [ "$DRV" -ge 550 ]; then
    echo "⚠ Treiber ok, aber nur nv-codec-headers 12.x kompatibel"
else
    echo "❌ Treiber zu alt (< 550): NVENC API Mismatch möglich!"
fi

# -----------------------------------------------
echo
echo "▶ CUDA Toolkit:"
if nvcc --version &>/dev/null; then
    nvcc --version | head -n 3
    echo "✔ CUDA Toolkit gefunden"
else
    echo "⚠ nvcc fehlt (kein CUDA Devel Image?)"
fi

# -----------------------------------------------
echo
echo "▶ PyTorch CUDA Status:"
python3 - << 'EOF'
try:
    import torch
    print("Torch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("❌ PyTorch nicht installiert:", e)
EOF

echo
echo "==============================================="
echo "              SYSTEM CHECK FERTIG"
echo "==============================================="
