#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v7.0 – Mit GPU Ken-Burns Zoom-Effekt

NEU in v7.0:
  - GPU-beschleunigter Ken-Burns Zoom für Bilder (PyTorch CUDA)
  - Konfigurierbare Zoom-Stärke (--kb-strength)
  - Zoom-Richtung (--kb-direction): none, left, right, up, down, diagonal
  - Easing-Funktionen (--kb-ease): linear, ease_in, ease_out, ease_in_out
  - Automatische GPU-Erkennung mit CPU-Fallback
  
Features von v6.3:
  - JSON-strikt, Intro=Szene 0, HyperTrail nur im Hintergrund
  - Outro-Support, frühe Einblendungen via Offsets
  - Weiche Gap-Blenden (Black-Fades), 1080p30
"""

import subprocess
from pathlib import Path
import json, argparse, shutil

# ---------- PyTorch für GPU Ken-Burns (optional) ----------
try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.io import read_image
    import numpy as np  # Für Fast-Pipe Rendering
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch nicht verfügbar - Ken-Burns Zoom deaktiviert")

# ---------- utils ----------
def has_nvenc() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False

def run(cmd, quiet=False):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        try:
            print(r.stderr.decode("utf-8", "ignore"))
        except Exception:
            print(r.stderr)
    return r.returncode == 0

def esc_txt(s: str) -> str:
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ken_burns_gpu_image(
    img_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    fi_start: float,
    fi_dur: float,
    fo_start: float,
    fo_dur: float,
    zoom_start: float = 1.0,
    zoom_end: float = 1.05,
    pan: str = "none",
    ease: str = "ease_in_out",
    use_fp16: bool = True,
    nvenc: bool = True
) -> Path:
    """
    GPU Ken-Burns mit temp PNG files.
    OPTIMIERT: Bild wird nur EINMAL geladen, nicht pro Frame!
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch nicht verfügbar")

    import tempfile
    import time
    
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
    
    num_frames = max(1, int(round(clip_dur * fps)))
    tmp_dir = Path(tempfile.mkdtemp(prefix="kb_"))
    
    # KRITISCH: Bild nur EINMAL laden!
    print(f"   📥 Lade Bild: {img_path.name}...")
    img = read_image(str(img_path)).to(device=device, dtype=dtype) / 255.0  # [C,H,W]
    C, H, W = img.shape
    print(f"   ✅ Bild geladen: {W}x{H}, {C} channels")
    
    # Easing-Funktion
    def ease_fn(t: float) -> float:
        if ease == "ease_in_out":
            return t*t*t*(t*(t*6 - 15) + 10)
        elif ease == "ease_in":
            return t*t
        elif ease == "ease_out":
            return 1 - (1-t)*(1-t)
        return t
    
    # Pan-Richtung
    pan_dx, pan_dy = 0.0, 0.0
    if pan in ("left", "right", "up", "down", "diag_tl", "diag_tr", "diag_bl", "diag_br"):
        mapping = {
            "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1),
            "diag_tl": (-1, -1), "diag_tr": (1, -1), "diag_bl": (-1, 1), "diag_br": (1, 1),
        }
        pan_dx, pan_dy = mapping[pan]
        norm = (pan_dx*pan_dx + pan_dy*pan_dy) ** 0.5
        if norm > 0:
            pan_dx, pan_dy = pan_dx / norm, pan_dy / norm
    
    print(f"   🎨 Ken-Burns GPU: {num_frames} frames @ {fps}fps on {device}")
    
    # GPU Batch-Processing für maximale Performance
    batch_size = 30 if device == "cuda" else 10
    frames_saved = 0
    last_report = time.time()
    
    # Pre-compute alle Transformations-Parameter
    print(f"   📐 Pre-compute Transformationen...")
    transforms = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0.0
        et = ease_fn(t)
        scale = zoom_start + (zoom_end - zoom_start) * et
        
        new_h, new_w = int(H * scale), int(W * scale)
        if new_h < height or new_w < width:
            scale_factor = max(height / new_h, width / new_w)
            new_h, new_w = int(new_h * scale_factor), int(new_w * scale_factor)
        
        max_off_x, max_off_y = max(0, new_w - width), max(0, new_h - height)
        cx = int(max_off_x * 0.5 * (1 + pan_dx * (2 * et - 1)))
        cy = int(max_off_y * 0.5 * (1 + pan_dy * (2 * et - 1)))
        off_x = clamp(cx - width // 2, 0, max_off_x)
        off_y = clamp(cy - height // 2, 0, max_off_y)
        
        tt = i / fps
        alpha = 1.0
        if fi_dur > 0 and tt >= fi_start:
            alpha = min(alpha, (tt - fi_start) / fi_dur)
        if fo_dur > 0 and tt >= fo_start:
            alpha = min(alpha, max(0.0, (clip_dur - tt) / fo_dur))
        alpha = float(clamp(alpha, 0.0, 1.0))
        
        transforms.append((new_h, new_w, off_x, off_y, alpha))
    
    print(f"   ✅ {num_frames} Transformationen berechnet")
    
    # Batch-Rendering mit direktem numpy save (10x schneller als PIL)
    try:
        from PIL import Image
        import numpy as np
        
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            
            for i in range(start_idx, end_idx):
                new_h, new_w, off_x, off_y, alpha = transforms[i]
                
                # GPU Transform
                zimg = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC)
                zimg = zimg[:, off_y:off_y+height, off_x:off_x+width]
                
                if zimg.shape[1] != height or zimg.shape[2] != width:
                    zimg = TF.center_crop(zimg, [height, width])
                
                # Apply alpha - DIREKT zu numpy uint8
                frame_tensor = (zimg.clamp(0, 1) * alpha * 255).to(dtype=torch.uint8).cpu()
                frame_np = frame_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
                
                # Schneller Save mit PIL (fromarray ist schneller als TF.to_pil_image)
                Image.fromarray(frame_np, mode='RGB').save(
                    tmp_dir / f"f_{i:06d}.png",
                    compress_level=1  # Minimal compression für Speed
                )
                frames_saved += 1
            
            # Progress
            now = time.time()
            elapsed = now - start_time
            fps_current = frames_saved / elapsed if elapsed > 0 else 0
            eta = (num_frames - frames_saved) / fps_current if fps_current > 0 else 0
            progress = (frames_saved / num_frames) * 100
            
            print(f"   ⏳ {frames_saved}/{num_frames} ({progress:.1f}%) | {fps_current:.1f} fps | ETA: {int(eta)}s", end='\r')
            
            # Memory cleanup nach jedem Batch
            if device == "cuda":
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        avg_fps = frames_saved / elapsed
        print(f"\n   ✅ {frames_saved} frames | {elapsed:.1f}s | avg {avg_fps:.1f} fps")
        
    except Exception as e:
        print(f"\n   ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n   🎬 Encoding {num_frames} frames (25 threads)...")
    
    # FFmpeg mit allen Cores
    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M"] if nvenc else \
          ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-threads", "25"]
    
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-threads", "25",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "f_%06d.png"),
        "-pix_fmt", "yuv420p",
        *enc,
        str(out_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"   ⚠️ FFmpeg Error: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
    else:
        print(f"   ✅ Video encoded: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return out_path



    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch nicht verfügbar")

    import numpy as np
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
    
    num_frames = max(1, int(round(clip_dur * fps)))
    
    # Bild laden
    img = read_image(str(img_path)).to(device=device, dtype=dtype) / 255.0  # [C,H,W]
    C, H, W = img.shape
    
    # Easing-Funktion
    def ease_fn(t: float) -> float:
        if ease == "ease_in_out":
            return t*t*t*(t*(t*6 - 15) + 10)
        elif ease == "ease_in":
            return t*t
        elif ease == "ease_out":
            return 1 - (1-t)*(1-t)
        return t
    
    # Pan-Richtung
    pan_dx, pan_dy = 0.0, 0.0
    if pan in ("left", "right", "up", "down", "diag_tl", "diag_tr", "diag_bl", "diag_br"):
        mapping = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, -1),
            "down": (0, 1),
            "diag_tl": (-1, -1),
            "diag_tr": (1, -1),
            "diag_bl": (-1, 1),
            "diag_br": (1, 1),
        }
        pan_dx, pan_dy = mapping[pan]
        norm = (pan_dx*pan_dx + pan_dy*pan_dy) ** 0.5
        if norm > 0:
            pan_dx, pan_dy = pan_dx / norm, pan_dy / norm
    
    print(f"   🎨 Ken-Burns Fast-Pipe: {num_frames} frames @ {fps}fps on {device}")
    
    # FFmpeg mit Pipe starten - KORRIGIERTE VERSION
    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M"] if nvenc else \
          ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-threads", "8"]
    
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",  # stdin pipe
        "-pix_fmt", "yuv420p",
        *enc,
        "-r", str(fps),
        str(out_path)
    ]
    
    print(f"   🔧 FFmpeg Command: {' '.join(ffmpeg_cmd[:15])}...")
    
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8  # 100MB buffer
    )
    
    try:
        # Frames direkt in FFmpeg schreiben
        for i in range(num_frames):
            # Progress
            if i % 30 == 0 or i == num_frames - 1:
                progress = ((i + 1) / num_frames) * 100
                print(f"   ⏳ {progress:.1f}% ({i+1}/{num_frames})", end='\r')
            
            t = i / (num_frames - 1) if num_frames > 1 else 0.0
            et = ease_fn(t)
            scale = zoom_start + (zoom_end - zoom_start) * et
            
            # Skaliertes Bild - IMMER größer als Ziel für Zoom-Out Effekt
            new_h, new_w = int(H * scale), int(W * scale)
            
            # Sicherstellen dass Bild mindestens Zielgröße hat
            if new_h < height or new_w < width:
                # Upscale falls zu klein
                scale_factor = max(height / new_h, width / new_w)
                new_h = int(new_h * scale_factor)
                new_w = int(new_w * scale_factor)
            
            zimg = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC)
            
            # Pan-Offset
            max_off_x = max(0, new_w - width)
            max_off_y = max(0, new_h - height)
            
            cx = int(max_off_x * 0.5 * (1 + pan_dx * (2 * et - 1)))
            cy = int(max_off_y * 0.5 * (1 + pan_dy * (2 * et - 1)))
            off_x = clamp(cx - width // 2, 0, max_off_x)
            off_y = clamp(cy - height // 2, 0, max_off_y)
            
            # Crop
            zimg = zimg[:, off_y:off_y+height, off_x:off_x+width]
            if zimg.shape[1] != height or zimg.shape[2] != width:
                zimg = TF.center_crop(zimg, [height, width])
            
            # Fade-In/Out
            tt = i / fps
            alpha = 1.0
            if fi_dur > 0 and tt >= fi_start:
                alpha = min(alpha, (tt - fi_start) / fi_dur)
            if fo_dur > 0 and tt >= fo_start:
                alpha = min(alpha, max(0.0, (clip_dur - tt) / fo_dur))
            alpha = float(clamp(alpha, 0.0, 1.0))
            
            # Frame zu RGB24 numpy
            frame = (zimg.clamp(0, 1) * alpha * 255).to(dtype=torch.uint8).cpu().numpy()
            frame = frame.transpose(1, 2, 0)  # CHW -> HWC
            
            # Sicherstellen dass Frame die richtige Größe hat
            if frame.shape[0] != height or frame.shape[1] != width:
                print(f"\n   ⚠️ Frame Size Mismatch: {frame.shape} vs {height}x{width}")
                continue
            
            # In FFmpeg Pipe schreiben
            try:
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
            except BrokenPipeError:
                print(f"\n   ❌ FFmpeg crashed at frame {i}")
                # Lese stderr für Fehlerdiagnose
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                print(f"   FFmpeg Error:\n{stderr[-500:]}")
                break
            
            # Memory cleanup
            if i % 30 == 0 and device == "cuda":
                torch.cuda.empty_cache()
        
        print(f"\n   ✅ Alle Frames geschrieben, warte auf FFmpeg...")
        
    except Exception as e:
        print(f"\n   ❌ Exception während Rendering: {e}")
    finally:
        try:
            process.stdin.close()
        except:
            pass
        
        process.wait(timeout=10)
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode('utf-8', errors='ignore')
            print(f"   ⚠️ FFmpeg Exit Code: {process.returncode}")
            print(f"   Last 500 chars of stderr:\n{stderr[-500:]}")
    
    return out_path

# ---------- renderer ----------
class StoryRenderer:
    def __init__(self, images_dir: Path, metadata_path: Path, output_dir: Path):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "temp_clips"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

        self.base_dir = self.output_dir.parent

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)

        self.nvenc_available = has_nvenc()
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        if self.nvenc_available:
            print("🎞️ GPU (NVENC) erkannt und aktiviert.")
        else:
            print("⚠️ Kein NVENC gefunden – verwende CPU (libx264).")
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🔥 PyTorch CUDA verfügbar: {gpu_name}")
        else:
            print("⚠️ PyTorch CUDA nicht verfügbar - Ken-Burns auf CPU (langsam)")

    @staticmethod
    def _is_video(p: Path) -> bool:
        return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    @staticmethod
    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    # ----------- builders -----------
    def _render_source_with_fades(
        self,
        src_path: Path | None,
        clip_dur: float,
        fade_in_start: float,
        fade_in_dur: float,
        fade_out_start: float,
        fade_out_dur: float,
        width: int,
        height: int,
        fps: int,
        idx: int,
        kb_strength: float = 0.0,
        kb_direction: str = "none",
        kb_ease: str = "ease_in_out"
    ) -> Path:
        """
        Renderer für Szenen mit optionalem Ken-Burns:
        - kb_strength > 0: GPU Ken-Burns Zoom
        - kb_strength == 0: Normale Darstellung
        """
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        clip_dur = max(0.0, float(clip_dur))

        # Ken-Burns für Bilder (wenn aktiviert)
        if (src_path and src_path.exists() and self._is_image(src_path) 
            and kb_strength > 0 and TORCH_AVAILABLE):
            
            # ZOOM-RICHTUNG: Start IMMER größer (fill screen), dann zoom out
            # Berechne initialen Zoom-Faktor damit Bild immer Screen füllt
            img_for_size = read_image(str(src_path))
            _, img_h, img_w = img_for_size.shape
            
            # Minimaler Zoom um Screen zu füllen
            min_zoom = max(height / img_h, width / img_w)
            
            # Zoom startet bei min_zoom (fills screen) + kb_strength Offset
            zoom_start = min_zoom * (1.0 + clamp(kb_strength, 0.0, 1.0) * 0.05)
            zoom_end = min_zoom  # Endet bei exakt screen-filling
            
            print(f"⚙️  Szene {idx:02d} - Ken-Burns GPU (zoom {zoom_start:.3f}→{zoom_end:.3f}, {kb_direction})")
            print(f"   📐 Bild: {img_w}x{img_h}, Screen: {width}x{height}, min_zoom: {min_zoom:.3f}")
            
            return ken_burns_gpu_image(
                img_path=src_path,
                out_path=outp,
                width=width,
                height=height,
                fps=fps,
                clip_dur=clip_dur,
                fi_start=fade_in_start,
                fi_dur=fade_in_dur,
                fo_start=fade_out_start,
                fo_dur=fade_out_dur,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                pan=kb_direction,
                ease=kb_ease,
                use_fp16=True,
                nvenc=self.nvenc_available
            )

        # Fallback: Standard-Rendering ohne Ken-Burns
        if src_path and src_path.exists():
            if self._is_image(src_path):
                inputs = [
                    "-loop", "1",
                    "-t", f"{clip_dur:.6f}",
                    "-r", str(fps),
                    "-i", str(src_path)
                ]
                base = (
                    f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
                )
            elif self._is_video(src_path):
                inputs = [
                    "-ss", "0",
                    "-t", f"{clip_dur:.6f}",
                    "-i", str(src_path)
                ]
                base = (
                    f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
                )
            else:
                inputs = [
                    "-f", "lavfi",
                    "-t", f"{clip_dur:.6f}",
                    "-i", f"color=c=black:s={width}x{height}:r={fps}"
                ]
                base = "[0:v]format=yuv420p"
        else:
            inputs = [
                "-f", "lavfi",
                "-t", f"{clip_dur:.6f}",
                "-i", f"color=c=black:s={width}x{height}:r={fps}"
            ]
            base = "[0:v]format=yuv420p"

        fade_in_start = max(0.0, fade_in_start)
        fade_in_dur = max(0.0, fade_in_dur)
        fade_out_start = max(0.0, fade_out_start)
        fade_out_dur = max(0.0, fade_out_dur)

        flt = (
            f"{base},"
            f"fade=t=in:st={fade_in_start:.6f}:d={fade_in_dur:.6f},"
            f"fade=t=out:st={fade_out_start:.6f}:d={fade_out_dur:.6f}[v]"
        )

        if self.nvenc_available:
            enc = [
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-b:v", "12M",
                "-pix_fmt", "yuv420p"
            ]
        else:
            enc = [
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p"
            ]

        print(f"⚙️  Szene {idx:02d} rendern … (Quelle: {src_path.name if src_path else 'BLACK'})")

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", flt,
            "-map", "[v]",
            "-r", str(fps),
            "-an",
            *enc,
            "-t", f"{clip_dur:.6f}",
            str(outp)
        ]
        run(cmd, quiet=True)
        return outp

    def _render_intro(
        self,
        intro_src: Path | None,
        intro_base_dur: float,
        width: int,
        height: int,
        fps: int,
        title: str,
        author: str,
        text_in_at: float,
        fade_out: float,
        fade_out_offset: float
    ) -> Path:
        """Intro OHNE drawtext (FFmpeg fehlt libfreetype)"""
        intro_base_dur = max(0.0, float(intro_base_dur))
        intro_clip_dur = intro_base_dur

        t_out_start = clamp(
            intro_base_dur + fade_out_offset,
            0.0,
            max(0.0, intro_clip_dur - fade_out)
        )

        outp = self.tmp_dir / "intro_0000.mp4"

        if intro_src and intro_src.exists():
            print(f"   📁 Intro-Quelle: {intro_src.name}")
            
            if self._is_video(intro_src):
                inputs = ["-stream_loop", "-1", "-i", str(intro_src), "-t", f"{intro_clip_dur:.6f}"]
            else:
                inputs = ["-loop", "1", "-t", f"{intro_clip_dur:.6f}", "-i", str(intro_src)]
        else:
            print(f"   ⚠️ Keine Intro-Quelle, nutze Schwarz")
            inputs = [
                "-f", "lavfi",
                "-t", f"{intro_clip_dur:.6f}",
                "-i", f"color=c=black:s={width}x{height}:r={fps}"
            ]

        # KEIN drawtext - nur Video processing
        flt = (
            f"[0:v]"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuv420p,setsar=1,"
            f"eq=brightness=-0.15,"
            f"fade=t=out:st={t_out_start:.6f}:d={fade_out:.6f}[v]"
        )

        if self.nvenc_available:
            enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"]
        else:
            enc = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"]

        print("   🎬 Rendere Intro (ohne Text-Overlay)...")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs,
            "-filter_complex", flt,
            "-map", "[v]",
            "-r", str(fps),
            "-an",
            *enc,
            str(outp)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            print(f"   ❌ FFmpeg Error:\n{stderr[:500]}")
        
        if outp.exists():
            print(f"   ✅ Intro: {outp.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"   ⚠️ Intro nicht erstellt - überspringe")
            
        return outp

    def _build_gap_black(self, duration: float, width: int, height: int, fps: int, idx: int) -> Path:
        """Schwarzer Zwischenclip mit weichem Fade-In/Out"""
        d = max(0.0, float(duration))
        outp = self.tmp_dir / f"gap_{idx:04d}.mp4"
        if d < 1e-3:
            d = 1.0 / max(1, fps)

        fade_each = min(0.5, d / 2.0)
        flt = (
            f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"
            f"fade=t=in:st=0:d={fade_each:.6f},"
            f"fade=t=out:st={(d - fade_each):.6f}:d={fade_each:.6f}[v]"
        )

        if self.nvenc_available:
            enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"]
        else:
            enc = ["-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-t", f"{d:.6f}",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-filter_complex", flt,
            "-map", "[v]",
            "-an",
            *enc,
            "-t", f"{d:.6f}",
            str(outp)
        ]
        run(cmd, quiet=True)
        return outp

    def _merge_concat(self, items, out_path: Path):
        """Concatenate clips ohne Re-Encode"""
        concat_file = out_path.parent / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in items:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        print(f"🔗 Verbinde {len(items)} Segmente (Concat, -c copy) …")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(out_path)
        ]
        run(cmd, quiet=False)
        return out_path

    # ----------- main render -----------
    def render(
        self,
        audiobook_file: Path,
        images_prefix="image_",
        width=1920,
        height=1080,
        fps=30,
        fade_in=1.5,
        fade_out=2.0,
        fade_in_offset=0.0,
        fade_out_offset=0.0,
        kb_strength=0.5,
        kb_direction="none",
        kb_ease="ease_in_out",
        overlay_file=None,
        overlay_opacity=0.35,
        quality="hd"
    ):
        scenes = self.meta.get("scenes", [])
        if not scenes:
            print("❌ Keine Szenen im JSON.")
            return None

        title = self.meta.get("title", "")
        author = self.meta.get("author", "")

        n = len(scenes)
        starts = [float(s["start_time"]) for s in scenes]
        ends = [float(s["end_time"]) for s in scenes]
        bases = [max(0.0, ends[i] - starts[i]) for i in range(n)]

        scene_clips, scene_durs = [], []

        for i, s in enumerate(scenes):
            sid = int(s.get("scene_id", i))
            stype = s.get("type", "scene")
            base = bases[i]

            used_fi_offset = float(fade_in_offset if i > 0 else 0.0)
            pre_extend = -min(0.0, used_fi_offset)
            clip_dur = base + pre_extend

            fi_start = max(
                0.0,
                (starts[i] + used_fi_offset) - (starts[i] - pre_extend)
            )
            fi_start = clamp(fi_start, 0.0, max(0.0, clip_dur - fade_in))

            fo_start_raw = (ends[i] + float(fade_out_offset)) - (starts[i] - pre_extend)
            fo_start = clamp(fo_start_raw, 0.0, max(0.0, clip_dur - fade_out))

            print(
                f"➡️ Szene {i+1}/{n}  ID={sid}  type={stype}  "
                f"base={base:.3f}s  clip_dur={clip_dur:.3f}s  "
                f"fi@{fi_start:.2f}/d{fade_in:.2f}  "
                f"fo@{fo_start:.2f}/d{fade_out:.2f}"
            )

            # ---------- Intro ----------
            if i == 0 or stype == "intro":
                intro_file = self.base_dir / "intro.mp4"
                img_intro = self.images_dir / f"{images_prefix}{sid:04d}.png"
                intro_src = (
                    intro_file if intro_file.exists()
                    else (img_intro if img_intro.exists() else None)
                )

                clip = self._render_intro(
                    intro_src=intro_src,
                    intro_base_dur=base,
                    width=width,
                    height=height,
                    fps=fps,
                    title=title,
                    author=author,
                    text_in_at=3.0,
                    fade_out=fade_out - 0.2,
                    fade_out_offset=fade_out_offset
                )
                scene_clips.append(clip)
                scene_durs.append(base)
                continue

            # ---------- Outro ----------
            if stype == "outro":
                outro_video = self.base_dir / "outro.mp4"
                outro_image = self.images_dir / "outro.png"
                img = self.images_dir / f"{images_prefix}{sid:04d}.png"

                if outro_video.exists():
                    src = outro_video
                elif outro_image.exists():
                    src = outro_image
                else:
                    src = img

                clip = self._render_source_with_fades(
                    src_path=src,
                    clip_dur=clip_dur,
                    fade_in_start=fi_start,
                    fade_in_dur=fade_in,
                    fade_out_start=fo_start,
                    fade_out_dur=fade_out,
                    width=width,
                    height=height,
                    fps=fps,
                    idx=i,
                    kb_strength=0.0  # Kein Zoom für Outro
                )
                scene_clips.append(clip)
                scene_durs.append(clip_dur)
                continue

            # ---------- Normale Szenen mit Ken-Burns ----------
            img = self.images_dir / f"{images_prefix}{sid:04d}.png"
            clip = self._render_source_with_fades(
                src_path=img,
                clip_dur=clip_dur,
                fade_in_start=fi_start,
                fade_in_dur=fade_in,
                fade_out_start=fo_start,
                fade_out_dur=fade_out,
                width=width,
                height=height,
                fps=fps,
                idx=i,
                kb_strength=kb_strength,
                kb_direction=kb_direction,
                kb_ease=kb_ease
            )
            scene_clips.append(clip)
            scene_durs.append(clip_dur)

        # ---------- Gaps ----------
        items = []
        for i in range(n):
            items.append(scene_clips[i])

            if i < n - 1:
                if i == 0 and scenes[i].get("type", "") in {"intro"}:
                    end_i = starts[i] + bases[i]
                else:
                    end_i = ends[i]

                gap_real = max(0.0, starts[i + 1] - end_i)
                next_in_offset = float(fade_in_offset if (i + 1) > 0 else 0.0)
                gap_eff = max(0.0, gap_real + next_in_offset)
                
                if gap_eff > 1e-3:
                    gap_clip = self._build_gap_black(gap_eff, width, height, fps, idx=i)
                    items.append(gap_clip)

        print(f"🔎 Merge-Check: Segmente={len(items)}  (Szenen + Gaps)")
        merged = self.output_dir / "_merged_master.mp4"
        self._merge_concat(items, merged)

        # ---------- Overlay ----------
        visual = merged
        overlay_path = Path(overlay_file) if overlay_file else None
        if overlay_path and overlay_path.exists():
            print("✨ Overlay anwenden …")
            ov_out = self.output_dir / "_visual_overlay.mp4"
            if overlay_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                ov_inputs = ["-stream_loop", "-1", "-i", str(overlay_path)]
            else:
                ov_inputs = ["-loop", "1", "-r", str(fps), "-i", str(overlay_path)]

            if self.nvenc_available:
                enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"]
            else:
                enc = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]

            cmd = [
                "ffmpeg", "-y",
                "-i", str(merged),
                *ov_inputs,
                "-filter_complex",
                f"[0:v]format=yuv420p[base];"
                f"[1:v]scale={width}:{height},format=rgba,"
                f"colorchannelmixer=aa={overlay_opacity:.3f}[ovr];"
                f"[base][ovr]overlay=0:0:shortest=1[out]",
                "-map", "[out]",
                "-an",
                *enc,
                str(ov_out)
            ]
            run(cmd, quiet=True)
            visual = ov_out

        # ---------- Audio-Mux ----------
        print("🔊 Muxe Video + Audio …")
        final_hd = self.output_dir / "story_final_hd.mp4"

        if not overlay_path or not overlay_path.exists():
            cmd_hd = [
                "ffmpeg", "-y",
                "-fflags", "+genpts",
                "-i", str(visual),
                "-i", str(audiobook_file),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-shortest",
                str(final_hd)
            ]
        else:
            if self.nvenc_available:
                enc_v = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "10M", "-pix_fmt", "yuv420p"]
            else:
                enc_v = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
            
            cmd_hd = [
                "ffmpeg", "-y",
                "-fflags", "+genpts",
                "-i", str(visual),
                "-i", str(audiobook_file),
                "-map", "0:v:0",
                "-map", "1:a:0",
                *enc_v,
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-shortest",
                str(final_hd)
            ]

        run(cmd_hd, quiet=True)

        # Cleanup
        try:
            shutil.rmtree(self.tmp_dir)
            print("🧹 Temporäre Dateien gelöscht.")
        except Exception:
            pass

        if quality == "sd":
            print("📦 Erzeuge SD-Derivat …")
            final_sd = self.output_dir / "story_final_sd.mp4"
            run(
                [
                    "ffmpeg", "-y",
                    "-i", str(final_hd),
                    "-vf", "scale=640:360:force_original_aspect_ratio=decrease,fps=30",
                    "-c:v", "libx264",
                    "-b:v", "600k",
                    "-c:a", "aac",
                    "-b:a", "96k",
                    "-movflags", "+faststart",
                    str(final_sd)
                ],
                quiet=True
            )

        print("✅ Fertig:", final_hd)
        return final_hd


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Story Renderer v7.0 (GPU Ken-Burns Zoom, JSON-strikt, optimierte Performance)"
    )
    ap.add_argument("--path", required=True, help="Projektbasis-Verzeichnis")
    ap.add_argument("--images", default=None, help="Bildordner (default: <path>/images)")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei (default: <path>/master.wav)")
    ap.add_argument("--metadata", default=None, help="Metadata JSON (default: <path>/audiobook/audiobook_metadata.json)")
    ap.add_argument("--output", default=None, help="Output-Ordner (default: <path>/story)")
    
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd", help="Ausgabe-Qualität")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    ap.add_argument("--fade-in", type=float, default=1.5, help="Fade-In Dauer (s)")
    ap.add_argument("--fade-out", type=float, default=2.0, help="Fade-Out Dauer (s)")
    ap.add_argument("--fade-in-offset", type=float, default=0.0,
                    help="Fade-In Offset: 0=Start bei Szene; -1=1s früher")
    ap.add_argument("--fade-out-offset", type=float, default=0.0,
                    help="Fade-Out Offset: 0=Start bei Szenenende; -1=1s früher")
    
    # Ken-Burns Parameter
    ap.add_argument("--kb-strength", type=float, default=0.5,
                    help="Ken-Burns Zoom-Stärke: 0=aus, 0.5=2.5%%, 1.0=5%% (default: 0.5)")
    ap.add_argument("--kb-direction", default="none",
                    choices=["none", "left", "right", "up", "down", "diag_tl", "diag_tr", "diag_bl", "diag_br"],
                    help="Pan-Richtung während Zoom (default: none)")
    ap.add_argument("--kb-ease", default="ease_in_out",
                    choices=["linear", "ease_in", "ease_out", "ease_in_out"],
                    help="Easing-Funktion für Zoom (default: ease_in_out)")
    
    ap.add_argument("--overlay", default="particel.mp4", help="Overlay-Video/Bild (optional)")
    ap.add_argument("--overlay-opacity", type=float, default=0.35, help="Overlay-Transparenz (0-1)")
    
    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base / "images")
    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")
    output = Path(args.output) if args.output else (base / "story")

    # Overlay-Default
    if args.overlay:
        overlay = Path(args.overlay)
    else:
        overlay = base / "particel.mp4"
        if not overlay.exists():
            overlay = None

    if not audiobook.exists():
        print(f"❌ Hörbuch nicht gefunden: {audiobook}")
        return
    if not metadata.exists():
        print(f"❌ Metadaten nicht gefunden: {metadata}")
        return

    print("\n" + "="*60)
    print("🎬 Story Renderer v7.0 - GPU Ken-Burns Edition")
    print("="*60)
    print(f"📁 Projekt: {base}")
    print(f"🖼️  Bilder: {images_dir}")
    print(f"🔊 Audio: {audiobook.name}")
    print(f"🎨 Ken-Burns: strength={args.kb_strength}, direction={args.kb_direction}, ease={args.kb_ease}")
    print("="*60 + "\n")

    r = StoryRenderer(images_dir, metadata, output)
    r.render(
        audiobook_file=audiobook,
        images_prefix="image_",
        width=1920,
        height=1080,
        fps=args.fps,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        fade_in_offset=args.fade_in_offset,
        fade_out_offset=args.fade_out_offset,
        kb_strength=args.kb_strength,
        kb_direction=args.kb_direction,
        kb_ease=args.kb_ease,
        overlay_file=overlay,
        overlay_opacity=args.overlay_opacity,
        quality=args.quality
    )

if __name__ == "__main__":
    main()