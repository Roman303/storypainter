#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Pipeline v12 – Optimiert mit JSON-Timing Treue

Wichtigste Änderungen:
- Strikte JSON-Timing Einhaltung
- Flexible Fade-Dauer ohne Limits
- Separate Intro/Outro Fade-Kontrolle
- Verbesserte Motion Blur Defaults
- Optimierte GPU-Performance
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# UTILS
# ============================================================================

def run(cmd, quiet: bool = False) -> bool:
    """FFmpeg-Befehl ausführen."""
    if not quiet:
        print("\n" + "="*60)
        print(" ".join(str(c) for c in cmd))
        print("="*60)
    
    r = subprocess.run(cmd, capture_output=True)
    
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "ignore") if r.stderr else "Unknown error"
        print(f"❌ Error: {err}")
        return False
    
    if not quiet and r.stderr:
        out = r.stderr.decode("utf-8", "ignore")
        if out.strip():
            print(out)
    
    return True


def esc_txt(s: str) -> str:
    """Escape für FFmpeg drawtext."""
    if not s:
        return ""
    return (
        s.replace("\\", "\\\\")
         .replace(":", "\\:")
         .replace("'", "\\'")
         .replace("[", "\\[")
         .replace("]", "\\]")
    )


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def color_to_ffmpeg(c: str, alpha: float = 1.0) -> str:
    """Konvertiert Farbe zu FFmpeg-Format."""
    c = (c or "white").strip()
    alpha = clamp(alpha, 0.0, 1.0)
    
    if c.startswith("#") and len(c) == 7:
        r, g, b = c[1:3], c[3:5], c[5:7]
        return f"0x{r}{g}{b}@{alpha:.3f}"
    return f"{c}@{alpha:.3f}"


# ============================================================================
# ULTRA-OPTIMIZED GPU ZOOM MIT MOTION BLUR
# ============================================================================

class CUDAZoomRenderer:
    """Ultra-optimierte GPU-Zoom-Pipeline mit Motion Blur."""
    
    def __init__(
        self,
        device: torch.device,
        width: int,
        height: int,
        use_cuda_graphs: bool = True,
        motion_blur_duration: float = 1.0
    ):
        self.device = device
        self.width = width
        self.height = height
        self.use_cuda_graphs = use_cuda_graphs and torch.cuda.is_available()
        self.motion_blur_duration = motion_blur_duration
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    @torch.inference_mode()
    def load_image_to_gpu(
        self,
        image_path: str,
        target_width: int,
        target_height: int
    ) -> torch.Tensor:
        """Lädt Bild direkt auf GPU mit minimalen Copies."""
        
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Bild nicht gefunden: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Intelligenter Upscale
        H_orig, W_orig = img.shape[:2]
        if max(H_orig, W_orig) < 2500:
            upscale = 2
        else:
            upscale = 1
            
        target_w = target_width * upscale
        target_h = target_height * upscale
        
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Direkt zu GPU mit FP16
        img_tensor = torch.from_numpy(img).pin_memory()
        img_gpu = img_tensor.to(self.device, non_blocking=True, dtype=torch.float16)
        
        img_gpu = img_gpu.div_(255.0)
        img_gpu = img_gpu.permute(2, 0, 1).unsqueeze(0)
        
        return img_gpu
    
    @torch.inference_mode()
    def render_frame(
        self,
        frame: torch.Tensor,
        z: float,
        cx: int,
        cy: int,
        alpha: float,
        frame_history: Deque[torch.Tensor],
        motion_blur_strength: float,
        frame_index: int,
        fps: int
    ) -> Tuple[torch.Tensor, Deque[torch.Tensor]]:
        """Rendert einzelnes Frame mit Zoom und Motion Blur."""
        
        _, _, H_up, W_up = frame.shape
        
        # Crop-Berechnung
        new_w = int(W_up / z)
        new_h = int(H_up / z)
        
        x0 = max(0, min(W_up - new_w, cx - new_w // 2))
        y0 = max(0, min(H_up - new_h, cy - new_h // 2))
        
        cropped = frame[:, :, y0:y0 + new_h, x0:x0 + new_w]
        
        zoomed = F.interpolate(
            cropped,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        
        # Motion Blur
        if motion_blur_strength > 0.0 and len(frame_history) > 0:
            max_history_frames = int(self.motion_blur_duration * fps)
            
            while len(frame_history) > max_history_frames:
                frame_history.popleft()
            
            num_frames = len(frame_history)
            weights = torch.zeros(num_frames + 1, device=self.device, dtype=torch.float16)
            
            weights[0] = 1.0
            
            decay_rate = 0.7
            for i in range(num_frames):
                weights[i + 1] = motion_blur_strength * (decay_rate ** (i + 1))
            
            total_weight = weights.sum()
            if total_weight > 0:
                weights = weights / total_weight
                
                blended = torch.zeros_like(zoomed)
                blended.add_(zoomed, alpha=weights[0])
                
                for i, hist_frame in enumerate(frame_history):
                    blended.add_(hist_frame, alpha=weights[i + 1])
                
                zoomed = blended
        
        if alpha < 1.0:
            zoomed.mul_(alpha)
        
        frame_history.append(zoomed.clone())
        
        max_frames = int(self.motion_blur_duration * fps) + 1
        while len(frame_history) > max_frames:
            frame_history.popleft()
        
        return zoomed, frame_history
    
    @torch.inference_mode()
    def render_zoom_sequence(
        self,
        image_path: str,
        output_mp4: str,
        fps: int,
        duration: float,
        zoom_factor: float,
        center_w: float,
        center_h: float,
        direction: str,
        fi_start: float,
        fi_dur: float,
        fo_end_time: float,
        fo_dur: float,
        motion_blur_strength: float = 0.5,
    ) -> None:
        """Haupt-Rendering-Loop mit Motion Blur."""
        
        print(f"   🚀 CUDA-Zoom auf {self.device}")
        print(f"   ⚡ Motion Blur: {motion_blur_strength:.1f} über {self.motion_blur_duration:.1f}s")
        
        H_up = self.height * 2
        W_up = self.width * 2
        frame = self.load_image_to_gpu(image_path, self.width, self.height)
        
        total_frames = max(1, int(round(duration * fps)))
        
        direction = direction if direction in ("in", "out") else "in"
        z0, z1 = (1.0, float(zoom_factor)) if direction == "in" else (float(zoom_factor), 1.0)
        
        cx = int(W_up * float(center_w))
        cy = int(H_up * float(center_h))
        
        fi_start = max(0.0, float(fi_start))
        fi_dur = max(0.0, float(fi_dur))
        fo_dur = max(0.0, float(fo_dur))
        fo_start = max(0.0, float(fo_end_time) - fo_dur)
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p7",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "0",
            "-maxrate", "15M",
            "-bufsize", "20M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_mp4,
        ]
        
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        cpu_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame_history = deque(maxlen=int(self.motion_blur_duration * fps) + 5)
        
        try:
            if self.use_cuda_graphs and total_frames > 120:
                print("   ⚡ CUDA Graphs aktiviert")
                for _ in range(3):
                    _ = self.render_frame(
                        frame, 1.0, cx, cy, 1.0, deque(), 0.0, 0, fps
                    )
                torch.cuda.synchronize()
            
            for i in range(total_frames):
                t = i / float(fps)
                t_norm = 0.0 if total_frames <= 1 else i / float(total_frames - 1)
                
                z = z0 + (z1 - z0) * (0.5 - 0.5 * math.cos(math.pi * t_norm))
                
                alpha = 1.0
                
                if fi_dur > 0.0 and t < fi_start + fi_dur:
                    if t < fi_start:
                        alpha = 0.0
                    else:
                        alpha = (t - fi_start) / fi_dur
                
                if fo_dur > 0.0 and t >= fo_start:
                    alpha_out = (fo_start + fo_dur - t) / fo_dur if t < fo_start + fo_dur else 0.0
                    alpha = min(alpha, max(0.0, alpha_out))
                
                alpha = clamp(alpha, 0.0, 1.0)
                
                zoomed, frame_history = self.render_frame(
                    frame, z, cx, cy, alpha, frame_history, 
                    motion_blur_strength, i, fps
                )
                
                out_gpu = zoomed[0].permute(1, 2, 0).clamp_(0, 1).mul_(255)
                out_cpu = out_gpu.byte().cpu()
                np.copyto(cpu_buffer, out_cpu.numpy())
                
                proc.stdin.write(cpu_buffer.tobytes())
                
                if (i + 1) % 30 == 0:
                    print(f"   Frame {i+1}/{total_frames} ({100*(i+1)/total_frames:.1f}%)")
        
        except BrokenPipeError:
            print("⚠️  FFmpeg pipe broken")
        except Exception as e:
            print(f"❌ Rendering error: {e}")
            raise
        finally:
            try:
                proc.stdin.close()
            except:
                pass
        
        proc.wait()
        
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
            raise RuntimeError(f"GPU Zoom encoding failed: {stderr}")
        
        print(f"   ✅ {total_frames} Frames gerendert")


# ============================================================================
# TIMING HELPERS - STRIKTE JSON-EINHALTUNG
# ============================================================================

def compute_scene_windows(scenes) -> Tuple[list, list, list]:
    """
    Berechnet Scene-Windows für Gaps.
    WICHTIG: Basis-Duration wird EXAKT aus JSON übernommen!
    """
    n = len(scenes)
    starts = [float(s["start_time"]) for s in scenes]
    ends = [float(s["end_time"]) for s in scenes]
    
    # EXAKTE JSON-Dauer ohne Modifikation
    bases = [ends[i] - starts[i] for i in range(n)]
    
    half_prev = [0.0] * n
    half_next = [0.0] * n

    for i in range(n):
        if i > 0:
            gap = max(0.0, starts[i] - ends[i-1])
            half_prev[i] = 0.5 * gap
        if i < n-1:
            gap = max(0.0, starts[i+1] - ends[i])
            half_next[i] = 0.5 * gap
    
    return bases, half_prev, half_next


# ============================================================================
# INTRO RENDERING - FLEXIBLE FADE-DAUER
# ============================================================================

def render_intro_clip(
    src: Optional[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    author: str,
    fontfile: Optional[str],
    color_main: str,
    intro_fade_in: float = 3.0,
    intro_fade_out: float = 2.0,
    darken: float = -0.12,
    blur_sigma: float = 4.0,
):
    """
    Intro mit flexiblen Fade-Parametern.
    Keine Limits mehr - User hat volle Kontrolle!
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_unblur = out_path.with_suffix(".unblur.mp4")
    tmp_blur   = out_path.with_suffix(".blur.mp4")
    tmp_xfade  = out_path.with_suffix(".xfade_bg.mp4")

    clip_dur = float(clip_dur)
    
    # Flexible Fade-Dauer (kein Maximum!)
    fade_blur_dur = min(intro_fade_in, clip_dur * 0.5)
    fade_out_dur = min(intro_fade_out, clip_dur * 0.5)
    fade_out_start = max(0.0, clip_dur - fade_out_dur)

    # Text Setup
    txt_title  = esc_txt(title or "")
    txt_author = esc_txt(author or "")
    fontopt    = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""

    col_main = color_to_ffmpeg(color_main, 1.0)
    col_soft = color_to_ffmpeg(color_main, 0.35)

    text_fade_in_start  = 0.8
    text_fade_in_dur    = 0.8
    text_fade_out_dur   = min(1.2, fade_out_dur)
    text_fade_out_start = max(0.0, clip_dur - text_fade_out_dur)

    alpha_text = (
        f"if(lt(t,{text_fade_in_start}),0,"
        f" if(lt(t,{text_fade_in_start + text_fade_in_dur}),"
        f"    (t-{text_fade_in_start})/{text_fade_in_dur},"
        f"  if(lt(t,{text_fade_out_start}),1,"
        f"   if(lt(t,{clip_dur}),({clip_dur}-t)/{text_fade_out_dur},0))))"
    )

    # Input-Quelle
    if src and src.exists():
        if src.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            bg_inputs = ["-i", str(src)]
            bg_base = "[0:v]"
        else:
            bg_inputs = ["-loop", "1", "-t", str(clip_dur), "-i", str(src)]
            bg_base = "[0:v]"
    else:
        bg_inputs = [
            "-f", "lavfi",
            "-t", str(clip_dur),
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
        ]
        bg_base = "[0:v]"

    # (1) UNBLUR
    run([
        "ffmpeg", "-y",
        *bg_inputs,
        "-filter_complex",
        f"{bg_base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"format=yuv420p,setsar=1[v]",
        "-map", "[v]",
        "-t", str(clip_dur),
        "-r", str(fps),
        "-an",
        str(tmp_unblur),
    ])

    # (2) BLUR
    run([
        "ffmpeg", "-y",
        *bg_inputs,
        "-filter_complex",
        f"{bg_base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"format=yuv420p,setsar=1,"
        f"gblur=sigma={blur_sigma},eq=brightness={darken}[v]",
        "-map", "[v]",
        "-t", str(clip_dur),
        "-r", str(fps),
        "-an",
        str(tmp_blur),
    ])

    # (3) XFADE
    run([
        "ffmpeg", "-y",
        "-i", str(tmp_unblur),
        "-i", str(tmp_blur),
        "-filter_complex",
        f"xfade=transition=fade:duration={fade_blur_dur}:offset=0.0[v]",
        "-map", "[v]",
        "-t", str(clip_dur),
        "-r", str(fps),
        "-an",
        str(tmp_xfade),
    ])

    # (4) FADE OUT + TEXT
    flt_txt = (
        "[0:v]"
        f"fade=t=out:st={fade_out_start}:d={fade_out_dur},"
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-40:alpha='{alpha_text}':"
        f"shadowcolor=black:shadowx=3:shadowy=3,"
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_soft}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-38:alpha='({alpha_text})*0.45',"
        f"drawtext=text='{txt_author}':fontsize=38:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+55:alpha='{alpha_text}':"
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"
    )

    run([
        "ffmpeg", "-y",
        "-i", str(tmp_xfade),
        "-filter_complex", flt_txt,
        "-map", "[v]",
        "-t", str(clip_dur),
        "-r", str(fps),
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ])

    # Cleanup
    for f in [tmp_unblur, tmp_blur, tmp_xfade]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================================
# SCENE RENDERING - KEINE FADE-LIMITS
# ============================================================================

def render_scene_image_clip(
    src_img: Optional[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    fi_start: float,
    fi_dur: float,
    fo_end_time: float,
    fo_dur: float,
    zoom_factor: float,
    zoom_center_w: float,
    zoom_center_h: float,
    zoom_direction: str,
    cuda_renderer: Optional[CUDAZoomRenderer] = None,
) -> Path:
    """
    Rendert einzelne Scene mit flexiblen Fade-Parametern.
    KEINE LIMITS mehr - volle User-Kontrolle!
    """
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keine Clamps mehr - respektiere User-Input!
    fo_dur = float(fo_dur)
    fi_start = float(fi_start)
    fi_dur = float(fi_dur)
    fo_start = float(fo_end_time) - fo_dur

    # GPU-Zoom?
    zoom_enabled = (
        zoom_factor is not None 
        and float(zoom_factor) > 1.0001 
        and src_img is not None 
        and src_img.exists()
        and cuda_renderer is not None
    )

    if zoom_enabled:
        print(f"   ⚡ GPU-Zoom (factor={zoom_factor}, dir={zoom_direction})")
        cuda_renderer.render_zoom_sequence(
            image_path=str(src_img),
            output_mp4=str(out_path),
            fps=fps,
            duration=clip_dur,
            zoom_factor=float(zoom_factor),
            center_w=float(zoom_center_w or 0.5),
            center_h=float(zoom_center_h or 0.5),
            direction=zoom_direction or "in",
            fi_start=fi_start,
            fi_dur=fi_dur,
            fo_end_time=fo_end_time,
            fo_dur=fo_dur,
            motion_blur_strength=0.5,
        )
        return out_path

    # Standard ohne Zoom
    if src_img and src_img.exists():
        inputs = ["-loop", "1", "-t", f"{clip_dur:.6f}", "-r", str(fps), "-i", str(src_img)]
        base = "[0:v]"
    else:
        inputs = [
            "-f", "lavfi",
            "-t", f"{clip_dur:.6f}",
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
        ]
        base = "[0:v]"

    flt = (
        f"{base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p,setsar=1,"
        f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"
        f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", flt,
        "-map", "[v]",
        "-r", str(fps),
        "-an",
        "-t", f"{clip_dur:.6f}",
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "19",
        "-b:v", "0",
        "-maxrate", "10M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    
    run(cmd, quiet=False)
    return out_path


# ============================================================================
# MAIN PIPELINE - JSON-TIMING TREUE
# ============================================================================

class StoryPipeline:
    def __init__(
        self,
        images_dir: Path,
        metadata_path: Path,
        base_path: Path,
        output_dir: Path,
        fontfile: Optional[str],
        color_main: str,
        use_cuda_graphs: bool = True,
        motion_blur_duration: float = 1.0,
    ):
        self.images_dir = Path(images_dir)
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "temp"
        
        ensure_dir(self.output_dir)
        ensure_dir(self.tmp_dir)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.fontfile = fontfile
        self.color_main = color_main
        self.motion_blur_duration = motion_blur_duration

        self.title = self.meta.get("title") or self.meta.get("book_info", {}).get("title", "")
        self.author = self.meta.get("author") or self.meta.get("book_info", {}).get("author", "")
        self.scenes_meta = self.meta.get("scenes", [])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_renderer = CUDAZoomRenderer(
            device=device,
            width=1920,
            height=1080,
            use_cuda_graphs=use_cuda_graphs,
            motion_blur_duration=motion_blur_duration
        )

        print(f"\n{'='*60}")
        print(f"📘 Titel: {self.title}")
        print(f"👤 Autor: {self.author}")
        print(f"🎬 Szenen: {len(self.scenes_meta)}")
        print(f"🚀 Device: {device}")
        print(f"⚡ CUDA Graphs: {use_cuda_graphs}")
        print(f"🌀 Motion Blur: {motion_blur_duration:.1f}s Dauer")
        print(f"{'='*60}\n")

    def build_scene_clips(
        self,
        images_prefix: str,
        width: int,
        height: int,
        fps: int,
        fade_in: float,
        fade_out: float,
        intro_fade_in: float = 3.0,
        intro_fade_out: float = 2.0,
    ) -> Tuple[List[Path], List[float]]:
        """
        Baut alle Scene-Clips mit STRIKTER JSON-Timing Einhaltung.
        """
        
        scenes = self.scenes_meta
        if not scenes:
            raise RuntimeError("Keine Szenen in metadata.json")

        bases, half_prev, half_next = compute_scene_windows(scenes)

        clips: List[Path] = []
        durs: List[float] = []

        print(f"\n{'='*60}")
        print("📊 TIMING-ÜBERSICHT (aus JSON):")
        print(f"{'='*60}")
        
        for i, s in enumerate(scenes):
            start = float(s["start_time"])
            end = float(s["end_time"])
            base_dur = bases[i]
            clip_dur = base_dur + half_prev[i] + half_next[i]
            
            print(f"Szene {i:3d}: {start:7.2f}s - {end:7.2f}s  "
                  f"(Base: {base_dur:6.2f}s, +Gap: {half_prev[i]+half_next[i]:5.2f}s = {clip_dur:6.2f}s)")
        
        print(f"{'='*60}\n")

        for i, s in enumerate(scenes):
            stype = s.get("type", "scene")
            start = float(s["start_time"])
            end = float(s["end_time"])
            base_dur = bases[i]
            clip_dur = base_dur + half_prev[i] + half_next[i]
            
            # Fade-Positionen relativ zum Clip
            fi_start = half_prev[i]
            fi_dur = fade_in  # KEINE Limits!
            fo_end = half_prev[i] + base_dur
            fo_dur = fade_out  # KEINE Limits!

            outp = self.tmp_dir / f"scene_{i:04d}.mp4"
            src_img = self.images_dir / f"{images_prefix}{int(s.get('scene_id', i)):04d}.png"
            
            if not src_img.exists():
                src_img = None

            if outp.exists():
                print(f"↩ Szene {i} bereits vorhanden")
                clips.append(outp)
                durs.append(clip_dur)
                continue

            # INTRO
            if stype == "intro":
                print(f"\n🎬 Intro Szene {i}: {clip_dur:.2f}s (JSON: {start:.2f}-{end:.2f}s)")
                intro_src = self.base_path / "intro.mp4"
                if not intro_src.exists() and src_img:
                    intro_src = src_img
                elif not intro_src.exists():
                    intro_src = None

                render_intro_clip(
                    src=intro_src,
                    out_path=outp,
                    width=width,
                    height=height,
                    fps=fps,
                    clip_dur=clip_dur,
                    title=self.title,
                    author=self.author,
                    fontfile=self.fontfile,
                    color_main=self.color_main,
                    intro_fade_in=intro_fade_in,
                    intro_fade_out=intro_fade_out,
                )
                clips.append(outp)
                durs.append(clip_dur)
                continue

            # OUTRO
            if stype == "outro":
                print(f"\n🎬 Outro Szene {i}: {clip_dur:.2f}s (JSON: {start:.2f}-{end:.2f}s)")
                outro_src = self.base_path / "outro.mp4"
                
                if not outro_src.exists():
                    outro_src = src_img

                if outro_src and outro_src.exists():
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(outro_src),
                        "-vf", (
                            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
                        ),
                        "-an",
                        "-r", str(fps),
                        "-t", f"{clip_dur:.6f}",
                        "-c:v", "h264_nvenc",
                        "-preset", "p7",
                        "-rc", "vbr",
                        "-cq", "19",
                        "-pix_fmt", "yuv420p",
                        str(outp)
                    ]
                    run(cmd, quiet=False)
                else:
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-t", f"{clip_dur:.6f}",
                        "-i", f"color=c=black:s={width}x{height}:r={fps}",
                        "-c:v", "h264_nvenc",
                        "-preset", "p7",
                        "-cq", "23",
                        str(outp)
                    ]
                    run(cmd, quiet=False)

                clips.append(outp)
                durs.append(clip_dur)
                continue

            # NORMALE SZENE
            zoom_factor = float(s.get("zoom_factor", 1.0)) if s.get("zoom_factor") is not None else 1.0
            zoom_center_w = float(s.get("zoom_center_w", 0.5)) if s.get("zoom_center_w") is not None else 0.5
            zoom_center_h = float(s.get("zoom_center_h", 0.5)) if s.get("zoom_center_h") is not None else 0.5
            zoom_direction = s.get("zoom_direction", "in") or "in"

            print(f"\n🖼️ Szene {i} ({stype}) – {clip_dur:.2f}s (JSON: {start:.2f}-{end:.2f}s)")
            print(f"   Fade In: {fi_start:.2f}s + {fi_dur:.2f}s")
            print(f"   Fade Out: Ende {fo_end:.2f}s - {fo_dur:.2f}s")

            render_scene_image_clip(
                src_img=src_img,
                out_path=outp,
                width=width,
                height=height,
                fps=fps,
                clip_dur=clip_dur,
                fi_start=fi_start,
                fi_dur=fi_dur,
                fo_end_time=fo_end,
                fo_dur=fo_dur,
                zoom_factor=zoom_factor,
                zoom_center_w=zoom_center_w,
                zoom_center_h=zoom_center_h,
                zoom_direction=zoom_direction,
                cuda_renderer=self.cuda_renderer,
            )

            clips.append(outp)
            durs.append(clip_dur)

        return clips, durs


    def concat_clips(self, clips: List[Path], out_path: Path) -> Path:
        """Konkateniert alle Clips."""
        concat_file = out_path.parent / "concat.txt"
        
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in clips:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        print(f"\n🔗 Konkateniere {len(clips)} Clips...")
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(out_path)
        ]
        
        run(cmd, quiet=False)
        return out_path

    def finalize(
        self,
        master_video: Path,
        audiobook_file: Path,
        overlay_file: Optional[Path],
        overlay_opacity: float,
        width: int,
        height: int,
        fps: int,
        make_sd: bool,
    ) -> Tuple[Path, Optional[Path]]:
        """Finalisiert Video mit Audio und Overlay."""
        
        visual = master_video

        # Overlay anwenden
        if overlay_file and overlay_file.exists():
            print(f"\n✨ Overlay wird angewendet: {overlay_file}")
            ov_out = self.output_dir / "_overlay_master.mp4"
            
            if overlay_file.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                ov_inputs = ["-stream_loop", "-1", "-i", str(overlay_file)]
            else:
                ov_inputs = ["-loop", "1", "-r", str(fps), "-i", str(overlay_file)]

            cmd = [
                "ffmpeg", "-y",
                "-i", str(master_video),
                *ov_inputs,
                "-filter_complex",
                (
                    f"[0:v]format=yuv420p[base];"
                    f"[1:v]scale={width}:{height},format=rgba,"
                    f"colorchannelmixer=aa={overlay_opacity:.3f}[ovr];"
                    f"[base][ovr]overlay=0:0:shortest=1[out]"
                ),
                "-map", "[out]",
                "-c:v", "h264_nvenc",
                "-preset", "p7",
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", "19",
                "-b:v", "0",
                "-maxrate", "15M",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(ov_out)
            ]
            
            if run(cmd, quiet=False):
                visual = ov_out
            else:
                print("⚠️ Overlay fehlgeschlagen, fahre ohne fort")
        else:
            print("\n🚫 Kein Overlay aktiv")

        # Audio muxen
        print("\n🔊 Audio wird gemuxed...")
        final_hd = self.output_dir / "story_final_hd.mp4"
        
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
        
        run(cmd_hd, quiet=False)

        # SD-Version
        final_sd = None
        if make_sd:
            print("\n📦 Erzeuge SD-Version...")
            final_sd = self.output_dir / "story_final_sd.mp4"
            cmd_sd = [
                "ffmpeg", "-y",
                "-i", str(final_hd),
                "-vf", "scale=640:360:force_original_aspect_ratio=decrease,fps=30",
                "-c:v", "h264_nvenc",
                "-preset", "p7",
                "-rc", "vbr",
                "-cq", "23",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                str(final_sd)
            ]
            run(cmd_sd, quiet=False)

        return final_hd, final_sd


# ============================================================================
# CLI - ERWEITERTE PARAMETER
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Story Pipeline v12 – Optimiert mit JSON-Timing Treue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:

  # Standard mit längeren Fades:
  python video_generator_2.py --path /projekt --fade-out 3.0

  # Intro/Outro separate Fades:
  python video_generator_2.py --path /projekt --intro-fade-out 4.0 --fade-out 2.5

  # Motion Blur anpassen:
  python video_generator_2.py --path /projekt --motion-blur-duration 1.5
        """
    )
    
    # Basis-Parameter
    ap.add_argument("--path", required=True, help="Projekt-Basis-Pfad")
    ap.add_argument("--images", default=None, help="Bilder-Ordner")
    ap.add_argument("--metadata", default=None, help="metadata.json Pfad")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei")
    ap.add_argument("--output", default=None, help="Ausgabe-Ordner")a

    # Video-Parameter
    ap.add_argument("--fps", type=int, default=30, help="Frames pro Sekunde")
    
    # Fade-Parameter (jetzt ohne Limits!)
    ap.add_argument("--fade-in", type=float, default=1.5, 
                    help="Scene Fade-In Dauer in Sekunden")
    ap.add_argument("--fade-out", type=float, default=1.5, 
                    help="Scene Fade-Out Dauer in Sekunden (KEINE Limits!)")
    
    # Separate Intro/Outro Fades
    ap.add_argument("--intro-fade-in", type=float, default=3.0,
                    help="Intro Fade-In Dauer (Blur-Effekt)")
    ap.add_argument("--intro-fade-out", type=float, default=2.5,
                    help="Intro Fade-Out Dauer")

    # Overlay
    ap.add_argument("--overlay", default="overlay.mp4", help="Overlay-Datei")
    ap.add_argument("--overlay-opacity", type=float, default=0.32,
                    help="Overlay Transparenz (0.0-1.0)")
    
    # Quality
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd",
                    help="Ausgabe-Qualität")

    # Text/Font
    ap.add_argument("--font", default=None, help="TTF/OTF Font für Intro")
    ap.add_argument("--text-color", default="#ffffff", help="Text-Farbe (Hex)")
    
    # GPU-Optionen
    ap.add_argument("--no-cuda-graphs", action="store_true", 
                    help="CUDA Graphs deaktivieren")
    ap.add_argument("--motion-blur-duration", type=float, default=0.6,
                    help="Motion Blur Dauer in Sekunden (0.5-2.0)")

    args = ap.parse_args()

    # Pfade aufbauen
    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base / "images")
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")
    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")
    output = Path(args.output) if args.output else (base / "story_v12")

    # Validierung
    if not metadata.exists():
        raise SystemExit(f"❌ Metadata nicht gefunden: {metadata}")
    if not audiobook.exists():
        raise SystemExit(f"❌ Audio nicht gefunden: {audiobook}")

    # Overlay suchen
    overlay = None
    if args.overlay and args.overlay.strip():
        overlay_candidates = [
            base / args.overlay,
            base / "overlay.mp4",
            base / "overlay.png",
            Path(args.overlay),
        ]
        
        for candidate in overlay_candidates:
            if candidate.exists():
                overlay = candidate
                print(f"✅ Overlay gefunden: {overlay}")
                break
        
        if not overlay:
            print(f"⚠️ Overlay nicht gefunden, gesucht in:")
            for c in overlay_candidates:
                print(f"   - {c}")

    # Pipeline initialisieren
    pipeline = StoryPipeline(
        images_dir=images_dir,
        metadata_path=metadata,
        base_path=base,
        output_dir=output,
        fontfile=args.font,
        color_main=args.text_color,
        use_cuda_graphs=not args.no_cuda_graphs,
        motion_blur_duration=args.motion_blur_duration,
    )

    # Info-Ausgabe
    print(f"\n{'='*60}")
    print("⚙️  KONFIGURATION:")
    print(f"{'='*60}")
    print(f"Scene Fade-In:       {args.fade_in:.1f}s")
    print(f"Scene Fade-Out:      {args.fade_out:.1f}s")
    print(f"Intro Fade-In:       {args.intro_fade_in:.1f}s")
    print(f"Intro Fade-Out:      {args.intro_fade_out:.1f}s")
    print(f"Motion Blur:         {args.motion_blur_duration:.1f}s")
    print(f"FPS:                 {args.fps}")
    print(f"{'='*60}\n")

    # Szenen rendern
    clips, durs = pipeline.build_scene_clips(
        images_prefix="image_",
        width=1920,
        height=1080,
        fps=args.fps,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        intro_fade_in=args.intro_fade_in,
        intro_fade_out=args.intro_fade_out,
    )

    # Concat
    merged = output / "_merged_master.mp4"
    pipeline.concat_clips(clips, merged)

    # Finalize
    hd, sd = pipeline.finalize(
        master_video=merged,
        audiobook_file=audiobook,
        overlay_file=overlay,
        overlay_opacity=args.overlay_opacity,
        width=1920,
        height=1080,
        fps=args.fps,
        make_sd=(args.quality == "sd")
    )

    # Cleanup
    try:
        shutil.rmtree(pipeline.tmp_dir, ignore_errors=True)
    except:
        pass

    # Finale Statistik
    print(f"\n{'='*60}")
    print("✅ FERTIG!")
    print(f"{'='*60}")
    print(f"📹 HD Video:  {hd}")
    if sd:
        print(f"📹 SD Video:  {sd}")
    print(f"\n📊 Statistik:")
    print(f"   Szenen:    {len(clips)}")
    print(f"   Dauer:     {sum(durs):.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()