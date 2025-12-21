#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Pipeline v10 â€“ GPU (NVENC), ohne Zoom, Blur/Darken bei Text

- Keine Torch-/CUDA-PyTorch-AbhÃ¤ngigkeit.
- Keine Ken-Burns/Zoom mehr.
- Nur ffmpeg + h264_nvenc (GPU-Encode).
- Pro Szene:
  * Bild (oder schwarz) wird auf 1920x1080 skaliert.
  * Fade-In/Fade-Out gemÃ¤ÃŸ metadata.json (mit Gap/2 vor/nach Szene).
  * WÃ¤hrend Titel/Text: Bild wird leicht abgedunkelt + weichgezeichnet (Blur).
  * Titel + Screen-Text mit sanfter Alpha-Animation (if/lt) und optional Glow.
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
from typing import List, Tuple, Optional


# ---------------- utils ----------------
def run(cmd, quiet: bool = False) -> bool:
    """Run a shell command; print stderr if it fails (unless quiet)."""
    print("\n----- FFmpeg CMD -----")
    print(" ".join(str(c) for c in cmd))
    print("----------------------")
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        try:
            err = r.stderr.decode("utf-8", "ignore")
        except Exception:
            err = str(r.stderr)
        print("Error :", err)
    elif not quiet:
        try:
            out = r.stderr.decode("utf-8", "ignore")
        except Exception:
            out = str(r.stderr)
        if out.strip():
            print(out)
    return r.returncode == 0

        
def make_gpu_zoom_filter(
    dur: float,
    zoom_factor: float,
    center_w: float,
    center_h: float,
    direction: str,
    width: int,
    height: int,
) -> str:

    if direction not in ("in", "out"):
        direction = "in"

    if direction == "in":
        z0 = 1.0
        z1 = zoom_factor
    else:
        z0 = zoom_factor
        z1 = 1.0

    # Smooth cosine easing
    zoom_expr = f"({z0} + ({z1 - z0})*(0.5 - 0.5*cos(PI*t/{dur})))"

    # Overscaled size
    scaled_w = f"{width}*{zoom_expr}"
    scaled_h = f"{height}*{zoom_expr}"

    # Crop coordinates (CPU crop accepts expressions!)
    crop_x = f"({scaled_w}-{width})*{center_w}"
    crop_y = f"({scaled_h}-{height})*{center_h}"

    return (
        "format=nv12,"
        "hwupload_cuda,"
        f"scale_npp={scaled_w}:{scaled_h},"
        # CPU crop can animate!
        f"crop={width}:{height}:{crop_x}:{crop_y},"
        "format=yuv420p"
    )


def get_video_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        raise RuntimeError(f"Konnte Videolänge nicht ermitteln: {path}")
        

def esc_txt(s: str) -> str:
    """Escape characters fÃ¼r drawtext."""
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
    """
    Accepts '#RRGGBB' or named colors; returns '0xRRGGBB@alpha' or 'name@alpha'
    suitable for drawtext fontcolor.
    """
    c = (c or "white").strip()
    alpha = clamp(alpha, 0.0, 1.0)
    if c.startswith("#") and len(c) == 7:
        r = c[1:3]; g = c[3:5]; b = c[5:7]
        return f"0x{r}{g}{b}@{alpha:.3f}"
    return f"{c}@{alpha:.3f}"


def render_zoom_scene_gpu(
    image_path: str,
    output_mp4: str,
    width: int,
    height: int,
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
    upscale_factor: int = 2,
    motion_blur_strength: float = 0.95
) -> None:
    """
    GPU-Ken-Burns:
    - Smooth Cosine-Zoom (in/out)
    - Motion-Blur
    - Fades exakt wie frÃ¼her (fi_start/fi_dur, fo_end_time/fo_dur) in Sekunden
    - LÃ¤uft komplett auf der GPU, Encoding via h264_nvenc
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Bild laden ---
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Bild nicht gefunden: {image_path}")
    H_orig, W_orig = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max(H_orig, W_orig) < 2400:
        upscale_factor = 2
    else:
        upscale_factor = 1

    # Hochskalieren fÃ¼r schÃ¶ne Details
    H_up = height * upscale_factor
    W_up = width * upscale_factor
    img = cv2.resize(img, (W_up, H_up), interpolation=cv2.INTER_CUBIC)        

    frame = torch.from_numpy(img).half().to(device) / 255.0
    frame = frame.permute(2, 0, 1).unsqueeze(0)               # [1,3,H,W]

    total_frames = max(1, int(round(duration * fps)))

    # Zoom-Parameter
    if direction not in ("in", "out"):
        direction = "in"

    if direction == "in":
        z0 = 1.0
        z1 = float(zoom_factor)
    else:
        z0 = float(zoom_factor)
        z1 = 1.0

    cx = int(W_up * float(center_w))
    cy = int(H_up * float(center_h))

    # Fades
    fi_start = max(0.0, float(fi_start))
    fi_dur   = max(0.0, float(fi_dur))
    fo_dur   = max(0.0, float(fo_dur))
    fo_start = max(0.0, float(fo_end_time) - fo_dur)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p3",
        "-rc:v", "vbr",
        "-cq:v", "20",
        "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    prev = None

    for i in range(total_frames):
        # Zeit in Sekunden
        t = i / float(fps)
        # Normalisierte Zeit fÃ¼r Zoom-Easing
        t_norm = 0.0 if total_frames <= 1 else i / float(total_frames - 1)

        # Cosine-Ease Zoom
        z = z0 + (z1 - z0) * (0.5 - 0.5 * math.cos(math.pi * t_norm))

        new_w = int(W_up / z)
        new_h = int(H_up / z)

        # Crop-Koordinaten, Center beachten, innerhalb des Bildes clampen
        x0 = max(0, min(W_up - new_w, cx - new_w // 2))
        y0 = max(0, min(H_up - new_h, cy - new_h // 2))

        cropped = frame[:, :, y0:y0 + new_h, x0:x0 + new_w]

        zoomed = F.interpolate(
            cropped,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).half()

        # Motion-Blur: einfacher temporal blend
        if prev is not None and motion_blur_strength > 0.0:
            s = float(motion_blur_strength)
            zoomed = zoomed * (1.0 - s) + prev * s
        prev = zoomed.clone().half()

        # Fade-Alpha (wie ffmpeg fade Filter, aber in PyTorch)
        alpha = 1.0

        # Fade-In
        if fi_dur > 0.0:
            if t < fi_start:
                alpha = 0.0
            elif t < fi_start + fi_dur:
                alpha = (t - fi_start) / fi_dur
            # danach bleibt alpha erstmal 1.0

        # Fade-Out
        if fo_dur > 0.0:
            if t >= fo_start:
                if t < fo_start + fo_dur:
                    alpha_out = (fo_start + fo_dur - t) / fo_dur
                else:
                    alpha_out = 0.0
                alpha = min(alpha, alpha_out)

        alpha = max(0.0, min(1.0, alpha))
        zoomed = zoomed * alpha

        out = (zoomed[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        try:
            proc.stdin.write(out.tobytes())
        except BrokenPipeError:
            break

    try:
        proc.stdin.close()
    except Exception:
        pass

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg GPU zoom encoding failed â€“ MP4 ist kaputt!")

# ------------- timing helpers -------------
def compute_scene_windows(scenes) -> Tuple[list, list, list]:
    """
    FÃ¼r jede Szene:
    bases[i]     = end - start (Original-SzenenlÃ¤nge)
    half_prev[i] = 1/2 Gap zur vorherigen Szene (0 bei i==0)
    half_next[i] = 1/2 Gap zur nÃ¤chsten Szene (0 bei i==last)
    """
    n = len(scenes)
    starts = [float(s["start_time"]) for s in scenes]
    ends   = [float(s["end_time"])   for s in scenes]
    bases  = [max(0.0, ends[i] - starts[i]) for i in range(n)]
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

def render_text_png(text, out_png, width=1920, height=1080, fontsize=72):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
    draw.text(
        ((width - text_w)//2, (height - text_h)//2),
        text,
        font=font,
        fill=(255,255,255,255)
    )

    img.save(out_png)


# --------- Intro mit Titel, weicher Text & (step) Blur/Darken ---------
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
    darken: float = -0.08,
    blur_sigma: float = 4.0,
):

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_unblur = out_path.with_suffix(".unblur.mp4")
    tmp_blur   = out_path.with_suffix(".blur.mp4")
    tmp_xfade  = out_path.with_suffix(".xfade_bg.mp4")

    # -----------------------------
    # Timing
    # -----------------------------
    clip_dur = float(clip_dur)
    fade_in_dur    = 0.3
    fade_out_dur   = 1.2
    fade_out_start = max(0.0, clip_dur - fade_out_dur)
    
    fade_blur_dur  = min(2.0, clip_dur * 0.4)

    # -----------------------------
    # TEXT SETUP (aus deinem Block)
    # -----------------------------
    txt_title  = esc_txt(title or "")
    txt_author = esc_txt(author or "")
    fontopt    = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""

    col_main = color_to_ffmpeg(color_main, 1.0)
    col_soft = color_to_ffmpeg(color_main, 0.35)

    text_fade_in_start  = 0.8
    text_fade_in_dur    = 0.8
    text_fade_out_dur   = 1.0
    text_fade_out_start = max(0.0, clip_dur -  1.2)

    alpha_text = (
        f"if(lt(t,{text_fade_in_start}),0,"
        f" if(lt(t,{text_fade_in_start + text_fade_in_dur}),"
        f"    (t-{text_fade_in_start})/{text_fade_in_dur},"
        f"  if(lt(t,{text_fade_out_start}),1,"
        f"   if(lt(t,{clip_dur}),({clip_dur}-t)/{text_fade_out_dur},0))))"
    )

    # ----------------- Input-Quelle -----------------
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

    # -----------------------------
    # (1) UNBLUR
    # -----------------------------
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

    # -----------------------------
    # (2) BLUR
    # -----------------------------
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

    # -----------------------------
    # (3) XFADE scharf â†’ blur
    # -----------------------------
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

    # -----------------------------
    # (4) FADE OUT + TEXT OVERLAY
    # -----------------------------
    flt_txt = (
        "[0:v]"
        f"fade=t=in:st=0:d={fade_in_dur},"
        f"fade=t=out:st={fade_out_start}:d={fade_out_dur},"
        # Titel
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-40:alpha='{alpha_text}':"
        f"shadowcolor=black:shadowx=3:shadowy=3,"
        # Glow
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_soft}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-38:alpha='({alpha_text})*0.45',"
        # Author
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

def render_intro_clip_with_cinematic_text(
    src: Optional[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    subtitle: str,
    fontfile: Optional[str],
    color_main: str,
    darken: float = -0.18,
    blur_sigma: float = 6.0,
):
    """
    Cinematic Intro:
    - Minimaler Zoom-In (1.00 â†’ 1.04)
    - Weicher Blur + Darken
    - GroÃŸer Kino-Titel + Untertitel
    - Saubere Fades
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    txt_title    = esc_txt(title or "")
    txt_subtitle = esc_txt(subtitle or "")
    fontopt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""

    col_main = color_to_ffmpeg(color_main, 1.0)
    col_soft = color_to_ffmpeg(color_main, 0.35)

    # --- Timing ---
    text_fade_in  = 0.8
    text_fade_out = 1.5
    zoom_end      = 1.04

    text_out_start = max(0.0, clip_dur - text_fade_out)

    alpha_text = (
        f"if(lt(t,{text_fade_in}),0,"
        f" if(lt(t,{text_fade_in+0.8}),(t-{text_fade_in})/0.8,"
        f"  if(lt(t,{text_out_start}),1,"
        f"   if(lt(t,{clip_dur}),({clip_dur}-t)/{text_fade_out},0))))"
    )

    # ----------------- Input -----------------
    if src and src.exists():
        if src.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            inputs = ["-i", str(src)]
            base = "[0:v]"
        else:
            inputs = ["-loop", "1", "-t", str(clip_dur), "-i", str(src)]
            base = "[0:v]"
    else:
        inputs = ["-f", "lavfi", "-t", str(clip_dur),
                  "-i", f"color=c=black:s={width}x{height}:r={fps}"]
        base = "[0:v]"

    # ----------------- Filtergraph -----------------
    flt = (
        f"{base}"
        # âœ… Minimaler Zoom-In (cine feel)
        f"scale={width}:{height},"
        f"zoompan=z='1+0.04*t/{clip_dur}':d=1:s={width}x{height},"
        f"format=yuv420p,setsar=1,"
        # âœ… Blur + Darken
        f"gblur=sigma={blur_sigma},"
        f"eq=brightness={darken},"
        # âœ… TITEL (Haupt)
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-40:alpha='{alpha_text}':"
        f"shadowcolor=black:shadowx=3:shadowy=3,"
        # âœ… TITEL-GLOW
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_soft}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-38:alpha='({alpha_text})*0.4',"
        # âœ… UNTERTITEL
        f"drawtext=text='{txt_subtitle}':fontsize=36:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+50:alpha='{alpha_text}'"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", flt,
        "-t", str(clip_dur),
        "-r", str(fps),
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p3",
        "-rc:v", "vbr",
        "-cq:v", "20",
        "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path)
    ]
    run(cmd)


# --------- Szenen: Bild + Blur/Darken + Text ---------
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
    # Text & Timing (werden aktuell ignoriert)
    screen_title: str,
    screen_text: str,
    title_start: float,
    title_duration: float,
    text_start: float,
    text_stop: float,
    # Blur & darken (optional, auch ohne Text nutzbar)
    darken: float,
    blur_sigma: float,
    fontfile: Optional[str],
    color_main: str,
    glow_amount: float,
    cinematic_text: bool,
    title_fontsize: int,
    text_fontsize: int,
    # Zoom
    zoom_factor: float,
    zoom_center_w: float,
    zoom_center_h: float,
    zoom_direction: str,
) -> Path:
    """
    Bild (oder schwarz) â†’ 1920x1080,
    - saubere Fades (fi_start/fi_dur, fo_end_time/fo_dur)
    - optionaler GPU-Ken-Burns-Zoom
    - KEIN Text-Overlay (Titel/Screen-Text werden ignoriert)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip_dur = float(clip_dur)

    # Fade-Parameter normalisieren
    fo_dur = max(0.0, float(fo_dur))
    fi_start = max(0.0, float(fi_start))
    fi_dur   = max(0.0, float(fi_dur))
    fo_start = max(0.0, float(fo_end_time) - fo_dur)

    # --- ZOOM-PFAD (GPU) ---
    zoom_enabled = zoom_factor is not None and float(zoom_factor) > 1.0001 and src_img is not None

    if zoom_enabled:
        print(f"   â†’ GPU-Zoom (factor={zoom_factor}, dir={zoom_direction})")
        render_zoom_scene_gpu(
            image_path=str(src_img),
            output_mp4=str(out_path),
            width=width,
            height=height,
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
            upscale_factor=2,
            motion_blur_strength=0.3,
        )
        return out_path

    # --- FFMPEG-PFAD (kein Zoom) ---
    # Bildquelle
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

    # Basis-Filter: Scale + Pad + SAR
    flt = (
        f"{base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p,setsar=1,"
        f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"
        f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}"
    )

    # Optional Darken/Blur global
    if darken != 0.0 or blur_sigma > 0.0:
        flt += f",eq=brightness={darken:.3f},gblur=sigma={blur_sigma}"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", flt,
        "-map", "[v]",
        "-r", str(fps),
        "-an",
        "-t", f"{clip_dur:.6f}",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    run(cmd, quiet=False)
    return out_path




# ------------- Hauptpipeline-Klasse -------------
class StoryV10:
    def __init__(
        self,
        images_dir: Path,
        metadata_path: Path,
        base_path: Path,
        output_dir: Path,
        fontfile: Optional[str],
        color_main: str,
        glow_amount: float,
        cinematic_text: bool,
        title_fontsize: int,
        text_fontsize: int
    ):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "temp_v10"
        ensure_dir(self.output_dir)
        ensure_dir(self.tmp_dir)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # book_scenes.json fÃ¼r Screen-Texte
        self.book_scenes = []
        book_json = Path(base_path) / "book_scenes.json"
        if book_json.exists():
            try:
                with open(book_json, "r", encoding="utf-8") as f:
                    self.book_scenes = json.load(f).get("scenes", [])
            except Exception:
                self.book_scenes = []
        else:
            print("â„¹ï¸  book_scenes.json nicht gefunden â€“ Szenentexte werden Ã¼bersprungen.")

        self.fontfile = fontfile
        self.color_main = color_main
        self.glow_amount = clamp(glow_amount, 0.0, 1.0)
        self.cinematic_text = cinematic_text
        self.title_fontsize = title_fontsize
        self.text_fontsize = text_fontsize

        # Titel/Autor optional aus meta oder book_info
        self.title = self.meta.get("title") or self.meta.get("book_info", {}).get("title", "")
        self.author = self.meta.get("author") or self.meta.get("book_info", {}).get("author", "")

        self.scenes_meta = self.meta.get("scenes", [])

        print("ðŸ“˜ Titel:", self.title)
        print("ðŸ‘¤ Autor:", self.author)
        print("ðŸ“¼ Szenen:", len(self.scenes_meta))
        print("ðŸ“ book_scenes:", len(self.book_scenes))

    @staticmethod
    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    def _book_fields_for_scene(self, meta_index: int, stype: str):
        """
        Zuordnung metadata.scene -> book_scenes:
        - Wenn erste Szene in metadata 'intro' ist und book_scenes nur die eigentlichen Szenen enthÃ¤lt:
          -> book_scene_index = meta_index - 1
        - Intro kriegt keine book_scenes-Texte.
        """
        if stype == "intro":
            return ("", "", 0.0, 0.0)

        if not self.book_scenes:
            return ("", "", 0.0, 0.0)

        scenes_meta = self.scenes_meta
        offset = 0
        if scenes_meta and scenes_meta[0].get("type") == "intro" and len(self.book_scenes) == max(0, len(scenes_meta)-1):
            offset = -1

        idx = meta_index + offset
        if 0 <= idx < len(self.book_scenes):
            bs = self.book_scenes[idx] or {}
            return (
                bs.get("screen_title", ""),
                bs.get("screen_text", ""),
                float(bs.get("screen_text_start", 0.0)),
                float(bs.get("screen_text_stop", 0.0)),
            )
        return ("", "", 0.0, 0.0)

    def step1_build_scene_clips(
        self,
        images_prefix: str,
        width: int,
        height: int,
        fps: int,
        fade_in: float,
        fade_out: float,
        base_path: Path
    ) -> Tuple[List[Path], List[float]]:
        scenes = self.scenes_meta
        if not scenes:
            raise RuntimeError("Keine Szenen im metadata.json.")

        bases, half_prev, half_next = compute_scene_windows(scenes)

        clips: List[Path] = []
        durs: List[float] = []

        for i, s in enumerate(scenes):

           
            stype = s.get("type", "scene")
            start = float(s["start_time"])
            end   = float(s["end_time"])
            base_dur = max(0.0, end - start)
            clip_dur = base_dur + half_prev[i] + half_next[i]
            
            fi_start = half_prev[i]
            fi_dur   = clamp(fade_in, 0.0, clip_dur)
            fo_end   = half_prev[i] + base_dur
            fo_dur   = clamp(fade_out, 0.0, clip_dur)

            outp = self.tmp_dir / f"scene_{i:04d}.mp4"
            src_img = self.images_dir / f"{images_prefix}{int(s.get('scene_id', i)):04d}.png"
            if not src_img.exists():
                src_img = None

            # ------------------------------------------------------
            # NEU: Szene nur rendern, wenn Temp-File noch nicht existiert
            # ------------------------------------------------------
            if outp.exists():
                print(f"â© Szene {i} bereits gerendert, verwende vorhandene Datei.")
                clips.append(outp)
                durs.append(clip_dur)
                continue

            # Intro
            if stype == "intro":
                print(f"ðŸŽ¬ Intro Szene {i}: {clip_dur:.2f}s")
                intro_src: Optional[Path] = None
                intro_mp4 = base_path / "intro.mp4"
                if intro_mp4.exists():
                    intro_src = intro_mp4
                elif src_img is not None:
                    intro_src = src_img
                else:
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
                    darken=-0.08,
                    blur_sigma=5.0
                )
                clips.append(outp)
                durs.append(clip_dur)
                continue
            # ---------------------------------------------
            # OUTRO â€“ einfach abspielen, unverÃ¤ndert
            # ---------------------------------------------
            if stype == "outro":
                print(f"ðŸŽ¬ Outro Szene {i}: {clip_dur:.2f}s")

                # Quelle suchen
                outro_mp4 = base_path / "outro.mp4"
                if outro_mp4.exists():
                    outro_src = outro_mp4
                    print("   â†’ outro.mp4 wird verwendet.")
                else:
                    # Fallback: Bild der Szene
                    outro_src = src_img
                    print("   â†’ kein outro.mp4 â€“ fallback auf Szenenbild.")

                # ffmpeg: nur skalieren + padding, keine Effekte
                if outro_src:
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
                        "-preset", "p3",
                        "-rc:v", "vbr",
                        "-cq:v", "20",
                        "-b:v", "8M",
                        "-pix_fmt", "yuv420p",
                        str(outp)
                    ]
                    run(cmd, quiet=False)
                else:
                    # schwarz rendern, falls gar nichts existiert
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-t", f"{clip_dur:.6f}",
                        "-i", f"color=c=black:s={width}x{height}:r={fps}",
                        "-c:v", "h264_nvenc",
                        "-preset", "p5",
                        "-b:v", "5M",
                        str(outp)
                    ]
                    run(cmd, quiet=False)

                clips.append(outp)
                durs.append(clip_dur)
                continue

            # Normale Szenen
            b_title, b_text, b_text_start_rel, b_text_stop_rel = self._book_fields_for_scene(i, stype)

            text_start = half_prev[i] + max(0.0, b_text_start_rel)
            text_stop  = half_prev[i] + (b_text_stop_rel if b_text_stop_rel > 0 else base_dur)
            text_start = clamp(text_start, 0.0, clip_dur)
            text_stop  = clamp(text_stop, 0.0, clip_dur)
            if text_stop < text_start:
                text_stop = text_start

            title_start = half_prev[i]
            title_duration = 2.5
            # -----------------------------
            # âœ… ZOOM-PARAMETER AUS metadata.json
            # -----------------------------
            zoom_factor = float(s.get("zoom_factor", 1.0)) if s.get("zoom_factor") is not None else 1.0
            zoom_center_w = float(s.get("zoom_center_w", 0.5)) if s.get("zoom_center_w") is not None else 0.5
            zoom_center_h = float(s.get("zoom_center_h", 0.5)) if s.get("zoom_center_h") is not None else 0.5
            zoom_direction = s.get("zoom_direction", "in") or "in"
            print(f"ðŸ–¼ï¸ Szene {i} ({stype}) â€“ {clip_dur:.2f}s")

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
                screen_title=b_title or "",
                screen_text=b_text or "",
                title_start=title_start,
                title_duration=title_duration,
                text_start=text_start,
                text_stop=text_stop,
                darken=-0.15,
                blur_sigma=3.5,
                fontfile=self.fontfile,
                color_main=self.color_main,
                glow_amount=self.glow_amount,
                cinematic_text=self.cinematic_text,
                title_fontsize=self.title_fontsize,
                text_fontsize=self.text_fontsize,
                zoom_factor=zoom_factor,
                zoom_center_w=zoom_center_w,
                zoom_center_h=zoom_center_h,
                zoom_direction=zoom_direction
            )

            clips.append(outp)
            durs.append(clip_dur)

        return clips, durs

    def step2_concat(self, segs: List[Path], out_path: Path) -> Path:
        concat_file = out_path.parent / "concat_v10.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in segs:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        print(f"ðŸ”— Concat {len(segs)} Segmente â€¦ (copy)")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(out_path)
        ]
        run(cmd, quiet=False)
        return out_path

    def step3_finalize(
        self,
        master_video: Path,
        audiobook_file: Path,
        overlay_file: Optional[Path],
        overlay_opacity: float,
        width: int,
        height: int,
        fps: int,
        make_sd: bool
    ) -> Tuple[Path, Optional[Path]]:
        visual = master_video

        # Optional Overlay über gesamte Länge (FAST: Precompute + Mix)
        if overlay_file and overlay_file.exists():
            print("✨ Overlay anwenden (FAST) …")
        
            ov_out = self.output_dir / "_overlay_master.mp4"
            overlay_cache = self.output_dir / "_overlay_cache.mp4"
        
            master_dur = get_video_duration(master_video)
            print(f"   → Master-Länge: {master_dur:.3f}s")
        
            is_video = overlay_file.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
        
            # Cache neu bauen wenn nicht da / zu klein / Overlay neuer als Cache
            cache_needs_build = (
                (not overlay_cache.exists()) or
                (overlay_cache.stat().st_size < 1024 * 1024) or
                (overlay_file.stat().st_mtime > overlay_cache.stat().st_mtime)
            )
        
            if cache_needs_build:
                print("⚡ Precompute Overlay (1080p, fps match, leicht decodierbar) …")
        
                if is_video:
                    ov_inputs = ["-stream_loop", "-1", "-i", str(overlay_file)]
                else:
                    ov_inputs = ["-loop", "1", "-i", str(overlay_file)]
        
                cmd_pre = [
                    "ffmpeg", "-y",
                    "-hide_banner", "-loglevel", "error",
                    *ov_inputs,
                    "-t", f"{master_dur:.6f}",
                    "-vf",
                        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                        f"fps={fps},format=yuv420p",
                    "-an",
                    "-c:v", "h264_nvenc",
                    "-preset", "p3",
                    "-rc:v", "vbr",
                    "-cq:v", "20",
                    "-b:v", "6M",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(overlay_cache)
                ]
                run(cmd_pre, quiet=True)
        
            # Mix (schnell, weil Overlay jetzt leicht ist)
            print("✨ Overlay anwenden (mix) …")
            cmd = [
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", str(master_video),
                "-i", str(overlay_cache),
                "-filter_complex",
                (
                    f"[0:v]format=yuv420p[base];"
                    f"[1:v]format=rgba,colorchannelmixer=aa={overlay_opacity:.3f}[ovr];"
                    f"[base][ovr]overlay=0:0:shortest=1[out]"
                ),
                "-map", "[out]",
                "-map", "0:a?",
                "-c:v", "h264_nvenc",
                "-preset", "p3",
                "-rc:v", "vbr",
                "-cq:v", "20",
                "-b:v", "8M",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-shortest",
                str(ov_out)
            ]
            run(cmd, quiet=True)
        
            visual = ov_out
        



        print("ðŸ”Š Audio muxen â€¦")
        final_hd = self.output_dir / "story_final_hd.mp4"
        cmd_hd = [
            "ffmpeg", "-y",
            "-fflags", "+genpts",
            "-i", str(visual),
            "-i", str(audiobook_file),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            str(final_hd)
        ]
        run(cmd_hd, quiet=True)

        final_sd = None
        if make_sd:
            print("ðŸ“¦ Erzeuge SD-Derivat (GPU) â€¦")
            final_sd = self.output_dir / "story_final_sd.mp4"
            cmd_sd = [
                "ffmpeg", "-y",
                "-i", str(final_hd),
                "-vf", "scale=640:360:force_original_aspect_ratio=decrease,fps=30",
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-b:v", "1.5M",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                str(final_sd)
            ]
            run(cmd_sd, quiet=True)

        return final_hd, final_sd


# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(
        description="Story Pipeline v10 â€“ metadata.json + book_scenes.json, Blur/Darken bei Text (ohne Zoom, GPU only)"
    )
    ap.add_argument("--path", required=True, help="Projektbasis (darin liegt book_scenes.json)")
    ap.add_argument("--images", default=None, help="Ordner mit Bildern (default: <path>/images)")
    ap.add_argument("--metadata", default=None, help="Pfad zur metadata.json mit Szenen-Timings")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei (volle LÃ¤nge, z.B. master.wav)")
    ap.add_argument("--output", default=None, help="Ausgabeordner (default: <path>/story_v10)")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.0)
    ap.add_argument("--fade-out", type=float, default=1.0)

    ap.add_argument("--overlay", default="overlay1.mp4", help="Overlay-Video/Bild Ã¼ber gesamte LÃ¤nge")
    ap.add_argument("--overlay-opacity", type=float, default=0.25)
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd", help="Erzeuge SD-Derivat zusÃ¤tzlich")

    # Styling
    ap.add_argument("--font", default=None, help="Pfad zu TTF/OTF Schrift (optional)")
    ap.add_argument("--text-color", default="#ffffff", help="z.B. '#ffffff' oder 'white'")
    ap.add_argument("--text-glow", type=float, default=0.6, help="0..1 â€“ IntensitÃ¤t der weichen GlÃ¼hebene")
    ap.add_argument("--cinematic-text", action="store_true",
                    help="Aktiviert Vorleuchten/Soft-Intro fÃ¼r Screentext")
    ap.add_argument("--title-fontsize", type=int, default=70)
    ap.add_argument("--text-fontsize", type=int, default=42)

    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base / "images")
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata_small.json")
    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")
    output = Path(args.output) if args.output else (base / "story_v10")
    overlay_path = Path(args.overlay) if args.overlay else None


    if not metadata.exists():
        raise SystemExit(f"Metadaten nicht gefunden: {metadata}")
    if not audiobook.exists():
        raise SystemExit(f"Audio nicht gefunden: {audiobook}")
    if not images_dir.exists():
        print(f"âš ï¸  Bildordner {images_dir} existiert nicht â€“ fehlende Szenen werden schwarz gerendert.")

    pipeline = StoryV10(
        images_dir=images_dir,
        metadata_path=metadata,
        base_path=base,
        output_dir=output,
        fontfile=args.font,
        color_main=args.text_color,
        glow_amount=args.text_glow,
        cinematic_text=args.cinematic_text,
        title_fontsize=args.title_fontsize,
        text_fontsize=args.text_fontsize
    )

    # Schritt 1: Szenenclips
    clips, durs = pipeline.step1_build_scene_clips(
        images_prefix="image_",
        width=1920,
        height=1080,
        fps=args.fps,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        base_path=base
    )

    # Schritt 2: Concat
    merged = output / "_merged_master.mp4"
    pipeline.step2_concat(clips, merged)

    # ---------------------------------
    # Overlay korrekt & robust auflösen
    # ---------------------------------
    overlay = None
    
    if args.overlay and args.overlay.strip():
        candidate = Path(args.overlay)
    
        search_paths = [
            candidate,                                  # exakt so wie angegeben
            base / candidate,                            # ✅ relativ zu --path (Projektbasis)
            Path.cwd() / candidate,                      # relativ zum aktuellen Ordner
            Path(__file__).resolve().parent / candidate  # relativ zum Script
        ]
            
        for p in search_paths:
            if p.exists():
                overlay = p
                break
    
        if overlay:
            print(f"🎬 Overlay aktiv: {overlay}")
        else:
            print(f"⚠️ Overlay-Datei nicht gefunden: {args.overlay} – Overlay wird deaktiviert.")
    else:
        print("🎬 Kein Overlay aktiv")


    hd, sd = pipeline.step3_finalize(
        master_video=merged,
        audiobook_file=audiobook,
        overlay_file=overlay,
        overlay_opacity=args.overlay_opacity,
        width=1920,
        height=1080,
        fps=args.fps,
        make_sd=(args.quality == "sd")
    )

    # Temp cleanup
    try:
        shutil.rmtree(pipeline.tmp_dir, ignore_errors=True)
    except Exception:
        pass

    print("âœ… Fertig â€“ HD:", hd)
    if sd:
        print("âœ… SD:", sd)


if __name__ == "__main__":
    main()