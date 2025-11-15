#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Pipeline v9 – Cinematic Edition

- Nutzt:
  * metadata.json (Timings: start_time, end_time, type, scene_id, …)
  * book_scenes.json (Texte: screen_title, screen_text, screen_text_start/stop)
- 3 Schritte:
  1) Szenenclips bauen (mit Gap/2 davor/danach, Fades und Ken-Burns)
  2) Clips concat (ohne Re-Encode)
  3) Overlay + Audio muxen, HD + optional SD ausgeben

Features:
- GPU-Ken-Burns für Bilder (PyTorch + CUDA, Bicubic)
- Intro abdunkeln & gblur, während Titel angezeigt wird
- Szenen:
  * optionaler Screen-Title (am Szenenanfang, 2.5s) mit sanfter Bewegung
  * Screen-Text zwischen screen_text_start/stop
  * während Text: Bild leicht abdunkeln & weichzeichnen
- Cinematic Text Styling:
  * frei wählbare Schrift (TTF/OTF) via --font
  * Textfarbe via --text-color (Hex oder Name)
  * weicher Glow via --text-glow (0..1)
  * optionaler „Vorleuchten“-Effekt (--cinematic-text)
  * Fontgrößen via --title-fontsize, --text-fontsize
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

# ---- optional: torch only if used for GPU Ken-Burns ----
try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.io import read_image
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ---------------- utils ----------------
def run(cmd, quiet: bool = False) -> bool:
    """Run a shell command; print stderr if it fails (unless quiet)."""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        try:
            print(r.stderr.decode("utf-8", "ignore"))
        except Exception:
            print(r.stderr)
    return r.returncode == 0


def has_nvenc() -> bool:
    """Check if ffmpeg has NVENC encoders."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True
        )
        return "h264_nvenc" in r.stdout or "hevc_nvenc" in r.stdout
    except Exception:
        return False


def esc_txt(s: str) -> str:
    """Escape characters that annoy drawtext."""
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


# ------------- timing helpers -------------
def compute_scene_windows(scenes) -> Tuple[list, list, list]:
    """
    Für jede Szene:
    bases[i]     = end - start (Original-Szenenlänge)
    half_prev[i] = 1/2 Gap zur vorherigen Szene (0 bei i==0)
    half_next[i] = 1/2 Gap zur nächsten Szene (0 bei i==last)
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


# ------------- GPU Ken Burns for images -------------
def ken_burns_gpu_image(
    img_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    fi_start: float,
    fi_dur: float,
    fo_end_time: float,
    fo_dur: float,
    zoom_start: float,
    zoom_end: float,
    pan: str = "none",
    ease: str = "linear",
    use_fp16: bool = True,
    nvenc: bool = True
) -> Path:
    """
    Rendert Ken-Burns-Video (aus Einzelbild) auf GPU (PyTorch):
    - Bicubic-Scaling
    - Pan (Richtung)
    - Fade-In & Fade-Out zeitlich exakt zum Clip
    - Encodiert mit ffmpeg (NVENC falls verfügbar).
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch/torchvision nicht verfügbar – installiere torch/torchvision für GPU-Ken-Burns.")

    import tempfile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

    num_frames = max(1, int(round(clip_dur * fps)))
    tmp_dir = Path(tempfile.mkdtemp(prefix="kb_frames_"))
    img = read_image(str(img_path)).to(device=device, dtype=dtype) / 255.0  # [C,H,W]
    C, H, W = img.shape

    def ease_fn(t: float) -> float:
        if ease == "ease_in_out":
            # smootherstep
            return t*t*t*(t*(t*6 - 15) + 10)
        elif ease == "ease_in":
            return t*t
        elif ease == "ease_out":
            return 1 - (1-t)*(1-t)
        return t

    # pan direction vector
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

    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0.0
        et = ease_fn(t)
        scale = zoom_start + (zoom_end - zoom_start) * et

        new_h, new_w = int(H * scale), int(W * scale)
        zimg = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC)

        max_off_x = max(0, new_w - width)
        max_off_y = max(0, new_h - height)

        # Pan: verschiebt den Crop innerhalb des vergrößerten Bildes
        cx = int(max_off_x * 0.5 * (1 + pan_dx * (2 * et - 1)))
        cy = int(max_off_y * 0.5 * (1 + pan_dy * (2 * et - 1)))
        off_x = clamp(cx - width // 2, 0, max_off_x)
        off_y = clamp(cy - height // 2, 0, max_off_y)

        zimg = zimg[:, off_y:off_y+height, off_x:off_x+width]
        if zimg.shape[1] != height or zimg.shape[2] != width:
            zimg = TF.center_crop(zimg, [height, width])

        tt = i / fps
        alpha = 1.0
        if fi_dur > 0 and tt >= fi_start:
            alpha = min(alpha, (tt - fi_start) / fi_dur)
        if fo_dur > 0 and tt >= (fo_end_time - fo_dur):
            alpha = min(alpha, max(0.0, (fo_end_time - tt) / fo_dur))
        alpha = float(clamp(alpha, 0.0, 1.0))

        frame = (zimg.clamp(0, 1) * alpha).to(dtype=torch.float32).cpu()
        TF.to_pil_image(frame).save(tmp_dir / f"f_{i:06d}.png")

    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M"] if nvenc else ["-c:v", "libx264", "-crf", "18", "-preset", "slow"]
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "f_%06d.png"),
        "-pix_fmt", "yuv420p",
        *enc,
        "-r", str(fps),
        str(out_path)
    ]
    run(cmd, quiet=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path


# --------- Intro mit Titel, Abdunkeln & Blur ---------
def render_intro_with_title(
    src: Optional[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    author: str,
    title_in: float = 2.0,
    title_hold: float = 6.0,
    title_out: float = 1.0,
    darken: float = -0.25,
    blur_sigma: float = 8.0,
    nvenc: bool = True,
    fontfile: Optional[str] = None,
    color_main: str = "white"
) -> Path:
    """
    Intro ohne HyperTrail:
    - Quelle (Video/Bild oder schwarz)
    - während der Titel-Phase: abdunkeln + blur
    - Titel/Autor mit Alpha-Curve.
    """
    t_in = title_in
    t_full = title_in + title_hold
    t_out = title_in + title_hold + title_out

    alpha_expr = (
        f"if(lt(t,{t_in}),0,"
        f" if(lt(t,{t_full}),(t-{t_in})/({max(0.0001, title_hold)}),"
        f"  if(lt(t,{t_out}),1-(t-{t_full})/({max(0.0001, title_out)}),0)))"
    )

    # Eingang
    if src and src.exists():
        if src.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            inputs = ["-i", str(src)]
            base = "[0:v]"
        else:
            inputs = ["-loop", "1", "-t", f"{clip_dur:.6f}", "-r", str(fps), "-i", str(src)]
            base = "[0:v]"
    else:
        inputs = ["-f", "lavfi", "-t", f"{clip_dur:.6f}", "-i", f"color=c=black:s={width}x{height}:r={fps}"]
        base = "[0:v]"

    txt_title = esc_txt(title or "")
    txt_author = esc_txt(author or "")
    fontopt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""
    col = color_to_ffmpeg(color_main, 1.0)

    flt = (
        f"{base}scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"
        f"eq=brightness={darken:.3f}:enable='between(t,{t_in},{t_out})',"
        f"gblur=sigma={blur_sigma}:enable='between(t,{t_in},{t_out})'[b];"
        f"[b]drawtext=text='{txt_title}':fontsize=72:fontcolor={col}{fontopt}:"
        f"alpha='{alpha_expr}':x=(w-text_w)/2:y=(h-text_h)/2-40:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        f"drawtext=text='{txt_author}':fontsize=36:fontcolor={col}{fontopt}:"
        f"alpha='{alpha_expr}':x=(w-text_w)/2:y=(h-text_h)/2+60:"
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"
    )

    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"] if nvenc else \
          ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", flt,
        "-map", "[v]",
        "-r", str(fps),
        "-an",
        *enc,
        "-t", f"{clip_dur:.6f}",
        str(out_path)
    ]
    run(cmd, quiet=True)
    return out_path


# --------- Cinematic Szenentext mit Blur/Darken ---------
def render_scene_with_dynamic_text(
    kb_video: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    screen_title: str,
    screen_text: str,
    title_start: float,
    title_duration: float,
    text_start: float,
    text_stop: float,
    darken: float,
    blur_sigma: float,
    nvenc: bool,
    fontfile: Optional[str],
    color_main: str,
    glow_amount: float,
    cinematic_text: bool,
    title_fontsize: int,
    text_fontsize: int
) -> Path:
    """
    Legt einen cineastischen Text-Layer über ein Ken-Burns-Video:
    - Screen-Title: ab title_start, title_duration lang, mit Alpha und leichter Bewegung.
    - Screen-Text: zwischen text_start und text_stop; Bild abdunkeln & blur.
    - Glow + optionales „Vorleuchten“ für Text.
    """
    txt_title = esc_txt(screen_title or "")
    txt_text  = esc_txt(screen_text or "")
    fontopt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""
    glow_amount = clamp(glow_amount, 0.0, 1.0)

    col_main = color_to_ffmpeg(color_main, 1.0)
    col_soft = color_to_ffmpeg(color_main, glow_amount * 0.66)

    # Title-Alpha mit Offset
    t_s = title_start
    t_d = title_duration
    t_in = 0.35
    t_out = 0.45
    t_mid_end = t_s + t_d - t_out

    alpha_title = (
        f"if(lt(t,{t_s}),0,"
        f" if(lt(t,{t_s + t_in}), (t-{t_s})/{t_in},"
        f"  if(lt(t,{t_mid_end}), 1,"
        f"   if(lt(t,{t_s + t_d}), 1-((t-({t_mid_end}))/{t_out}), 0))))"
    )
    # leichte Bewegung nach oben über die Titeldauer
    y_title = f"(h-text_h)/2-60 - 10*clip((t-{t_s})/{max(0.0001, t_d)},0,1)"

    # Text-Alpha in [text_start, text_stop]
    ts = text_start
    te = text_stop
    mid_in = ts + 0.4
    mid_out = max(ts + 0.4, te - 0.4)

    alpha_text = (
        f"if(lt(t,{ts}),0,"
        f" if(lt(t,{mid_in}), (t-{ts})/0.4,"
        f"  if(lt(t,{mid_out}), 1,"
        f"   if(lt(t,{te}), 1-((t-({mid_out}))/0.4), 0))))"
    )
    # Text gleitet leicht nach oben
    y_text = f"(h-text_h)/2+70 - 12*clip((t-{ts})/{max(0.0001, te-ts)},0,1)"

    enable_tx = f"between(t,{ts},{te})"

    # optionales Vorleuchten für cinematic_text
    if cinematic_text:
        pre_start = max(0.0, ts - 0.25)
        pre_mid   = ts
        pre_end   = ts + 0.25
        pre_alpha_text = (
            f"if(lt(t,{pre_start}),0,"
            f" if(lt(t,{pre_mid}), (t-{pre_start})/0.25,"
            f"  if(lt(t,{pre_end}), 1-((t-{pre_mid})/0.25), 0)))"
        )
    else:
        pre_alpha_text = "0"

    flt = (
        f"[0:v]format=yuv420p,"
        f"eq=brightness={darken}:enable='{enable_tx}',"
        f"gblur=sigma={blur_sigma}:enable='{enable_tx}'[b];"
        # Vorleuchten-Schicht für Text
        f"[b]drawtext=text='{txt_text}':fontsize={text_fontsize}:fontcolor={col_soft}{fontopt}:"
        f"alpha='{pre_alpha_text}':x=(w-text_w)/2:y={y_text}:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        # Haupttitel – helle Ebene
        f"drawtext=text='{txt_title}':fontsize={title_fontsize}:fontcolor={col_main}{fontopt}:"
        f"alpha='{alpha_title}':x=(w-text_w)/2:y={y_title}:"
        f"shadowcolor=black:shadowx=2:shadowy=2,"
        # Glow-Layer für Titel
        f"drawtext=text='{txt_title}':fontsize={title_fontsize}:fontcolor={col_soft}{fontopt}:"
        f"alpha='{alpha_title}*{glow_amount}':x=(w-text_w)/2:y={y_title}+1:"
        f"shadowcolor=black:shadowx=0:shadowy=0,"
        # Haupttext – helle Ebene
        f"drawtext=text='{txt_text}':fontsize={text_fontsize}:fontcolor={col_main}{fontopt}:"
        f"alpha='{alpha_text}':x=(w-text_w)/2:y={y_text}:"
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"
    )

    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"] if nvenc else \
          ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(kb_video),
        "-filter_complex", flt,
        "-map", "[v]",
        "-r", str(fps),
        "-an",
        *enc,
        "-t", f"{clip_dur:.6f}",
        str(out_path)
    ]
    run(cmd, quiet=True)
    return out_path


# ------------- Hauptpipeline-Klasse -------------
class StoryV9:
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
        self.tmp_dir = self.output_dir / "temp_v9"
        ensure_dir(self.output_dir)
        ensure_dir(self.tmp_dir)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # book_scenes.json für Screen-Texte
        self.book_scenes = []
        book_json = Path(base_path) / "book_scenes.json"
        if book_json.exists():
            try:
                with open(book_json, "r", encoding="utf-8") as f:
                    self.book_scenes = json.load(f).get("scenes", [])
            except Exception:
                self.book_scenes = []
        else:
            print("ℹ️  book_scenes.json nicht gefunden – Szenentexte werden übersprungen.")

        self.nvenc = has_nvenc()
        print("🎞️ NVENC:", "aktiv" if self.nvenc else "nicht gefunden (Fallback CPU)")

        self.fontfile = fontfile
        self.color_main = color_main
        self.glow_amount = clamp(glow_amount, 0.0, 1.0)
        self.cinematic_text = cinematic_text
        self.title_fontsize = title_fontsize
        self.text_fontsize = text_fontsize

        # Titel/Autor optional aus meta oder book_info
        self.title = self.meta.get("title") or self.meta.get("book_info", {}).get("title", "")
        self.author = self.meta.get("author") or self.meta.get("book_info", {}).get("author", "")

    @staticmethod
    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    def _book_fields_for_index(self, idx: int):
        """Hole Screen-Texte & Timing aus book_scenes für Szene idx."""
        if idx < len(self.book_scenes):
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
        kb_strength: float,
        kb_direction: str,
        kb_ease: str
    ) -> Tuple[List[Path], List[float]]:
        scenes = self.meta.get("scenes", [])
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

            # Fades: in/out relativ zum Clip
            fi_start = half_prev[i]
            fi_dur   = clamp(fade_in, 0.0, clip_dur)
            fo_end   = half_prev[i] + base_dur
            fo_dur   = clamp(fade_out, 0.0, clip_dur)

            outp = self.tmp_dir / f"scene_{i:04d}.mp4"
            src_img = self.images_dir / f"{images_prefix}{int(s.get('scene_id', i)):04d}.png"

            # Screen-Texte aus book_scenes
            b_title, b_text, b_text_start_rel, b_text_stop_rel = self._book_fields_for_index(i)

            # Text-Timings relativ zum Clip:
            # Szene beginnt im Clip bei half_prev[i]
            text_start = half_prev[i] + max(0.0, b_text_start_rel)
            text_stop  = half_prev[i] + (b_text_stop_rel if b_text_stop_rel > 0 else base_dur)
            text_start = clamp(text_start, 0.0, clip_dur)
            text_stop  = clamp(text_stop, 0.0, clip_dur)
            if text_stop < text_start:
                text_stop = text_start

            # Title (screen_title) am Szenenanfang (nicht Clipanfang):
            title_start = half_prev[i]
            title_duration = 2.5

            if stype == "intro":
                # Intro: Titel des Buches / Autors, Blur+Darken
                print(f"🎬 Intro Szene {i}: {clip_dur:.2f}s")
                # ggf. Intro-Quelle (Video/Bild) – optional
                intro_mp4 = self.output_dir.parent / "intro.mp4"
                src = intro_mp4 if intro_mp4.exists() else (src_img if src_img.exists() else None)
                render_intro_with_title(
                    src=src,
                    out_path=outp,
                    width=width,
                    height=height,
                    fps=fps,
                    clip_dur=clip_dur,
                    title=self.title,
                    author=self.author,
                    title_in=2.0,
                    title_hold=max(1.0, base_dur - 3.0),
                    title_out=1.0,
                    darken=-0.25,
                    blur_sigma=8.0,
                    nvenc=self.nvenc,
                    fontfile=self.fontfile,
                    color_main=self.color_main
                )

            elif stype == "outro":
                # Outro: leichtes Ken-Burns auf Bild oder schwarz
                print(f"🏁 Outro Szene {i}: {clip_dur:.2f}s")
                if src_img.exists() and self._is_image(src_img):
                    ken_burns_gpu_image(
                        img_path=src_img,
                        out_path=outp,
                        width=width,
                        height=height,
                        fps=fps,
                        clip_dur=clip_dur,
                        fi_start=fi_start,
                        fi_dur=fi_dur,
                        fo_end_time=fo_end,
                        fo_dur=fo_dur,
                        zoom_start=1.0,
                        zoom_end=1.02,
                        pan="none",
                        ease=kb_ease,
                        use_fp16=True,
                        nvenc=self.nvenc
                    )
                else:
                    fo_start = max(0.0, fo_end - fo_dur)
                    flt = (
                        f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"
                        f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"
                        f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}[v]"
                    )
                    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-
pix_fmt", "yuv420p"] if self.nvenc else \
                          ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-t", f"{clip_dur:.6f}", "-i", f"color=c=black:s={width}x{height}:r={fps}",
                        "-filter_complex", flt,
                        "-map", "[v]",
                        "-r", str(fps),
                        "-an",
                        *enc,
                        "-t", f"{clip_dur:.6f}",
                        str(outp)
                    ]
                    run(cmd, quiet=True)

            else:
                # Normale Szene (Bild)
                if src_img.exists() and self._is_image(src_img):
                    print(f"🖼️  Szene {i} – Bild, Clipdauer {clip_dur:.2f}s, KB+Text")
                    kb_temp = outp.parent / f"_kb_{i:04d}.mp4"
                    z_start = 1.0
                    z_end   = 1.0 + clamp(kb_strength, 0.0, 1.0) * 0.05

                    # 1) Ken-Burns (GPU)
                    ken_burns_gpu_image(
                        img_path=src_img,
                        out_path=kb_temp,
                        width=width,
                        height=height,
                        fps=fps,
                        clip_dur=clip_dur,
                        fi_start=fi_start,
                        fi_dur=fi_dur,
                        fo_end_time=fo_end,
                        fo_dur=fo_dur,
                        zoom_start=z_start,
                        zoom_end=z_end,
                        pan=kb_direction,
                        ease=kb_ease,
                        use_fp16=True,
                        nvenc=self.nvenc
                    )

                    # 2) Optional Texte + Blur/Darken
                    if (b_title and b_title.strip()) or (b_text and b_text.strip()):
                        render_scene_with_dynamic_text(
                            kb_video=kb_temp,
                            out_path=outp,
                            width=width,
                            height=height,
                            fps=fps,
                            clip_dur=clip_dur,
                            screen_title=b_title or "",
                            screen_text=b_text or "",
                            title_start=title_start,
                            title_duration=title_duration,
                            text_start=text_start,
                            text_stop=text_stop,
                            darken=-0.15,
                            blur_sigma=3.5,
                            nvenc=self.nvenc,
                            fontfile=self.fontfile,
                            color_main=self.color_main,
                            glow_amount=self.glow_amount,
                            cinematic_text=self.cinematic_text,
                            title_fontsize=self.title_fontsize,
                            text_fontsize=self.text_fontsize
                        )
                        try:
                            kb_temp.unlink(missing_ok=True)
                        except Exception:
                            pass
                    else:
                        # Keine Screen-Texte -> direkt Ken-Burns-Video übernehmen
                        shutil.move(kb_temp, outp)
                else:
                    # Fallback: schwarzer Hintergrund mit Fades
                    print(f"⚠️  Szene {i}: Bild {src_img.name} nicht gefunden → schwarzer Hintergrund.")
                    fo_start = max(0.0, fo_end - fo_dur)
                    flt = (
                        f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"
                        f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"
                        f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}[v]"
                    )
                    enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"] if self.nvenc else \
                          ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-t", f"{clip_dur:.6f}", "-i", f"color=c=black:s={width}x{height}:r={fps}",
                        "-filter_complex", flt,
                        "-map", "[v]",
                        "-r", str(fps),
                        "-an",
                        *enc,
                        "-t", f"{clip_dur:.6f}",
                        str(outp)
                    ]
                    run(cmd, quiet=True)

            clips.append(outp)
            durs.append(clip_dur)

        return clips, durs

    def step2_concat(self, segs: List[Path], out_path: Path) -> Path:
        concat_file = out_path.parent / "concat_v9.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in segs:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        print(f"🔗 Concat {len(segs)} Segmente … (copy)")
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

        # Optional Overlay über gesamte Länge
        if overlay_file and overlay_file.exists():
            print("✨ Overlay anwenden (volle Länge) …")
            ov_out = self.output_dir / "_overlay_master.mp4"
            if overlay_file.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                ov_inputs = ["-stream_loop", "-1", "-i", str(overlay_file)]
            else:
                ov_inputs = ["-loop", "1", "-r", str(fps), "-i", str(overlay_file)]

            enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M", "-pix_fmt", "yuv420p"] if self.nvenc else \
                  ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]

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
                "-an",
                *enc,
                str(ov_out)
            ]
            run(cmd, quiet=True)
            visual = ov_out

        print("🔊 Audio muxen …")
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
            print("📦 Erzeuge SD-Derivat …")
            final_sd = self.output_dir / "story_final_sd.mp4"
            cmd_sd = [
                "ffmpeg", "-y",
                "-i", str(final_hd),
                "-vf", "scale=640:360:force_original_aspect_ratio=decrease,fps=30",
                "-c:v", "libx264", "-b:v", "600k",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                str(final_sd)
            ]
            run(cmd_sd, quiet=True)

        return final_hd, final_sd


# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(
        description="Story Pipeline v9 – metadata.json + book_scenes.json mit Cinematic-Text-Styling"
    )
    ap.add_argument("--path", required=True, help="Projektbasis (darin liegt book_scenes.json)")
    ap.add_argument("--images", default=None, help="Ordner mit Bildern (default: <path>/images)")
    ap.add_argument("--metadata", default=None, help="Pfad zur metadata.json mit Szenen-Timings")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei (volle Länge, z.B. master.wav)")
    ap.add_argument("--output", default=None, help="Ausgabeordner (default: <path>/story_v9)")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.0)
    ap.add_argument("--fade-out", type=float, default=1.0)

    ap.add_argument("--kb-strength", type=float, default=0.5, help="0..1 → ca. 0..5% Zoom")
    ap.add_argument("--kb-direction", default="none",
                    choices=["none", "left", "right", "up", "down", "diag_tl", "diag_tr", "diag_bl", "diag_br"])
    ap.add_argument("--kb-ease", default="ease_in_out", choices=["linear", "ease_in", "ease_out", "ease_in_out"])

    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild über gesamte Länge")
    ap.add_argument("--overlay-opacity", type=float, default=0.25)
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd", help="Erzeuge SD-Derivat zusätzlich")

    # v9 Styling
    ap.add_argument("--font", default=None, help="Pfad zu TTF/OTF Schrift (optional)")
    ap.add_argument("--text-color", default="#ffffff", help="z.B. '#ffffff' oder 'white'")
    ap.add_argument("--text-glow", type=float, default=0.6, help="0..1 – Intensität der weichen Glühebene")
    ap.add_argument("--cinematic-text", action="store_true",
                    help="Aktiviert Vorleuchten/Soft-Intro für Screentext")
    ap.add_argument("--title-fontsize", type=int, default=70)
    ap.add_argument("--text-fontsize", type=int, default=42)

    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base / "images")
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")
    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")
    output = Path(args.output) if args.output else (base / "story_v9")

    if not metadata.exists():
        raise SystemExit(f"Metadaten nicht gefunden: {metadata}")
    if not audiobook.exists():
        raise SystemExit(f"Audio nicht gefunden: {audiobook}")
    if not images_dir.exists():
        print(f"⚠️  Bildordner {images_dir} existiert nicht – fehlende Szenen werden schwarz gerendert.")

    pipeline = StoryV9(
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
        kb_strength=args.kb_strength,
        kb_direction=args.kb_direction,
        kb_ease=args.kb_ease
    )

    # Schritt 2: Concat
    merged = output / "_merged_master.mp4"
    pipeline.step2_concat(clips, merged)

    # Schritt 3: Overlay + Audio + HD/SD
    overlay = Path(args.overlay) if args.overlay else None
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

    print("✅ Fertig – HD:", hd)
    if sd:
        print("✅ SD:", sd)


if __name__ == "__main__":
    main()
