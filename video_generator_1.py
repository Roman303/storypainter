#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Pipeline v10.5 – Blur/Darken + Text (2-Pass Intro & Szenen, NVENC)

- Keine Torch / kein Zoom.
- Alles über ffmpeg:
  * Intro:
      1) intro_bg.mp4: Hintergrund (intro.mp4 oder Bild) mit Fade-in/out,
         sanftem Blur + Abdunkeln ab t=2s.
      2) intro_text.mov: Titel/Autor als Alphakanal-Video.
      3) Merge per overlay.
  * Szenen:
      1) scene_bg_XXXX.mp4: Bild-Hintergrund mit Fade-in/out,
         sanftem Blur+Dunkel während Text.
      2) scene_txt_XXXX.mov: Titel + Screen-Text mit Alphakanal.
      3) Merge per overlay.

- GPU-Encoding:
  * Szenen/Intro/Overlay: h264_nvenc, -b:v 8M, -preset p5, -pix_fmt yuv420p
  * SD-Derivat: h264_nvenc, -b:v 600k

- Daten:
  * metadata.json (oder audiobook_metadata*.json): Szenen (type:intro/scene/outro, start_time, end_time, scene_id, …)
  * book_scenes.json: screen_title, screen_text, screen_text_start, screen_text_stop
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------- utils ----------------
def run(cmd, quiet: bool = False) -> bool:
    """Run a shell command; print stderr if it fails (unless quiet)."""
    print("\n----- FFmpeg CMD -----")
    print(" ".join(str(c) for c in cmd))
    print("----------------------\n")

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        try:
            print(r.stderr.decode("utf-8", "ignore"))
        except Exception:
            print(r.stderr)
    return r.returncode == 0


def esc_txt(s: str) -> str:
    """Escape characters für drawtext."""
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


# ---------------- Low-Level Renderfunktionen ----------------
def render_background(
    src_img: Optional[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    fade_in_start: float,
    fade_in_dur: float,
    fade_out_start: float,
    fade_out_dur: float,
    blur_start: float,
    blur_dur: float,
    blur_sigma: float,
    dark_value: float,
):
    """
    Hintergrund ohne Text:
    - Input: Bild oder Video (falls src_img mp4/etc.), sonst schwarze Fläche.
    - Fade-in/out.
    - Sanfter Blur+Dunkel-Fade (sigma & brightness animiert).
    """
    blur_start = max(0.0, blur_start)
    blur_dur   = max(0.0001, blur_dur)
    blur_end   = blur_start + blur_dur

    # Expressions für Blur & Dunkel (sanft)
    blur_expr = (
        f"if(lt(t,{blur_start}),0,"
        f" if(lt(t,{blur_end}), (t-{blur_start})/{blur_dur}*{blur_sigma},"
        f"  {blur_sigma}))"
    )
    bright_expr = (
        f"if(lt(t,{blur_start}),0,"
        f" if(lt(t,{blur_end}), (t-{blur_start})/{blur_dur}*{dark_value},"
        f"  {dark_value}))"
    )

    # Input
    if src_img is not None and src_img.exists():
        if src_img.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            in_args = ["-i", str(src_img)]
        else:
            in_args = ["-loop", "1", "-t", f"{clip_dur:.6f}", "-r", str(fps), "-i", str(src_img)]
    else:
        in_args = [
            "-f", "lavfi",
            "-t", f"{clip_dur:.6f}",
            "-i", f"color=c=black:s={width}x{height}:r={fps}"
        ]

    fc = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"format=yuv420p,setsar=1,"
        f"fade=t=in:st={fade_in_start}:d={fade_in_dur},"
        f"fade=t=out:st={fade_out_start}:d={fade_out_dur},"
        f"gblur=sigma='{blur_expr}',"
        f"eq=brightness='{bright_expr}'"
        f"[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        *in_args,
        "-filter_complex", fc,
        "-map", "[v]", "-an",
        "-t", f"{clip_dur:.6f}",
        "-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "8M",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path)
    ]
    run(cmd)


def render_textlayer_intro(
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    author: str,
    fontfile: Optional[str],
    color_main: str
):
    """
    Transparentes Intro-Textvideo (qtrle mit Alpha):
    - ab t=2s wird der Text eingeblendet,
    - Titel: Fade-In 0.8s ab t=2, Fade-Out 1s, endet 2.5s vor Clipende.
    - Autor: Fade-In 1.0s ab t=2.5, gleicher Fade-Out-Bereich.
    """
    title_t = esc_txt(title or "")
    author_t = esc_txt(author or "")
    font_opt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""
    col = color_to_ffmpeg(color_main, 1.0)

    t_in_start = 2.0
    t_in_dur   = 0.8

    a_in_start = 2.5
    a_in_dur   = 1.0

    text_end   = max(0.0, clip_dur - 2.5)
    text_out_dur = 1.0
    text_out_start = max(0.0, text_end - text_out_dur)

    alpha_title = (
        f"if(lt(t,{t_in_start}),0,"
        f" if(lt(t,{t_in_start + t_in_dur}), (t-{t_in_start})/{t_in_dur},"
        f"  if(lt(t,{text_out_start}),1,"
        f"   if(lt(t,{text_end}), ({text_end}-t)/{text_out_dur},"
        f"    0))))"
    )

    alpha_author = (
        f"if(lt(t,{a_in_start}),0,"
        f" if(lt(t,{a_in_start + a_in_dur}), (t-{a_in_start})/{a_in_dur},"
        f"  if(lt(t,{text_out_start}),1,"
        f"   if(lt(t,{text_end}), ({text_end}-t)/{text_out_dur},"
        f"    0))))"
    )

    fc = (
        f"color=c=black@0:s={width}x{height}:r={fps}[base];"
        f"[base]drawtext=text='{title_t}':"
        f"fontsize=70:fontcolor={col}{font_opt}:"
        f"alpha='{alpha_title}':"
        f"x=(w-text_w)/2:y=h/2-80"
        f"[t1];"
        f"[t1]drawtext=text='{author_t}':"
        f"fontsize=42:fontcolor={col}{font_opt}:"
        f"alpha='{alpha_author}':"
        f"x=(w-text_w)/2:y=h/2+20"
        f"[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black@0:s={width}x{height}:r={fps}",
        "-filter_complex", fc,
        "-map", "[v]", "-an",
        "-t", f"{clip_dur:.6f}",
        "-c:v", "qtrle",
        str(out_path)
    ]
    run(cmd)


def render_textlayer_scene(
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    text: str,
    title_start: float,
    text_start: float,
    text_end: float,
    fontfile: Optional[str],
    color_main: str
):
    """
    Transparentes Textvideo für Szenen:
    - Titel: Fade-In 0.8s ab title_start, Fade-Out 1s vor text_end.
    - Text:  Fade-In 0.6s ab text_start, Fade-Out 1s vor text_end.
    """
    title_t = esc_txt(title or "")
    text_t  = esc_txt(text or "")
    font_opt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""
    col = color_to_ffmpeg(color_main, 1.0)

    text_end = clamp(text_end, 0.0, clip_dur)
    out_dur = 1.0
    out_start = max(0.0, text_end - out_dur)

    # Title alpha
    t_fi = 0.8
    alpha_title = (
        f"if(lt(t,{title_start}),0,"
        f" if(lt(t,{title_start + t_fi}), (t-{title_start})/{t_fi},"
        f"  if(lt(t,{out_start}),1,"
        f"   if(lt(t,{text_end}),({text_end}-t)/{out_dur},0))))"
    )

    # Text alpha
    txt_fi = 0.6
    alpha_text = (
        f"if(lt(t,{text_start}),0,"
        f" if(lt(t,{text_start + txt_fi}), (t-{text_start})/{txt_fi},"
        f"  if(lt(t,{out_start}),1,"
        f"   if(lt(t,{text_end}),({text_end}-t)/{out_dur},0))))"
    )

    fc = (
        f"color=c=black@0:s={width}x{height}:r={fps}[base];"
        f"[base]drawtext=text='{title_t}':"
        f"fontsize=70:fontcolor={col}{font_opt}:"
        f"alpha='{alpha_title}':"
        f"x=(w-text_w)/2:y=h/2-80"
        f"[t1];"
        f"[t1]drawtext=text='{text_t}':"
        f"fontsize=42:fontcolor={col}{font_opt}:"
        f"alpha='{alpha_text}':"
        f"x=(w-text_w)/2:y=h/2+30"
        f"[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black@0:s={width}x{height}:r={fps}",
        "-filter_complex", fc,
        "-map", "[v]", "-an",
        "-t", f"{clip_dur:.6f}",
        "-c:v", "qtrle",
        str(out_path)
    ]
    run(cmd)


def merge_layers(bg: Path, text: Path, out_path: Path, fps: int, dur: float):
    """Hintergrund (bg) + Text (mit Alpha) → fertiger Clip."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(bg),
        "-i", str(text),
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-an",
        "-t", f"{dur:.6f}",
        "-r", str(fps),
        "-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "8M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path)
    ]
    run(cmd)


# ---------------- Wrapper für Intro & Szenen ----------------
def render_intro(
    src_img: Optional[Path],
    tmp_dir: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    author: str,
    fontfile: Optional[str],
    color_main: str
):
    """Baut Intro aus Hintergrund + Textlayer."""
    bg = tmp_dir / "intro_bg.mp4"
    txt = tmp_dir / "intro_text.mov"

    # Hintergrund: Blur+Dunkel langsam ab t=2s
    render_background(
        src_img=src_img,
        out_path=bg,
        width=width,
        height=height,
        fps=fps,
        clip_dur=clip_dur,
        fade_in_start=0.0,
        fade_in_dur=1.0,
        fade_out_start=max(0.0, clip_dur - 1.5),
        fade_out_dur=1.5,
        blur_start=2.0,
        blur_dur=2.0,
        blur_sigma=8.0,
        dark_value=-0.20,
    )

    # Textlayer
    render_textlayer_intro(
        out_path=txt,
        width=width,
        height=height,
        fps=fps,
        clip_dur=clip_dur,
        title=title,
        author=author,
        fontfile=fontfile,
        color_main=color_main
    )

    merge_layers(bg, txt, out_path, fps=fps, dur=clip_dur)


def render_scene(
    src_img: Optional[Path],
    tmp_dir: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    fi_start: float,
    fi_dur: float,
    fo_start: float,
    fo_dur: float,
    title: str,
    text: str,
    title_start: float,
    text_start: float,
    text_end: float,
    fontfile: Optional[str],
    color_main: str,
    scene_id: int
):
    """Baut Szenenclip aus Hintergrund (mit Blur+Dunkel) + Textlayer."""
    bg = tmp_dir / f"scene_bg_{scene_id:04d}.mp4"
    txt = tmp_dir / f"scene_txt_{scene_id:04d}.mov"

    # Wann soll Blur/Dunkel langsam hochfahren?
    # Wir starten leicht vor dem Text (0.5s vorher).
    blur_start = max(0.0, text_start - 0.5)
    blur_dur   = 1.2

    render_background(
        src_img=src_img,
        out_path=bg,
        width=width,
        height=height,
        fps=fps,
        clip_dur=clip_dur,
        fade_in_start=fi_start,
        fade_in_dur=fi_dur,
        fade_out_start=fo_start,
        fade_out_dur=fo_dur,
        blur_start=blur_start,
        blur_dur=blur_dur,
        blur_sigma=6.0,
        dark_value=-0.15,
    )

    render_textlayer_scene(
        out_path=txt,
        width=width,
        height=height,
        fps=fps,
        clip_dur=clip_dur,
        title=title,
        text=text,
        title_start=title_start,
        text_start=text_start,
        text_end=text_end,
        fontfile=fontfile,
        color_main=color_main
    )

    merge_layers(bg, txt, out_path, fps=fps, dur=clip_dur)


# ------------- Hauptpipeline-Klasse -------------
class StoryV10_5:
    def __init__(
        self,
        images_dir: Path,
        metadata_path: Path,
        base_path: Path,
        output_dir: Path,
        fontfile: Optional[str],
        color_main: str,
    ):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "temp_v10_5"
        ensure_dir(self.output_dir)
        ensure_dir(self.tmp_dir)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # book_scenes.json (oder _test)
        self.book_scenes = []
        book_json = Path(base_path) / "book_scenes.json"
        if not book_json.exists():
            alt = Path(base_path) / "book_scenes_test.json"
            if alt.exists():
                book_json = alt
        if book_json.exists():
            try:
                with open(book_json, "r", encoding="utf-8") as f:
                    self.book_scenes = json.load(f).get("scenes", [])
            except Exception:
                self.book_scenes = []
        else:
            print("ℹ️ book_scenes.json nicht gefunden – Szenentexte werden übersprungen.")

        self.fontfile = fontfile
        self.color_main = color_main

        # Titel/Autor aus meta oder book_info
        self.title = self.meta.get("title") or self.meta.get("book_info", {}).get("title", "")
        self.author = self.meta.get("author") or self.meta.get("book_info", {}).get("author", "")

        self.scenes_meta = self.meta.get("scenes", [])

        print("📘 Titel:", self.title)
        print("👤 Autor:", self.author)
        print("📼 Szenen:", len(self.scenes_meta))
        print("📝 book_scenes:", len(self.book_scenes))

    @staticmethod
    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    def _book_fields_for_scene(self, meta_index: int, stype: str):
        """
        Zuordnung metadata.scene -> book_scenes:
        - Falls erste Szene 'intro' ist und book_scenes nur echte Szenen enthält:
          -> book_scene_index = meta_index - 1
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
            fo_start = half_prev[i] + base_dur - fade_out
            fo_dur   = clamp(fade_out, 0.0, clip_dur)

            outp = self.tmp_dir / f"scene_{i:04d}.mp4"

            # Bild anhand scene_id
            scene_id = int(s.get("scene_id", i))
            src_img = self.images_dir / f"{images_prefix}{scene_id:04d}.png"
            if not src_img.exists():
                src_img = None

            # INTRO
            if stype == "intro":
                print(f"🎬 Intro {i}: {clip_dur:.2f}s")
                intro_mp4 = base_path / "intro.mp4"
                if intro_mp4.exists():
                    intro_src = intro_mp4
                else:
                    intro_src = src_img  # evtl. image_0001 als Fallback

                render_intro(
                    src_img=intro_src,
                    tmp_dir=self.tmp_dir,
                    out_path=outp,
                    width=width,
                    height=height,
                    fps=fps,
                    clip_dur=clip_dur,
                    title=self.title,
                    author=self.author,
                    fontfile=self.fontfile,
                    color_main=self.color_main
                )
                clips.append(outp)
                durs.append(clip_dur)
                continue

            # Outtro behandeln wie Szene ohne Text
            b_title, b_text, b_txt_start, b_txt_stop = self._book_fields_for_scene(i, stype)

            # Textzeiten im Clip
            title_start = half_prev[i]
            text_start  = half_prev[i] + max(0.0, b_txt_start)
            text_end    = half_prev[i] + (b_txt_stop if b_txt_stop > 0 else base_dur)
            text_start  = clamp(text_start, 0.0, clip_dur)
            text_end    = clamp(text_end, 0.0, clip_dur)
            if text_end < text_start:
                text_end = text_start

            print(f"🎬 Szene {i} ({stype}): {clip_dur:.2f}s")

            if stype == "outro":
                # Outro ohne Text: einfach nur Hintergrund mit Fades, ohne Textlayer
                render_background(
                    src_img=src_img,
                    out_path=outp,
                    width=width,
                    height=height,
                    fps=fps,
                    clip_dur=clip_dur,
                    fade_in_start=fi_start,
                    fade_in_dur=fi_dur,
                    fade_out_start=fo_start,
                    fade_out_dur=fo_dur,
                    blur_start=clip_dur - 2.0,
                    blur_dur=1.5,
                    blur_sigma=4.0,
                    dark_value=-0.10,
                )
            else:
                # Normale Szene mit Textlayer
                render_scene(
                    src_img=src_img,
                    tmp_dir=self.tmp_dir,
                    out_path=outp,
                    width=width,
                    height=height,
                    fps=fps,
                    clip_dur=clip_dur,
                    fi_start=fi_start,
                    fi_dur=fi_dur,
                    fo_start=fo_start,
                    fo_dur=fo_dur,
                    title=b_title or "",
                    text=b_text or "",
                    title_start=title_start,
                    text_start=text_start,
                    text_end=text_end,
                    fontfile=self.fontfile,
                    color_main=self.color_main,
                    scene_id=i
                )

            clips.append(outp)
            durs.append(clip_dur)

        return clips, durs

    def step2_concat(self, segs: List[Path], out_path: Path) -> Path:
        concat_file = out_path.parent / "concat_v10_5.txt"
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

        # Optional Overlay
        if overlay_file and overlay_file.exists():
            print("✨ Overlay anwenden (volle Länge) …")
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
                "-map", "[out]", "-an",
                "-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "8M",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
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
                "-c:v", "h264_nvenc", "-b:v", "600k",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                str(final_sd)
            ]
            run(cmd_sd, quiet=True)

        return final_hd, final_sd


# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(
        description="Story Pipeline v10.5 – Blur/Darken + Text, 2-Pass Intro & Szenen, NVENC"
    )
    ap.add_argument("--path", required=True, help="Projektbasis (darin liegt book_scenes.json)")
    ap.add_argument("--images", default=None, help="Ordner mit Bildern (default: <path>/images)")
    ap.add_argument("--metadata", default=None, help="Pfad zur metadata.json mit Szenen-Timings")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei (volle Länge, z.B. master.wav)")
    ap.add_argument("--output", default=None, help="Ausgabeordner (default: <path>/story_v10_5)")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.0)
    ap.add_argument("--fade-out", type=float, default=1.0)

    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild über gesamte Länge")
    ap.add_argument("--overlay-opacity", type=float, default=0.25)
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd", help="Erzeuge SD-Derivat zusätzlich")

    # Styling
    ap.add_argument("--font", default=None, help="Pfad zu TTF/OTF Schrift (optional)")
    ap.add_argument("--text-color", default="#ffffff", help="z.B. '#ffffff' oder 'white'")

    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base / "images")

    # Metadata: test → fallback
    if args.metadata:
        metadata = Path(args.metadata)
    else:
        m1 = base / "audiobook" / "audiobook_metadata_test.json"
        m2 = base / "audiobook" / "audiobook_metadata.json"
        metadata = m1 if m1.exists() else m2

    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")
    output = Path(args.output) if args.output else (base / "story_v10_5")

    if not metadata.exists():
        raise SystemExit(f"Metadaten nicht gefunden: {metadata}")
    if not audiobook.exists():
        raise SystemExit(f"Audio nicht gefunden: {audiobook}")
    if not images_dir.exists():
        print(f"⚠️ Bildordner {images_dir} existiert nicht – Szenen ohne Bilder werden schwarz gerendert.")

    pipeline = StoryV10_5(
        images_dir=images_dir,
        metadata_path=metadata,
        base_path=base,
        output_dir=output,
        fontfile=args.font,
        color_main=args.text_color,
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
