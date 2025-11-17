#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Pipeline v10 – Cinematic Edition (ohne Zoom, VAST-freundlich)

- Kein Ken-Burns/Zoom, keine Torch-/CUDA-Abhängigkeit.
- Nur ffmpeg/libx264 (CPU, preset=veryfast).
- Pro Szene:
  * Bild (oder schwarz) wird auf 1920x1080 skaliert.
  * Fade-In/Fade-Out gemäß metadata.json (mit Gap/2 vor/nach Szene).
  * Während Titel/Text: Bild wird leicht abgedunkelt + weichgezeichnet (Blur).
  * Titel + Screen-Text mit sanfter Alpha-Animation & leichter Bewegung, optional Glow/Vorleuchten.

Datenquellen:
- metadata.json: Szenen-Timings (start_time, end_time, type, scene_id, …)
- book_scenes.json: Texte (screen_title, screen_text, screen_text_start/stop)
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
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        try:
            print(r.stderr.decode("utf-8", "ignore"))
        except Exception:
            print(r.stderr)
    return r.returncode == 0


def esc_txt(s: str) -> str:
    """Escape characters, damit drawtext nicht ausrastet."""
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


# --------- Intro mit Titel, Abdunkeln & Blur (ohne Zoom) ---------
def render_intro_clip(
    src_img: Optional[Path],
    out_path: Path,
    width: int,
    height: int,
    fps: int,
    clip_dur: float,
    title: str,
    author: str,
    fontfile: Optional[str],
    color_main: str,
    darken: float = -0.25,
    blur_sigma: float = 8.0
):
    """
    Rendert ein statisches Intro:
    - Bild oder Video als Hintergrund
    - Abdunkeln + Blur über gesamte Clipdauer
    - Titel + Autor mittig gesetzt
    - Kein Zoom, keine Bewegung
    """

    # Input-Quelle definieren
    if src_img is not None and src_img.exists():
        if src_img.suffix.lower() in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
            inputs = ["-i", str(src_img)]
            base = "[0:v]"
        else:
            inputs = [
                "-loop", "1",
                "-t", f"{clip_dur:.6f}",
                "-r", str(fps),
                "-i", str(src_img)
            ]
            base = "[0:v]"
    else:
        # Schwarzfallback
        inputs = [
            "-f", "lavfi",
            "-t", f"{clip_dur:.6f}",
            "-i", f"color=c=black:s={width}x{height}:r={fps}"
        ]
        base = "[0:v]"

    txt_title = esc_txt(title or "")
    txt_author = esc_txt(author or "")
    font_opt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""
    col = color_to_ffmpeg(color_main, 1.0)

    # Falls beide leer → nur BG rendern, kein drawtext
    has_title = bool(txt_title.strip())
    has_author = bool(txt_author.strip())

    # Basis-Kette (ohne Text)
    base_chain = (
        f"{base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,"
        f"eq=brightness={darken},"
        f"gblur=sigma={blur_sigma}"
    )

    if not (has_title or has_author):
        flt = base_chain + "[v]"
    else:
        # Text-Layer
        text_chain = "[b]"
        if has_title:
            text_chain += (
                f"drawtext=text='{txt_title}':fontsize=72:fontcolor={col}{font_opt}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2-60:"
                f"shadowcolor=black:shadowx=2:shadowy=2,"
            )
        if has_author:
            text_chain += (
                f"drawtext=text='{txt_author}':fontsize=40:fontcolor={col}{font_opt}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2+40:"
                f"shadowcolor=black:shadowx=2:shadowy=2,"
            )
        # letztes Komma weg & [v] anhängen
        if text_chain.endswith(","):
            text_chain = text_chain[:-1]
        text_chain += "[v]"
        flt = base_chain + "[b];" + text_chain

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", flt,
        "-map", "[v]",
        "-r", str(fps),
        "-t", f"{clip_dur:.6f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-movflags", "+faststart",
        str(out_path)
    ]
    run(cmd, quiet=False)


# --------- Szenentext mit Blur/Darken (ohne Zoom) ---------
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
    # Text & Timing
    screen_title: str,
    screen_text: str,
    title_start: float,
    title_duration: float,
    text_start: float,
    text_stop: float,
    # Blur & darken
    darken: float,
    blur_sigma: float,
    fontfile: Optional[str],
    color_main: str,
    glow_amount: float,
    cinematic_text: bool,
    title_fontsize: int,
    text_fontsize: int
) -> Path:
    """
    Bild (oder schwarz) → 1920x1080, Fade-In/Out, Blur+Darken während Text/Titel,
    dazu cineastische Text-Overlays.
    """
    txt_title = esc_txt(screen_title or "")
    txt_text  = esc_txt(screen_text or "")
    has_title = bool(txt_title.strip())
    has_text  = bool(txt_text.strip())

    fontopt = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""
    glow_amount = clamp(glow_amount, 0.0, 1.0)

    col_main = color_to_ffmpeg(color_main, 1.0)
    col_soft = color_to_ffmpeg(color_main, glow_amount * 0.66)

    fo_dur = max(0.0, fo_dur)
    fo_start = max(0.0, fo_end_time - fo_dur)
    fi_start = max(0.0, fi_start)
    fi_dur   = max(0.0, fi_dur)

    # Title-Alpha
    t_s = title_start
    t_d = max(0.1, title_duration)
    t_in = min(0.5, t_d / 3.0)
    t_out = min(0.5, t_d / 3.0)
    t_mid_end = t_s + t_d - t_out

    alpha_title = (
        f"if(lt(t,{t_s}),0,"
        f" if(lt(t,{t_s + t_in}), (t-{t_s})/{t_in},"
        f"  if(lt(t,{t_mid_end}), 1,"
        f"   if(lt(t,{t_s + t_d}), 1-((t-({t_mid_end}))/{t_out}), 0))))"
    )
    y_title = f"(h-text_h)/2-60 - 10*clip((t-{t_s})/{max(0.0001,t_d)},0,1)"

    # Text-Alpha
    ts = text_start
    te = text_stop
    if te < ts:
        te = ts
    mid_in = ts + 0.4
    mid_out = max(ts + 0.4, te - 0.4)

    alpha_text = (
        f"if(lt(t,{ts}),0,"
        f" if(lt(t,{mid_in}), (t-{ts})/0.4,"
        f"  if(lt(t,{mid_out}), 1,"
        f"   if(lt(t,{te}), 1-((t-({mid_out}))/0.4), 0))))"
    )
    y_text = f"(h-text_h)/2+70 - 12*clip((t-{ts})/{max(0.0001, te-ts)},0,1)"

    # Blur/Darken-Fenster: sobald irgendein Text/Titel aktiv ist
    if has_title or has_text:
        bg_start_candidates = []
        bg_end_candidates = []
        if has_title:
            bg_start_candidates.append(t_s)
            bg_end_candidates.append(t_s + t_d)
        if has_text:
            bg_start_candidates.append(ts)
            bg_end_candidates.append(te)
        bg_start = clamp(min(bg_start_candidates), 0.0, clip_dur)
        bg_end   = clamp(max(bg_end_candidates), 0.0, clip_dur)
    else:
        bg_start, bg_end = 0.0, 0.0  # kein Effekt

    enable_bg = f"between(t,{bg_start},{bg_end})"

    # optionales „Vorleuchten“ für Screentext
    if cinematic_text and has_text:
        pre_start = max(0.0, ts - 0.25)
        pre_mid   = ts
        pre_end   = ts + 0.25
        pre_alpha_text = (
            f"if(lt(t,{pre_start}),0,"
            f" if(lt(t,{pre_mid}), (t-{pre_start})/0.25,"
            f"  if(lt(t,{pre_end}), 1-((t-{pre_mid})/0.25), 0)))"
        )
    else:
        pre_alpha_text = None  # kein Vorleuchten

    # Bildquelle
    if src_img and src_img.exists():
        inputs = ["-loop", "1", "-t", f"{clip_dur:.6f}", "-r", str(fps), "-i", str(src_img)]
        base = "[0:v]"
    else:
        inputs = ["-f", "lavfi", "-t", f"{clip_dur:.6f}", "-i",
                  f"color=c=black:s={width}x{height}:r={fps}"]
        base = "[0:v]"

    # 1) Basis-Bildkette
    chain = (
        f"{base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"
        f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"
        f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}"
    )

    if has_title or has_text:
        # Blur und Darken nur wenn Text/Titel existiert
        chain += (
            f",eq=brightness={darken:.3f}:enable='between(t,{bg_start},{bg_end})'"
            f",gblur=sigma={blur_sigma}:enable='between(t,{bg_start},{bg_end})'[b]"
        )
    else:
        chain += "[v]"  # kein weiterer Text-Filter

    # 2) Text-Kette (nur wenn nötig)
    if has_title or has_text:
        text_chain = "[b]"
        # Vorleuchten (Screentext)
        if has_text and pre_alpha_text is not None:
            text_chain += (
                f"drawtext=text='{txt_text}':fontsize={text_fontsize}:fontcolor={col_soft}{fontopt}:"
                f"alpha={pre_alpha_text}:x=(w-text_w)/2:y={y_text}:"
                f"shadowcolor=black:shadowx=2:shadowy=2,"
            )
        # Haupttitel
        if has_title:
            text_chain += (
                f"drawtext=text='{txt_title}':fontsize={title_fontsize}:fontcolor={col_main}{fontopt}:"
                f"alpha={alpha_title}:x=(w-text_w)/2:y={y_title}:"
                f"shadowcolor=black:shadowx=2:shadowy=2,"
            )
            # Glow-Titel
            if glow_amount > 0.0:
                text_chain += (
                    f"drawtext=text='{txt_title}':fontsize={title_fontsize}:fontcolor={col_soft}{fontopt}:"
                    f"alpha={alpha_title}*{glow_amount}:x=(w-text_w)/2:y={y_title}+1:"
                    f"shadowcolor=black:shadowx=0:shadowy=0,"
                )
        # Haupttext
        if has_text:
            text_chain += (
                f"drawtext=text='{txt_text}':fontsize={text_fontsize}:fontcolor={col_main}{fontopt}:"
                f"alpha={alpha_text}:x=(w-text_w)/2:y={y_text}:"
                f"shadowcolor=black:shadowx=2:shadowy=2,"
            )

        # letztes Komma entfernen & [v] anhängen
        if text_chain.endswith(","):
            text_chain = text_chain[:-1]
        text_chain += "[v]"

        flt = chain + ";" + text_chain
    else:
        flt = chain  # already ends with [v]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", flt,
        "-map", "[v]",
        "-r", str(fps),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-movflags", "+faststart",
        "-t", f"{clip_dur:.6f}",
        str(out_path)
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
        - Wenn erste Szene in metadata 'intro' ist und book_scenes nur die eigentlichen Szenen enthält:
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

            # Intro-Szene
            if stype == "intro":
                print(f"🎬 Intro Szene {i}: {clip_dur:.2f}s")

                # Introquelle bestimmen:
                intro_src: Optional[Path] = None
                intro_mp4 = base_path / "intro.mp4"
                if intro_mp4.exists():
                    intro_src = intro_mp4
                elif src_img is not None and src_img.exists():
                    intro_src = src_img
                else:
                    intro_src = None

                render_intro_clip(
                    src_img=intro_src,
                    out_path=outp,
                    width=width,
                    height=height,
                    fps=fps,
                    clip_dur=clip_dur,
                    title=self.title,
                    author=self.author,
                    fontfile=self.fontfile,
                    color_main=self.color_main,
                    darken=-0.25,
                    blur_sigma=8.0
                )

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

            print(f"🖼️ Szene {i} ({stype}) – {clip_dur:.2f}s")

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
                text_fontsize=self.text_fontsize
            )

            clips.append(outp)
            durs.append(clip_dur)

        return clips, durs

    def step2_concat(self, segs: List[Path], out_path: Path) -> Path:
        concat_file = out_path.parent / "concat_v10.txt"
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
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-movflags", "+faststart",
                str(ov_out)
            ]
            run(cmd, quiet=False)
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
        run(cmd_hd, quiet=False)

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
            run(cmd_sd, quiet=False)

        return final_hd, final_sd


# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(
        description="Story Pipeline v10 – metadata.json + book_scenes.json, Blur/Darken bei Text (ohne Zoom)"
    )
    ap.add_argument("--path", required=True, help="Projektbasis (darin liegt book_scenes.json)")
    ap.add_argument("--images", default=None, help="Ordner mit Bildern (default: <path>/images)")
    ap.add_argument("--metadata", default=None, help="Pfad zur metadata.json mit Szenen-Timings")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei (volle Länge, z.B. master.wav)")
    ap.add_argument("--output", default=None, help="Ausgabeordner (default: <path>/story_v10)")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.0)
    ap.add_argument("--fade-out", type=float, default=1.0)

    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild über gesamte Länge")
    ap.add_argument("--overlay-opacity", type=float, default=0.25)
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd", help="Erzeuge SD-Derivat zusätzlich")

    # Styling
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
    output = Path(args.output) if args.output else (base / "story_v10")

    if not metadata.exists():
        raise SystemExit(f"Metadaten nicht gefunden: {metadata}")
    if not audiobook.exists():
        raise SystemExit(f"Audio nicht gefunden: {audiobook}")
    if not images_dir.exists():
        print(f"⚠️  Bildordner {images_dir} existiert nicht – fehlende Szenen werden schwarz gerendert.")

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
