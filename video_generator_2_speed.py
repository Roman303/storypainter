#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Pipeline v12 – MAXIMUM SPEED EDITION
- KEIN CUDA Zoom (deaktiviert)
- KEINE CUDA Graphs (Overkill)
- KEIN Motion Blur (fps-Killer)
- DIREKTER FFmpeg-Stream (kein Python-Rendering)
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import math
from pathlib import Path
from typing import List, Tuple, Optional


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
# TIMING HELPERS - STRIKTE JSON-EINHALTUNG
# ============================================================================

def compute_scene_windows(scenes) -> Tuple[list, list, list]:
    """
    Berechnet Scene-Windows für Gaps.
    """
    n = len(scenes)
    starts = [float(s["start_time"]) for s in scenes]
    ends = [float(s["end_time"]) for s in scenes]
    
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
# INTRO RENDERING - OPTIMIERT
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
):
    """
    Intro mit flexiblen Fade-Parametern - ALLES IN EINEM FFMPEG DURCHLAUF!
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_dur = float(clip_dur)
    
    # Flexible Fade-Dauer
    fade_in_dur = min(intro_fade_in, clip_dur * 0.5)
    fade_out_dur = min(intro_fade_out, clip_dur * 0.5)
    fade_out_start = max(0.0, clip_dur - fade_out_dur)

    # Text Setup
    txt_title  = esc_txt(title or "")
    txt_author = esc_txt(author or "")
    fontopt    = f":fontfile='{esc_txt(fontfile)}'" if fontfile else ""

    col_main = color_to_ffmpeg(color_main, 1.0)
    col_soft = color_to_ffmpeg(color_main, 0.35)

    # Input-Quelle
    if src and src.exists():
        if src.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            # Video als Input
            cmd_base = ["ffmpeg", "-y", "-i", str(src)]
            filter_start = "[0:v]"
        else:
            # Bild als Input
            cmd_base = ["ffmpeg", "-y", "-loop", "1", "-t", str(clip_dur), "-i", str(src)]
            filter_start = "[0:v]"
    else:
        # Fallback: schwarzer Hintergrund
        cmd_base = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-t", str(clip_dur),
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
        ]
        filter_start = "[0:v]"

    # ALLES IN EINEM FILTERGRAPH - KEINE ZWISCHENDATEIEN!
    filter_complex = (
        f"{filter_start}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p,setsar=1,"
        f"fade=t=in:st=0:d={fade_in_dur},"
        f"fade=t=out:st={fade_out_start}:d={fade_out_dur},"
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-90:"
        f"shadowcolor=black:shadowx=3:shadowy=3,"
        f"drawtext=text='{txt_title}':fontsize=78:fontcolor={col_soft}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-88,"
        f"drawtext=text='{txt_author}':fontsize=38:fontcolor={col_main}{fontopt}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+5:"
        f"shadowcolor=black:shadowx=2:shadowy=2"
        f"[v]"
    )

    cmd = [
        *cmd_base,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-t", str(clip_dur),
        "-r", str(fps),
        "-an",
        "-c:v", "h264_nvenc",
        "-preset", "p4",  # Schneller als p5/p7
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "21",  # Etwas niedrigere Qualität = schneller
        "-b:v", "0",
        "-maxrate", "12M",
        "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    
    run(cmd, quiet=False)


# ============================================================================
# SCENE RENDERING - KEIN ZOOM, KEIN CUDA, MAXIMALE GESCHWINDIGKEIT
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
    zoom_factor: float = 1.0,  # Wird ignoriert!
    zoom_center_w: float = 0.5,
    zoom_center_h: float = 0.5,
    zoom_direction: str = "in",
    cuda_renderer=None,  # Wird ignoriert!
) -> Path:
    """
    Rendert einzelne Scene - EXTREM SCHNELL (kein Zoom, nur FFmpeg)
    """
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fade-Parameter
    fo_dur = float(fo_dur)
    fi_start = float(fi_start)
    fi_dur = float(fi_dur)
    fo_start = float(fo_end_time) - fo_dur

    # Bild vorhanden?
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

    # EIN FILTER - KEINE KOMPLEXEN OPERATIONEN!
    flt = (
        f"{base}"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p,setsar=1,"
        f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"
        f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}"
        f"[v]"
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
        "-preset", "p4",  # Schneller!
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "21",  # Etwas niedrigere Qualität = schneller
        "-b:v", "0",
        "-maxrate", "12M",
        "-bufsize", "16M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    
    run(cmd, quiet=False)
    return out_path


# ============================================================================
# MAIN PIPELINE - MAXIMALE GESCHWINDIGKEIT
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
        # CUDA-Parameter entfernt/ignoriert!
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

        self.title = self.meta.get("title") or self.meta.get("book_info", {}).get("title", "")
        self.author = self.meta.get("author") or self.meta.get("book_info", {}).get("author", "")
        self.scenes_meta = self.meta.get("scenes", [])
        
        # Track intro/outro timing
        self.intro_end_time = 0.0
        self.outro_start_time = float('inf')
        for s in self.scenes_meta:
            stype = s.get("type", "scene")
            if stype == "intro":
                self.intro_end_time = max(self.intro_end_time, float(s.get("end_time", 0)))
            elif stype == "outro":
                self.outro_start_time = min(self.outro_start_time, float(s.get("start_time", float('inf'))))

        print(f"\n{'='*60}")
        print(f"📘 Titel: {self.title}")
        print(f"👤 Autor: {self.author}")
        print(f"🎬 Szenen: {len(self.scenes_meta)}")
        print(f"🚀 Modus: MAXIMUM SPEED (KEIN Zoom, KEIN CUDA)")
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
        Baut alle Scene-Clips mit MAXIMALER GESCHWINDIGKEIT.
        """
        
        scenes = self.scenes_meta
        if not scenes:
            raise RuntimeError("Keine Szenen in metadata.json")

        bases, half_prev, half_next = compute_scene_windows(scenes)

        clips: List[Path] = []
        durs: List[float] = []

        print(f"\n{'='*60}")
        print("📊 TIMING-ÜBERSICHT:")
   
        
        for i, s in enumerate(scenes):
            start = float(s["start_time"])
            end = float(s["end_time"])
            base_dur = bases[i]
            clip_dur = base_dur + half_prev[i] + half_next[i]
            
            print(f"Szene {i:3d}: {start:7.2f}s - {end:7.2f}s  "
                  f"(Dauer: {clip_dur:6.2f}s)")
        
        print(f"{'='*60}\n")

        for i, s in enumerate(scenes):
            stype = s.get("type", "scene")
            start = float(s["start_time"])
            end = float(s["end_time"])
            base_dur = bases[i]
            clip_dur = base_dur + half_prev[i] + half_next[i]
            
            # Fade-Positionen relativ zum Clip
            fi_start = half_prev[i]
            fi_dur = fade_in
            fo_end = half_prev[i] + base_dur
            fo_dur = fade_out

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
                print(f"\n🎬 Intro Szene {i}: {clip_dur:.2f}s")
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
                print(f"\n🎬 Outro Szene {i}: {clip_dur:.2f}s")
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
                        "-preset", "p4",
                        "-rc", "vbr",
                        "-cq", "21",
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
                        "-preset", "p4",
                        "-cq", "23",
                        str(outp)
                    ]
                    run(cmd, quiet=False)

                clips.append(outp)
                durs.append(clip_dur)
                continue

            # NORMALE SZENE - ZOOM WIRD IGNORIERT!
            print(f"\n🖼️ Szene {i} ({stype}) – {clip_dur:.2f}s")
            print(f"   Fade In: {fi_start:.2f}s + {fi_dur:.2f}s")
            print(f"   Fade Out: Ende {fo_end:.2f}s - {fo_dur:.2f}s")
            print(f"   ⚡ Zoom: DEAKTIVIERT (maximale Geschwindigkeit)")

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
                zoom_factor=1.0,  # Zoom immer aus!
            )

            clips.append(outp)
            durs.append(clip_dur)

        return clips, durs


    def concat_clips(self, clips: List[Path], out_path: Path) -> Path:
        """
        Konkateniert alle Clips - EXTREM SCHNELL (keine Transitions).
        """
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
            "-c", "copy",  # Direktes Kopieren = sofort!
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
        """Finalisiert Video mit Audio."""
        
        visual = master_video

        # Overlay anwenden (nur wenn wirklich nötig)
        if overlay_file and overlay_file.exists():
            print(f"\n✨ Overlay wird angewendet...")
            
            ov_out = self.output_dir / "_overlay_master.mp4"
            
            if overlay_file.suffix.lower() in {".mp4", ".mov", ".mkv"}:
                ov_inputs = ["-stream_loop", "-1", "-i", str(overlay_file)]
            else:
                ov_inputs = ["-loop", "1", "-r", str(fps), "-i", str(overlay_file)]

            # Einfaches Overlay - kein komplexer Fade wenn nicht nötig
            if self.intro_end_time > 0 or self.outro_start_time < float('inf'):
                # Mit Intro/Outro Erkennung
                fade_duration = 1.0
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(master_video),
                    *ov_inputs,
                    "-filter_complex",
                    (
                        f"[0:v]format=yuv420p[base];"
                        f"[1:v]scale={width}:{height},format=rgba,"
                        f"colorchannelmixer=aa={overlay_opacity:.3f},"
                        f"fade=t=in:st={self.intro_end_time:.3f}:d={fade_duration},"
                        f"fade=t=out:st={max(0, self.outro_start_time - fade_duration):.3f}:d={fade_duration}[ovr];"
                        f"[base][ovr]overlay=0:0:shortest=1[out]"
                    ),
                    "-map", "[out]",
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",
                    "-cq", "21",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(ov_out)
                ]
            else:
                # Ohne Intro/Outro - einfaches Overlay
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
                    "-preset", "p4",
                    "-cq", "21",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(ov_out)
                ]
            
            if run(cmd, quiet=False):
                visual = ov_out
            else:
                print("⚠️ Overlay fehlgeschlagen, fahre ohne fort")

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

        # SD-Version (optional)
        final_sd = None
        if make_sd:
            print("\n📦 Erzeuge SD-Version...")
            final_sd = self.output_dir / "story_final_sd.mp4"
            cmd_sd = [
                "ffmpeg", "-y",
                "-i", str(final_hd),
                "-vf", "scale=640:360:force_original_aspect_ratio=decrease,fps=30",
                "-c:v", "h264_nvenc",
                "-preset", "p4",
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
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Story Pipeline v12 – MAXIMUM SPEED EDITION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Basis-Parameter
    ap.add_argument("--path", required=True, help="Projekt-Basis-Pfad")
    ap.add_argument("--images", default=None, help="Bilder-Ordner")
    ap.add_argument("--metadata", default=None, help="metadata.json Pfad")
    ap.add_argument("--audiobook", default=None, help="Audio-Datei")
    ap.add_argument("--output", default=None, help="Ausgabe-Ordner")

    # Video-Parameter
    ap.add_argument("--fps", type=int, default=30, help="Frames pro Sekunde")
    
    # Fade-Parameter
    ap.add_argument("--fade-in", type=float, default=1.5, help="Scene Fade-In Dauer")
    ap.add_argument("--fade-out", type=float, default=1.5, help="Scene Fade-Out Dauer")
    ap.add_argument("--intro-fade-in", type=float, default=3.0, help="Intro Fade-In")
    ap.add_argument("--intro-fade-out", type=float, default=2.5, help="Intro Fade-Out")

    # Overlay
    ap.add_argument("--overlay", default="overlay.mp4", help="Overlay-Datei")
    ap.add_argument("--overlay-opacity", type=float, default=0.32, help="Overlay Transparenz")
    
    # Quality
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd", help="Ausgabe-Qualität")

    # Text/Font
    ap.add_argument("--font", default=None, help="TTF/OTF Font für Intro")
    ap.add_argument("--text-color", default="#ffffff", help="Text-Farbe (Hex)")

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

    # Pipeline initialisieren (KEINE CUDA-Parameter mehr!)
    pipeline = StoryPipeline(
        images_dir=images_dir,
        metadata_path=metadata,
        base_path=base,
        output_dir=output,
        fontfile=args.font,
        color_main=args.text_color,
    )

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

    # Concat (keine Transitions!)
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

    print(f"\n{'='*60}")
    print("✅ FERTIG!")
    print(f"{'='*60}")
    print(f"📹 HD Video:  {hd}")
    if sd:
        print(f"📹 SD Video:  {sd}")
    print(f"📊 Dauer:     {sum(durs):.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()