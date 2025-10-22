#!/usr/bin/env python3
"""
Story-Renderer - Erstellt Video aus Hörbuch + Bildern

Neu:
- Titel-Overlay über dem Intro-Video (nicht separater Clip)
- Dynamischer Blur unter dem Titel (Ein-/Ausblenden synchron zum Titel)
- Parametrisierbares Timing für Titel: Offset, Fade-In, Fade-Out, Blur-Stärke
- Automatisches Wrapping & Positionierung (oberes Drittel; 2 Zeilen -> ~10% höher)
- Weiterhin: Overlay (Loop), Szenen aus Metadaten, separates Outro-Video

Voraussetzung:
- ffmpeg verfügbar
- book_context.json enthält:
  {
    "book_info": { "title": "...", "author": "..." }
  }
"""

import subprocess
from pathlib import Path
import json
import textwrap
import argparse

class StoryRenderer:
    def __init__(self, images_dir, audiobook_metadata, output_dir, book_context=None):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(audiobook_metadata, 'r', encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.book_info = {}
        if book_context and Path(book_context).exists():
            with open(book_context, 'r', encoding="utf-8") as f:
                ctx = json.load(f)
                self.book_info = ctx.get("book_info", {})

        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg nicht gefunden!")

    # --------------------------- Utilities ---------------------------

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_video_file(filepath):
        if not filepath:
            return False
        p = Path(filepath)
        return p.exists() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    @staticmethod
    def _is_image_file(filepath):
        if not filepath:
            return False
        p = Path(filepath)
        return p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    @staticmethod
    def _escape_drawtext(s: str) -> str:
        # Escape für drawtext text='...'
        # Wichtige Escapes: backslash, single-quote, colon, percent
        return (
            s.replace("\\", r"\\")
             .replace("'", r"\'")
             .replace(":", r"\:")
             .replace("%", r"\%")
        )

    def _wrap_title_and_positions(self, raw_title: str, width_chars=35):
        """Wrappt den Titel und gibt Y-Positionen für Titel/Autor zurück."""
        if not raw_title:
            return "", "(h/3)", "(h/3+120)"
        wrapped = textwrap.fill(raw_title, width=width_chars)
        lines = wrapped.count("\n") + 1
        y_title = "(h/3)"
        y_author = "(h/3+120)"
        if lines > 1:
            # ca. 10% höher; Author entsprechend nachrücken
            y_title = "(h/3 - 0.10*h)"
            y_author = "(h/3 + 90)"
        return wrapped, y_title, y_author

    # --------------------------- Core rendering ---------------------------

    def render_story_video(
        self,
        audiobook_file,
        quality="hd",
        transition="fade",
        vignette=False,
        overlay_file=None,
        overlay_opacity=0.3,
        fps=30,
        intro_file=None,
        intro_duration=10.0,
        outro_file=None,
        fade_duration=1.0,
        include_title_overlay=True,
        title_offset=3.0,
        title_fadein=0.8,
        title_fadeout=0.8,
        title_blur_sigma=8.0,
        title_fontsize=72,
        author_fontsize=42,
        title_color="white",
        author_color="white",
    ):
        """
        Baut das komplette Video: Intro(+Titel-Overlay) + Szenen + Outro und muxed mit dem Hörbuch-Audio.
        - Titel-Overlay liegt über dem Intro-Clip, Blur wirkt auf den Intro-Clip unter dem Text.
        - Blur & Text folgen einem zeitlichen Verlauf (Offset -> FadeIn -> Hold -> FadeOut).
        """

        print("=" * 60)
        print("🎬 STORY-RENDERER (Titel-Overlay über Intro)")
        print("=" * 60)

        # Qualität bestimmen
        if quality == "hd":
            width, height = 1920, 1080
            bitrate = "8M"
            audio_bitrate = "192k"
            preset = "medium"
            suffix = ""
        else:
            width, height = 854, 480
            bitrate = "2M"
            audio_bitrate = "128k"
            preset = "fast"
            suffix = "_sd"

        output_file = self.output_dir / f"story{suffix}.mp4"

        # Szenen laden
        scenes = self.metadata.get("scenes", [])
        head_silence = float(self.metadata.get("head_silence", 0.0))
        tail_silence = float(self.metadata.get("tail_silence", 0.0))
        print(f"ℹ️  Head-Silence: {head_silence:.2f}s, Tail-Silence: {tail_silence:.2f}s, Szenen: {len(scenes)}")

        # Segmente definieren: INTRO -> SCENES -> OUTRO
        segments = []

        # INTRO: Immer als Video/Clip (Fallback: Schwarz)
        intro_src = intro_file if (intro_file and Path(intro_file).exists()) else None
        if not intro_src:
            print("⚠️  Kein Intro-Video angegeben – nutze schwarzen Hintergrund.")
        segments.append({
            "type": "intro",
            "src": intro_src,
            "duration": float(intro_duration),
        })

        # SCENES
        for s in scenes:
            sid = int(s["scene_id"])
            dur = float(s["duration"])
            img = self.images_dir / f"image_{sid:04d}.png"
            segments.append({"type": "scene", "src": str(img), "duration": dur})

        # OUTRO
        if outro_file and Path(outro_file).exists():
            segments.append({"type": "outro", "src": str(outro_file), "duration": 10.0})
        elif tail_silence > 0:
            # Falls kein Outro-Video, aber Tail-Silence vorhanden, fülle visuell mit Schwarz
            segments.append({"type": "outro_black", "src": None, "duration": tail_silence})

        print(f"🧩 Segmente insgesamt: {len(segments)}")

        # Temp-Verzeichnis
        temp_dir = self.output_dir / "temp_clips"
        temp_dir.mkdir(exist_ok=True)
        clip_files = []

        # Buchinfos
        title_raw = self.book_info.get("title", "").strip()
        author_raw = self.book_info.get("author", "").strip()

        title_wrapped, y_title, y_author = self._wrap_title_and_positions(title_raw, width_chars=35)
        title_text = self._escape_drawtext(title_wrapped.replace("\n", r"\n"))
        author_text = self._escape_drawtext(author_raw)

        # Pro Segment einzelnen Clip bauen
        for idx, seg in enumerate(segments, 1):
            seg_type = seg["type"]
            seg_dur = float(seg["duration"])
            src = seg["src"]
            clip_path = temp_dir / f"clip_{idx:04d}.mp4"

            inputs = []
            filters = []

            # Quelle (Hintergrund)
            if seg_type in ("scene",):
                if src and Path(src).exists() and self._is_image_file(src):
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(src)]
                else:
                    # Fallback: Schwarz
                    inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                               f"color=c=black:s={width}x{height}:r={fps}"]
                bg_label = "[0:v]"
            elif seg_type in ("intro", "outro"):
                if src and self._is_video_file(src):
                    inputs += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(src)]
                elif src and self._is_image_file(src):
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(src)]
                else:
                    inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                               f"color=c=black:s={width}x{height}:r={fps}"]
                bg_label = "[0:v]"
            else:
                # outro_black
                inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                           f"color=c=black:s={width}x{height}:r={fps}"]
                bg_label = "[0:v]"

            # Optionales Overlay (global)
            ov_in = None
            if overlay_file and Path(overlay_file).exists():
                if self._is_video_file(overlay_file):
                    inputs += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(overlay_file)]
                    ov_in = "[1:v]"
                elif self._is_image_file(overlay_file):
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(overlay_file)]
                    ov_in = "[1:v]"

            # Basis-Scale/Pad
            base = (
                f"{bg_label}scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[base0]"
            )

            # Vignette optional
            if vignette:
                base += ";[base0]vignette=PI/6[base1]"
                current = "[base1]"
            else:
                current = "[base0]"

            # -------- INTRO: Titel-Overlay + dynamischer Blur --------
            if seg_type == "intro" and include_title_overlay and (title_text or author_text):
                # Zeitfunktion für Alpha (Text) und Sigma (Blur)
                # Alpha(t): 0 .. fade-in .. 1 .. (halten) .. fade-out .. 0
                # Blur(t):  0 .. fade-in .. max .. (halten) .. fade-out .. 0
                off = max(0.0, float(title_offset))
                fi = max(0.05, float(title_fadein))
                fo = max(0.05, float(title_fadeout))
                sigma = max(0.0, float(title_blur_sigma))
                seg = seg_dur  # nur für Formel lesbarer

                alpha_expr = (
                    f"if(lt(t,{off}),0,"
                    f" if(lt(t,{off+fi}),(t-{off})/{fi},"
                    f"  if(lt(t,{seg-fo}),1,"
                    f"   if(lt(t,{seg}),({seg}-t)/{fo},0)"
                    f"  )"
                    f" )"
                    f")"
                )

                blur_expr = (
                    f"if(lt(t,{off}),0,"
                    f" if(lt(t,{off+fi}),(t-{off})/{fi}*{sigma},"
                    f"  if(lt(t,{seg-fo}),{sigma},"
                    f"   if(lt(t,{seg}),({seg}-t)/{fo}*{sigma},0)"
                    f"  )"
                    f" )"
                    f")"
                )

                # Blur NUR auf Videobasis anwenden, Text danach zeichnen
                base += f";{current}gblur=sigma='{blur_expr}'[intro_blur]"
                current = "[intro_blur]"

                # Drawtext-Filter: Titel und Autor
                title_font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
                author_font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

                if title_text:
                    base += (
                        f";{current}drawtext=fontfile='{title_font}':"
                        f"text='{title_text}':fontcolor={title_color}:fontsize={int(title_fontsize)}:"
                        f"x=(w-text_w)/2:y={y_title}:alpha='{alpha_expr}'[intro_t1]"
                    )
                    current = "[intro_t1]"

                if author_text:
                    base += (
                        f";{current}drawtext=fontfile='{author_font}':"
                        f"text='{author_text}':fontcolor={author_color}:fontsize={int(author_fontsize)}:"
                        f"x=(w-text_w)/2:y={y_author}:alpha='{alpha_expr}'[intro_t2]"
                    )
                    current = "[intro_t2]"

            # -------- Szene: Fades (optional) --------
            if seg_type == "scene" and transition == "fade" and fade_duration > 0:
                base += (
                    f";{current}fade=t=in:st=0:d={float(fade_duration):.3f}"
                    f",fade=t=out:st={max(0.0, seg_dur - float(fade_duration)):.3f}:d={float(fade_duration):.3f}[sc_fade]"
                )
                current = "[sc_fade]"

            # -------- Overlay drüber --------
            if ov_in:
                base += f";{ov_in}scale={width}:{height},format=rgba,colorchannelmixer=aa={float(overlay_opacity):.3f}[ov]"
                base += f";{current}[ov]overlay=0:0:shortest=1[outv]"
                final_v = "[outv]"
            else:
                final_v = current

            filters.append(base)

            # ffmpeg Kommando
            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", ";".join(filters),
                "-map", final_v,
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-t", f"{seg_dur:.3f}",
                str(clip_path)
            ]

            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0 or not clip_path.exists():
                print(f"   ✗ Segment {idx} fehlgeschlagen → {seg_type}")
                print(res.stderr.decode()[:400])
                continue

            clip_files.append(clip_path)
            print(f"   ✓ Segment {idx}/{len(segments)} erstellt ({seg_type}, {seg_dur:.3f}s)")

        if not clip_files:
            print("\n❌ Keine Clips erstellt – Abbruch")
            return None

        # Concat & Mux mit Audiobook
        concat_file = self.output_dir / "temp_clips" / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clip_files:
                f.write(f"file '{clip.as_posix()}'\n")

        cmd_final = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audiobook_file),
            "-c:v", "libx264", "-preset", preset, "-b:v", bitrate,
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            "-shortest",
            str(output_file)
        ]
        result = subprocess.run(cmd_final, capture_output=True)

        # Cleanup
        try:
            for c in clip_files:
                c.unlink(missing_ok=True)
            concat_file.unlink(missing_ok=True)
            (self.output_dir / "temp_clips").rmdir()
        except Exception:
            pass

        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / (1024 ** 2)
            print(f"\n✅ Video: {output_file.name} ({size_mb:.1f} MB)")
            return output_file
        else:
            print("\n❌ Rendering fehlgeschlagen")
            print(result.stderr.decode()[:600])
            return None

    # Convenience: beide Qualitäten
    def render_both_qualities(self, audiobook_file, **kwargs):
        results = {}
        print("📹 Rendere HD-Version…")
        hd = self.render_story_video(audiobook_file, quality="hd", **kwargs)
        if hd:
            results["hd"] = hd
        print("\n📹 Rendere SD-Version…")
        sd = self.render_story_video(audiobook_file, quality="sd", **kwargs)
        if sd:
            results["sd"] = sd

        print(f"\n{'=' * 60}")
        print("✅ VIDEO-RENDERING FERTIG!")
        if "hd" in results:
            print(f"   HD: {results['hd']}")
        if "sd" in results:
            print(f"   SD: {results['sd']}")
        print(f"{'=' * 60}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Story-Renderer mit Titel-Overlay über Intro-Video")

    # Nur path ist Pflicht
    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")

    # Alle anderen Parameter sind optional und überschreibbar
    parser.add_argument("--images", default=None, help="Verzeichnis mit Bildern (image_XXXX.png)")
    parser.add_argument("--audiobook", default=None, help="Komplettes Hörbuch-Audio (WAV)")
    parser.add_argument("--metadata", default=None, help="Audiobook-Metadaten (JSON)")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis")
    parser.add_argument("--book-context", default=None, help="Pfad zu book_context.json")

    parser.add_argument("--quality", choices=["hd", "sd", "both"], default="both", help="Video-Qualität (Standard: both)")
    parser.add_argument("--transition", choices=["none", "fade"], default="fade", help="Szenen-Übergang")
    parser.add_argument("--fade-duration", type=float, default=1.0, help="Fade-Dauer für Szenen in Sekunden")
    parser.add_argument("--vignette", action="store_true", help="Vignette-Effekt aktivieren")

    parser.add_argument("--overlay", help="Overlay-Datei (Video/Bild), wird geloopt")
    parser.add_argument("--overlay-opacity", type=float, default=0.3, help="Overlay-Transparenz 0.0–1.0")

    parser.add_argument("--fps", type=int, default=30, help="Ziel-FPS")
    parser.add_argument("--intro", help="Intro-Video")
    parser.add_argument("--intro-duration", type=float, default=10.0, help="Intro-Dauer in Sekunden")
    parser.add_argument("--outro", help="Outro-Video")

    # Titel-Overlay Parameter
    parser.add_argument("--title-enable", dest="title_enable", action="store_true", default=True,
                        help="Titel-Overlay aktivieren (Standard: an)")
    parser.add_argument("--title-offset", type=float, default=3.0,
                        help="Sekunden bis Titel-Einblendung beginnt (Standard: 3.0)")
    parser.add_argument("--title-fadein", type=float, default=0.8, help="Fade-In Dauer des Titels (Sek.)")
    parser.add_argument("--title-fadeout", type=float, default=0.8, help="Fade-Out Dauer des Titels (Sek.)")
    parser.add_argument("--title-blur", type=float, default=8.0, help="Maximale Blur-Stärke (Sigma) während des Titels")
    parser.add_argument("--title-fontsize", type=int, default=72, help="Schriftgröße Titel")
    parser.add_argument("--author-fontsize", type=int, default=42, help="Schriftgröße Autor")

    args = parser.parse_args()
    base_path = args.path

    # --- CONFIG ---
    CONFIG = {
        # Dynamisch aus Basis-Pfad
        "images": args.images or os.path.join(base_path, "images"),
        "audiobook": args.audiobook or os.path.join(base_path, "audiobook", "complete_audiobook.wav"),
        "metadata": args.metadata or os.path.join(base_path, "audiobook", "audiobook_metadata.json"),
        "output": args.output or os.path.join(base_path, "story"),
        "book_context": args.book_context or os.path.join(base_path, "book_context.json"),

        # Video- und Rendering-Einstellungen
        "quality": args.quality,
        "transition": args.transition,
        "fade_duration": args.fade_duration,
        "vignette": args.vignette,
        "overlay": args.overlay,
        "overlay_opacity": args.overlay_opacity,
        "fps": args.fps,
        "intro": args.intro,
        "intro_duration": args.intro_duration,
        "outro": args.outro,

        # Titel-Overlay
        "title_enable": args.title_enable,
        "title_offset": args.title_offset,
        "title_fadein": args.title_fadein,
        "title_fadeout": args.title_fadeout,
        "title_blur": args.title_blur,
        "title_fontsize": args.title_fontsize,
        "author_fontsize": args.author_fontsize,
    }

    renderer = StoryRenderer(
        images_dir=args.images,
        audiobook_metadata=args.metadata,
        output_dir=args.output,
        book_context=args.book_context
    )

    common = dict(
        transition=args.transition,
        fade_duration=args.fade_duration,
        vignette=args.vignette,
        overlay_file=args.overlay,
        overlay_opacity=args.overlay_opacity,
        fps=args.fps,
        intro_file=args.intro,
        intro_duration=args.intro_duration,
        outro_file=args.outro,
        include_title_overlay=args.title_enable,
        title_offset=args.title_offset,
        title_fadein=args.title_fadein,
        title_fadeout=args.title_fadeout,
        title_blur_sigma=args.title_blur,
        title_fontsize=args.title_fontsize,
        author_fontsize=args.author_fontsize,
    )

    if args.quality == "both":
        renderer.render_both_qualities(audiobook_file=args.audiobook, **common)
    else:
        renderer.render_story_video(audiobook_file=args.audiobook, quality=args.quality, **common)

    print("\n🎉 Story-Rendering abgeschlossen!")


if __name__ == "__main__":
    main()
