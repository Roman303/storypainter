#!/usr/bin/env python3
"""
Story-Renderer - Erstellt Video aus Hörbuch + Bildern
- Nutzt komplettes Hörbuch-Audio
- Synchronisiert Bilder mit Szenen-Timestamps (frame-konsistent)
- Optionales Intro- und Outro-Video (separat definierbar mit Dauer)
- Optionales Overlay (Video oder Bild), das in einer Endlosschleife läuft
- Sanfte Fades zwischen Szenen (konfigurierbar)
"""

import subprocess
from pathlib import Path
import json
import os
import argparse


class StoryRenderer:
    def __init__(self, images_dir, audiobook_metadata, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(audiobook_metadata, 'r', encoding="utf-8") as f:
            self.metadata = json.load(f)

        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg nicht gefunden!")

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _is_video_file(self, filepath):
        if not filepath:
            return False
        p = Path(filepath)
        return p.exists() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def _is_image_file(self, filepath):
        if not filepath:
            return False
        p = Path(filepath)
        return p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    # --------------------------- Core Rendering ---------------------------

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
        outro_duration=None,
        fade_duration=1.0,
    ):
        print("=" * 60)
        print("🎬 STORY-RENDERER")
        print("=" * 60)
        print(f"\n📁 Bilder:   {self.images_dir}")
        print(f"🎵 Hörbuch:  {audiobook_file}")
        print(f"📁 Output:   {self.output_dir}")
        print(f"🎨 Qualität: {quality.upper()}")
        print(f"🎭 Transition: {transition} (fade_d={fade_duration:.2f}s)")
        print(f"🎞️ FPS: {fps}")

        # Qualitätsprofile
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

        # --- Segmentliste aufbauen ---
        segments = []
        head_silence = float(self.metadata.get("head_silence", 0.0))
        tail_silence = float(self.metadata.get("tail_silence", 0.0))
        scenes = self.metadata.get("scenes", [])

        # --- Intro ---
        if intro_file and Path(intro_file).exists() and intro_duration > 0:
            if abs(head_silence - intro_duration) > 0.25:
                print(f"⚠️  Hinweis: head_silence ({head_silence:.2f}s) ≠ intro_duration ({intro_duration:.2f}s)")
            segments.append({
                "type": "intro",
                "src": intro_file,
                "duration": intro_duration,
            })
        elif head_silence > 0:
            segments.append({
                "type": "intro_silence",
                "src": None,
                "duration": head_silence,
            })

        # --- Szenen ---
        for s in scenes:
            scene_id = int(s["scene_id"])
            duration = float(s["duration"])
            image_file = self.images_dir / f"image_{scene_id:04d}.png"
            segments.append({
                "type": "scene",
                "src": str(image_file),
                "scene_id": scene_id,
                "duration": duration,
            })

        # --- Outro ---
        if outro_file and Path(outro_file).exists():
            dur = outro_duration or tail_silence or intro_duration
            if abs(tail_silence - dur) > 0.25:
                print(f"⚠️  Hinweis: tail_silence ({tail_silence:.2f}s) ≠ outro_duration ({dur:.2f}s)")
            segments.append({
                "type": "outro",
                "src": outro_file,
                "duration": dur,
            })
        elif tail_silence > 0:
            segments.append({
                "type": "outro_silence",
                "src": None,
                "duration": tail_silence,
            })

        print(f"\n🧩 Segmente: {len(segments)}")
        print(f"   Intro: {intro_file or '–'} ({intro_duration:.2f}s)")
        print(f"   Outro: {outro_file or '–'} ({(outro_duration or tail_silence):.2f}s)")
        print(f"   Szenen: {len(scenes)}  | Audio: {self.metadata.get('total_duration', 0)/60:.1f} Min")

        # --- Temp Clips ---
        temp_dir = self.output_dir / "temp_clips"
        temp_dir.mkdir(exist_ok=True)
        clip_files = []

        for idx, seg in enumerate(segments, 1):
            seg_type = seg["type"]
            seg_dur = float(seg["duration"])
            src = seg["src"]
            clip_path = temp_dir / f"clip_{idx:04d}.mp4"

            inputs = []
            filters = []

            # --- Hintergrundquelle ---
            if seg_type in ("scene",):
                if not src or not Path(src).exists():
                    inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                               f"color=c=black:s={width}x{height}:r={fps}"]
                    bg_label = "[0:v]"
                else:
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(src)]
                    bg_label = "[0:v]"
            elif seg_type in ("intro", "outro"):
                if self._is_video_file(src):
                    inputs += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(src)]
                elif self._is_image_file(src):
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(src)]
                else:
                    inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                               f"color=c=black:s={width}x{height}:r={fps}"]
                bg_label = "[0:v]"
            else:
                inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                           f"color=c=black:s={width}x{height}:r={fps}"]
                bg_label = "[0:v]"

            # --- Overlay ---
            ov_in = None
            if overlay_file and Path(overlay_file).exists():
                if self._is_video_file(overlay_file):
                    inputs += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(overlay_file)]
                    ov_in = "[1:v]"
                elif self._is_image_file(overlay_file):
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(overlay_file)]
                    ov_in = "[1:v]"

            # --- Filter ---
            base = f"{bg_label}scale={width}:{height}:force_original_aspect_ratio=decrease," \
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
            if vignette:
                base += ",vignette=PI/6"
            if transition == "fade" and seg_type == "scene" and fade_duration > 0:
                base += f",fade=t=in:st=0:d={fade_duration:.3f}"
                if seg_dur > fade_duration:
                    base += f",fade=t=out:st={max(0.0, seg_dur - fade_duration):.3f}:d={fade_duration:.3f}"
            base += "[base]"
            filters.append(base)

            if ov_in:
                filters.append(f"{ov_in}scale={width}:{height},format=rgba,"
                               f"colorchannelmixer=aa={overlay_opacity:.3f}[ovr]")
                filters.append("[base][ovr]overlay=0:0:shortest=1[out]")
                out_label = "[out]"
            else:
                out_label = "[base]"

            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", ";".join(filters),
                "-map", out_label,
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-t", f"{seg_dur:.3f}",
                str(clip_path)
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not clip_path.exists():
                print(f"   ✗ Segment {idx} fehlgeschlagen ({seg_type})")
                print(result.stderr.decode()[:300])
                continue

            clip_files.append(clip_path)
            print(f"   ✓ Segment {idx}/{len(segments)} erstellt ({seg_type}, {seg_dur:.3f}s)")

        if not clip_files:
            print("\n❌ Keine Clips erstellt – Abbruch")
            return None

        print(f"\n📹 {len(clip_files)} Clips erstellt, kombiniere…")

        # --- Concat ---
        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clip_files:
                f.write(f"file '{clip.as_posix()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audiobook_file),
            "-c:v", "libx264", "-preset", preset, "-b:v", bitrate,
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            "-shortest",
            str(output_file)
        ]
        result = subprocess.run(cmd, capture_output=True)

        # Cleanup
        try:
            for clip in clip_files:
                clip.unlink(missing_ok=True)
            concat_file.unlink(missing_ok=True)
            temp_dir.rmdir()
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
    parser = argparse.ArgumentParser(description="Story-Renderer: Video aus Hörbuch + Bildern (+ Intro/Outro + Overlay)")
    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")
    parser.add_argument("--images", default=None, help="Verzeichnis mit Bildern (image_XXXX.png)")
    parser.add_argument("--audiobook", default=None, help="Komplettes Hörbuch-Audio (WAV)")
    parser.add_argument("--metadata", default=None, help="Audiobook-Metadaten (JSON)")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis")

    parser.add_argument("--quality", choices=["hd", "sd", "both"], default="both", help="Video-Qualität")
    parser.add_argument("--transition", choices=["none", "fade"], default="fade", help="Szenen-Übergang")
    parser.add_argument("--fade-duration", type=float, default=1.0, help="Fade-Dauer in Sekunden")
    parser.add_argument("--vignette", action="store_true", help="Vignette-Effekt aktivieren")

    parser.add_argument("--overlay", help="Overlay-Datei (Video oder Bild, geloopt)")
    parser.add_argument("--overlay-opacity", type=float, default=0.3, help="Overlay-Transparenz 0.0–1.0")
    parser.add_argument("--fps", type=int, default=30, help="Ziel-Framerate")

    # NEU: Getrennte Intro-/Outro-Konfiguration
    parser.add_argument("--intro", help="Intro-Datei (Video/Bild)")
    parser.add_argument("--intro-duration", type=float, default=10.0, help="Intro-Dauer in Sekunden")
    parser.add_argument("--outro", help="Outro-Datei (Video/Bild)")
    parser.add_argument("--outro-duration", type=float, default=None, help="Outro-Dauer in Sekunden")

    args = parser.parse_args()
    base_path = args.path

    # Pfade prüfen
    if not Path(args.audiobook).exists():
        print(f"❌ Hörbuch nicht gefunden: {args.audiobook}")
        return
    if not Path(args.metadata).exists():
        print(f"❌ Metadaten nicht gefunden: {args.metadata}")
        return

    renderer = StoryRenderer(
        images_dir=args.images or os.path.join(base_path, "images"),
        audiobook_metadata=args.metadata or os.path.join(base_path, "audiobook", "audiobook_metadata.json"),
        output_dir=args.output or os.path.join(base_path, "story")
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
        outro_duration=args.outro_duration,
    )

    if args.quality == "both":
        renderer.render_both_qualities(audiobook_file=args.audiobook, **common)
    else:
        renderer.render_story_video(audiobook_file=args.audiobook, quality=args.quality, **common)

    print("\n🎉 Story-Rendering abgeschlossen!")


if __name__ == "__main__":
    main()
