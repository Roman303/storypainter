#!/usr/bin/env python3
"""
Story-Renderer - Erstellt Video aus Hörbuch + Bildern (GAP/XFADE version)
- Nutzt komplettes Hörbuch-Audio (WAV wird NICHT geschnitten)
- Szenen-Video wird so gebaut, dass es exakt die Audiolänge abdeckt
- Einheitlicher GAP zwischen den Szenen wird aus JSON berechnet
- Jeder Scene-Chunk wird um GAP am Anfang und am Ende verlängert
- Beim Mergen werden die Chunks mit 2*GAP überblendet (xfade),
  so dass die effektiven Szenen-Start/Endpunkte (ohne Verlängerung)
  nahtlos zusammenlaufen
- Fades innerhalb der Szene sind so gelegt, dass das Einfaden
  mit einer Verzögerung (transition_in_delay) startet und das
  Ausfaden mit einer Verzögerung (transition_out_delay) am JSON-Ende beginnt.
- Optionales Intro/Outro/Overlay bleiben erhalten

Wichtige Annahmen fürs JSON der Szenen (mindestens eins der Felder muss passen):
- Szenenobjekte enthalten entweder "start" und "end" (Sekunden), ODER
- "start" und "duration", ODER
- es existiert ein globaler "uniform_gap" in metadata (Sekunden)

Wenn mehrere Gaps berechnet werden können, wird der Median verwendet und
auf Konsistenz geprüft (Warnung, wenn die Streuung > 20 ms ist).
"""

import subprocess
from pathlib import Path
import json
import os
import argparse
import statistics


class StoryRenderer:
    def __init__(self, images_dir, audiobook_metadata, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_dir = self.output_dir.parent

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

    def _resolve(self, maybe_path):
        if not maybe_path:
            return None
        p = Path(maybe_path)
        return p if p.is_absolute() else (self.base_dir / p)

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

    def _compute_uniform_gap(self, scenes):
        ug = self.metadata.get("uniform_gap")
        if isinstance(ug, (int, float)) and ug >= 0:
            return float(ug)

        gaps = []
        for i in range(1, len(scenes)):
            a = scenes[i - 1]
            b = scenes[i]
            if ("end" in a or "duration" in a) and ("start" in b or "start_time" in b):
                a_end = None
                if "end" in a:
                    a_end = float(a["end"])
                elif "duration" in a and ("start" in a or "start_time" in a):
                    a_start = float(a.get("start", a.get("start_time", 0.0)))
                    a_end = a_start + float(a["duration"])

                b_start = float(b.get("start", b.get("start_time", 0.0)))

                if a_end is not None:
                    g = b_start - a_end
                    if g >= -0.05:
                        gaps.append(max(0.0, g))

        if gaps:
            med = statistics.median(gaps)
            if len(gaps) > 1:
                spread = max(gaps) - min(gaps)
                if spread > 0.02:
                    print(f"⚠️  Hinweis: GAPs variieren (min={min(gaps):.3f}s, max={max(gaps):.3f}s, median={med:.3f}s). Nutze Median.")
            return float(med)

        return 0.0

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
        transition_in_delay=0.0,
        transition_out_delay=0.0,
    ):
        print("=" * 60)
        print("🎬 STORY-RENDERER (GAP/XFADE)")
        print("=" * 60)
        print(f"\n📁 Bilder:   {self.images_dir}")
        print(f"🎵 Hörbuch:  {audiobook_file}")
        print(f"📁 Output:   {self.output_dir}")
        print(f"🎨 Qualität: {quality.upper()}")
        print(f"🎭 Transition: {transition} (fade_d={fade_duration:.2f}s)")
        print(f"🎞️ FPS: {fps}")

        if quality == "hd":
            width, height = 1920, 1080
            bitrate = "8M"
            audio_bitrate = "192k"
            preset = "medium"
            suffix = ""
        else:
            width, height = 640, 360
            bitrate = "800k"
            audio_bitrate = "96k"
            preset = "veryfast"
            suffix = "_sd"

        output_file = self.output_dir / f"story{suffix}.mp4"

        head_silence = float(self.metadata.get("head_silence", 0.0))
        tail_silence = float(self.metadata.get("tail_silence", 0.0))
        scenes = self.metadata.get("scenes", [])
        total_duration = float(self.metadata.get("total_duration", 0.0))

        uniform_gap = self._compute_uniform_gap(scenes)
        print(f"\n⏱️  Uniform GAP: {uniform_gap:.3f} s")

        intro_file = self._resolve(intro_file)
        outro_file = self._resolve(outro_file)
        overlay_file = self._resolve(overlay_file)

        segments = []

        if intro_file and intro_file.exists() and intro_duration > 0:
            segments.append({"type": "intro", "src": str(intro_file), "duration": float(intro_duration)})
        elif head_silence > 0:
            segments.append({"type": "intro_silence", "src": None, "duration": float(head_silence)})

        for s in scenes:
            scene_id = int(s.get("scene_id", s.get("id", 0)))
            if "duration" in s:
                base_dur = float(s["duration"])
            elif "start" in s and "end" in s:
                base_dur = float(s["end"]) - float(s["start"])
            else:
                raise ValueError("Szenen benötigen entweder 'duration' oder ('start' und 'end').")

            seg_dur = max(0.01, base_dur + 2 * uniform_gap)
            image_file = self.images_dir / f"image_{scene_id:04d}.png"
            segments.append({
                "type": "scene",
                "src": str(image_file),
                "scene_id": scene_id,
                "duration": seg_dur,
                "base_duration": base_dur,
            })

        if outro_file and outro_file.exists():
            dur = float(outro_duration or tail_silence or intro_duration)
            segments.append({"type": "outro", "src": str(outro_file), "duration": dur})
        elif tail_silence > 0:
            segments.append({"type": "outro_silence", "src": None, "duration": float(tail_silence)})

        temp_dir = self.output_dir / "temp_clips"
        temp_dir.mkdir(exist_ok=True)
        clip_files = []
        clip_durations = []

        for idx, seg in enumerate(segments, 1):
            seg_type = seg["type"]
            seg_dur = float(seg["duration"])
            src = seg["src"]
            clip_path = temp_dir / f"clip_{idx:04d}.mp4"

            inputs = []
            filters = []

            if seg_type == "scene":
                if not src or not Path(src).exists():
                    inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                               f"color=c=black:s={width}x{height}:r={fps}"]
                    bg_label = "[0:v]"
                else:
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(src)]
                    bg_label = "[0:v]"
            else:
                inputs += ["-f", "lavfi", "-t", f"{seg_dur:.3f}", "-i",
                           f"color=c=black:s={width}x{height}:r={fps}"]
                bg_label = "[0:v]"

            ov_in = None
            if overlay_file and overlay_file.exists():
                if self._is_video_file(overlay_file):
                    inputs += ["-stream_loop", "-1", "-t", f"{seg_dur:.3f}", "-i", str(overlay_file)]
                    ov_in = f"[1:v]"
                elif self._is_image_file(overlay_file):
                    inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-r", str(fps), "-i", str(overlay_file)]
                    ov_in = f"[1:v]"

            base = (
                f"{bg_label}scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
            )
            if vignette:
                base += ",vignette=PI/6"

            if transition == "fade" and seg_type == "scene" and fade_duration > 0:
                fade_in_start = max(0.0, uniform_gap - fade_duration + transition_in_delay)
                fade_out_start = max(0.0, seg_dur - uniform_gap + transition_out_delay)
                base += f",fade=t=in:st={fade_in_start:.3f}:d={fade_duration:.3f}"
                base += f",fade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f}"

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
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-t", f"{seg_dur:.3f}",
                str(clip_path)
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not clip_path.exists() or clip_path.stat().st_size < 1000:
                print(f"   ✗ Segment {idx} fehlgeschlagen ({seg_type})")
                continue

            clip_files.append(clip_path)
            clip_durations.append(seg_dur)

        merged_video = self._xfade_merge(
            clip_files=clip_files,
            durations=clip_durations,
            overlap_duration=max(0.0, 2 * uniform_gap),
            fps=fps,
            preset=preset,
            bitrate=bitrate,
        )

        if not merged_video:
            print("\n❌ XFADE-Merge fehlgeschlagen!")
            return None

        hard_cap = total_duration if total_duration > 0 else None
        cmd = [
            "ffmpeg", "-y",
            "-i", str(merged_video),
            "-i", str(audiobook_file),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", preset, "-b:v", bitrate,
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            "-shortest",
        ]
        if hard_cap:
            cmd += ["-t", f"{hard_cap:.3f}"]
        cmd += [str(output_file)]

        subprocess.run(cmd, capture_output=True)
        return output_file

    def _xfade_merge(self, clip_files, durations, overlap_duration, fps, preset, bitrate):
        if len(clip_files) == 1:
            return clip_files[0]

        cmd = ["ffmpeg", "-y"]
        for p in clip_files:
            cmd += ["-i", str(p)]

        filter_parts = []
        last_label = f"[0:v]"
        filter_parts.append(f"{last_label}fps={fps},format=yuv420p[v0]")
        current_label = "[v0]"

        for i in range(1, len(clip_files)):
            prev_dur = float(durations[i - 1])
            this_dur = float(durations[i])
            eff_overlap = min(overlap_duration, prev_dur * 0.3, this_dur * 0.3)
            eff_overlap = max(0.0, eff_overlap)
            offset = max(0.0, prev_dur - eff_overlap)

            filter_parts.append(f"[{i}:v]fps={fps},format=yuv420p[v{i}in]")
            filter_parts.append(
                f"{current_label}[v{i}in]xfade=transition=fade:duration={eff_overlap:.6f}:offset={offset:.6f}[v{i}]"
            )
            current_label = f"[v{i}]"

        out_label = current_label
        cmd += [
            "-filter_complex", ";".join(filter_parts),
            "-map", out_label,
            "-an",
            "-c:v", "libx264", "-preset", preset, "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            str(self.output_dir / "_merged_video_tmp.mp4")
        ]

        subprocess.run(cmd, capture_output=True)
        return self.output_dir / "_merged_video_tmp.mp4"


def main():
    parser = argparse.ArgumentParser(description="Story-Renderer mit dynamischen Fades und GAP/XFADE")
    parser.add_argument("--path", required=True)
    parser.add_argument("--images", default=None)
    parser.add_argument("--audiobook", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quality", choices=["hd", "sd"], default="sd")
    parser.add_argument("--transition", choices=["none", "fade"], default="fade")
    parser.add_argument("--fade-duration", type=float, default=2.0)
    parser.add_argument("--transition-in-delay", type=float, default=0.0)
    parser.add_argument("--transition-out-delay", type=float, default=0.0)
    parser.add_argument("--vignette", action="store_false")
    parser.add_argument("--overlay", default="particel.mp4")
    parser.add_argument("--overlay-opacity", type=float, default=0.3
    parser.add_argument("--fps", type=int, default=30, help="Ziel-Framerate")

    parser.add_argument("--intro", default="intro.mp4", help="Intro-Datei (Video/Bild)")
    parser.add_argument("--intro-duration", type=float, default=10.0, help="Intro-Dauer in Sekunden")
    parser.add_argument("--outro", default="outro.mp4", help="Outro-Datei (Video/Bild)")
    parser.add_argument("--outro-duration", type=float, default=10.0, help="Outro-Dauer in Sekunden")

    args = parser.parse_args()
    base_path = args.path

    audiobook_path = args.audiobook or os.path.join(base_path, "audiobook", "complete_audiobook.wav")
    metadata_path = args.metadata or os.path.join(base_path, "audiobook", "audiobook_metadata.json")

    if not Path(audiobook_path).exists():
        print(f"❌ Hörbuch nicht gefunden: {audiobook_path}")
        return
    if not Path(metadata_path).exists():
        print(f"❌ Metadaten nicht gefunden: {metadata_path}")
        return

    renderer = StoryRenderer(
        images_dir=args.images or os.path.join(base_path, "images"),
        audiobook_metadata=metadata_path,
        output_dir=args.output or os.path.join(base_path, "story")
    )

    common = dict(
        transition=args.transition,
        fade_duration=args.fade_duration,
        transition_gap=args.transition_gap,
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
        renderer.render_both_qualities(audiobook_file=audiobook_path, **common)
    else:
        renderer.render_story_video(audiobook_file=audiobook_path, quality=args.quality, **common)

    print("\n🎉 Story-Rendering abgeschlossen!")


if __name__ == "__main__":
    main()
