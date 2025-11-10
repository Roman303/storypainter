#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v6.3 – JSON-strikt, Intro=Szene 0, HyperTrail nur im Hintergrund,
Outro-Support, frühe Einblendungen via Offsets, weiche Gap-Blenden (Black-Fades),
1080p30, Auto-GPU, optimierte Performance.

Intro:
  - Dauer = JSON-Intro-Dauer (z.B. 12s), egal wie lang intro.mp4 wirklich ist.
  - Per tpad wird IMMER genug Material erzeugt, danach hart mit -t auf JSON-Länge
    geschnitten → garantiert exakte Dauer.
  - HyperTrail-FX nur auf dem Hintergrund, blendet in den letzten 2 Sekunden aus.
  - Titel/Autor wie in v1:
      * Titel: fontsize 72, weiß, zentriert, Shadow
      * Autor: fontsize 36, weiß, zentriert, Shadow
      * Alpha-Kurve:
          if(lt(t,2),0,
             if(lt(t,3.5),(t-2)/1.5,
                if(lt(t,9),1,
                   if(lt(t,10),1-(t-9)/1,0))))
Gaps:
  - Nur aus Timeline:
        gap = max(0, start_time(scene[i+1]) - end_time(scene[i]))
    → JSON pause_duration wird ignoriert.
Overlay:
  - Wenn kein --overlay übergeben, wird base_dir/"overlay.mp4" als Default benutzt
    (wenn vorhanden).
"""

import subprocess
from pathlib import Path
import json, argparse, shutil

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

# ---------- renderer ----------
class StoryRenderer:
    def __init__(self, images_dir: Path, metadata_path: Path, output_dir: Path):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "temp_clips"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

        # Basis-Verzeichnis (z.B. <base>/story → base = parent)
        self.base_dir = self.output_dir.parent

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)

        self.nvenc_available = has_nvenc()
        if self.nvenc_available:
            print("🎞️ GPU (NVENC) erkannt und aktiviert.")
        else:
            print("⚠️ Kein NVENC gefunden – verwende CPU (libx264).")

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
        idx: int
    ) -> Path:
        """
        Generic Renderer für Szenen (inkl. Outro):
          - Bild: loop + scale/pad + Fades
          - Video: -ss 0 -t clip_dur + scale/pad + Fades
          - Fallback: Schwarzer Background mit Fades
        """
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        clip_dur = max(0.0, float(clip_dur))

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
                # Videoquelle, hart auf clip_dur begrenzt
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
                # unbekannter Typ → schwarzes Bild
                inputs = [
                    "-f", "lavfi",
                    "-t", f"{clip_dur:.6f}",
                    "-i", f"color=c=black:s={width}x{height}:r={fps}"
                ]
                base = "[0:v]format=yuv420p"
        else:
            # Kein File → schwarzes Bild
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
        """
        Intro mit HyperTrail (v6.3):
          - tpad verlängert die Quelle IMMER mehr als genug.
          - -t schneidet hart auf JSON-Intro-Länge (intro_base_dur).
          - HyperTrail läuft über den Hintergrund und blendet in den letzten 2s aus.
          - Titel/Autor werden NICHT getrailed (oben drauf, wie v1).
        """
        intro_base_dur = max(0.0, float(intro_base_dur))
        intro_clip_dur = intro_base_dur

        # Wir erzeugen einfach "zu viel" Trail-Material und schneiden es später ab:
        # stop_duration = intro_base_dur sorgt dafür, dass die Quelle
        # mindestens intro_base_dur länger nachläuft als ihr eigenes Ende.
        # Da wir danach mit -t intro_clip_dur abschneiden, ist garantiert
        # genug Material für die vollen JSON-Sekunden vorhanden.
        pad_dur = intro_base_dur
        tpad = f"tpad=stop_mode=clone:stop_duration={pad_dur:.3f},"
        print(f"✨ Intro tpad aktiv (clone +{pad_dur:.2f}s, danach hart auf {intro_clip_dur:.2f}s geschnitten)")

        # Globaler Fade-Out (für das ganze Intro-Bild inkl. Text)
        t_out_start = clamp(
            intro_base_dur + fade_out_offset,
            0.0,
            max(0.0, intro_clip_dur - fade_out)
        )
        # HyperTrail-Out: letzte 2 Sekunden des Intros
        trail_fade_dur = min(2.0, intro_clip_dur)
        trail_fade_start = max(0.0, intro_clip_dur - trail_fade_dur)

        outp = self.tmp_dir / "intro_0000.mp4"
        t1, t2 = esc_txt(title), esc_txt(author)

        # Alpha-Kurve wie in deiner v1
        alpha_expr = (
            "if(lt(t,2),0,"
            "if(lt(t,3.5),(t-2)/1.5,"
            "if(lt(t,9),1,"
            "if(lt(t,10),1-(t-9)/1,0))))"
        )

        if intro_src and intro_src.exists():
            inputs = ["-i", str(intro_src)]
        else:
            inputs = [
                "-f", "lavfi",
                "-t", f"{intro_clip_dur:.6f}",
                "-i", f"color=c=black:s={width}x{height}:r={fps}"
            ]

        # Filtergraph:
        #  [0:v]tpad → scale/pad → abdunkeln → global fade out → split
        #  → tmix HyperTrail → Trail-Fade-Out → overlay auf Base → Text oben drauf
        flt = (
            f"[0:v]{tpad}"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"
            f"eq=brightness=-0.25,"
            f"fade=t=out:st={t_out_start:.6f}:d={fade_out:.6f}[b0];"
            f"[b0]split[base][trailsrc];"
            f"[trailsrc]tmix=frames=60:weights='1 1 1 1 1',"
            f"format=rgba,fade=t=out:st={trail_fade_start:.6f}:d={trail_fade_dur:.6f}:alpha=1[trail];"
            f"[base]format=rgba[base_rgba];"
            f"[base_rgba][trail]overlay=0:0:shortest=1[bg];"
            f"[bg]drawtext=text='{t1}':fontsize=72:fontcolor=white:"
            f"alpha='{alpha_expr}':x=(w-text_w)/2:y=(h-text_h)/2-40:"
            f"shadowcolor=black:shadowx=2:shadowy=2,"
            f"drawtext=text='{t2}':fontsize=36:fontcolor=white:"
            f"alpha='{alpha_expr}':x=(w-text_w)/2:y=(h-text_h)/2+60:"
            f"shadowcolor=black:shadowx=2:shadowy=2[v]"
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

        print("🎬 Intro (Szene 0) mit HyperTrail rendern …")
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", flt,
            "-map", "[v]",
            "-r", str(fps),
            "-an",
            *enc,
            "-t", f"{intro_clip_dur:.6f}",  # Hart auf JSON-Länge schneiden
            str(outp)
        ]
        run(cmd, quiet=True)
        return outp

    def _build_gap_black(self, duration: float, width: int, height: int, fps: int, idx: int) -> Path:
        """Schwarzer Zwischenclip mit weichem Fade-In/Out (kurz, proportional zur Länge)."""
        d = max(0.0, float(duration))
        outp = self.tmp_dir / f"gap_{idx:04d}.mp4"
        if d < 1e-3:
            # Nullgap -> 1 Frame schwarz
            d = 1.0 / max(1, fps)

        fade_each = min(0.5, d / 2.0)  # bis 0.5s weiche Blende je Seite
        flt = (
            f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"
            f"fade=t=in:st=0:d={fade_each:.6f},"
            f"fade=t=out:st={(d - fade_each):.6f}:d={fade_each:.6f}[v]"
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

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-t", f"{d:.6f}",
            "-i", "anullsrc=r=48000:cl=stereo",   # Dummy-Audio, wird verworfen
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
        """
        Items = Liste von Pfaden (Szenen- und Gap-Clips) → sauber concatenieren.

        Performance-Optimierung:
          - Kein Re-Encode mehr, sondern `-c copy`.
          - Alle Segmente wurden vorher mit gleichen Settings erzeugt
            (Auflösung, FPS, Codec), daher concatfähig.
        """
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

            # Clipstart wird vorgezogen, wenn fade_in_offset < 0 (außer bei Szene 0 / Intro)
            used_fi_offset = float(fade_in_offset if i > 0 else 0.0)
            pre_extend = -min(0.0, used_fi_offset)  # nur bei negativem Offset > 0
            clip_dur = base + pre_extend  # keine Post-Extension

            # Fade-In beginnt bei scene.start + offset (relativ zu Clipstart)
            fi_start = max(
                0.0,
                (starts[i] + used_fi_offset) - (starts[i] - pre_extend)
            )
            fi_start = clamp(fi_start, 0.0, max(0.0, clip_dur - fade_in))

            # Fade-Out beginnt bei scene.end + offset (relativ zu Clipstart); geclamped innerhalb des Clips
            fo_start_raw = (ends[i] + float(fade_out_offset)) - (starts[i] - pre_extend)
            fo_start = clamp(fo_start_raw, 0.0, max(0.0, clip_dur - fade_out))

            print(
                f"➡️ Szene {i+1}/{n}  ID={sid}  type={stype}  "
                f"base={base:.3f}s  clip_dur={clip_dur:.3f}s  "
                f"fi@{fi_start:.2f}/d{fade_in:.2f}  "
                f"fo@{fo_start:.2f}/d{fade_out:.2f}"
            )

            # ---------- Intro (Szene 0 / type=intro) ----------
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

            # ---------- Outro (type=outro) ----------
            if stype == "outro":
                outro_video = self.base_dir / "outro.mp4"
                outro_image = self.images_dir / "outro.png"
                img = self.images_dir / f"{images_prefix}{sid:04d}.png"

                if outro_video.exists():
                    src = outro_video
                elif outro_image.exists():
                    src = outro_image
                else:
                    src = img  # Fallback auf image_{sid}.png

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
                    idx=i
                )
                scene_clips.append(clip)
                scene_durs.append(clip_dur)
                continue

            # ---------- Normale Szenen ----------
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
                idx=i
            )
            scene_clips.append(clip)
            scene_durs.append(clip_dur)

        # ---------- Gaps mit weichen Blenden (Black-Clips) ----------
        items = []
        for i in range(n):
            items.append(scene_clips[i])

            if i < n - 1:
                # Falls Intro künstlich verlängert wurde → rechne mit JSON-Intro-Länge
                if i == 0 and scenes[i].get("type", "") in {"intro"}:
                    end_i = starts[i] + bases[i]  # bases[0] = JSON intro-length
                else:
                    end_i = ends[i]

                gap_real = max(0.0, starts[i + 1] - end_i)

                # Frühe Einblendung der nächsten Szene verkürzt die Pause real.
                next_in_offset = float(fade_in_offset if (i + 1) > 0 else 0.0)
                gap_eff = max(0.0, gap_real + next_in_offset)
                if gap_eff > 1e-3:
                    gap_clip = self._build_gap_black(gap_eff, width, height, fps, idx=i)
                    items.append(gap_clip)

        print(f"🔎 Merge-Check: Segmente={len(items)}  (Szenen + Gaps)")
        merged = self.output_dir / "_merged_master.mp4"
        self._merge_concat(items, merged)

        # ---------- Overlay (optional) ----------
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
                enc = [
                    "-c:v", "h264_nvenc",
                    "-preset", "p5",
                    "-b:v", "12M",
                    "-pix_fmt", "yuv420p"
                ]
            else:
                enc = [
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p"
                ]

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

        # Wenn kein Overlay benutzt wurde: Video-Stream 1:1 übernehmen (copy)
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
            # Mit Overlay wurde Video bereits neu encodiert → jetzt nur noch muxen
            if self.nvenc_available:
                enc_v = [
                    "-c:v", "h264_nvenc",
                    "-preset", "p5",
                    "-cq", "19",
                    "-b:v", "10M",
                    "-pix_fmt", "yuv420p"
                ]
            else:
                enc_v = [
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p"
                ]
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
        description="Story Renderer v6.3 (JSON-strikt, Intro-HyperTrail, Outro-Support, optimierte Performance)"
    )
    ap.add_argument("--path", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--audiobook", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--quality", choices=["hd", "sd"], default="sd")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.5, help="Fade-In Dauer (s)")
    ap.add_argument("--fade-out", type=float, default=2.0, help="Fade-Out Dauer (s)")
    ap.add_argument("--fade-in-offset", type=float, default=0.0,
                    help="0=Start bei Szene; -1=1s früher usw.")
    ap.add_argument("--fade-out-offset", type=float, default=0.0,
                    help="0=Start bei Szenenende; -1=1s früher usw.")
    ap.add_argument("--overlay", default="particel.mp4", help="Overlay-Video/Bild (optional)")
    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base / "images")
    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")
    output = Path(args.output) if args.output else (base / "story")

    # Overlay-Default: base/overlay.mp4, falls --overlay nicht gesetzt
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
        overlay_file=overlay,
        overlay_opacity=0.35,
        quality=args.quality
    )

if __name__ == "__main__":
    main()
