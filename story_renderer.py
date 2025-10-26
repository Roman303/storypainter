#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v4 – Gap-gesyncte XFADEs, Fade-Offsets, Intro-Blur/Text, 1080p60, Auto-GPU.

Kernlogik pro Szene i:
  base_dur = end_i - start_i
  gap_before = (i==1 ? head_silence : start_i - end_{i-1})
  gap_after  = (i==n ? tail_silence : start_{i+1} - end_i)
  clip_dur   = gap_before + base_dur + gap_after

  Fade-In:
    fade_in_start = clamp(gap_before - fade_in + fade_in_offset, 0, gap_before - fade_in)
    (endet also spätestens an start_i)

  Fade-Out:
    fade_out_start = clamp(gap_before + base_dur + fade_out_offset,
                           gap_before + base_dur,
                           clip_dur - fade_out)

Merge:
  xfade duration = jeweiliger Gap (Intro→Szene1: head_silence; Szene i→i+1: start_{i+1}-end_i)
  xfade offset   = (prev_clip_duration - xfade_duration)
"""

import subprocess
from pathlib import Path
import json, argparse, shutil

# ---------- utils ----------
def has_nvenc() -> bool:
    try:
        r = subprocess.run(["ffmpeg","-hide_banner","-encoders"],
                           capture_output=True, text=True, check=True)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False

def run(cmd, quiet=False):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        print(r.stderr.decode("utf-8","ignore"))
    return r.returncode == 0

def esc_txt(s: str) -> str:
    if not s: return ""
    return s.replace("\\","\\\\").replace(":","\\:").replace("'","\\'")

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

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        subprocess.run(["ffmpeg","-version"], check=True, capture_output=True)

        self.nvenc_available = has_nvenc()
        if self.nvenc_available:
            print("🎞️ GPU (NVENC) erkannt und aktiviert.")
        else:
            print("⚠️ Kein NVENC gefunden – verwende CPU (libx264).")

    # ----------- helpers -----------
    @staticmethod
    def _is_video(p: Path): return p.suffix.lower() in {".mp4",".mov",".mkv",".avi",".webm"}
    @staticmethod
    def _is_image(p: Path): return p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}

    # ----------- clip builders -----------
    def _render_still_with_fades(self, img_path: Path, clip_dur: float,
                                 fade_in_start: float, fade_in_dur: float,
                                 fade_out_start: float, fade_out_dur: float,
                                 width:int, height:int, fps:int,
                                 zoom0:float, zoom1:float, idx:int) -> Path:
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"

        if img_path.exists() and self._is_image(img_path):
            inputs = ["-loop","1","-t",f"{clip_dur:.6f}","-r",str(fps),"-i",str(img_path)]
            total = max(1, int(round(fps * clip_dur)))
            zlin = f"({zoom0:.6f}+({(zoom1-zoom0):.6f})*on/{max(1,total-1)})"
            zoom = (f"zoom='{zlin}':x='trunc(iw/2 - iw/zoom/2)':"
                    f"y='trunc(ih/2 - ih/zoom/2)':d={total}:s={width}x{height}:fps={fps}")
            base = (f"[0:v]format=rgb24,zoompan={zoom},scale={width}:{height}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")
        else:
            inputs = ["-f","lavfi","-t",f"{clip_dur:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]
            base = "[0:v]format=yuv420p"

        # Fades bei frei wählbaren Startzeiten
        flt = (f"{base},"
               f"fade=t=in:st={max(0.0,fade_in_start):.6f}:d={max(0.0,fade_in_dur):.6f},"
               f"fade=t=out:st={max(0.0,fade_out_start):.6f}:d={max(0.0,fade_out_dur):.6f}[v]")

        cmd = ["ffmpeg","-y", *inputs,
               "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an",
               "-c:v","libx264","-crf","18","-preset","ultrafast",
               "-pix_fmt","yuv420p",
               "-t", f"{clip_dur:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _render_intro(self, intro_src: Path|None, intro_dur: float,
                      width:int, height:int, fps:int,
                      title:str, author:str) -> Path:
        outp = self.tmp_dir / "intro_0001.mp4"
        t_blur_in, blur_dur = 2.0, 2.0
        t1_in, t1_out = 2.0, 8.6
        t2_in, t2_out = 2.5, 8.6
        t1, t2 = esc_txt(title), esc_txt(author)

        if intro_src and intro_src.exists():
            if self._is_video(intro_src):
                inputs = ["-ss","0","-t",f"{intro_dur:.6f}","-i",str(intro_src)]
            else:
                inputs = ["-loop","1","-t",f"{intro_dur:.6f}","-r",str(fps),"-i",str(intro_src)]
        else:
            inputs = ["-f","lavfi","-t",f"{intro_dur:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]

        flt = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,split[base][blur];"
            f"[blur]gblur=sigma=10,eq=brightness=-0.3[bl];"
            f"[base][bl]xfade=transition=fade:duration={blur_dur}:offset={t_blur_in}[intro];"
            f"[intro]drawtext=text='{t1}':fontcolor=white:fontsize=40:"
            f"x=(w-text_w)/2:y=(h*0.45-text_h):"
            f"alpha='if(lt(t,{t1_in}),0, if(lt(t,{t1_in+1}), (t-{t1_in})/1, "
            f"if(lt(t,{t1_out}),1, if(lt(t,{t1_out+1}), 1-((t-{t1_out})/1), 0))))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=(h*0.45+text_h+10):"
            f"alpha='if(lt(t,{t2_in}),0, if(lt(t,{t2_in+1}), (t-{t2_in})/1, "
            f"if(lt(t,{t2_out}),1, if(lt(t,{t2_out+1}), 1-((t-{t2_out})/1), 0))))'[v]"
        )

        cmd = ["ffmpeg","-y", *inputs, "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an",
               "-c:v","libx264","-crf","18","-preset","ultrafast","-pix_fmt","yuv420p",
               "-t", f"{intro_dur:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    # --------- safe xfade merge ----------
    def _merge_xfade(self, clips, durations, fade_times, out_path: Path):
        """Verkettet Clips mit individuellen xfade-Dauern (aus Gaps)."""
        if len(clips) == 1:
            shutil.copy(clips[0], out_path)
            return out_path

        cmd = ["ffmpeg","-y"]
        for c in clips:
            cmd += ["-i", str(c)]

        flt = []
        last = "[0:v]"
        for i in range(1, len(clips)):
            prev_dur = durations[i-1]
            fade_dur = max(0.0, float(fade_times[i-1]))
            offset = max(0.0, prev_dur - fade_dur)
            flt.append(f"{last}[{i}:v]xfade=transition=fade:duration={fade_dur:.6f}:offset={offset:.6f}[v{i}]")
            last = f"[v{i}]"

        enc = ["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"] if self.nvenc_available else \
              ["-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p"]

        cmd += ["-filter_complex",";".join(flt),"-map",last,*enc,str(out_path)]
        print("🔗 Merge mit XFade läuft …")
        run(cmd)
        print("✅ XFade Merge abgeschlossen.")
        return out_path

    # ----------- main render -----------
    def render(self, audiobook_file: Path,
               images_prefix="image_",
               width=1920, height=1080, fps=60,
               zoom_depth=0.06, zoom_direction="in",
               fade_in=1.5, fade_out=2.0,
               fade_in_offset=0.0, fade_out_offset=0.0,
               overlay_file=None, overlay_opacity=0.35,
               quality="hd",
               use_intro=True,  # nutzt head_silence-Intro (mit Text/Blur)
               respect_json_silence=True):

        scenes = self.meta.get("scenes", [])
        if not scenes:
            print("❌ Keine Szenen im JSON."); return None

        head_silence = float(self.meta.get("head_silence", scenes[0].get("start_time",0.0)))
        tail_silence = float(self.meta.get("tail_silence", 0.0))

        parts   = []
        p_durs  = []
        titles  = self.meta.get("title","")
        author  = self.meta.get("author","")

        # ---------- Intro (optional) ----------
        if use_intro and respect_json_silence and head_silence > 0:
            print(f"🎬 Intro (head_silence = {head_silence:.3f}s) …")
            intro_file = Path(self.output_dir.parent / "intro.mp4")
            intro_clip = self._render_intro(intro_file if intro_file.exists() else None,
                                            head_silence, width, height, fps, titles, author)
            parts.append(intro_clip)
            p_durs.append(head_silence)

        # ---------- Szenen bauen ----------
        print("🎬 Szenenclips mit Gap-Padding & Fade-Offsets …")
        scene_clips, scene_durs = [], []
        n = len(scenes)

        # Vorberechnungen
        starts = [float(s["start_time"]) for s in scenes]
        ends   = [float(s["end_time"])   for s in scenes]
        bases  = [ends[i]-starts[i] for i in range(n)]

        for i in range(n):
            sid = int(scenes[i].get("scene_id", i+1))
            base = bases[i]
            gap_before = head_silence if i==0 else (starts[i] - ends[i-1])
            gap_after  = tail_silence  if i==n-1 else (starts[i+1] - ends[i])

            # Clipdauer
            clip_dur = gap_before + base + gap_after

            # Fade-Startzeiten (geclamped in die Gaps)
            fi_start = clamp(gap_before - fade_in + fade_in_offset, 0.0, max(0.0, gap_before - fade_in))
            fo_start = clamp(gap_before + base + fade_out_offset,
                             gap_before + base,
                             max(0.0, clip_dur - fade_out))

            # Zoomrichtung
            if zoom_direction == "in":
                z0, z1 = (1 - zoom_depth/2, 1 + zoom_depth/2)
            elif zoom_direction == "out":
                z0, z1 = (1 + zoom_depth/2, 1 - zoom_depth/2)
            else:
                z0, z1 = ((1 - zoom_depth/2, 1 + zoom_depth/2) if sid % 2 == 0 else (1 + zoom_depth/2, 1 - zoom_depth/2))

            img = self.images_dir / f"{images_prefix}{sid:04d}.png"
            print(f"➡️ Szene {i+1}/{n} (ID {sid})  clip_dur={clip_dur:.3f}s  fi_st={fi_start:.3f}  fo_st={fo_start:.3f}")
            clip = self._render_still_with_fades(
                img_path=img, clip_dur=clip_dur,
                fade_in_start=fi_start, fade_in_dur=fade_in,
                fade_out_start=fo_start, fade_out_dur=fade_out,
                width=width, height=height, fps=fps,
                zoom0=z0, zoom1=z1, idx=i+1
            )
            scene_clips.append(clip); scene_durs.append(clip_dur)
            print(f"✔ Szene {i+1}/{n} fertig.")

        # ---------- Merge via XFADE (Durations = Gaps) ----------
        clips = []
        durs  = []
        fades = []

        # Intro→Szene1
        if parts:
            clips.append(parts[0]); durs.append(p_durs[0])
            gap0 = head_silence
            fades.append(gap0)

        # Szene i → Szene i+1
        for i in range(n):
            clips.append(scene_clips[i]); durs.append(scene_durs[i])
            if i < n-1:
                fades.append(starts[i+1] - ends[i])

        merged = self.output_dir / "_merged_master.mp4"
        self._merge_xfade(clips, durs, fades, merged)

        # ---------- Overlay (optional) ----------
        visual = merged
        if overlay_file and Path(overlay_file).exists():
            print("✨ Overlay anwenden …")
            ov_out = self.output_dir / "_visual_overlay.mp4"
            ov_inputs = (["-stream_loop","-1","-i",str(overlay_file)]
                         if Path(overlay_file).suffix.lower() in {".mp4",".mov",".mkv",".avi",".webm"}
                         else ["-loop","1","-r",str(fps),"-i",str(overlay_file)])
            enc = ["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"] \
                  if self.nvenc_available else \
                  ["-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p"]
            cmd = ["ffmpeg","-y","-i",str(merged), *ov_inputs,
                   "-filter_complex",
                   f"[0:v]format=yuv420p[base];"
                   f"[1:v]scale={width}:{height},format=rgba,colorchannelmixer=aa={overlay_opacity:.3f}[ovr];"
                   f"[base][ovr]overlay=0:0:shortest=1[out]",
                   "-map","[out]","-an",*enc,str(ov_out)]
            run(cmd, quiet=True)
            visual = ov_out

        # ---------- Audio-Mux ----------
        print("🔊 Muxe Video + Audio …")
        final_hd = self.output_dir / "story_final_hd.mp4"
        enc = ["-c:v","h264_nvenc","-preset","p5","-cq","19","-b:v","12M","-pix_fmt","yuv420p"] \
              if self.nvenc_available else \
              ["-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p"]
        cmd_hd = ["ffmpeg","-y","-fflags","+genpts","-i",str(visual),"-i",str(audiobook_file),
                  "-map","0:v:0","-map","1:a:0",*enc,
                  "-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",str(final_hd)]
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
            run(["ffmpeg","-y","-i",str(final_hd),
                 "-vf","scale=640:360:force_original_aspect_ratio=decrease,fps=30",
                 "-c:v","libx264","-b:v","600k","-c:a","aac","-b:a","96k",
                 "-movflags","+faststart",str(final_sd)], quiet=True)

        print("✅ Fertig:", final_hd)
        return final_hd


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Story Renderer v4 (Gap-XFade, Fade-Offsets, Intro Blur/Text, GPU)")
    ap.add_argument("--path", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--audiobook", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--quality", choices=["hd","sd"], default="hd")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--zoom-depth", type=float, default=0.06)
    ap.add_argument("--zoom-direction", choices=["in","out","alt"], default="in")
    ap.add_argument("--fade-in", type=float, default=1.5, help="Fade-In Dauer (s)")
    ap.add_argument("--fade-out", type=float, default=2.0, help="Fade-Out Dauer (s)")
    ap.add_argument("--fade-in-offset", type=float, default=0.0, help="In-Offset (s, + später)")
    ap.add_argument("--fade-out-offset", type=float, default=0.0, help="Out-Offset (s, + später)")
    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild (optional)")
    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base/"images")
    audiobook  = Path(args.audiobook) if args.audiobook else (base/"audiobook"/"complete_audiobook.wav")
    metadata   = Path(args.metadata) if args.metadata else (base/"audiobook"/"audiobook_metadata.json")
    output     = Path(args.output) if args.output else (base/"story")
    overlay    = Path(args.overlay) if args.overlay else (base/"particel.mp4")

    if not audiobook.exists():
        print(f"❌ Hörbuch nicht gefunden: {audiobook}"); return
    if not metadata.exists():
        print(f"❌ Metadaten nicht gefunden: {metadata}"); return

    r = StoryRenderer(images_dir, metadata, output)
    r.render(
        audiobook_file=audiobook,
        images_prefix="image_",
        width=1920, height=1080, fps=args.fps,
        zoom_depth=args.zoom_depth, zoom_direction=args.zoom_direction,
        fade_in=args.fade_in, fade_out=args.fade_out,
        fade_in_offset=args.fade_in_offset, fade_out_offset=args.fade_out_offset,
        overlay_file=overlay if overlay.exists() else None,
        overlay_opacity=0.35,
        quality=args.quality,
        use_intro=True, respect_json_silence=True
    )

if __name__ == "__main__":
    main()
