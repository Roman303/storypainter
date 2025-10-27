#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v6 – JSON-strikt, Intro=Szene 0, frühe Einblendungen via Offsets,
weiche Gap-Blenden (Black-Fades), 1080p30, Auto-GPU.

Offset-Definition (vom Nutzer vorgegeben):
  - fade_in_offset:  0   → Fade beginnt exakt bei scene.start_time
                      <0 → Fade startet früher (z.B. -1 = 1s vor Szenenbeginn)
                      >0 → Fade startet später
  - fade_out_offset: 0   → Fade beginnt exakt bei scene.end_time
                      <0 → Fade startet vor Szenenende (sichtbar im Clip)
                      >0 → Fade startet nach Szenenende (nur sichtbar, wenn <= Clipende)

Wichtig:
  - Clipbeginn wird bei NEGATIVEM fade_in_offset entsprechend vorgezogen,
    damit das Bild vor dem Szenenbeginn schon sichtbar ist.
  - Clipende bleibt bei scene.end_time (keine Post-Extension), damit es zu
    keinen Überlappungen kommt. Ein Fade-Out nach Szeneende (positiver Offset)
    wird geclamped und ist ggf. nicht sichtbar.
  - Gaps zwischen Szenen: gap_eff = (start_{i+1} - end_i) + fade_in_offset_{i+1}, min. 0
    → dadurch verkürzt eine frühe Einblendung der nächsten Szene die Pause real.
"""

import subprocess
from pathlib import Path
import json, argparse, shutil

# ---------- utils ----------
def has_nvenc() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, check=True)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False

def run(cmd, quiet=False):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        print(r.stderr.decode("utf-8", "ignore"))
    return r.returncode == 0

def esc_txt(s: str) -> str:
    if not s: return ""
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

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)

        self.nvenc_available = has_nvenc()
        if self.nvenc_available:
            print("🎞️ GPU (NVENC) erkannt und aktiviert.")
        else:
            print("⚠️ Kein NVENC gefunden – verwende CPU (libx264).")

    @staticmethod
    def _is_video(p: Path): return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    @staticmethod
    def _is_image(p: Path): return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    # ----------- builders -----------
    def _render_still_with_fades(self, img_path: Path, clip_dur: float,
                                 fade_in_start: float, fade_in_dur: float,
                                 fade_out_start: float, fade_out_dur: float,
                                 width:int, height:int, fps:int,
                                 idx:int) -> Path:
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"

        if img_path.exists() and self._is_image(img_path):
            inputs = ["-loop","1","-t",f"{clip_dur:.6f}","-r",str(fps),"-i",str(img_path)]
            base = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")
        else:
            inputs = ["-f","lavfi","-t",f"{clip_dur:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]
            base = "[0:v]format=yuv420p"

        flt = (f"{base},"
               f"fade=t=in:st={max(0.0,fade_in_start):.6f}:d={max(0.0,fade_in_dur):.6f},"
               f"fade=t=out:st={max(0.0,fade_out_start):.6f}:d={max(0.0,fade_out_dur):.6f}[v]")

        enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]
                if self.nvenc_available else
                ["-c:v","libx264","-crf","18","-preset","ultrafast","-pix_fmt","yuv420p"])
        print(f"⚙️ Szene {idx:02d} rendern …")

        cmd = ["ffmpeg","-y", *inputs,
               "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an",
               *enc,
               "-t", f"{clip_dur:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _render_intro(self, intro_src: Path|None, intro_base_dur: float,
                      width:int, height:int, fps:int,
                      title:str, author:str,
                      text_in_at: float,
                      fade_out: float,
                      fade_out_offset: float) -> Path:
        """
        Intro-Design:
          - Quelle skaliert, leicht abgedunkelt + geblurt.
          - Titel/Autor blenden ab 3s ein.
          - Globales Ausblenden vor Ende des Intros (Offset-Logik respektiert).
        """
        # Fade-Out für Intro: Start = end_time + offset → innerhalb des Clips clampen
        intro_clip_dur = intro_base_dur  # wir verlängern Intro nicht nach hinten
        t_out_start = clamp(intro_base_dur + fade_out_offset, 0.0, max(0.0, intro_clip_dur - fade_out))

        outp = self.tmp_dir / "intro_0001.mp4"
        t1, t2 = esc_txt(title), esc_txt(author)

        if intro_src and intro_src.exists():
            if self._is_video(intro_src):
                inputs = ["-ss","0","-t",f"{intro_clip_dur:.6f}","-i",str(intro_src)]
            else:
                inputs = ["-loop","1","-t",f"{intro_clip_dur:.6f}","-r",str(fps),"-i",str(intro_src)]
        else:
            inputs = ["-f","lavfi","-t",f"{intro_clip_dur:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]

        flt = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"
            f"gblur=sigma=8,eq=brightness=-0.25[b];"
            f"[b]fade=t=out:st={t_out_start:.6f}:d={fade_out:.6f}[b1];"
            f"[b1]drawtext=text='{t1}':fontcolor=white:fontsize=40:"
            f"x=(w-text_w)/2:y=(h*0.45-text_h):"
            f"alpha='if(lt(t,{text_in_at}),0, if(lt(t,{text_in_at+1}), (t-{text_in_at})/1, "
            f"if(lt(t,{t_out_start}),1, 1-((t-{t_out_start})/{fade_out}))))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=(h*0.45+text_h+10):"
            f"alpha='if(lt(t,{text_in_at+0.5}),0, if(lt(t,{text_in_at+1.5}), (t-{text_in_at-0.5})/1, "
            f"if(lt(t,{t_out_start}),1, 1-((t-{t_out_start})/{fade_out}))))'[v]"
        )

        enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]
                if self.nvenc_available else
                ["-c:v","libx264","-crf","18","-preset","ultrafast","-pix_fmt","yuv420p"])

        print("🎬 Intro (Szene 0) rendern …")
        cmd = ["ffmpeg","-y", *inputs, "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an", *enc,
               "-t", f"{intro_clip_dur:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _build_gap_black(self, duration: float, width:int, height:int, fps:int, idx:int) -> Path:
        """Schwarzer Zwischenclip mit weichem Fade-In/Out (kurz, proportional zur Länge)."""
        d = max(0.0, float(duration))
        outp = self.tmp_dir / f"gap_{idx:04d}.mp4"
        if d < 1e-3:
            # Nullgap -> 1 Frame schwarz
            d = 1.0 / max(1, fps)

        fade_each = min(0.5, d/2.0)  # bis 0.5s weiche Blende je Seite
        flt = (f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"
               f"fade=t=in:st=0:d={fade_each:.6f},"
               f"fade=t=out:st={(d-fade_each):.6f}:d={fade_each:.6f}[v]")

        enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]
               if self.nvenc_available else
               ["-c:v","libx264","-crf","18","-preset","ultrafast","-pix_fmt","yuv420p"])

        cmd = ["ffmpeg","-y","-f","lavfi","-t",f"{d:.6f}","-i","anullsrc=r=48000:cl=stereo",
               "-filter_complex", flt, "-map","[v]","-an", *enc, "-t", f"{d:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _merge_concat(self, items, out_path: Path):
        """Items = Liste von Pfaden (Szenen- und Gap-Clips) → sauber concatenieren."""
        concat_file = out_path.parent / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in items:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]
               if self.nvenc_available else
               ["-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p"])

        print(f"🔗 Verbinde {len(items)} Segmente (Concat) …")
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat_file), *enc, str(out_path)], quiet=False)
        return out_path

    # ----------- main render -----------
    def render(self, audiobook_file: Path,
               images_prefix="image_",
               width=1920, height=1080, fps=30,
               fade_in=1.5, fade_out=2.0,
               fade_in_offset=0.0, fade_out_offset=0.0,
               overlay_file=None, overlay_opacity=0.35,
               quality="hd"):

        scenes = self.meta.get("scenes", [])
        if not scenes:
            print("❌ Keine Szenen im JSON."); return None

        title = self.meta.get("title","")
        author = self.meta.get("author","")

        n = len(scenes)
        starts = [float(s["start_time"]) for s in scenes]
        ends   = [float(s["end_time"])   for s in scenes]
        bases  = [max(0.0, ends[i]-starts[i]) for i in range(n)]

        scene_clips, scene_durs = [], []

        for i, s in enumerate(scenes):
            sid   = int(s.get("scene_id", i))
            base  = bases[i]

            # Clipstart wird vorgezogen, wenn fade_in_offset < 0
            pre_extend = -min(0.0, float(fade_in_offset if i>0 else 0.0))  # Szene 0 nicht vor t=0 vorziehen
            clip_dur = base + pre_extend  # keine Post-Extension (siehe Header)

            # Fade-In beginnt bei scene.start + offset (relativ zu Clipstart)
            fi_start = max(0.0, (starts[i] + (fade_in_offset if i>0 else 0.0)) - (starts[i] - pre_extend))
            fi_start = clamp(fi_start, 0.0, max(0.0, clip_dur - fade_in))

            # Fade-Out beginnt bei scene.end + offset (relativ zu Clipstart); geclamped innerhalb des Clips
            fo_start_raw = (ends[i] + fade_out_offset) - (starts[i] - pre_extend)
            fo_start = clamp(fo_start_raw, 0.0, max(0.0, clip_dur - fade_out))

            print(f"➡️ Szene {i+1}/{n}  ID={sid}  base={base:.3f}s  clip_dur={clip_dur:.3f}s  "
                  f"fi@{fi_start:.2f} d{fade_in:.2f}  fo@{fo_start:.2f} d{fade_out:.2f}")

            if i == 0:
                # Intro = Szene 0 (nicht vor t=0 ziehen)
                intro_file = (self.output_dir.parent / "intro.mp4")
                img_intro  = self.images_dir / f"{images_prefix}{sid:04d}.png"
                intro_src  = intro_file if intro_file.exists() else (img_intro if img_intro.exists() else None)
                # Für Intro: globales Fade-Out „vor Ende“ via fade_out_offset
                clip = self._render_intro(
                    intro_src=intro_src, intro_base_dur=base,
                    width=width, height=height, fps=fps,
                    title=title, author=author,
                    text_in_at=3.0,
                    fade_out=fade_out-0.2, fade_out_offset=fade_out_offset
                )
                # für Konsistenz die „Clipdauer“ als base (Intro ohne Pre-Extension)
                scene_clips.append(clip); scene_durs.append(base)
            else:
                img = self.images_dir / f"{images_prefix}{sid:04d}.png"
                clip = self._render_still_with_fades(
                    img_path=img, clip_dur=clip_dur,
                    fade_in_start=fi_start, fade_in_dur=fade_in,
                    fade_out_start=fo_start, fade_out_dur=fade_out,
                    width=width, height=height, fps=fps, idx=i
                )
                scene_clips.append(clip); scene_durs.append(clip_dur)

        # ---------- Gaps mit weichen Blenden (Black-Clips) ----------
        items = []
        for i in range(n):
            items.append(scene_clips[i])

            if i < n-1:
                gap_real = max(0.0, starts[i+1] - ends[i])
                # Frühe Einblendung der nächsten Szene verkürzt die Pause.
                next_in_offset = float(fade_in_offset if (i+1) > 0 else 0.0)
                gap_eff = max(0.0, gap_real + next_in_offset)  # bei negativem Offset wird die Pause kürzer
                if gap_eff > 1e-3:
                    gap_clip = self._build_gap_black(gap_eff, width, height, fps, idx=i)
                    items.append(gap_clip)

        print(f"🔎 Merge-Check: segmente={len(items)}  (Szenen + Gaps)")
        merged = self.output_dir / "_merged_master.mp4"
        self._merge_concat(items, merged)

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
        enc = ["-c:v","h264_nvenc","-preset","p5","-cq","19","-b:v","10M","-pix_fmt","yuv420p"] \
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
    ap = argparse.ArgumentParser(description="Story Renderer v6 (JSON-strikt, frühe Einblendungen per Offset, Gap-Fades)")
    ap.add_argument("--path", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--audiobook", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--quality", choices=["hd","sd"], default="sd")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.5, help="Fade-In Dauer (s)")
    ap.add_argument("--fade-out", type=float, default=2.0, help="Fade-Out Dauer (s)")
    ap.add_argument("--fade-in-offset", type=float, default=0.0, help="0=Start bei Szene; -1=1s früher usw.")
    ap.add_argument("--fade-out-offset", type=float, default=0.0, help="0=Start bei Szenenende; -1=1s früher usw.")
    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild (optional)")
    args = ap.parse_args()

    base = Path(args.path)
    images_dir = Path(args.images) if args.images else (base/"images")
    audiobook  = Path(args.audiobook) if args.audiobook else (base/"audiobook"/"complete_audiobook.wav")
    metadata   = Path(args.metadata) if args.metadata else (base/"audiobook"/"audiobook_metadata.json")
    output     = Path(args.output) if args.output else (base/"story")
    overlay    = Path(args.overlay) if args.overlay else None

    if not audiobook.exists():
        print(f"❌ Hörbuch nicht gefunden: {audiobook}"); return
    if not metadata.exists():
        print(f"❌ Metadaten nicht gefunden: {metadata}"); return

    r = StoryRenderer(images_dir, metadata, output)
    r.render(
        audiobook_file=audiobook,
        images_prefix="image_",
        width=1920, height=1080, fps=args.fps,
        fade_in=args.fade_in, fade_out=args.fade_out,
        fade_in_offset=args.fade_in_offset, fade_out_offset=args.fade_out_offset,
        overlay_file=overlay,
        overlay_opacity=0.35,
        quality=args.quality
    )

if __name__ == "__main__":
    main()
