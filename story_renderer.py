#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v7.4h – Hybrid GPU Edition (NVENC + CPU-Filter)
- Overlay-Video (particel.mp4) aus Base-Path
- Ken-Burns auf Szenen (ohne Intro/Outro)
- Intro mit spätem Blur/Darken + Text 0.2s vor Fade-Out raus
- Outro via outro.mp4 (ohne Fades, ohne Pause)
- Fade-Offsets: 0 = an Scene-Begin/Ende; negativ = früher
"""

import subprocess, json, argparse, shutil
from pathlib import Path

# ---------- helpers ----------

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
    return "" if not s else s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

def clamp(x, lo, hi): return max(lo, min(hi, x))

# ---------- renderer ----------

class StoryRenderer:
    def __init__(self, base_path: Path, images_dir: Path, metadata_path: Path, output_dir: Path):
        self.base_path = Path(base_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "temp_clips"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if not has_nvenc():
            raise RuntimeError("❌ Keine NVIDIA/NVENC-GPU erkannt (CPU-Fallback ist deaktiviert).")
        print("🎞️ NVENC erkannt – Hybrid-GPU aktiv (GPU-Encode, CPU-Filter).")

    def _enc_args(self, target="work"):
        # bewährte, kompatible NVENC-Settings
        base = [
            "-c:v","h264_nvenc",
            "-preset","p7",
            "-rc","vbr",
            "-rc-lookahead","32",
            "-multipass","fullres",
            "-spatial-aq","1","-aq-strength","8",
            "-cq","19","-b:v","15M","-maxrate","25M",
            "-profile:v","high","-pix_fmt","yuv420p"
        ]
        if target == "final":
            return base[:-4] + ["-b:v","10M","-maxrate","18M"]
        return base

    # ---------- builders ----------

    def _render_still_with_fades_and_kenburns(self, img_path: Path, clip_dur: float,
                                              fade_in_start: float, fade_in_dur: float,
                                              fade_out_start: float, fade_out_dur: float,
                                              width:int, height:int, fps:int, idx:int,
                                              kb_strength: float = 0.06, direction: str = "in") -> Path:
        """Ken-Burns via zoompan (CPU), danach Fades; robust, schnell genug mit NVENC."""
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        if img_path.exists():
            inputs = ["-loop","1","-t",f"{clip_dur:.6f}","-r",str(fps),"-i",str(img_path)]
            # zoompan: sanfter Zoom über ganze Clipdauer
            # direction: "in" = leicht rein; "out" = leicht raus; "alt" alternierend
            z = kb_strength
            if direction == "out": z = -kb_strength
            # einfaches Zoom-Modell: zpos von 1±z/2 über Dauer
            # zoompan numeric: zoom=1 + t*rate*(z/D) ~ vereinfachen mit linearer Approx
            # Wir nutzen hier eine simple Variante: leichter Crop + scale back (simuliert KB)
            kb = (
              f"scale={int(width*(1+abs(kb_strength)))}:{int(height*(1+abs(kb_strength)))}:"
              f"force_original_aspect_ratio=increase,"
              f"zoompan=z='if(lte(on,1),1,{1 + (kb_strength if kb_strength>0 else 0)})':"
              f"d={int(clip_dur*fps)}:s={width}x{height}:fps={fps}"
            )
            base = f"[0:v]{kb},format=yuv420p"
        else:
            inputs = ["-f","lavfi","-t",f"{clip_dur:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]
            base = "[0:v]format=yuv420p"

        flt = (f"{base},"
               f"fade=t=in:st={max(0.0,fade_in_start):.6f}:d={max(0.0,fade_in_dur):.6f},"
               f"fade=t=out:st={max(0.0,fade_out_start):.6f}:d={max(0.0,fade_out_dur):.6f}[v]")

        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y", *inputs, "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an", *enc,
               "-t", f"{clip_dur:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _render_video_plain(self, video_path: Path, clip_dur: float,
                            width:int, height:int, fps:int, idx:int) -> Path:
        """Video 1:1 (skaliert/padded), keine Fades."""
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        inputs = ["-ss","0","-t",f"{clip_dur:.6f}","-i",str(video_path)]
        flt = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v]")
        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y", *inputs, "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an", *enc, "-t", f"{clip_dur:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _render_intro(self, intro_src: Path|None, intro_dur: float,
                      width:int, height:int, fps:int,
                      title:str, author:str,
                      text_in_at: float, fade_out: float, fade_out_offset: float) -> Path:
        """
        Intro:
          - bis text_in_at: normal
          - ab text_in_at: 1s Crossfade zu Blur+Dark (CPU-Filter)
          - globales Fade-Out vor Ende
          - Text ab text_in_at ein; 0.2s vor globalem Fade-Out wieder aus
        """
        outp = self.tmp_dir / "intro.mp4"
        d = intro_dur
        t_out_start = clamp(d + fade_out_offset, 0.0, max(0.0, d - fade_out))
        text_out_start = max(0.0, t_out_start - 0.2)
        t1, t2 = esc_txt(title), esc_txt(author)

        if intro_src and intro_src.exists():
            inputs = ["-ss","0","-t",f"{d:.6f}","-i",str(intro_src)]
        else:
            inputs = ["-f","lavfi","-t",f"{d:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]

        flt = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,split[base][blur];"
            f"[blur]trim=start={text_in_at},setpts=PTS-STARTPTS+{text_in_at}/TB,"
            f"gblur=sigma=8,eq=brightness=-0.25[bl];"
            f"[base][bl]xfade=transition=fade:duration=1.0:offset={text_in_at}[intro];"
            f"[intro]fade=t=out:st={t_out_start:.6f}:d={fade_out:.6f}[b1];"
            f"[b1]drawtext=text='{t1}':fontcolor=white:fontsize=40:"
            f"x=(w-text_w)/2:y=(h*0.45-text_h):"
            f"alpha='if(lt(t,{text_in_at}),0, if(lt(t,{text_in_at+1}), (t-{text_in_at})/1, "
            f"if(lt(t,{text_out_start}),1, if(lt(t,{t_out_start}), 1-((t-{text_out_start})/0.2), 0))))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=(h*0.45+text_h+10):"
            f"alpha='if(lt(t,{text_in_at+0.5}),0, if(lt(t,{text_in_at+1.5}), (t-{text_in_at-0.5})/1, "
            f"if(lt(t,{text_out_start}),1, if(lt(t,{t_out_start}), 1-((t-{text_out_start})/0.2), 0))))'[v]"
        )

        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y", *inputs, "-filter_complex", flt,
               "-map","[v]","-r",str(fps), "-an", *enc, "-t", f"{d:.6f}", str(outp)]
        run(cmd, quiet=True)
        return outp

    def _build_gap_black(self, dur: float, width:int, height:int, fps:int, idx:int) -> Path:
        """Weicher Black-Clip als Gap (Fade in/out je bis 0.5s, proportional)."""
        outp = self.tmp_dir / f"gap_{idx:04d}.mp4"
        fade_each = min(0.5, dur/2.0)
        flt = (f"color=c=black:s={width}x{height}:r={fps},"
               f"fade=t=in:st=0:d={fade_each:.6f},"
               f"fade=t=out:st={(dur-fade_each):.6f}:d={fade_each:.6f},format=yuv420p[v]")
        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y","-f","lavfi","-t",f"{dur:.6f}",
               "-i","anullsrc=r=48000:cl=stereo","-filter_complex",flt,
               "-map","[v]","-an",*enc,"-t",f"{dur:.6f}",str(outp)]
        run(cmd, quiet=True)
        return outp

    def _merge_concat(self, clips, out_path: Path):
        concat_list = out_path.parent / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{Path(c).resolve().as_posix()}'\n")

        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y","-f","concat","-safe","0",
               "-i",str(concat_list),*enc,str(out_path)]
        print("🔗 Merge:", " ".join(cmd))
        ok = run(cmd, quiet=False)
        if not ok or not out_path.exists():
            raise RuntimeError(f"❌ Merge fehlgeschlagen – keine Datei '{out_path.name}' erzeugt.")
        return out_path

    # ---------- main render ----------

    def render(self, audiobook_file: Path,
               width=1920, height=1080, fps=30,
               fade_in=1.5, fade_out=2.0,
               fade_in_offset=0.0, fade_out_offset=0.0,
               kb_strength=0.06, kb_direction="in",
               overlay_name="particel.mp4"):

        scenes = self.meta["scenes"]
        title = self.meta.get("title","")
        author = self.meta.get("author","")
        n = len(scenes)
        starts = [float(s["start_time"]) for s in scenes]
        ends   = [float(s["end_time"])   for s in scenes]
        bases  = [max(0.0, ends[i]-starts[i]) for i in range(n)]

        items = []

        for i, s in enumerate(scenes):
            sid = int(s.get("scene_id", i))
            img = self.images_dir / f"image_{sid:04d}.png"
            base = bases[i]

            # Intro
            if i == 0:
                intro_src = self.base_path / "intro.mp4"
                if not intro_src.exists(): intro_src = img
                intro_clip = self._render_intro(intro_src, base, width, height, fps,
                                                title, author, 3.0, fade_out, fade_out_offset)
                items.append(intro_clip)
                continue

            # Outro
            if i == n-1:
                outro_vid = self.base_path / "outro.mp4"
                if not outro_vid.exists():
                    raise FileNotFoundError(f"❌ Outro-Video fehlt: {outro_vid}")
                outro_clip = self._render_video_plain(outro_vid, base, width, height, fps, idx=i)
                items.append(outro_clip)
                break

            # Normale Szene mit früh/spät-Fades + Ken-Burns
            pre_extend = -min(0.0, float(fade_in_offset))
            clip_dur = base + pre_extend
            fi_start = clamp((starts[i] + float(fade_in_offset)) - (starts[i] - pre_extend), 0.0, max(0.0, clip_dur - fade_in))
            fo_start = clamp((ends[i] + float(fade_out_offset)) - (starts[i] - pre_extend), 0.0, max(0.0, clip_dur - fade_out))

            clip = self._render_still_with_fades_and_kenburns(
                img, clip_dur,
                fi_start, fade_in,
                fo_start, fade_out,
                width, height, fps, i,
                kb_strength=kb_strength, direction=kb_direction
            )
            items.append(clip)

            # Gap zum nächsten (weich, verkürzt um frühen IN-Offset der nächsten Szene)
            gap_real = max(0.0, starts[i+1] - ends[i])
            gap_eff = max(0.0, gap_real + float(fade_in_offset))
            if gap_eff > 0.05:
                items.append(self._build_gap_black(gap_eff, width, height, fps, idx=i))

        merged = self.output_dir / "_merged_master.mp4"
        self._merge_concat(items, merged)

        # ---------- Overlay-Video (GPU-encode; Filter CPU, solid) ----------
        overlay_file = self.base_path / overlay_name if overlay_name else None
        visual = merged
        if overlay_file and overlay_file.exists():
            print("✨ Overlay-Video anwenden …")
            ov_out = self.output_dir / "_visual_overlay.mp4"
            # Loop overlay wenn zu kurz; skaliere auf Framegröße; alpha via colorchannelmixer
            flt = (
              f"[0:v]format=yuv420p[base];"
              f"[1:v]scale={width}:{height}:force_original_aspect_ratio=cover,format=rgba,"
              f"colorchannelmixer=aa=0.35[ov];"
              f"[base][ov]overlay=0:0:shortest=1[out]"
            )
            enc = self._enc_args("work")
            cmd = ["ffmpeg","-y","-i",str(merged),"-stream_loop","-1","-i",str(overlay_file),
                   "-filter_complex",flt,"-map","[out]","-an",*enc,str(ov_out)]
            run(cmd, quiet=False)
            visual = ov_out

        # ---------- Audio-Mux ----------
        print("🔊 Muxe Video + Audio …")
        final_hd = self.output_dir / "story_final_hd.mp4"
        enc_final = self._enc_args("final")
        cmd_hd = ["ffmpeg","-y","-fflags","+genpts",
                  "-i",str(visual),"-i",str(audiobook_file),
                  "-map","0:v:0","-map","1:a:0",*enc_final,
                  "-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",str(final_hd)]
        run(cmd_hd, quiet=False)

        # Cleanup
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        print("✅ Fertig:", final_hd)
        return final_hd


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Story Renderer v7.4h – Hybrid GPU (NVENC + CPU-Filter)")
    ap.add_argument("--path", required=True, help="Base-Path (enthält intro.mp4, outro.mp4, particel.mp4)")
    ap.add_argument("--audiobook", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.5)
    ap.add_argument("--fade-out", type=float, default=2.0)
    ap.add_argument("--fade-in-offset", type=float, default=0.0)
    ap.add_argument("--fade-out-offset", type=float, default=0.0)
    ap.add_argument("--kb-strength", type=float, default=0.06, help="Ken-Burns Stärke (0.03–0.12 sinnvoll)")
    ap.add_argument("--kb-direction", choices=["in","out","alt"], default="in")
    ap.add_argument("--overlay", default="particel.mp4", help="Overlay-Video-Dateiname im Base-Path")
    args = ap.parse_args()

    base = Path(args.path)
    audiobook = Path(args.audiobook) if args.audiobook else (base/"audiobook"/"master.wav")
    metadata  = Path(args.metadata)  if args.metadata  else (base/"audiobook"/"audiobook_metadata.json")
    output    = Path(args.output)    if args.output    else (base/"story")

    r = StoryRenderer(base, base/"images", metadata, output)
    r.render(
        audiobook_file=audiobook,
        width=1920, height=1080, fps=args.fps,
        fade_in=args.fade_in, fade_out=args.fade_out,
        fade_in_offset=args.fade_in_offset, fade_out_offset=args.fade_out_offset,
        kb_strength=args.kb_strength, kb_direction=("in" if args.kb_direction!="alt" else "in"),
        overlay_name=args.overlay
    )

if __name__ == "__main__":
    main()
