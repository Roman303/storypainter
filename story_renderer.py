#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v9 – Micro-Fades (schnell & präzise)
- Intro: smoother Blur/Darken ab 3.0s (xfade=1.5s), Text 0.2s später & 0.4s aus
- Szenen: Standbild + leichte(n) Ken-Burns (deaktivierbar), Fades mit Offsets
- Gaps: weiche Black-Clips (Offset der kommenden Szene verkürzt Pause)
- Overlay: particel.mp4 in einem Pass
- Outro: outro.mp4 1:1, keine Fades/Pause
- Ziel: story/story_final_hd.mp4
"""

import subprocess, json, argparse, shutil
from pathlib import Path

# ---------- helpers ----------

def has_nvenc() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True)
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

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

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
            raise RuntimeError("❌ Keine NVIDIA/NVENC-GPU erkannt (CPU-Fallback deaktiviert).")
        print("🎞️ NVENC aktiv – Micro-Fades-Renderer.")

    # NVENC args
    def _enc_args(self, target="work"):
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

    # ---------- building blocks ----------

    def _still_scaled(self, src, w, h):
        # scale/pad + yuv420p
        return (f"[{src}]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")

    def _render_scene_still(self, img_path: Path, clip_dur: float,
                            fi_st: float, fi_d: float,
                            fo_st: float, fo_d: float,
                            width: int, height: int, fps: int,
                            idx: int, kb_strength: float, kb_dir: str) -> Path:
        """Statisches Szenenbild – günstige Fades, optional leichter Ken-Burns."""
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp

        if img_path.exists():
            inputs = ["-loop","1","-t",f"{clip_dur:.6f}","-r",str(fps),"-i",str(img_path)]
            # Ken-Burns (deaktivierbar)
            if kb_strength and kb_strength > 0.0:
                k = float(kb_strength)
                # einfacher, performanter KB: leicht skalieren und zurückcroppen
                pre_scale = 1.0 + (k if kb_dir == "in" else 0.0)
                zoom_w = int(width * pre_scale)
                zoom_h = int(height * pre_scale)
                pre = f"scale={zoom_w}:{zoom_h}:force_original_aspect_ratio=increase"
                base = f"[0:v]{pre},{self._still_scaled('v0', width, height).replace('[v0]', '')}"
                # pragmatisch: pre-scale und danach auf Zielgröße bringen
                flt = (f"[0:v]{pre},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,"
                       f"fade=t=in:st={fi_st:.3f}:d={fi_d:.3f},"
                       f"fade=t=out:st={fo_st:.3f}:d={fo_d:.3f}[v]")
            else:
                # kein Ken-Burns → direkt skalieren
                flt = (f"{self._still_scaled('0:v', width, height)},"
                       f"fade=t=in:st={fi_st:.3f}:d={fi_d:.3f},"
                       f"fade=t=out:st={fo_st:.3f}:d={fo_d:.3f}[v]")
        else:
            inputs = ["-f","lavfi","-t",f"{clip_dur:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]
            flt = (f"[0:v]format=yuv420p,"
                   f"fade=t=in:st={fi_st:.3f}:d={fi_d:.3f},"
                   f"fade=t=out:st={fo_st:.3f}:d={fo_d:.3f}[v]")

        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y",*inputs,"-filter_complex",flt,
               "-map","[v]","-r",str(fps),"-an",*enc,"-t",f"{clip_dur:.6f}",str(outp)]
        run(cmd, quiet=True)
        return outp

    def _render_intro(self, intro_src: Path|None, intro_dur: float,
                      width:int, height:int, fps:int,
                      title:str, author:str,
                      text_in_at: float,
                      fade_out: float, fade_out_offset: float) -> Path:
        """Intro: bis t_in klar; ab t_in smoother xfade (1.5s) zu Blur+Dark; Text 0.2s später & 0.4s lang aus."""
        outp = self.tmp_dir / "intro_0000.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp

        d = intro_dur
        t_in = float(text_in_at)
        blur_sw_dur = 1.5  # smoother
        t_out_start = clamp(d + float(fade_out_offset), 0.0, max(0.0, d - fade_out))
        # Text-out: 0.2s später als zuvor (vorher -0.2), jetzt +0.2 in die Fade-Out-Phase hinein
        text_out_start = clamp(t_out_start + 0.2, 0.0, max(0.0, d - 0.4))
        text_out_dur = 0.4

        t1, t2 = esc_txt(title), esc_txt(author)

        if intro_src and intro_src.exists():
            inputs = ["-ss","0","-t",f"{d:.6f}","-i",str(intro_src)]
        else:
            inputs = ["-f","lavfi","-t",f"{d:.6f}",
                      "-i",f"color=c=black:s={width}x{height}:r={fps}"]

        # scale/pad, split → [base][blur]; blur/darken verzögert ab t_in; xfade 1.5s
        flt = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,split[base][blur];"
            f"[blur]trim=start={t_in},setpts=PTS-STARTPTS+{t_in}/TB,"
            f"gblur=sigma=8,eq=brightness=-0.25[bl];"
            f"[base][bl]xfade=transition=fade:duration={blur_sw_dur}:offset={t_in}[intro];"
            f"[intro]fade=t=out:st={t_out_start:.6f}:d={fade_out:.6f}[b1];"
            # Text: Ein ab 3.0s, Aus: +0.2s später starten, 0.4s lang
            f"[b1]drawtext=text='{t1}':fontcolor=white:fontsize=40:"
            f"x=(w-text_w)/2:y=(h*0.45-text_h):"
            f"alpha='if(lt(t,{t_in}),0,"
            f" if(lt(t,{t_in+1.0}), (t-{t_in})/1.0,"
            f"  if(lt(t,{text_out_start}),1.0,"
            f"      if(lt(t,{text_out_start+text_out_dur}), 1.0-((t-{text_out_start})/{text_out_dur}), 0.0))))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=(h*0.45+text_h+10):"
            f"alpha='if(lt(t,{t_in+0.5}),0,"
            f" if(lt(t,{t_in+1.5}), (t-{t_in-0.5})/1.0,"
            f"  if(lt(t,{text_out_start}),1.0,"
            f"      if(lt(t,{text_out_start+text_out_dur}), 1.0-((t-{text_out_start})/{text_out_dur}), 0.0))))'[v]"
        )

        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y",*inputs,"-filter_complex",flt,
               "-map","[v]","-r",str(fps),"-an",*enc,"-t",f"{d:.6f}",str(outp)]
        run(cmd, quiet=True)
        return outp

    def _render_video_plain(self, video_path: Path, clip_dur: float,
                            width:int, height:int, fps:int, idx:int) -> Path:
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp
                    
        inputs = ["-ss","0","-t",f"{clip_dur:.6f}","-i",str(video_path)]
        flt = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v]")
        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y",*inputs,"-filter_complex",flt,
               "-map","[v]","-r",str(fps),"-an",*enc,"-t",f"{clip_dur:.6f}",str(outp)]
        run(cmd, quiet=True)
        return outp

    def _build_gap_black(self, dur: float, width:int, height:int, fps:int, idx:int) -> Path:
        outp = self.tmp_dir / f"gap_{idx:04d}.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp
                    
        d = max(0.0, float(dur))
        if d < 1e-3:
            d = 1.0 / max(1, fps)
        fade_each = min(0.5, d/2.0)
        flt = (f"color=c=black:s={width}x{height}:r={fps},"
               f"fade=t=in:st=0:d={fade_each:.6f},"
               f"fade=t=out:st={(d-fade_each):.6f}:d={fade_each:.6f},format=yuv420p[v]")
        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y","-f","lavfi","-t",f"{d:.6f}","-i","anullsrc=r=48000:cl=stereo",
               "-filter_complex",flt,"-map","[v]","-an",*enc,"-t",f"{d:.6f}",str(outp)]
        run(cmd, quiet=True)
        return outp

    def _merge_concat(self, clips, out_path: Path):
        concat_list = out_path.parent / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{Path(c).resolve().as_posix()}'\n")
        enc = self._enc_args("work")
        cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat_list),*enc,str(out_path)]
        print("🔗 Merge:", " ".join(cmd))
        ok = run(cmd, quiet=False)
        if not ok or not out_path.exists():
            raise RuntimeError(f"❌ Merge fehlgeschlagen – keine Datei '{out_path.name}' erzeugt.")
        return out_path

    # ---------- main render ----------

    def render(
            self,
            audiobook_file: Path,
            width=1920, height=1080, fps=30,
            fade_in=1.5, fade_out=2.0,
            fade_in_offset=0.0, fade_out_offset=0.0,
            kb_strength=0.06, kb_direction="in",
            overlay_name="particel.mp4"
        ):
        # hier innen wird das Attribut gesetzt
        self.reuse_existing = True  # vorhandene Clips wiederverwenden

        scenes = self.meta["scenes"]
        title = self.meta.get("title","")
        author = self.meta.get("author","")
        n = len(scenes)
        starts = [float(s["start_time"]) for s in scenes]
        ends   = [float(s["end_time"])   for s in scenes]
        bases  = [max(0.0, ends[i]-starts[i]) for i in range(n)]

        items = []
        reuse_existing = True  # True = vorhandene Clips wiederverwenden


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

            # Normale Szene (Standbild + Fades, optional Ken-Burns)
            # Pre-Extension (frühere Einblendung) verlängert Clip nach vorne
            pre_extend = -min(0.0, float(fade_in_offset))
            clip_dur = base + pre_extend

            fi_start = clamp((starts[i] + float(fade_in_offset)) - (starts[i] - pre_extend), 0.0, max(0.0, clip_dur - fade_in))
            fo_start = clamp((ends[i] + float(fade_out_offset)) - (starts[i] - pre_extend), 0.0, max(0.0, clip_dur - fade_out))

            kb_dir = kb_direction if kb_direction in ("in","out") else ("in" if (i % 2 == 0) else "out")
            scene_clip = self._render_scene_still(
                img_path=img, clip_dur=clip_dur,
                fi_st=fi_start, fi_d=fade_in,
                fo_st=fo_start, fo_d=fade_out,
                width=width, height=height, fps=fps, idx=i,
                kb_strength=kb_strength, kb_dir=kb_dir
            )
            items.append(scene_clip)

            # Gap (verkürzt um frühen IN-Offset der nächsten Szene)
            if i < n-1:
                gap_real = max(0.0, starts[i+1] - ends[i])
                gap_eff = max(0.0, gap_real + float(fade_in_offset))  # negativer Offset kürzt Gap
                if gap_eff > 0.05:
                    items.append(self._build_gap_black(gap_eff, width, height, fps, idx=i))

        merged = self.output_dir / "_merged_master.mp4"
        self._merge_concat(items, merged)

        # ---------- Overlay (particel.mp4) ----------
        overlay_file = self.base_path / overlay_name if overlay_name else None
        visual = merged
        if overlay_file and overlay_file.exists():
            print("✨ Overlay anwenden …")
            ov_out = self.output_dir / "_visual_overlay.mp4"
        
            # Hier kein stream_loop, sondern tpad verlängert den Overlay-Stream bis 10h falls nötig
            flt = (
                f"[0:v]format=yuv420p[base];"
                f"[1:v]tpad=stop_mode=clone:stop_duration=36000,"
                f"scale={width}:{height}:force_original_aspect_ratio=cover,format=yuv420p,"
                f"colorchannelmixer=aa=0.35[ov];"
                f"[base][ov]overlay=0:0:shortest=0[out]"
            )
        
            enc = self._enc_args("work")
            cmd = [
                "ffmpeg","-y",
                "-i", str(merged),
                "-i", str(overlay_file),
                "-filter_complex", flt,
                "-map","[out]",
                "-an", *enc, str(ov_out)
            ]
            ok = run(cmd, quiet=False)
            if ok and ov_out.exists():
                visual = ov_out
                print(f"✅ Overlay erfolgreich: {ov_out}")
            else:
                print(f"⚠️ Overlay fehlgeschlagen – fahre ohne Overlay fort (nutze {merged}).")
        else:
            print(f"⚠️ Overlay-Video '{overlay_name}' nicht gefunden im Base-Path {self.base_path} – ohne Overlay weiter.")


        # Audio-Mux → story_final_hd.mp4
        final_hd = self.output_dir / "story_final_hd.mp4"
        enc_final = self._enc_args("final")
        cmd_hd = ["ffmpeg","-y","-fflags","+genpts",
                  "-i",str(visual),"-i",str(audiobook_file),
                  "-map","0:v:0","-map","1:a:0",*enc_final,
                  "-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",str(final_hd)]
        run(cmd_hd, quiet=False)

        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        print("__ Fertig:", final_hd)
        return final_hd


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Story Renderer v9 – Micro-Fades (schnell & präzise)")
    ap.add_argument("--path", required=True, help="Base-Path (enthält intro.mp4, outro.mp4, particel.mp4)")
    ap.add_argument("--audiobook", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.5)
    ap.add_argument("--fade-out", type=float, default=2.0)
    ap.add_argument("--fade-in-offset", type=float, default=0.0, help="0=bei Szenenstart; -1=1s früher")
    ap.add_argument("--fade-out-offset", type=float, default=0.0, help="0=bei Szenenende; -1=1s früher")
    ap.add_argument("--kb-strength", type=float, default=0.0, help="Ken-Burns Stärke; 0.0 deaktiviert")
    ap.add_argument("--kb-direction", choices=["in","out","alt"], default="in")
    ap.add_argument("--overlay", default="particel.mp4", help="Overlay-Video im Base-Path")
    args = ap.parse_args()

    base = Path(args.path)
    audiobook = Path(args.audiobook) if args.audiobook else (base/"audiobook"/"complete_audiobook.wav")
    metadata  = Path(args.metadata)  if args.metadata  else (base/"audiobook"/"audiobook_metadata.json")
    output    = Path(args.output)    if args.output    else (base/"story")

    r = StoryRenderer(base, base/"images", metadata, output)
    r.render(
        audiobook_file=audiobook,
        width=1920, height=1080, fps=args.fps,
        fade_in=args.fade_in, fade_out=args.fade_out,
        fade_in_offset=args.fade_in_offset, fade_out_offset=args.fade_out_offset,
        kb_strength=args.kb_strength,
        kb_direction=("in" if args.kb_direction!="alt" else "in"),
        overlay_name=args.overlay
    )

if __name__ == "__main__":
    main()
