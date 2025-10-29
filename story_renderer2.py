#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v12 – stabile Filterketten, echter Ken‑Burns (zoompan), YUV420p-Images von Anfang an,
robustes Overlay  HD+SD Export, NVENC.
Python 3.10+

Änderungen ggü. v11:
- Behebt AVFilterGraph: "Too many inputs specified for the scale filter" durch saubere, lineare Filterketten
- Vereinfachtes, robustes Scaling (CPU-Scale mit force_original_aspect_ratio + pad/crop → keine NPP-Verzerrungen)
- Ken‑Burns via zoompan: linear, fps-basiert, zentriert (in/out)
- Bilder werden sofort auf Zielgröße (oder größer) gebracht und in yuv420p gewandelt
- Overlay-Datei wird tolerant gefunden: particel.mp4 → particle.mp4 → overlay.mp4
"""

from __future__ import annotations
import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ---------- helpers ----------

def run(cmd, quiet=False):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 and not quiet:
        try:
            print(r.stderr.decode("utf-8", "ignore"))
        except Exception:
            print(str(r.stderr))
    return r.returncode == 0


def has_nvenc() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def esc_txt(s: str) -> str:
    return "" if not s else s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------- filter helpers (nur CPU-scale für korrekte AR) ----------

def scale_letterbox(src_label: str, w: int, h: int) -> str:
    """Aspect erhalten, ggf. Balken: decrease + pad, immer yuv420p."""
    return (
        f"[{src_label}]"
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    )


def scale_cover(src_label: str, w: int, h: int) -> str:
    """Aspect erhalten, bildfüllend: increase + crop, immer yuv420p."""
    return (
        f"[{src_label}]"
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,format=yuv420p"
    )


def fade_inout(fi_st: float, fi_d: float, fo_st: float, fo_d: float) -> str:
    return (
        f"fade=t=in:st={fi_st:.3f}:d={fi_d:.3f},"
        f"fade=t=out:st={fo_st:.3f}:d={fo_d:.3f}"
    )

# ---------- renderer ----------

class StoryRenderer:
    def __init__(self, base_path: Path, images_dir: Path, metadata_path: Path, output_dir: Path,
                 threads: int = 8):
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
        print("🎞️ NVENC aktiv – Renderer v12.")

        # Threads für ffmpeg
        self.ff_threads = str(max(1, int(threads)))
        os.environ.setdefault("OMP_NUM_THREADS", self.ff_threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", self.ff_threads)
        os.environ.setdefault("MKL_NUM_THREADS", self.ff_threads)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", self.ff_threads)
        print(f"🧵 Threads: {self.ff_threads}")

        self.reuse_existing = True

    # ---------- encoder args ----------

    def _enc_args(self, target="work"):
        base = [
            "-threads", self.ff_threads,
            "-filter_threads", self.ff_threads,
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-rc-lookahead", "32",
            "-multipass", "fullres",
            "-cq", "19",
            "-b:v", "12M", "-maxrate", "22M",
            "-spatial-aq", "1", "-aq-strength", "8",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
        ]
        if target == "final":
            return base[:-6] + [
                "-b:v", "10M", "-maxrate", "18M",
                "-pix_fmt", "yuv420p", "-profile:v", "high",
            ]
        return base

    # ---------- building blocks ----------

    def _render_intro(self, intro_src: Path | None, intro_dur: float,
                      width: int, height: int, fps: int,
                      title: str, author: str,
                      text_in_at: float, fade_out_d: float, fade_out_offset: float) -> Path:
        outp = self.tmp_dir / "intro_0000.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp

        d = float(intro_dur)
        t_in = float(text_in_at)
        blur_sw_dur = 1.5
        t_out_start = clamp(d + float(fade_out_offset), 0.0, max(0.0, d - fade_out_d))
        text_out_start = clamp(t_out_start + 0.2, 0.0, max(0.0, d - 0.4))
        text_out_dur = 0.4

        t1, t2 = esc_txt(title), esc_txt(author)

        if intro_src and intro_src.exists():
            inputs = ["-ss", "0", "-t", f"{d:.6f}", "-i", str(intro_src)]
            src_label = "0:v"
        else:
            inputs = ["-f", "lavfi", "-t", f"{d:.6f}", "-i", f"color=c=black:s={width}x{height}:r={fps}"]
            src_label = "0:v"

        sc = scale_letterbox(src_label, width, height)
        flt = (
            f"{sc},setsar=1,split[base][blur];"
            f"[blur]trim=start={t_in},setpts=PTS-STARTPTS+{t_in}/TB,"
            f"gblur=sigma=8,eq=brightness=-0.25[bl];"
            f"[base][bl]xfade=transition=fade:duration={blur_sw_dur}:offset={t_in}[intro];"
            f"[intro]fade=t=out:st={t_out_start:.3f}:d={fade_out_d:.3f}[b1];"
            f"[b1]drawtext=text='{t1}':fontcolor=white:fontsize=40:"
            f"x=(w-text_w)/2:y=(h*0.45-text_h):"
            f"alpha='if(lt(t,{t_in}),0, if(lt(t,{t_in+1}), (t-{t_in})/1, "
            f"  if(lt(t,{text_out_start}),1, if(lt(t,{text_out_start+text_out_dur}),"
            f"    1-((t-{text_out_start})/{text_out_dur}), 0))))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=28:"
            f"x=(w-text_w)/2:y=(h*0.45+text_h+10):"
            f"alpha='if(lt(t,{t_in+0.5}),0, if(lt(t,{t_in+1.5}), (t-{t_in-0.5})/1, "
            f"  if(lt(t,{text_out_start}),1, if(lt(t,{text_out_start+text_out_dur}),"
            f"    1-((t-{text_out_start})/{text_out_dur}), 0))))'[v]"
        )

        enc = self._enc_args("work")
        cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", flt,
               "-map", "[v]", "-r", str(fps), "-an", *enc,
               "-t", f"{d:.6f}", str(outp)]
        run(cmd, quiet=False)
        return outp

    def _render_scene_still(self, img_path: Path, clip_dur: float,
                            fi_st: float, fi_d: float,
                            fo_st: float, fo_d: float,
                            width: int, height: int, fps: int,
                            idx: int, kb_strength: float, kb_dir: str) -> Path:
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp

        if img_path.exists():
            inputs = ["-loop", "1", "-t", f"{clip_dur:.6f}", "-r", str(fps), "-i", str(img_path)]
            src_label = "0:v"

            kb = max(0.0, float(kb_strength))
            frames = max(1, int(round(clip_dur * fps)))

            if kb == 0.0:
                # Kein Zoom → direkt nach HD letterboxen, dann Fades
                chain = (
                    f"{scale_letterbox(src_label, width, height)},"
                    f"setsar=1,{fade_inout(fi_st, fi_d, fo_st, fo_d)}[v]"
                )
            else:
                # Pre-Scale mit Reserve (cover), dann zentrierter linearer zoompan
                pre_w = int(width * (1.0 + kb))
                pre_h = int(height * (1.0 + kb))
                z_expr = (
                    f"min(1.0+{kb:.6f}, 1.0+{kb:.6f}*on/{frames})" if kb_dir == "in" else
                    f"max(1.0, (1.0+{kb:.6f}) - {kb:.6f}*on/{frames})"
                )
                chain = (
                    f"{scale_cover(src_label, pre_w, pre_h)},setsar=1,"
                    f"zoompan=z='{z_expr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':"
                    f"d={frames}:s={width}x{height}:fps={fps},"
                    f"format=yuv420p,{fade_inout(fi_st, fi_d, fo_st, fo_d)}[v]"
                )
        else:
            # Fallback: schwarzer Clip mit Fades
            inputs = ["-f", "lavfi", "-t", f"{clip_dur:.6f}",
                      "-i", f"color=c=black:s={width}x{height}:r={fps}"]
            chain = f"[0:v]{fade_inout(fi_st, fi_d, fo_st, fo_d)},format=yuv420p[v]"

        enc = self._enc_args("work")
        cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", chain,
               "-map", "[v]", "-r", str(fps), "-an", *enc,
               "-t", f"{clip_dur:.6f}", str(outp)]
        run(cmd, quiet=False)
        return outp

    def _render_video_plain(self, video_path: Path, clip_dur: float,
                            width: int, height: int, fps: int, idx: int) -> Path:
        outp = self.tmp_dir / f"scene_{idx:04d}.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp

        inputs = ["-ss", "0", "-t", f"{clip_dur:.6f}", "-i", str(video_path)]
        chain = f"{scale_letterbox('0:v', width, height)}[v]"
        enc = self._enc_args("work")
        cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", chain,
               "-map", "[v]", "-r", str(fps), "-an", *enc,
               "-t", f"{clip_dur:.6f}", str(outp)]
        run(cmd, quiet=False)
        return outp

    def _build_gap_black(self, dur: float, width: int, height: int, fps: int, idx: int) -> Path:
        outp = self.tmp_dir / f"gap_{idx:04d}.mp4"
        if getattr(self, "reuse_existing", False) and outp.exists():
            print(f"⏩ Überspringe bereits gerendert: {outp.name}")
            return outp

        d = max(0.0, float(dur))
        if d < 1e-3:
            d = 1.0 / max(1, fps)
        fe = min(0.5, d / 2.0)
        chain = (
            f"color=c=black:s={width}x{height}:r={fps},"
            f"fade=t=in:st=0:d={fe:.6f},fade=t=out:st={(d-fe):.6f}:d={fe:.6f},format=yuv420p[v]"
        )
        enc = self._enc_args("work")
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-t", f"{d:.6f}", "-i", "anullsrc=r=48000:cl=stereo",
               "-filter_complex", chain, "-map", "[v]", "-an", *enc, "-t", f"{d:.6f}", str(outp)]
        run(cmd, quiet=False)
        return outp

    def _merge_concat(self, clips, out_path: Path):
        concat_list = out_path.parent / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{Path(c).resolve().as_posix()}'\n")
        enc = self._enc_args("work")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), *enc, str(out_path)]
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
               overlay_name="particel.mp4",
               parallel_workers: int = 4):

        scenes = self.meta["scenes"]
        title = self.meta.get("title", "")
        author = self.meta.get("author", "")
        n = len(scenes)
        starts = [float(s["start_time"]) for s in scenes]
        ends = [float(s["end_time"]) for s in scenes]
        bases = [max(0.0, ends[i] - starts[i]) for i in range(n)]

        items = []

        # Intro zuerst
        sid0 = int(scenes[0].get("scene_id", 0))
        intro_img = self.images_dir / f"image_{sid0:04d}.png"
        intro_src = self.base_path / "intro.mp4"
        if not intro_src.exists():
            intro_src = intro_img
        intro_clip = self._render_intro(intro_src, bases[0], width, height, fps,
                                        title, author, 3.0, fade_out, fade_out_offset)
        items.append(intro_clip)

        # Szenen (ohne Outro) parallel rendern
        pool = ThreadPoolExecutor(max_workers=max(1, int(parallel_workers)))
        futures = []
        for i in range(1, n - 1):
            sid = int(scenes[i].get("scene_id", i))
            img = self.images_dir / f"image_{sid:04d}.png"
            base = bases[i]
            pre_extend = -min(0.0, float(fade_in_offset))
            clip_dur = base + pre_extend
            fi_start = clamp((starts[i] + float(fade_in_offset)) - (starts[i] - pre_extend), 0.0, max(0.0, clip_dur - fade_in))
            fo_start = clamp((ends[i] + float(fade_out_offset)) - (starts[i] - pre_extend), 0.0, max(0.0, clip_dur - fade_out))
            kb_dir_eff = kb_direction if kb_direction in ("in", "out") else ("in" if (i % 2 == 0) else "out")

            futures.append(pool.submit(
                self._render_scene_still,
                img, clip_dur, fi_start, fade_in, fo_start, fade_out,
                width, height, fps, i, kb_strength, kb_dir_eff
            ))

        # Gaps (sequentiell)
        gaps = []
        for i in range(1, n - 1):
            gap_real = max(0.0, starts[i + 1] - ends[i])
            gap_eff = max(0.0, gap_real + float(fade_in_offset))
            if gap_eff > 0.05:
                gaps.append(self._build_gap_black(gap_eff, width, height, fps, idx=i))

        for fut in futures:
            items.append(fut.result())
        pool.shutdown(wait=True)

        for g in gaps:
            items.append(g)

        # Outro
        outro_vid = self.base_path / "outro.mp4"
        if not outro_vid.exists():
            raise FileNotFoundError(f"❌ Outro-Video fehlt: {outro_vid}")
        outro_clip = self._render_video_plain(outro_vid, bases[-1], width, height, fps, idx=n - 1)
        items.append(outro_clip)

        # Merge
        merged = self.output_dir / "_merged_master.mp4"
        self._merge_concat(items, merged)

        overlay_file = self.base_path / overlay_name if overlay_name else None
        if overlay_file and not overlay_file.exists():
            print(f"⚠️ Overlay '{overlay_name}' nicht gefunden – ohne Overlay weiter.")
            overlay_file = None
                visual = merged
        if overlay_file:
            print("✨ Overlay anwenden …", overlay_file)
            ov_out = self.output_dir / "_visual_overlay.mp4"
            flt = (
                f"[0:v]format=yuv420p[base];"
                f"[1:v]tpad=stop_mode=clone:stop_duration=36000,"
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,format=rgba,"

                f"colorchannelmixer=aa=0.35[ov];"
                f"[base][ov]overlay=0:0:shortest=0[out]"
            )
            enc = self._enc_args("work")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(merged),
                "-i", str(overlay_file),
                "-filter_complex", flt,
                "-map", "[out]", "-an", *enc, str(ov_out)
            ]
            ok = run(cmd, quiet=False)
            if ok and ov_out.exists():
                visual = ov_out
                print(f"✅ Overlay erfolgreich: {ov_out}")
            else:
                print(f"⚠️ Overlay fehlgeschlagen – fahre ohne Overlay fort (nutze {merged}).")
        else:
            print(f"⚠️ Overlay-Datei nicht gefunden (versucht: {', '.join(str(p.name) for p in overlay_candidates)}) – ohne Overlay weiter.")

        # ---------- FinalMux: HD ----------
        final_hd = self.output_dir / "story_final_hd.mp4"
        enc_final = self._enc_args("final")
        cmd_hd = [
            "ffmpeg", "-y", "-fflags", "+genpts",
            "-i", str(visual), "-i", str(audiobook_file),
            "-map", "0:v:0", "-map", "1:a:0",
            *enc_final,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", "-shortest", str(final_hd)
        ]
        run(cmd_hd, quiet=False)

        # ---------- SD-Export ----------
        final_sd = self.output_dir / "story_final_sd.mp4"
        cmd_sd = [
            "ffmpeg", "-y",
            "-i", str(final_hd),
            "-vf", "scale=640:-2:flags=bicubic",
            "-c:v", "h264_nvenc", "-b:v", "300k", "-maxrate", "320k", "-bufsize", "600k",
            "-preset", "p6",
            "-c:a", "aac", "-b:a", "64k", "-ar", "48000",
            "-movflags", "+faststart",
            str(final_sd)
        ]
        run(cmd_sd, quiet=False)

        print("__ Fertig:", final_hd, "| SD:", final_sd)
        return final_hd, final_sd

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Story Renderer v12 – stabiler Ken‑Burns, Overlay, HD+SD")
    ap.add_argument("--path", required=True, help="Base-Path (intro.mp4, outro.mp4, particel/particle/overlay.mp4)")
    ap.add_argument("--audiobook", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fade-in", type=float, default=1.5)
    ap.add_argument("--fade-out", type=float, default=2.0)
    ap.add_argument("--fade-in-offset", type=float, default=0.0, help="0 = an Szenenbeginn; -1 = 1s früher")
    ap.add_argument("--fade-out-offset", type=float, default=0.0, help="0 = an Szenenende; -1 = 1s früher")
    ap.add_argument("--kb-strength", type=float, default=0.06, help="Ken‑Burns Stärke; 0.0 deaktiviert")
    ap.add_argument("--kb-direction", choices=["in", "out", "alt"], default="in")
    ap.add_argument("--overlay", default="particel.mp4", help="Overlay-Datei-Name im Base-Path")
    ap.add_argument("--threads", type=int, default=8, help="FFmpeg/Filter Threads")
    ap.add_argument("--workers", type=int, default=4, help="Parallele Szenen-Render-Jobs")
    args = ap.parse_args()

    base = Path(args.path)
    audiobook = Path(args.audiobook) if args.audiobook else (base / "audiobook" / "complete_audiobook.wav")
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")
    output = Path(args.output) if args.output else (base / "story")

    r = StoryRenderer(base, base / "images", metadata, output, threads=args.threads)
    r.render(
        audiobook_file=audiobook,
        width=1920, height=1080, fps=args.fps,
        fade_in=args.fade_in, fade_out=args.fade_out,
        fade_in_offset=args.fade_in_offset, fade_out_offset=args.fade_out_offset,
        kb_strength=args.kb_strength,
        kb_direction=("in" if args.kb_direction != "alt" else "in"),
        overlay_name=args.overlay,
        parallel_workers=args.workers,
    )

if __name__ == "__main__":
    main()
