#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer – IMG • WAV • GPU • VIDEO (NVENC)

Builds a full video from a JSON storyboard, a folder of images, an audiobook WAV,
optionally an intro/outro clip and a looping particle overlay. Designed for NVIDIA GPUs.

Key features
- Smooth Ken‑Burns style zoom for images (zoom in/out/alternate) with frame-accurate timing
- Nice blur-dim fade-in/out for images (composited sharp over blurred base)
- Intro scene: blur/dim from t=2 → t=5, title+author drawtext with fade in/out
- Outro scene: play as-is with no fades
- Overlay video looped across entire runtime
- Audiobook WAV aligned from t=0, trimmed to total runtime
- Overlap safety check for fade offsets across adjacent scenes (json & CLI)
- GPU encoding via h264_nvenc or hevc_nvenc; high quality defaults
- Optional SD downscale (450px wide, ~300k video bitrate)

Inputs & layout
base_path/
  ├─ images/                # image_0001.png, image_0002.png, ... matching scene ids
  ├─ audiobook/
  │    ├─ complete_audiobook.wav
  │    └─ audiobook_metadata.json  # storyboard
  ├─ intro.mp4              # optional, used if present or via --intro-video
  ├─ outro.mp4              # optional, used if present or via --outro-video
  └─ particel.mp4           # optional looping overlay, or override via --overlay

Usage example
python story_renderer_nvenc.py \
  --path /path/to/project \
  --quality hd --fps 30 \
  --zoom-depth 0.06 --zoom-direction in \
  --fade-in 1.5 --fade-out 2.0 \
  --fade-in-offset 0.0 --fade-out-offset 0.0

Notes
- Requires ffmpeg with NVENC support (nvidia drivers & ffmpeg compiled with --enable-nonfree --enable-nvenc).
- Timing model defaults to absolute start/end in JSON; will also work if only durations are provided.
- If per-scene fade offsets exist in JSON (scene.fade_in_offset / scene.fade_out_offset), they override CLI.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

# --------------------------- helpers ---------------------------

def run(cmd: str):
    print("\n[ffmpeg]", cmd)
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def time_s(val):
    return float(val)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# --------------------------- core ---------------------------

def build():
    ap = argparse.ArgumentParser(description="Story Renderer IMG WAV GPU VIDEO")
    ap.add_argument("--path", required=True)
    ap.add_argument("--quality", choices=["hd","sd"], default="hd")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zoom-depth", type=float, default=0.06)
    ap.add_argument("--zoom-direction", choices=["in","out","alt"], default="in")
    ap.add_argument("--fade-in", type=float, default=1.5, help="Fade-In Dauer (s)")
    ap.add_argument("--fade-out", type=float, default=2.0, help="Fade-Out Dauer (s)")
    ap.add_argument("--fade-in-offset", type=float, default=0.0, help="In-Offset (s, + später)")
    ap.add_argument("--fade-out-offset", type=float, default=0.0, help="Out-Offset (s, + später)")
    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild (optional)")
    ap.add_argument("--intro-video", default=None)
    ap.add_argument("--outro-video", default=None)
    ap.add_argument("--encoder", choices=["h264","hevc"], default="h264")
    ap.add_argument("--pixfmt", default="yuv420p")
    ap.add_argument("--crf", type=int, default=19, help="Target quality for NVENC CQ mode (h264 only)")
    args = ap.parse_args()

    base = Path(args.path)
    images_dir = base / "images"
    audiobook = base / "audiobook" / "complete_audiobook.wav"
    metadata  = base / "audiobook" / "audiobook_metadata.json"
    output_dir = base / "story"
    overlay_default = base / "particel.mp4"

    intro_video = Path(args.intro_video) if args.intro_video else (base / "intro.mp4")
    outro_video = Path(args.outro_video) if args.outro_video else (base / "outro.mp4")
    overlay_path = Path(args.overlay) if args.overlay else overlay_default

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON storyboard
    with open(metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    title = meta.get("title", "")
    author = meta.get("author", "")
    scenes = meta.get("scenes", [])
    pause_duration = float(meta.get("pause_duration", 0))
    timestamps_base = meta.get("timestamps_base", "absolute")

    # Resolution & bitrates
    if args.quality == "hd":
        width, height = 1920, 1080
        v_bitrate = "8M"  # conservative, NVENC CQ handles quality
    else:
        width, height = 450, -2  # keep aspect
        v_bitrate = "300k"

    fps = args.fps

    # NVENC encoder opts (balanced quality)
    if args.encoder == "h264":
        vcodec = "h264_nvenc"
        # p5 slow (good quality), vbr_hq with cq target
        vopts = f"-rc vbr_hq -cq {args.crf} -preset p5 -b:v {v_bitrate} -maxrate {v_bitrate} -bufsize {v_bitrate}"
    else:
        vcodec = "hevc_nvenc"
        vopts = f"-rc vbr_hq -cq 21 -preset p5 -b:v {v_bitrate} -maxrate {v_bitrate} -bufsize {v_bitrate}"

    # validate images exist
    def img_path_for(scene_id: int) -> Path:
        return images_dir / f"image_{scene_id:04d}.png"

    # Overlap safety (using absolute times if provided)
    if timestamps_base == "absolute":
        for i in range(len(scenes)-1):
            a = scenes[i]
            b = scenes[i+1]
            # per-scene overrides
            a_fin_off = float(a.get("fade_in_offset", args.fade_in_offset))
            a_fout_off = float(a.get("fade_out_offset", args.fade_out_offset))
            b_fin_off = float(b.get("fade_in_offset", args.fade_in_offset))
            # Compute fade windows (start of alpha ramp)
            a_fade_out_start = a["end_time"] + a_fout_off - args.fade_out
            b_fade_in_start  = b["start_time"] + b_fin_off
            if a_fade_out_start > b_fade_in_start:
                raise SystemExit(
                    f"\n[ERROR] Fade overlap zwischen Szene {a['scene_id']} und {b['scene_id']} erkannt. "
                    f"out_start={a_fade_out_start:.3f}s > next_in_start={b_fade_in_start:.3f}s. "
                    "Passe fade offsets oder Pausen an."
                )

    # temp directory for segments
    work = Path(tempfile.mkdtemp(prefix="story_nvenc_"))
    segments = []

    # Helper to create a color pause segment
    def build_pause(dur: float, label: str):
        if dur <= 0:
            return None
        out = work / f"pause_{label}.mp4"
        cmd = (
            f"ffmpeg -y -f lavfi -t {dur:.3f} -i color=c=black:s={width}x{height}:r={fps} "
            f"-c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(out))}"
        )
        run(cmd)
        return out

    # Build INTRO if present in JSON (type:intro)
    intro_scene = next((s for s in scenes if s.get("type") == "intro"), None)
    if intro_scene and intro_video.exists():
        d = float(intro_scene.get("duration", intro_scene.get("end_time", 0) - intro_scene.get("start_time", 0)))
        out = work / f"scene_intro.mp4"
        # Title/author positions (title slightly above center)
        title_size = 64
        author_size = 38
        
        # Alpha ramps for title/author
        # title: fade in at 2.5s over 0.7s; fade out from 8.5s over 0.7s
        t_in, t_d, t_out, t_od = 2.5, 0.7, 8.5, 0.7
        # author starts 0.3s later
        a_in, a_d, a_out, a_od = t_in+0.3, 0.7, t_out, 0.7

        # Intro blur/dim from t=2 → t=5
        blur_st, blur_en = 2.0, 5.0

        # Filtergraph
        draw_title = (
            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text={shlex.quote(title)}:"
            f"fontsize={title_size}:fontcolor=white:shadowcolor=black:shadowx=2:shadowy=2:"
            f"x=(w-text_w)/2:y=h*0.36:"
            f"alpha='if(lt(t,{t_in}),0, if(lt(t,{t_in+t_d}),(t-{t_in})/{t_d}, if(lt(t,{t_out} ),1, if(lt(t,{t_out+t_od} ), 1-(t-{t_out})/{t_od}, 0))))'"
        )
        draw_author = (
            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text={shlex.quote(author)}:"
            f"fontsize={author_size}:fontcolor=white:shadowcolor=black:shadowx=2:shadowy=2:"
            f"x=(w-text_w)/2:y=h*0.46:"
            f"alpha='if(lt(t,{a_in}),0, if(lt(t,{a_in+a_d}),(t-{a_in})/{a_d}, if(lt(t,{a_out} ),1, if(lt(t,{a_out+a_od} ), 1-(t-{a_out})/{a_od}, 0))))'"
        )
        # Base chain: ensure duration, scale, blur/dim window via blend over blurred copy
        vf = (
            f"[0:v]scale={width}:{height},trim=duration={d:.3f},setpts=PTS-STARTPTS[base];"
            f"[base]boxblur=20:1,eq=brightness=-0.25,trim=duration={d:.3f},setpts=PTS-STARTPTS[blur];"
            f"[base]format=yuva420p,fade=in:st=0:d=0.001:alpha=1,fade=out:st=0.001:d=0.001:alpha=1[alphabase];"  # keep alpha track
            f"[blur][alphabase]overlay=format=auto,"
            f"eq=brightness='if(between(t,{blur_st},{blur_en}),-0.2,0)'[pretext];"
            f"[pretext]{draw_title}[titled];[titled]{draw_author}[vout]"
        )
        cmd = (
            f"ffmpeg -y -i {shlex.quote(str(intro_video))} -filter_complex {shlex.quote(vf)} "
            f"-map [vout] -r {fps} -c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(out))}"
        )
        run(cmd)
        segments.append(out)
        # add pause after intro if JSON implies (we trust meta.pause_duration)
        p = build_pause(pause_duration, "after_intro")
        if p: segments.append(p)
    elif intro_scene:
        raise SystemExit("Intro-Szene im JSON vorhanden, aber intro.mp4 nicht gefunden. Nutze --intro-video …")

    # Build "scene" images
    for s in scenes:
        if s.get("type") != "scene":
            continue
        sid = int(s["scene_id"])  # must match image id
        d = float(s.get("duration", s.get("end_time", 0) - s.get("start_time", 0)))
        frames = int(round(d * fps))
        img = img_path_for(sid)
        if not img.exists():
            raise SystemExit(f"Bild fehlt: {img}")

        fin = float(s.get("fade_in", args.fade_in))
        fout = float(s.get("fade_out", args.fade_out))
        fin_off = float(s.get("fade_in_offset", args.fade_in_offset))
        fout_off = float(s.get("fade_out_offset", args.fade_out_offset))

        # Effective fade start times within the scene timeline (0..d)
        fin_start = max(0.0, 0.0 + fin_off)
        fout_start = max(0.0, d + fout_off - fout)
        fout_start = min(fout_start, max(0.0, d - 0.001))  # clamp

        # Zoom direction
        dir_mode = args.zoom_direction
        if dir_mode == "alt":
            dir_mode = "in" if (sid % 2 == 1) else "out"
        depth = float(s.get("zoom_depth", args.zoom_depth))
        z_start = 1.0 if dir_mode == "in" else (1.0 + depth)
        z_end   = (1.0 + depth) if dir_mode == "in" else 1.0
        dz_per_frame = (z_end - z_start) / max(1, frames)

        out = work / f"scene_{sid:04d}.mp4"

        # Filtergraph for Ken Burns + blur-dim fades (sharp over blurred base)
        # Build a frame sequence from the still using fps & frames control
        # Use zoompan for stable per-frame zoom; center framing
        zoompan = (
            f"zoompan=z='if(eq(on,1),{z_start},zoom+{dz_per_frame:.10f})':"
            f"d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )
        vf = (
            f"[0:v]{zoompan},format=yuv420p[sharp];"
            f"[sharp]boxblur=20:1,eq=brightness=-0.25[blur];"
            # fade the SHARP layer's alpha in/out, then overlay on blurred base
            f"[sharp]format=yuva420p,"
            f"fade=t=in:st={fin_start:.3f}:d={fin:.3f}:alpha=1,"
            f"fade=t=out:st={fout_start:.3f}:d={fout:.3f}:alpha=1[sharpfade];"
            f"[blur][sharpfade]overlay=format=auto[vout]"
        )

        cmd = (
            f"ffmpeg -y -loop 1 -t {d:.3f} -i {shlex.quote(str(img))} "
            f"-filter_complex {shlex.quote(vf)} -map [vout] -r {fps} "
            f"-c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(out))}"
        )
        run(cmd)
        segments.append(out)

        # pause after each scene except the last before outro (we'll add outro below)
        p = build_pause(pause_duration, f"after_{sid:04d}")
        if p: segments.append(p)

    # OUTRO
    outro_scene = next((s for s in scenes if s.get("type") == "outro"), None)
    if outro_scene and outro_video.exists():
        d = float(outro_scene.get("duration", outro_scene.get("end_time", 0) - outro_scene.get("start_time", 0)))
        out = work / f"scene_outro.mp4"
        cmd = (
            f"ffmpeg -y -i {shlex.quote(str(outro_video))} -an "
            f"-vf scale={width}:{height},trim=duration={d:.3f},setpts=PTS-STARTPTS "
            f"-r {fps} -c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(out))}"
        )
        run(cmd)
        segments.append(out)
    elif outro_scene:
        raise SystemExit("Outro-Szene im JSON vorhanden, aber outro.mp4 nicht gefunden. Nutze --outro-video …")

    # Concatenate all segments (video only), then add audio + overlay in a final pass
    concat_list = work / "segments.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"file '{seg.as_posix()}'\n")

    concat_out = work / "video_concat.mp4"
    run(
        f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(concat_list))} "
        f"-c copy {shlex.quote(str(concat_out))}"
    )

    # Determine total duration from JSON if present, else probe concat file
    total_duration = float(meta.get("total_duration", 0))
    if total_duration <= 0:
        # probe using ffprobe
        import json as _json
        p = subprocess.run(
            f"ffprobe -v error -show_entries format=duration -of json {shlex.quote(str(concat_out))}",
            shell=True, capture_output=True, text=True
        )
        try:
            total_duration = float(_json.loads(p.stdout)["format"]["duration"]) if p.returncode == 0 else 0
        except Exception:
            total_duration = 0

    # Final assembly: add audiobook WAV (trim), loop overlay, re-encode once with NVENC
    final_out = (output_dir / ("story_hd.mp4" if args.quality == "hd" else "story_sd.mp4")).as_posix()

    overlay_input = ""
    overlay_chain = "[basev]copy[v0]"  # default passthrough

    if overlay_path.exists():
        overlay_input = f"-stream_loop -1 -i {shlex.quote(str(overlay_path))} "
        # Scale overlay to canvas and alpha blend subtly; loop then trim
        overlay_chain = (
            f"[1:v]scale={width}:{height},format=rgba,trim=duration={total_duration:.3f},setpts=PTS-STARTPTS[ol];"
            f"[0:v]format=rgba[basev];"
            f"[basev][ol]overlay=shortest=1,format=yuv420p[v0]"
        )

    # Audio trim
    audio_input = f"-i {shlex.quote(str(audiobook))}"

    vf = overlay_chain

    # Build final command
    final_cmd = (
        f"ffmpeg -y -i {shlex.quote(str(concat_out))} {overlay_input}{audio_input} "
        f"-filter_complex {shlex.quote(vf)} -map [v0] -map 2:a:0 "
        f"-c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} -c:a aac -b:a 160k -shortest "
        f"{shlex.quote(final_out)}"
    )
    run(final_cmd)

    # Optional SD transcode if requested (and not already sd)
    if args.quality == "sd":
        # Already SD. Ensure bitrate and width are correct just in case
        sd_out = output_dir / "story_sd.mp4"
        run(
            f"ffmpeg -y -i {shlex.quote(final_out)} -vf scale=450:-2 -c:v {vcodec} -b:v 300k -maxrate 300k -bufsize 300k "
            f"-c:a aac -b:a 96k -pix_fmt {args.pixfmt} {shlex.quote(str(sd_out))}"
        )

    print(f"\n✅ Fertig. Ausgabe: {final_out}")


if __name__ == "__main__":
    try:
        build()
    except Exception as e:
        print(f"\n✖ Fehler: {e}")
        sys.exit(1)
