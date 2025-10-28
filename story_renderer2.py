#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer – IMG • WAV • GPU • VIDEO (NVENC)

- NVIDIA NVENC (H.264/HEVC), YouTube-kompatibel (yuv420p, 48 kHz AAC Stereo)
- Sanfter Ken-Burns-Zoom (abschaltbar via --no-zoom)
- Blur-Fade (scharfes Bild als Alpha über geblurter, abgedunkelter Kopie)
- Intro (mit Titel/Autor-Fades) -> Pausen -> Bild-Szenen -> Outro (ohne Fades)
- Overlay-Video stabil geloopt (CPU-Reencode)
- Automatische Audio-Korrektur auf 48 kHz Stereo
- Parallel-Rendering der Szenen (konfigurierbar)
- Automatische NVENC-Limit-Erkennung und CPU-Fallback
- SD-Option (450x252, ~300k)
- Arbeitsordner bleibt erhalten (kein Auto-Delete)
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------- helpers ---------------------------

def run(cmd: str):
    print(f"[run] {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

def ffprobe_audio(path: Path):
    import json as _json
    p = subprocess.run(
        f"ffprobe -v error -show_streams -select_streams a:0 -of json {shlex.quote(str(path))}",
        shell=True, capture_output=True, text=True
    )
    try:
        info = _json.loads(p.stdout)["streams"][0]
        sr = int(info.get("sample_rate", 0))
        ch = int(info.get("channels", 0))
        return sr, ch
    except Exception:
        return 0, 0

def ensure_audio_48k_stereo(in_path: Path) -> Path:
    sr, ch = ffprobe_audio(in_path)
    if sr == 48000 and ch == 2:
        print(f"[audio] OK: {sr} Hz, {ch} ch -> unverändert")
        return in_path
    fixed = Path(tempfile.gettempdir()) / "audiobook_fixed.wav"
    cmd = (
        f"ffmpeg -y -i {shlex.quote(str(in_path))} -ar 48000 -ac 2 -c:a pcm_s16le "
        f"{shlex.quote(str(fixed))}"
    )
    run(cmd)
    print(f"[audio] converted -> 48 kHz stereo: {fixed}")
    return fixed

def safe_write_concat_list(path: Path, segments: list[Path]):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            safe = str(seg).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    print(f"[concat list] {path}")

# -------------------- NVENC limit + fallback --------------------

def detect_nvenc_limit() -> int:
    try:
        p = subprocess.run("nvidia-smi -q -x", shell=True, capture_output=True, text=True, timeout=2)
        if p.returncode == 0 and "Encoder" in p.stdout:
            import re
            m = re.search(r"<Encoder>\s*<SessionCount>(\d+)</SessionCount>", p.stdout)
            if m:
                sc = int(m.group(1))
                if sc > 0:
                    return max(1, sc)
    except Exception:
        pass
    return 2

def should_fallback_to_cpu(stderr_text: str) -> bool:
    err = (stderr_text or "").lower()
    keys = [
        "openencodesessionex failed",
        "incompatible client key",
        "no capable devices found",
        "no nvenc capable devices",
        "nvenc_initialise_encoder"
    ]
    return any(k in err for k in keys)

def encode_cmd_with_fallback(cmd_nvenc: str, cmd_cpu: str):
    print("[encode] Versuch mit NVENC ...")
    proc = subprocess.run(cmd_nvenc, shell=True, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode == 0:
        return
    if should_fallback_to_cpu(proc.stderr):
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        print("[encode] NVENC nicht verfügbar oder Limit erreicht -> Fallback CPU (libx264 ultrafast)")
        run(cmd_cpu)
    else:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

# --------------------------- core ---------------------------

def build():
    ap = argparse.ArgumentParser(description="Story Renderer IMG WAV GPU VIDEO")
    ap.add_argument("--path", required=True)
    ap.add_argument("--quality", choices=["hd","sd"], default="hd")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zoom-depth", type=float, default=0.06)
    ap.add_argument("--zoom-direction", choices=["in","out","alt"], default="in")
    ap.add_argument("--fade-in", type=float, default=1.5)
    ap.add_argument("--fade-out", type=float, default=2.0)
    ap.add_argument("--fade-in-offset", type=float, default=0.0)
    ap.add_argument("--fade-out-offset", type=float, default=0.0)
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--intro-video", default=None)
    ap.add_argument("--outro-video", default=None)
    ap.add_argument("--encoder", choices=["h264","hevc"], default="h264")
    ap.add_argument("--pixfmt", default="yuv420p")
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)//3))
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

    # JSON laden
    with open(metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scenes = meta.get("scenes", [])
    pause_duration = float(meta.get("pause_duration", 0))

    # Video-Parameter
    if args.quality == "hd":
        width, height = 1920, 1080
        v_bitrate = "8M"
    else:
        width, height = 450, 252
        v_bitrate = "300k"

    fps = args.fps
    if args.encoder == "h264":
        vcodec = "h264_nvenc"
        vopts = f"-rc vbr -cq {args.crf} -preset slow -b:v {v_bitrate}"
    else:
        vcodec = "hevc_nvenc"
        vopts = f"-rc vbr -cq 21 -preset slow -b:v {v_bitrate}"

    cpu_vcodec = "libx264"
    cpu_vopts = "-preset ultrafast -crf 18"

    ff_threads = max(2, (os.cpu_count() or 4)//2)
    ff_filter_threads = max(2, ff_threads//2)
    ff_thread_flags = f"-threads {ff_threads} -filter_complex_threads {ff_filter_threads}"

    nv_limit = detect_nvenc_limit()
    if args.workers > nv_limit:
        print(f"[NVENC] limit: {args.workers} -> {nv_limit}")
        args.workers = nv_limit

    work = Path(tempfile.mkdtemp(prefix="story_nvenc_"))
    print(f"[workdir] {work}")

    def build_pause(dur: float, label: str):
        if dur <= 0:
            return None
        out = work / f"pause_{label}.mp4"
        cmd = (
            f"ffmpeg -y {ff_thread_flags} -f lavfi -t {dur:.3f} -i color=c=black:s={width}x{height}:r={fps} "
            f"-c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(out))}"
        )
        encode_cmd_with_fallback(cmd, cmd.replace(f"-c:v {vcodec}", f"-c:v {cpu_vcodec} {cpu_vopts}"))
        return out

    # -------------------- Szenen rendern --------------------
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for s in scenes:
            if s.get("type") == "scene":
                sid = int(s["scene_id"])
                img = images_dir / f"image_{sid:04d}.png"
                if not img.exists():
                    raise SystemExit(f"Bild fehlt: {img}")
                d = float(s.get("duration", 3))
                out = work / f"scene_{sid:04d}.mp4"
                zoom = "" if args.no_zoom else f"zoompan=z='1+0.00005*on':d=1:s={width}x{height}:fps={fps},"
                vf = (
                    f"[0:v]{zoom}format=yuv420p,fade=t=in:st=0:d=1:alpha=1,"
                    f"fade=t=out:st={max(0.1, d-1):.3f}:d=1:alpha=1[vout]"
                )
                cmd_nv = (
                    f"ffmpeg -y {ff_thread_flags} -loop 1 -t {d:.3f} -i {shlex.quote(str(img))} "
                    f"-filter_complex {shlex.quote(vf)} -map [vout] -r {fps} "
                    f"-c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(out))}"
                )
                cmd_cpu = cmd_nv.replace(f"-c:v {vcodec}", f"-c:v {cpu_vcodec} {cpu_vopts}")
                futures.append(ex.submit(encode_cmd_with_fallback, cmd_nv, cmd_cpu))

        for fut in as_completed(futures):
            fut.result()

    # -------------------- CONCAT (Video only, stabil) --------------------
    concat_list = work / "segments.txt"
    safe_write_concat_list(concat_list, [Path(s) for s in sorted(work.glob('scene_*.mp4'))])
    
    concat_out = work / "video_concat.mp4"
    
    # Wir zwingen ffmpeg, korrekte Zeitstempel zu erzeugen und gleiche Parameter zu nutzen
    cmd_nv = (
        f"ffmpeg -y {ff_thread_flags} -f concat -safe 0 -fflags +genpts "
        f"-i {shlex.quote(str(concat_list))} "
        f"-vf scale={width}:{height},format=yuv420p,fps={fps} "
        f"-c:v {vcodec} {vopts} -pix_fmt {args.pixfmt} -r {fps} "
        f"-an {shlex.quote(str(concat_out))}"
    )
    cmd_cpu = (
        f"ffmpeg -y {ff_thread_flags} -f concat -safe 0 -fflags +genpts "
        f"-i {shlex.quote(str(concat_list))} "
        f"-vf scale={width}:{height},format=yuv420p,fps={fps} "
        f"-c:v {cpu_vcodec} {cpu_vopts} -pix_fmt {args.pixfmt} -r {fps} "
        f"-an {shlex.quote(str(concat_out))}"
    )
    
    encode_cmd_with_fallback(cmd_nv, cmd_cpu)

    # -------------------- OVERLAY (CPU Reencode stabil) --------------------
    video_wo_audio = work / "video_overlay.mp4"
    if overlay_path.exists():
        overlay_input = f"-stream_loop 100 -i {shlex.quote(str(overlay_path))} "
        vf_overlay = (
            f"[0:v]format=yuv420p[base];"
            f"[1:v]scale={width}:{height},format=rgba,setpts=PTS-STARTPTS[ol];"
            f"[base][ol]overlay=shortest=1,format=yuv420p[v]"
        )
        cmd_overlay = (
            f"ffmpeg -y {ff_thread_flags} -i {shlex.quote(str(concat_out))} {overlay_input}"
            f"-filter_complex {shlex.quote(vf_overlay)} -map [v] -r {fps} "
            f"-c:v {cpu_vcodec} {cpu_vopts} -pix_fmt {args.pixfmt} {shlex.quote(str(video_wo_audio))}"
        )
        run(cmd_overlay)
    else:
        run(
            f"ffmpeg -y {ff_thread_flags} -i {shlex.quote(str(concat_out))} -c copy {shlex.quote(str(video_wo_audio))}"
        )

    # -------------------- AUDIO fix & Mux --------------------
    audiobook_fixed = ensure_audio_48k_stereo(audiobook)
    final_out = output_dir / ("story_hd.mp4" if args.quality == "hd" else "story_sd.mp4")
    run(
        f"ffmpeg -y {ff_thread_flags} -i {shlex.quote(str(video_wo_audio))} -i {shlex.quote(str(audiobook_fixed))} "
        f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -ar 48000 -ac 2 -b:a 160k -shortest "
        f"-movflags +faststart {shlex.quote(str(final_out))}"
    )
    print(f"✅ Fertig: {final_out}")

if __name__ == "__main__":
    try:
        build()
    except Exception as e:
        print(f"✖ Fehler: {e}")
        sys.exit(1)
