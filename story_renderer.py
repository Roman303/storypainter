#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v13.9 – 30fps Lock • 48 kHz Audio • Timing-Fix • Clean Intro/Outro (no vignette)
- Globale 30 fps (Intro/Szenen/Outro/Overlay/Merge)
- Audio Output 48 kHz (AAC 192k), Video NVENC H.264 (yuv420p)
- Szenen exakt nach JSON: hartes Trim nach ALLEN Effekten, keine Drift/Überlänge
- Gaps aus JSON (Start > letztes Ende) → automatische Black-Gap-Clips
- Intro/Outro OHNE Vignette (stabil, farbecht); Szenen optional Soft-Blur-Vignette (--no-vignette schaltet global aus)
- Intro: Blur/Dim Blend 2s→4s, Titel 50px (höher), Autor 34px, saubere Farben (format=yuv420p)
- Szenen: Ken-Burns (+HyperTrail tmix=60, per --no-hypertrail abschaltbar)
- Overlay: CUDA-Overlay (nv12) mit robustem Software-Fallback
- Merge: NVENC Re-encode, Größencheck (>=100KB & >=10% Summe) + Re-Merge
- Parallel-Rendering der Szenen, dezente Logs
"""

import os, json, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

MIN_BYTES = 100 * 1024  # 100 KB
FPS_LOCK = 30  # global 30 fps

# ---------------------- Helpers ----------------------

def run(cmd, quiet=False, timeout=None):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if r.returncode != 0 and not quiet:
            print(r.stderr.decode('utf-8', 'ignore'))
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print('⏱️ FFmpeg Timeout – Vorgang abgebrochen.')
        return False

def has_nvenc():
    try:
        r = subprocess.run(['ffmpeg','-hide_banner','-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in r.stdout
    except: return False

def has_overlay_cuda():
    try:
        r = subprocess.run(['ffmpeg','-hide_banner','-filters'], capture_output=True, text=True)
        return 'overlay_cuda' in r.stdout and 'scale_cuda' in r.stdout and 'hwupload_cuda' in r.stdout
    except: return False

def probe_duration(path: Path) -> float:
    try:
        r = subprocess.run([
            'ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)
        ], capture_output=True, text=True)
        return max(0.0, float(r.stdout.strip())) if r.returncode == 0 and r.stdout.strip() else 0.0
    except: return 0.0

def clamp(x, lo, hi): return max(lo, min(hi, x))

# ---------------------- Filter Bausteine ----------------------

def scale_cover(src, w, h):
    return f"[{src}]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,format=yuv420p"

def fade_inout(fi_st, fi_d, fo_st, fo_d):
    return f"fade=t=in:st={fi_st:.3f}:d={fi_d:.3f},fade=t=out:st={fo_st:.3f}:d={fo_d:.3f}"

# Fade-Zeitpunkte strikt innerhalb der Szenendauer (Offsets verschieben nur den Beginn)
# Bei fade_out_offset=0 endet Fade-Out EXAKT bei duration

def compute_fades(duration: float, fi: float, fo: float, fi_off: float, fo_off: float):
    fi_st = clamp(0.0 + fi_off, 0.0, duration)
    fi_d  = max(0.0, min(fi, duration - fi_st))
    if abs(fo_off) < 1e-6:
        fo_st = max(0.0, duration - fo)
    else:
        fo_st = clamp(duration - fo + fo_off, 0.0, duration)
    fo_d  = max(0.0, min(fo, duration - fo_st))
    fi_end = fi_st + fi_d
    if fi_end > fo_st:
        gap = 0.001
        if abs(fo_off) < 1e-6:
            fi_d = max(0.0, fo_st - fi_st - gap)
        else:
            fo_st = clamp(fi_end + gap, 0.0, duration)
            fo_d = max(0.0, min(fo, duration - fo_st))
        print(f"⚠️ Fade-Konflikt geclamped: fi_st={fi_st:.3f}, fi_d={fi_d:.3f}, fo_st={fo_st:.3f}, fo_d={fo_d:.3f}")
    return fi_st, fi_d, fo_st, fo_d

# Soft-Blur-Vignette-Block (für Szenen – Intro/Outro nutzen ihn NICHT)

def soft_vignette_chain(label_in: str) -> str:
    return (
        f"[{label_in}]split[vb1][vb2];"
        f"[vb2]vignette=angle=0:mode=forward:eval=frame,eq=brightness=-0.20,gblur=sigma=8[vbmask];"
        f"[vb1][vbmask]blend=all_expr='A*(1-0.25)+B*0.25',format=yuv420p[vout]"
    )

# ---------------------- Renderer ----------------------

class StoryRenderer:
    def __init__(self, base: Path, images: Path, meta: Path, out: Path, threads: int = 8,
                 hypertrail: bool = True, vignette: bool = True):
        self.base, self.images, self.out = Path(base), Path(images), Path(out)
        self.tmp = self.out / 'temp_clips'
        self.out.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(exist_ok=True)
        with open(meta, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        if not has_nvenc():
            raise RuntimeError('❌ Keine NVIDIA/NVENC-GPU erkannt.')
        self.cuda_overlay = has_overlay_cuda()
        self.hypertrail = hypertrail
        self.vignette = vignette
        print(f"🎞️ NVENC aktiv – Renderer v13.9 (CUDA-Overlay={self.cuda_overlay}, HyperTrail={self.hypertrail}, Vignette={self.vignette})")

    def _enc(self, target='work'):
        if target == 'final':
            return ['-c:v','h264_nvenc','-preset','p5','-rc','vbr','-b:v','10M','-maxrate','18M','-pix_fmt','yuv420p']
        else:
            return ['-c:v','h264_nvenc','-preset','p5','-rc','vbr','-cq','19','-b:v','12M','-maxrate','22M','-pix_fmt','yuv420p']

    # ---------- Intro (ohne Vignette) ----------
    def _render_intro(self, dur: float, w: int, h: int, fps: int, title: str, author: str) -> Path:
        outp = self.tmp / 'intro_0000.mp4'
        if outp.exists() and outp.stat().st_size >= MIN_BYTES:
            return outp
        print('[intro] Rendering intro …')
        intro_src = self.base / 'intro.mp4'
        if intro_src.exists():
            inp = ['-ss','0','-t',f'{dur:.3f}','-r',str(FPS_LOCK),'-i',str(intro_src)]
            base = f"{scale_cover('0:v', w, h)},setsar=1"
        else:
            inp = ['-f','lavfi','-t',f'{dur:.3f}','-i',f'color=c=black:s={w}x{h}:r={FPS_LOCK}']
            base = '[0:v]setsar=1'
        fadeout_st = max(0.0, dur - 1.5)
        t1 = (title or '').replace("'","\\'")
        t2 = (author or '').replace("'","\\'")
        wexpr = "if(lt(T,2),0, if(lt(T,4),(T-2)/2,1))"
        flt = (
            f"{base},split=2[cl][bl];"
            f"[bl]gblur=sigma=5,eq=brightness=-0.11[bd];" # --- darken ---
            f"[cl][bd]blend=all_expr='A*(1-({wexpr}))+B*({wexpr})':shortest=1,format=yuv420p[base];"
            f"[base]drawtext=text='{t1}':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=(h*0.38-text_h):"
            f"alpha='if(lt(t,3),0,if(lt(t,5),(t-3)/2,1))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=34:x=(w-text_w)/2:y=(h*0.40+text_h+12):"
            f"alpha='if(lt(t,3),0,if(lt(t,5),(t-3)/2,1))',format=yuv420p,"
            f"fade=t=out:st={fadeout_st:.3f}:d=1.5[v]"
        )
        cmd = ['ffmpeg','-y',*inp,'-filter_complex', flt,'-map','[v]','-an',*self._enc('work'),'-t',f'{dur:.3f}',str(outp)]
        if not run(cmd) or not outp.exists() or outp.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"__ Intro fehlgeschlagen oder zu klein: {outp}")
        print('✅ intro done.')
        return outp

    # ---------- Outro (ohne Vignette) ----------
    def _render_outro(self, dur: float, w: int, h: int, fps: int) -> Path:
        outp = self.tmp / 'outro_final.mp4'
        if outp.exists() and outp.stat().st_size >= MIN_BYTES:
            return outp
        print('[outro] Rendering outro …')
        outro_src = self.base / 'outro.mp4'
        if outro_src.exists():
            inp = ['-ss','0','-t',f'{dur:.3f}','-r',str(FPS_LOCK),'-i',str(outro_src)]
            base = f"{scale_cover('0:v', w, h)}"
        else:
            inp = ['-f','lavfi','-t',f'{dur:.3f}','-i',f'color=c=black:s={w}x{h}:r={FPS_LOCK}']
            base = '[0:v]'
        flt = f"{base},format=yuv420p[v]"
        cmd = ['ffmpeg','-y',*inp,'-filter_complex', flt,'-map','[v]','-an',*self._enc('work'),'-t',f'{dur:.3f}',str(outp)]
        if not run(cmd) or not outp.exists() or outp.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"__ Outro fehlgeschlagen oder zu klein: {outp}")
        print('✅ outro done.')
        return outp

    # ---------- Szenen (Ken-Burns + Trail, nach Effekten trimmen; Vignette optional) ----------
    def _render_scene(self, img: Path, dur: float, fi: float, fo: float, fi_off: float, fo_off: float,
                      w: int, h: int, fps: int, idx: int, kb: float, kb_dir: str, kb_ease: bool) -> Path:
        outp = self.tmp / f'scene_{idx:04d}.mp4'
        if outp.exists() and outp.stat().st_size >= MIN_BYTES:
            return outp
        print(f"[{idx}] Rendering {img.name if img.exists() else 'scene'} …")
        fi_st, fi_d, fo_st, fo_d = compute_fades(dur, fi, fo, fi_off, fo_off)
        if img.exists():
            frames = max(1, int(round(dur * FPS_LOCK)))
            if kb == 0:
                chain = f"{scale_cover('0:v',w,h)},setsar=1,{fade_inout(fi_st,fi_d,fo_st,fo_d)}[vbase]"
            else:
                if kb_dir == 'out':
                    zexpr = (f"max(1.0,(1+{kb:.4f})-(1-cos(PI*on/{frames-1}))/2*{kb:.4f})" if kb_ease else f"max(1.0,(1+{kb:.4f})-{kb:.4f}*on/{frames-1})")
                else:
                    zexpr = (f"1+{kb:.4f}*(1-cos(PI*on/{frames-1}))/2" if kb_ease else f"1+{kb:.4f}*on/{frames-1}")
                chain = (
                    f"{scale_cover('0:v',w,h)},"
                    f"zoompan=z='{zexpr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d={frames}:s={w}x{h}:fps={FPS_LOCK},"
                    f"format=yuv420p,{fade_inout(fi_st,fi_d,fo_st,fo_d)}[vbase]"
                )
                if self.hypertrail:
                    chain += ";[vbase]tmix=frames=60[vbase]"
            # Vignette nur bei Szenen
            if self.vignette:
                chain += ";" + soft_vignette_chain('vbase') + f";[vout]trim=duration={dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            else:
                chain += f";[vbase]trim=duration={dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            cmd = ['ffmpeg','-y','-loop','1','-t',f'{dur:.3f}','-r',str(FPS_LOCK),'-i',str(img),'-filter_complex',chain,'-map','[v]','-an',*self._enc('work'),'-t',f'{dur:.3f}',str(outp)]
        else:
            chain = f"[0:v]{fade_inout(fi_st,fi_d,fo_st,fo_d)},format=yuv420p[vbase]"
            if self.vignette:
                chain += ";" + soft_vignette_chain('vbase') + f";[vout]trim=duration={dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            else:
                chain += f";[vbase]trim=duration={dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            cmd = ['ffmpeg','-y','-f','lavfi','-t',f'{dur:.3f}','-i',f'color=c=black:s={w}x{h}:r={FPS_LOCK}','-filter_complex',chain,'-map','[v]','-an',*self._enc('work'),'-t',f'{dur:.3f}',str(outp)]
        if not run(cmd) or not outp.exists() or outp.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"__ Szene {idx} fehlgeschlagen oder zu klein: {outp}")
        print(f"✅ Scene {idx} done.")
        return outp

    # ---------- Merge (NVENC, Validierung & Re-Merge) ----------
    def _merge(self, clips, outp: Path):
        lst = self.out / 'concat_list.txt'
        total = 0
        with open(lst, 'w', encoding='utf-8') as f:
            for c in clips:
                p = Path(c)
                if p.exists() and p.stat().st_size >= MIN_BYTES:
                    total += p.stat().st_size
                    safe = p.as_posix().replace("'", "'\\''")
                    f.write(f"file '{safe}'\n")
        def do_merge():
            cmd = ['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-r',str(FPS_LOCK),'-i',str(lst),
                   *self._enc('work'), '-movflags','+faststart', str(outp)]
            return run(cmd)
        if not do_merge():
            raise RuntimeError(f'Merge fehlgeschlagen: {outp}')
        size = outp.stat().st_size if outp.exists() else 0
        if total > 0 and (size < MIN_BYTES or size < 0.10 * total):
            print(f"⚠️ Merge-Output klein ({size} B < 10% von {total}). Versuche Re-Merge…")
            outp.unlink(missing_ok=True)
            if not do_merge() or not outp.exists() or outp.stat().st_size < max(MIN_BYTES, int(0.10*total)):
                raise FileNotFoundError(f'Kein gültiger Merge-Output: {outp}')

    # ---------- Render Pipeline ----------
    def render(self, audio: Path, width=1920, height=1080, fps=FPS_LOCK, fade_in=1.5, fade_out=2.0,
               fade_in_offset=0.0, fade_out_offset=0.0, kb_strength=0.06, kb_direction='in',
               overlay_name='overlay.mp4', workers=4, kb_ease=False):
        sc = self.meta['scenes']
        title = self.meta.get('title','')
        author = self.meta.get('author','')
        n = len(sc)
        starts = [float(s['start_time']) for s in sc]
        ends = [float(s['end_time']) for s in sc]
        bases = [max(0, ends[i]-starts[i]) for i in range(n)]
        clips = []
        # Intro
        clips.append(self._render_intro(bases[0], width, height, fps, title, author))
        # Szenen parallel
        pool = ThreadPoolExecutor(max_workers=max(1,int(workers)))
        futures = []
        for i in range(1, n-1):
            s = sc[i]
            sid = int(s['scene_id'])
            img = self.images / f"image_{sid:04d}.png"
            base = bases[i]
            kbdir = kb_direction if kb_direction in ('in','out') else ('in' if i%2==0 else 'out')
            futures.append((i, pool.submit(self._render_scene, img, base, fade_in, fade_out, fade_in_offset, fade_out_offset, width, height, FPS_LOCK, i, kb_strength, kbdir, kb_ease)))
            # Gaps einfügen falls nötig
            if i < n-2:
                gap = starts[i+1] - ends[i]
                if gap > 0.0005:
                    gap_out = self.tmp / f"gap_after_{i:04d}.mp4"
                    run(['ffmpeg','-y','-f','lavfi','-t', f'{gap:.3f}', '-i', f'color=c=black:s={width}x{height}:r={FPS_LOCK}',
                         '-c:v','h264_nvenc','-pix_fmt','yuv420p', str(gap_out)])
                    clips.append(gap_out)
        pool.shutdown(wait=True)
        for i, f in sorted(futures, key=lambda x:x[0]):
            clips.append(f.result())
        # Outro
        clips.append(self._render_outro(bases[-1], width, height, fps))
        print('✅ All scenes rendered correctly (intro/scenes/outro).')

        # Merge
        merged = self.out / '_merged_master.mp4'
        self._merge(clips, merged)

        # Overlay (CUDA → Soft-Fallback)
        visual = merged
        if merged.exists() and merged.stat().st_size >= MIN_BYTES:
            ov = self.base / overlay_name
            if ov.exists():
                print('✨ Overlay anwenden …')
                main_dur = probe_duration(merged)
                ov_out = self.out / '_visual_overlay.mp4'
                if self.cuda_overlay:
                    flt = (
                        f"[1:v]trim=duration={main_dur:.3f},setpts=PTS-STARTPTS,"
                        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                        f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,format=nv12[ov_cpu];"
                        f"[0:v]hwupload_cuda,scale_cuda={width}:{height},format=nv12[base];"
                        f"[ov_cpu]hwupload_cuda,format=nv12[ov];"
                        f"[base][ov]overlay_cuda=0:0:repeatlast=0[out]"
                    )
                    ok = run(['ffmpeg','-y','-r',str(FPS_LOCK),'-i', str(merged), '-r',str(FPS_LOCK),'-i', str(ov), '-filter_complex', flt,
                              '-map','[out]','-an', *self._enc('work'), str(ov_out)], timeout=600)
                else:
                    ok = False
                if not ok:
                    print('⚠️ CUDA-Overlay fehlgeschlagen – Soft-Overlay als Fallback …')
                    flt_sw = (
                        f"[1:v]trim=duration={main_dur:.3f},setpts=PTS-STARTPTS,"
                        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                        f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,format=rgba,colorchannelmixer=aa=0.35[ovt];"
                        f"[0:v][ovt]overlay=0:0:shortest=1[out]"
                    )
                    run(['ffmpeg','-y','-r',str(FPS_LOCK),'-i', str(merged), '-r',str(FPS_LOCK),'-i', str(ov), '-filter_complex', flt_sw,
                         '-map','[out]','-an', *self._enc('work'), str(ov_out)], timeout=600)
                if ov_out.exists() and ov_out.stat().st_size >= MIN_BYTES:
                    visual = ov_out
                    print('✅ Overlay erfolgreich.')
                else:
                    print('⚠️ Overlay fehlgeschlagen oder Timeout – benutze Merge-Output.')
            else:
                print(f'⚠️ Overlay {overlay_name} nicht gefunden – ohne Overlay weiter.')
        else:
            raise RuntimeError('❌ Merge-Output ist nicht valide – Overlay abgebrochen.')

        # Final Mux (48 kHz Audio, 30 fps)
        final = self.out / 'story_final_hd.mp4'
        cmd = ['ffmpeg','-y','-fflags','+genpts','-r',str(FPS_LOCK),'-i', str(visual), '-i', str(audio),
               '-map','0:v:0','-map','1:a:0', *self._enc('final'),
               '-c:a','aac','-b:a','192k','-ac','2','-ar','48000','-movflags','+faststart','-shortest', str(final)]
        run(cmd)
        if not final.exists() or final.stat().st_size < MIN_BYTES:
            raise RuntimeError('❌ Finales Video wurde nicht korrekt erzeugt.')
        print('__ Fertig:', final)
        return final

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description='Story Renderer v13.9 – 30fps • 48kHz • timing-correct')
    ap.add_argument('--path', required=True)
    ap.add_argument('--audiobook', default=None)
    ap.add_argument('--metadata', default=None)
    ap.add_argument('--output', default=None)
    ap.add_argument('--fps', type=int, default=FPS_LOCK)
    ap.add_argument('--fade-in', type=float, default=1.5)
    ap.add_argument('--fade-out', type=float, default=2.0)
    ap.add_argument('--fade-in-offset', type=float, default=0.0)
    ap.add_argument('--fade-out-offset', type=float, default=0.0)
    ap.add_argument('--kb-strength', type=float, default=0.06)
    ap.add_argument('--kb-direction', choices=['in','out'], default='in')
    ap.add_argument('--kb-ease', action='store_true')
    ap.add_argument('--overlay', default='overlay.mp4')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-hypertrail', action='store_true', help='Deaktiviert HyperTrail (tmix=60)')
    ap.add_argument('--no-vignette', action='store_true', help='Deaktiviert Soft-Blur-Vignette (Scenes)')
    a = ap.parse_args()

    base = Path(a.path)
    audio = Path(a.audiobook) if a.audiobook else base / 'audiobook' / 'master.wav'
    meta = Path(a.metadata) if a.metadata else base / 'audiobook' / 'audiobook_metadata.json'
    out = Path(a.output) if a.output else base / 'story'

    r = StoryRenderer(base, base / 'images', meta, out, threads=a.threads,
                      hypertrail=not a.no_hypertrail, vignette=not a.no_vignette)
    r.render(audio, width=1920, height=1080, fps=a.fps, fade_in=a.fade_in, fade_out=a.fade_out,
             fade_in_offset=a.fade_in_offset, fade_out_offset=a.fade_out_offset, kb_strength=a.kb_strength,
             kb_direction=a.kb_direction, overlay_name=a.overlay, workers=a.workers, kb_ease=a.kb_ease)

if __name__ == '__main__':
    main()
