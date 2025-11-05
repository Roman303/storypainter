#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Renderer v14.0 – Timeline-Accurate • GPU-Optimized • Dual Output
- Clips sind duration + pause_duration lang
- Clips treffen sich in der Mitte der Pause (keine Gaps)
- Fades sind Transparenz-Overlays INNERHALB des Clips
- Offsets verschieben nur Fade-Zeitpunkte, NICHT die Timeline
- Cross-Scene Overlap Detection mit Warnungen
- Maximale GPU-Nutzung (NVENC + CUDA filters wo möglich)
- Dual Output: HD (1920x1080) + Mobile (854x480, 300kbps)
"""

import os, json, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

MIN_BYTES = 100 * 1024  # 100 KB
FPS_LOCK = 30

# ---------------------- Helpers ----------------------

def run(cmd, quiet=False, timeout=None):
    try:
        if not quiet:
            print(f"   🔧 Running: {' '.join(cmd[:5])}...", flush=True)
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        if r.returncode != 0 and not quiet:
            print(f"⚠️ FFmpeg Error (return code {r.returncode}):", flush=True)
            print(r.stderr[:500], flush=True)  # Erste 500 Zeichen
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f'⏱️ FFmpeg Timeout ({timeout}s) – Vorgang abgebrochen.', flush=True)
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}", flush=True)
        return False

def has_nvenc():
    try:
        r = subprocess.run(['ffmpeg','-hide_banner','-encoders'], 
                          capture_output=True, text=True, timeout=5)
        return 'h264_nvenc' in r.stdout
    except: 
        return False

def has_cuda_filters():
    try:
        r = subprocess.run(['ffmpeg','-hide_banner','-filters'], 
                          capture_output=True, text=True, timeout=5)
        return all(x in r.stdout for x in ['overlay_cuda', 'scale_cuda', 'hwupload_cuda'])
    except: 
        return False

def probe_duration(path: Path) -> float:
    try:
        r = subprocess.run([
            'ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=noprint_wrappers=1:nokey=1',str(path)
        ], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return max(0.0, float(r.stdout.strip()))
    except: 
        pass
    return 0.0

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

# ---------------------- Timeline-Accurate Fade Logic ----------------------

def compute_fades_v2(scene_duration: float, pause_duration: float, 
                     fi: float, fo: float, fi_off: float, fo_off: float) -> Tuple[float, float, float, float]:
    """
    Berechnet Fade-Zeitpunkte im Clip-Kontext.
    
    Clip-Struktur:
    - Clip = [0, scene_duration + pause_duration]
    - Sichtbarer Content: [pause_duration/2, pause_duration/2 + scene_duration]
    
    Fade-Regeln:
    - Fade-In mit offset=0: startet bei pause_duration/2 (= start_time im absoluten Timeline)
    - Fade-Out mit offset=0: endet bei pause_duration/2 + scene_duration (= end_time)
    - Offsets verschieben die Zeitpunkte relativ zu diesen Ankern
    """
    half_pause = pause_duration / 2.0
    clip_duration = scene_duration + pause_duration
    
    # Fade-In: Startet bei half_pause + fi_off, läuft fi Sekunden
    fi_st = clamp(half_pause + fi_off, 0.0, clip_duration)
    fi_d = max(0.0, min(fi, clip_duration - fi_st))
    fi_end = fi_st + fi_d
    
    # Fade-Out: Endet bei half_pause + scene_duration + fo_off, läuft fo Sekunden zurück
    fo_end = clamp(half_pause + scene_duration + fo_off, 0.0, clip_duration)
    fo_d = max(0.0, min(fo, fo_end))
    fo_st = fo_end - fo_d
    
    # Overlap-Check innerhalb des Clips
    if fi_end > fo_st:
        gap = 0.05  # 50ms Mindestabstand
        fi_d = max(0.0, fo_st - fi_st - gap)
        print(f"⚠️ Intra-Clip Fade Overlap: fade_in gekürzt auf {fi_d:.3f}s")
    
    return fi_st, fi_d, fo_st, fo_d

def check_cross_scene_overlaps(scenes: List[Dict], pause_dur: float, 
                               fi: float, fo: float, fi_off: float, fo_off: float):
    """
    Prüft ob Fade-Overlaps zwischen aufeinanderfolgenden Szenen existieren.
    
    Timeline-Logik:
    - Clip N: [start_time - pause/2, start_time + duration + pause/2]
    - Clip N+1: [start_time_next - pause/2, start_time_next + duration_next + pause/2]
    - Clips treffen sich in der Mitte bei: end_time_N + pause/2 = start_time_N+1 - pause/2
    """
    warnings = []
    half_pause = pause_dur / 2.0
    
    for i in range(1, len(scenes) - 2):  # Nur content scenes (nicht intro/outro)
        curr = scenes[i]
        next_scene = scenes[i + 1]
        
        curr_start = float(curr['start_time'])
        curr_end = float(curr['end_time'])
        curr_dur = curr_end - curr_start
        
        next_start = float(next_scene['start_time'])
        next_end = float(next_scene['end_time'])
        next_dur = next_end - next_start
        
        # Berechne Fade-Zeitpunkte im Clip-Kontext
        _, _, fo_st, fo_d = compute_fades_v2(curr_dur, pause_dur, fi, fo, fi_off, fo_off)
        fi_st_next, fi_d_next, _, _ = compute_fades_v2(next_dur, pause_dur, fi, fo, fi_off, fo_off)
        
        # Konvertiere zu absoluter Timeline
        curr_clip_start = curr_start - half_pause
        fade_out_abs_start = curr_clip_start + fo_st
        fade_out_abs_end = fade_out_abs_start + fo_d
        
        next_clip_start = next_start - half_pause
        fade_in_abs_start = next_clip_start + fi_st_next
        fade_in_abs_end = fade_in_abs_start + fi_d_next
        
        # Check Overlap
        if fade_out_abs_end > fade_in_abs_start:
            overlap = fade_out_abs_end - fade_in_abs_start
            meeting_point = curr_end + half_pause
            warnings.append(
                f"⚠️ Cross-Scene Fade Overlap detected:\n"
                f"   Scene {curr['scene_id']} Fade-Out: {fade_out_abs_start:.3f}s - {fade_out_abs_end:.3f}s\n"
                f"   Scene {next_scene['scene_id']} Fade-In: {fade_in_abs_start:.3f}s - {fade_in_abs_end:.3f}s\n"
                f"   Overlap: {overlap:.3f}s (Clips treffen sich bei {meeting_point:.3f}s)\n"
                f"   Tipp: Reduziere fade_out oder nutze fade_out_offset < 0"
            )
    
    return warnings

# ---------------------- Filter Building Blocks ----------------------

def scale_cover(src, w, h):
    """Scale and crop to cover exact dimensions"""
    return (f"[{src}]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,format=yuv420p")

def fade_inout(fi_st, fi_d, fo_st, fo_d):
    return f"fade=t=in:st={fi_st:.3f}:d={fi_d:.3f},fade=t=out:st={fo_st:.3f}:d={fo_d:.3f}"

def soft_vignette_chain(label_in: str) -> str:
    """Soft blur vignette für Szenen"""
    return (
        f"[{label_in}]split[vb1][vb2];"
        f"[vb2]vignette=angle=0:mode=forward:eval=frame,eq=brightness=-0.20,gblur=sigma=8[vbmask];"
        f"[vb1][vbmask]blend=all_expr='A*(1-0.25)+B*0.25',format=yuv420p[vout]"
    )

# ---------------------- Renderer Class ----------------------

class StoryRenderer:
    def __init__(self, base: Path, images: Path, meta: Path, out: Path, 
                 threads: int = 8, hypertrail: bool = True, vignette: bool = True):
        self.base, self.images, self.out = Path(base), Path(images), Path(out)
        self.tmp = self.out / 'temp_clips'
        self.out.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(exist_ok=True)
        
        with open(meta, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        
        if not has_nvenc():
            raise RuntimeError('❌ Keine NVIDIA/NVENC-GPU erkannt.')
        
        self.cuda_filters = has_cuda_filters()
        self.hypertrail = hypertrail
        self.vignette = vignette
        self.threads = threads
        
        print(f"🎞️ Renderer v14.0 initialisiert:")
        print(f"   • NVENC: ✅")
        print(f"   • CUDA Filters: {'✅' if self.cuda_filters else '❌ (CPU fallback)'}")
        print(f"   • HyperTrail: {'✅' if self.hypertrail else '❌'}")
        print(f"   • Vignette: {'✅' if self.vignette else '❌'}")
        print(f"   • Threads: {self.threads}")

    def _enc(self, target='work', bitrate_mbps=12):
        """NVENC encoding presets - optimiert für Qualität und Performance"""
        if target == 'final':
            return ['-c:v','h264_nvenc','-preset','p6','-rc','vbr',
                   '-b:v','10M','-maxrate','18M','-bufsize','20M',
                   '-pix_fmt','yuv420p','-g',str(FPS_LOCK*2)]
        elif target == 'mobile':
            return ['-c:v','h264_nvenc','-preset','p5','-rc','vbr',
                   '-b:v','300k','-maxrate','500k','-bufsize','600k',
                   '-pix_fmt','yuv420p','-g',str(FPS_LOCK)]
        else:  # work
            return ['-c:v','h264_nvenc','-preset','p5','-rc','vbr','-cq','19',
                   '-b:v',f'{bitrate_mbps}M','-maxrate',f'{bitrate_mbps+8}M',
                   '-bufsize',f'{bitrate_mbps+10}M','-pix_fmt','yuv420p',
                   '-g',str(FPS_LOCK*2)]

    def _render_intro(self, clip_dur: float, w: int, h: int, title: str, author: str) -> Path:
        """Render intro (duration = scene_duration + pause_duration)"""
        outp = self.tmp / 'intro_0000.mp4'
        if outp.exists() and outp.stat().st_size >= MIN_BYTES:
            print(f'⚡ Intro bereits vorhanden: {outp}')
            return outp
        
        print(f'[intro] Rendering intro clip ({clip_dur:.3f}s)...')
        print(f'[intro] Output: {outp}')
        intro_src = self.base / 'intro.mp4'
        
        if intro_src.exists():
            inp = ['-ss','0','-t',f'{clip_dur:.3f}','-r',str(FPS_LOCK),'-i',str(intro_src)]
            base = f"{scale_cover('0:v', w, h)},setsar=1"
        else:
            inp = ['-f','lavfi','-t',f'{clip_dur:.3f}','-i',
                  f'color=c=black:s={w}x{h}:r={FPS_LOCK}']
            base = '[0:v]setsar=1'
        
        fadeout_st = max(0.0, clip_dur - 1.5)
        t1 = (title or '').replace("'", "\\'")
        t2 = (author or '').replace("'", "\\'")
        wexpr = "if(lt(T,2),0,if(lt(T,4),(T-2)/2,1))"
        
        flt = (
            f"{base},split=2[cl][bl];"
            f"[bl]gblur=sigma=5,eq=brightness=-0.11[bd];"
            f"[cl][bd]blend=all_expr='A*(1-({wexpr}))+B*({wexpr})':shortest=1,format=yuv420p[base];"
            f"[base]drawtext=text='{t1}':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=(h*0.38-text_h):"
            f"alpha='if(lt(t,3),0,if(lt(t,5),(t-3)/2,1))',"
            f"drawtext=text='{t2}':fontcolor=white:fontsize=34:x=(w-text_w)/2:y=(h*0.40+text_h+12):"
            f"alpha='if(lt(t,3),0,if(lt(t,5),(t-3)/2,1))',format=yuv420p,"
            f"fade=t=out:st={fadeout_st:.3f}:d=1.5[v]"
        )
        
        cmd = ['ffmpeg','-y',*inp,'-filter_complex',flt,'-map','[v]','-an',
               *self._enc('work'),'-t',f'{clip_dur:.3f}',str(outp)]
        
        if not run(cmd) or not outp.exists() or outp.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"❌ Intro failed: {outp}")
        
        print('✅ Intro done.')
        return outp

    def _render_outro(self, clip_dur: float, w: int, h: int) -> Path:
        """Render outro (duration = scene_duration + pause_duration)"""
        outp = self.tmp / 'outro_final.mp4'
        if outp.exists() and outp.stat().st_size >= MIN_BYTES:
            return outp
        
        print(f'[outro] Rendering outro clip ({clip_dur:.3f}s)...')
        outro_src = self.base / 'outro.mp4'
        
        if outro_src.exists():
            inp = ['-ss','0','-t',f'{clip_dur:.3f}','-r',str(FPS_LOCK),'-i',str(outro_src)]
            base = f"{scale_cover('0:v', w, h)}"
        else:
            inp = ['-f','lavfi','-t',f'{clip_dur:.3f}','-i',
                  f'color=c=black:s={w}x{h}:r={FPS_LOCK}']
            base = '[0:v]'
        
        flt = f"{base},format=yuv420p[v]"
        cmd = ['ffmpeg','-y',*inp,'-filter_complex',flt,'-map','[v]','-an',
               *self._enc('work'),'-t',f'{clip_dur:.3f}',str(outp)]
        
        if not run(cmd) or not outp.exists() or outp.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"❌ Outro failed: {outp}")
        
        print('✅ Outro done.')
        return outp

    def _render_scene(self, img: Path, scene_dur: float, pause_dur: float,
                     fi: float, fo: float, fi_off: float, fo_off: float, 
                     w: int, h: int, idx: int, kb: float, kb_dir: str, kb_ease: bool) -> Path:
        """
        Render scene mit Ken Burns und optional Vignette.
        
        Clip-Struktur:
        - clip_duration = scene_duration + pause_duration
        - Sichtbarer Content: [pause/2, pause/2 + scene_duration]
        """
        outp = self.tmp / f'scene_{idx:04d}.mp4'
        if outp.exists() and outp.stat().st_size >= MIN_BYTES:
            return outp
        
        clip_dur = scene_dur + pause_dur
        print(f"[{idx:04d}] Rendering {img.name if img.exists() else 'black'} (clip={clip_dur:.3f}s, scene={scene_dur:.3f}s)...")
        
        # Timeline-accurate fade computation
        fi_st, fi_d, fo_st, fo_d = compute_fades_v2(scene_dur, pause_dur, fi, fo, fi_off, fo_off)
        
        if img.exists():
            frames = max(1, int(round(clip_dur * FPS_LOCK)))
            
            # Ken Burns logic
            if kb == 0:
                chain = f"{scale_cover('0:v',w,h)},setsar=1,{fade_inout(fi_st,fi_d,fo_st,fo_d)}[vbase]"
            else:
                if kb_dir == 'out':
                    zexpr = (f"max(1.0,(1+{kb:.4f})-(1-cos(PI*on/{frames-1}))/2*{kb:.4f})" 
                            if kb_ease else 
                            f"max(1.0,(1+{kb:.4f})-{kb:.4f}*on/{frames-1})")
                else:
                    zexpr = (f"1+{kb:.4f}*(1-cos(PI*on/{frames-1}))/2" 
                            if kb_ease else 
                            f"1+{kb:.4f}*on/{frames-1}")
                
                chain = (
                    f"{scale_cover('0:v',w,h)},"
                    f"zoompan=z='{zexpr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':"
                    f"d={frames}:s={w}x{h}:fps={FPS_LOCK},"
                    f"format=yuv420p,{fade_inout(fi_st,fi_d,fo_st,fo_d)}[vbase]"
                )
                
                if self.hypertrail:
                    chain += ";[vbase]tmix=frames=60[vbase]"
            
            # Vignette (optional)
            if self.vignette:
                chain += ";" + soft_vignette_chain('vbase')
                chain += f";[vout]trim=duration={clip_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            else:
                chain += f";[vbase]trim=duration={clip_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            
            cmd = ['ffmpeg','-y','-loop','1','-t',f'{clip_dur:.3f}','-r',str(FPS_LOCK),
                  '-i',str(img),'-filter_complex',chain,'-map','[v]','-an',
                  *self._enc('work'),'-t',f'{clip_dur:.3f}',str(outp)]
        else:
            # Black placeholder
            chain = f"[0:v]{fade_inout(fi_st,fi_d,fo_st,fo_d)},format=yuv420p[vbase]"
            if self.vignette:
                chain += ";" + soft_vignette_chain('vbase')
                chain += f";[vout]trim=duration={clip_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            else:
                chain += f";[vbase]trim=duration={clip_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]"
            
            cmd = ['ffmpeg','-y','-f','lavfi','-t',f'{clip_dur:.3f}','-i',
                  f'color=c=black:s={w}x{h}:r={FPS_LOCK}','-filter_complex',chain,
                  '-map','[v]','-an',*self._enc('work'),'-t',f'{clip_dur:.3f}',str(outp)]
        
        if not run(cmd) or not outp.exists() or outp.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"❌ Scene {idx} failed: {outp}")
        
        return outp

    def _merge_clips(self, clips: List[Path], outp: Path):
        """Merge clips mit Validierung und Retry-Logik"""
        lst = self.out / 'concat_list.txt'
        total_size = 0
        
        with open(lst, 'w', encoding='utf-8') as f:
            for c in clips:
                if c and c.exists() and c.stat().st_size >= MIN_BYTES:
                    total_size += c.stat().st_size
                    safe = c.as_posix().replace("'", "'\\''")
                    f.write(f"file '{safe}'\n")
        
        def do_merge():
            cmd = ['ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat',
                  '-safe','0','-r',str(FPS_LOCK),'-i',str(lst),*self._enc('work'),
                  '-movflags','+faststart',str(outp)]
            return run(cmd, timeout=600)
        
        if not do_merge():
            raise RuntimeError(f'❌ Merge failed: {outp}')
        
        output_size = outp.stat().st_size if outp.exists() else 0
        
        # Validation: output sollte mindestens 10% der Eingabe sein
        if total_size > 0 and (output_size < MIN_BYTES or output_size < 0.10 * total_size):
            print(f"⚠️ Merge output klein ({output_size} B < 10% von {total_size}). Retry...")
            outp.unlink(missing_ok=True)
            if not do_merge() or not outp.exists() or outp.stat().st_size < max(MIN_BYTES, int(0.10*total_size)):
                raise RuntimeError(f'❌ Merge validation failed: {outp}')
        
        print(f'✅ Merge successful: {output_size / (1024*1024):.2f} MB')

    def _apply_overlay(self, base_video: Path, main_dur: float, w: int, h: int, 
                      overlay_name: str) -> Path:
        """Apply overlay mit CUDA acceleration und software fallback"""
        ov = self.base / overlay_name
        if not ov.exists():
            print(f'ℹ️ Overlay {overlay_name} nicht gefunden – überspringe.')
            return base_video
        
        print('✨ Applying overlay...')
        ov_out = self.out / '_visual_overlay.mp4'
        
        # Try CUDA first
        if self.cuda_filters:
            flt = (
                f"[1:v]trim=duration={main_dur:.3f},setpts=PTS-STARTPTS,"
                f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,format=nv12[ov_cpu];"
                f"[0:v]hwupload_cuda,scale_cuda={w}:{h},format=nv12[base];"
                f"[ov_cpu]hwupload_cuda,format=nv12[ov];"
                f"[base][ov]overlay_cuda=0:0:repeatlast=0[out]"
            )
            cmd = ['ffmpeg','-y','-r',str(FPS_LOCK),'-i',str(base_video),'-r',str(FPS_LOCK),
                  '-i',str(ov),'-filter_complex',flt,'-map','[out]','-an',
                  *self._enc('work'),str(ov_out)]
            
            if run(cmd, timeout=600) and ov_out.exists() and ov_out.stat().st_size >= MIN_BYTES:
                print('✅ CUDA overlay erfolgreich.')
                return ov_out
            
            print('⚠️ CUDA overlay fehlgeschlagen – CPU fallback...')
        
        # CPU fallback
        flt_sw = (
            f"[1:v]trim=duration={main_dur:.3f},setpts=PTS-STARTPTS,"
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,format=rgba,colorchannelmixer=aa=0.35[ovt];"
            f"[0:v][ovt]overlay=0:0:shortest=1[out]"
        )
        cmd = ['ffmpeg','-y','-r',str(FPS_LOCK),'-i',str(base_video),'-r',str(FPS_LOCK),
              '-i',str(ov),'-filter_complex',flt_sw,'-map','[out]','-an',
              *self._enc('work'),str(ov_out)]
        
        if run(cmd, timeout=600) and ov_out.exists() and ov_out.stat().st_size >= MIN_BYTES:
            print('✅ CPU overlay erfolgreich.')
            return ov_out
        
        print('⚠️ Overlay fehlgeschlagen – nutze base video.')
        return base_video

    def _create_mobile_version(self, hd_video: Path):
        """Create kleine mobile version (854x480, 300kbps)"""
        mobile = self.out / 'story_final_mobile.mp4'
        print('📱 Creating mobile version (854x480, ~300kbps)...')
        
        cmd = ['ffmpeg','-y','-i',str(hd_video),'-vf',
              'scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2',
              *self._enc('mobile'),'-c:a','aac','-b:a','64k','-ac','2','-ar','44100',
              '-movflags','+faststart',str(mobile)]
        
        if run(cmd, timeout=300) and mobile.exists() and mobile.stat().st_size >= MIN_BYTES:
            size_mb = mobile.stat().st_size / (1024*1024)
            print(f'✅ Mobile version: {size_mb:.2f} MB')
            return mobile
        else:
            print('⚠️ Mobile version creation failed.')
            return None

    def render(self, audio: Path, width=1920, height=1080, fade_in=1.5, fade_out=2.0,
              fade_in_offset=0.0, fade_out_offset=0.0, kb_strength=0.06, kb_direction='in',
              overlay_name='overlay.mp4', workers=4, kb_ease=False):
        """Main rendering pipeline"""
        sc = self.meta['scenes']
        title = self.meta.get('title', '')
        author = self.meta.get('author', '')
        pause_dur = float(self.meta.get('pause_duration', 2.0))
        n = len(sc)
        
        print(f"\n📋 Timeline Info:")
        print(f"   • Total scenes: {n}")
        print(f"   • Pause duration: {pause_dur}s")
        print(f"   • Fade-in: {fade_in}s (offset: {fade_in_offset}s)")
        print(f"   • Fade-out: {fade_out}s (offset: {fade_out_offset}s)")
        
        # Check cross-scene fade overlaps
        warnings = check_cross_scene_overlaps(sc, pause_dur, fade_in, fade_out, 
                                             fade_in_offset, fade_out_offset)
        if warnings:
            print("\n⚠️ TIMELINE WARNINGS:")
            for w in warnings:
                print(w)
            print()
        
        # Berechne Clip-Dauer für jede Szene (scene_duration + pause_duration)
        clip_durations = []
        for i, s in enumerate(sc):
            scene_dur = float(s['end_time']) - float(s['start_time'])
            # Intro und Outro haben die volle pause_duration, content scenes auch
            clip_dur = scene_dur + pause_dur
            clip_durations.append(clip_dur)
            
            print(f"   Scene {i} (ID {s['scene_id']}): scene={scene_dur:.3f}s → clip={clip_dur:.3f}s")
        
        clips = []
        
        # Render intro
        clips.append(self._render_intro(clip_durations[0], width, height, title, author))
        
        # Render content scenes parallel
        print(f'\n🎬 Rendering {n-2} content scenes mit {workers} workers...\n')
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = {}
        
        for i in range(1, n-1):
            s = sc[i]
            sid = int(s['scene_id'])
            img = self.images / f"image_{sid:04d}.png"
            scene_dur = float(s['end_time']) - float(s['start_time'])
            kbdir = kb_direction if kb_direction in ('in','out') else ('in' if i%2==0 else 'out')
            
            future = pool.submit(
                self._render_scene, img, scene_dur, pause_dur, fade_in, fade_out, 
                fade_in_offset, fade_out_offset, width, height, i, kb_strength, kbdir, kb_ease
            )
            futures[future] = i
        
        # Collect results in order
        print(f"\n⏳ Waiting for {len(futures)} scenes to complete...")
        scene_results = {}
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            completed += 1
            try:
                result = future.result()
                scene_results[idx] = result
                print(f'✅ Scene {idx:04d} completed ({completed}/{len(futures)}) - {result.stat().st_size / 1024:.1f} KB')
            except Exception as e:
                print(f'❌ Scene {idx} failed: {e}')
                import traceback
                traceback.print_exc()
                raise
        
        pool.shutdown(wait=True)
        print(f"✅ Worker pool closed.")
        
        # Add scenes in correct order
        for i in sorted(scene_results.keys()):
            clips.append(scene_results[i])
        
        # Render outro
        print(f"\n🎬 Phase 3: Outro")
        outro_dur = float(sc[-1]['end_time']) - float(sc[-1]['start_time'])
        outro_clip = self._render_outro(outro_dur + pause_dur, width, height)
        clips.append(outro_clip)
        print(f"   Outro clip size: {outro_clip.stat().st_size / 1024:.1f} KB")
        
        print(f'\n✅ All clips rendered: {len(clips)} total')
        
        # Merge
        print(f'\n🎬 Phase 4: Merging {len(clips)} clips...')
        merged = self.out / '_merged_master.mp4'
        self._merge_clips(clips, merged)
        merged_size = merged.stat().st_size / (1024*1024) if merged.exists() else 0
        print(f"   Merged size: {merged_size:.2f} MB")
        
        # Overlay
        print(f"\n🎬 Phase 5: Overlay")
        visual = merged
        if merged.exists() and merged.stat().st_size >= MIN_BYTES:
            main_dur = probe_duration(merged)
            print(f"   Merged duration: {main_dur:.3f}s")
            visual = self._apply_overlay(merged, main_dur, width, height, overlay_name)
            if visual != merged:
                print(f"   Overlay size: {visual.stat().st_size / (1024*1024):.2f} MB")
        else:
            raise RuntimeError('❌ Merge output nicht valide – Overlay abgebrochen.')
        
        # Final Mux (48 kHz Audio, 30 fps)
        print(f'\n🎬 Phase 6: Final Mux (HD)')
        print(f"   Video: {visual}")
        print(f"   Audio: {audio}")
        final = self.out / 'story_final_hd.mp4'
        cmd = ['ffmpeg','-y','-fflags','+genpts','-r',str(FPS_LOCK),'-i',str(visual),
               '-i',str(audio),'-map','0:v:0','-map','1:a:0',*self._enc('final'),
               '-c:a','aac','-b:a','192k','-ac','2','-ar','48000',
               '-movflags','+faststart','-shortest',str(final)]
        
        print(f"   Running FFmpeg mux...")
        if not run(cmd, timeout=600):
            raise RuntimeError('❌ Final video creation failed.')
        
        if not final.exists() or final.stat().st_size < MIN_BYTES:
            raise RuntimeError('❌ Finales Video wurde nicht korrekt erzeugt.')
        
        size_mb = final.stat().st_size / (1024*1024)
        print(f'✅ HD Video fertig: {size_mb:.2f} MB')
        
        # Create mobile version
        print(f'\n🎬 Phase 7: Mobile Version')
        self._create_mobile_version(final)
        
        print(f'\n🎉 === FERTIG ===')
        print(f'   • HD: {final}')
        mobile_path = self.out / "story_final_mobile.mp4"
        if mobile_path.exists():
            print(f'   • Mobile: {mobile_path} ({mobile_path.stat().st_size / (1024*1024):.2f} MB)')
        
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
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-hypertrail', action='store_true', help='Deaktiviert HyperTrail (tmix=60)')
    ap.add_argument('--no-vignette', action='store_true', help='Deaktiviert Soft-Blur-Vignette (Scenes)')
    a = ap.parse_args()

    base = Path(a.path)
    audio = Path(a.audiobook) if a.audiobook else base / 'audiobook' / 'master.wav'
    meta = Path(a.metadata) if a.metadata else base / 'audiobook' / 'audiobook_metadata_test.json'
    out = Path(a.output) if a.output else base / 'story'

    r = StoryRenderer(base, base / 'images', meta, out, threads=a.threads,
                      hypertrail=not a.no_hypertrail, vignette=not a.no_vignette)
    r.render(audio, width=1920, height=1080, fade_in=a.fade_in, fade_out=a.fade_out,
             fade_in_offset=a.fade_in_offset, fade_out_offset=a.fade_out_offset, kb_strength=a.kb_strength,
             kb_direction=a.kb_direction, overlay_name=a.overlay, workers=a.workers, kb_ease=a.kb_ease)

if __name__ == '__main__':
    main()