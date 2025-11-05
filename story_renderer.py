#!/usr/bin/env python3
"""
Story Renderer v13.9.1 – Absolute Timeline with NVENC
───────────────────────────────────────────────────────
- 30fps @ 1920×1080 (Full HD)
- 48kHz Audio
- Timing-korrekte Intro/Outro/Scene Clips
- HyperTrail (tmix=60) + Soft Vignette
- CUDA-accelerated overlays
"""

import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

FPS_LOCK = 30
AUDIO_RATE = 48000
RES_HD = (1920, 1080)
RES_SD = (640, 360)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Detection
# ═══════════════════════════════════════════════════════════════════════════════

def has_nvenc():
    """Check if NVENC (NVIDIA hardware encoder) is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except:
        return False


def has_overlay_cuda():
    """Check if overlay_cuda filter is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-filters'],
            capture_output=True, text=True, timeout=5
        )
        return 'overlay_cuda' in result.stdout
    except:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Filter Generators
# ═══════════════════════════════════════════════════════════════════════════════

def soft_vignette_chain(label_in):
    """
    Soft vignette: darkens edges with blur blend.
    """
    return (
        f"[{label_in}]split[vb1][vb2];"
        f"[vb2]vignette=angle=0:mode=forward:eval=frame,eq=brightness=-0.20,gblur=sigma=8[vbmask];"
        f"[vb1][vbmask]blend=all_expr='A*(1-0.25)+B*0.25',format=yuv420p[vout]"
    )


def ken_burns_expr(dur, strength=0.06, direction='in', ease=False, fps=30):
    total_frames = int(dur * fps)
    if ease:
        t_norm = f"1-pow(1-(on/{total_frames}),3)"
    else:
        t_norm = f"(on/{total_frames})"

    if direction == 'in':
        return f"1+{strength}*({t_norm})"
    else:
        return f"1+{strength}*(1-({t_norm}))"


# ═══════════════════════════════════════════════════════════════════════════════
# StoryRenderer Class
# ═══════════════════════════════════════════════════════════════════════════════

class StoryRenderer:
    def __init__(self, base: Path, audio: Path, meta: Path, out: Path, args):
        self.base = Path(base)
        self.audio = Path(audio)
        self.out = Path(out)
        self.tmp = self.out / 'temp_clips'
        self.images = self.base / 'images'
        
        self.out.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(exist_ok=True)
        
        with open(meta, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        
        if not has_nvenc():
            raise RuntimeError('❌ Keine NVIDIA/NVENC-GPU erkannt.')
        
        self.cuda_overlay = has_overlay_cuda()
        self.hypertrail = not args.no_hypertrail
        self.vignette = not args.no_vignette
        self.args = args
        
        print(f"🎞️  NVENC aktiv – Renderer v13.9.1")
        print(f"    CUDA-Overlay: {self.cuda_overlay}")
        print(f"    HyperTrail: {self.hypertrail}")
        print(f"    Vignette: {self.vignette}")
        print(f"    FPS: {args.fps} | Fade-In: {args.fade_in}s | Fade-Out: {args.fade_out}s")
    
    def _enc(self):
        """NVENC encoder settings for high quality."""
        return [
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-rc', 'vbr',
            '-b:v', '12M',
            '-maxrate', '22M',
            '-pix_fmt', 'yuv420p'
        ]
    
    def _get_clip_times(self, scene):
        """Berechnet Start/Ende inkl. Pausen-Offsets."""
        half_pause = self.meta.get('pause_duration', 0) / 2.0
        stype = scene["type"]
    
        if stype == "intro":
            return scene["start_time"], scene["end_time"] + half_pause
        elif stype == "outro":
            return scene["start_time"] - half_pause, scene["end_time"]
        else:
            return scene["start_time"] - half_pause, scene["end_time"] + half_pause
    
    
        
    def _is_image(self, path):
        """Check if file is an image (not used anymore - always PNG)."""
        return True
    
    def _find_source(self, scene):
        sid = scene['scene_id']
        stype = scene['type']
    
        if stype == 'intro':
            vid = self.base / 'intro.mp4'
            return vid if vid.exists() else self.images / 'intro.png'
        elif stype == 'outro':
            vid = self.base / 'outro.mp4'
            return vid if vid.exists() else self.images / 'outro.png'
        else:
            return self.images / f"image_{sid:04d}.png"

    
    def render_unit(self, scene):
        """
        Render a single scene clip with proper timing and effects.
        """
        sid = scene['scene_id']
        stype = scene['type']
        clip_start, clip_end = self._get_clip_times(scene)
        clip_dur = clip_end - clip_start

        #SPECIAL HACK todo
        if scene["scene_id"] == 1:
            clip_end += 2.0
            clip_dur = clip_end - clip_start
            print("🕒 Scene 1 extended by 2s for test")
                
        out_file = self.tmp / f"scene_{sid:04d}.mp4"
        
        # Find source file
        src = self._find_source(scene)
        
        if not src or not src.exists():
            print(f"❌ Scene {sid} ({stype}): File not found: {src}")
            return None
        
        print(f"📂 Scene {sid} ({stype}): {src.name} → {clip_dur:.2f}s")
        
        is_img = True  # Always PNG images
        
        # ───────────────────────────────────────────────────────────────────────
        # Build filter chain
        # ───────────────────────────────────────────────────────────────────────
        
        filters = []

        is_img = src.suffix.lower() in ['.png', '.jpg', '.jpeg']
        
        if is_img:
            # Statisches Bild → Ken Burns Zoom
            zoom_expr = ken_burns_expr(
                clip_dur,
                strength=self.args.kb_strength,
                direction=self.args.kb_direction,
                ease=self.args.kb_ease,
                fps=self.args.fps
            )
            filters.append(
                f"scale={RES_HD[0]}:{RES_HD[1]}:force_original_aspect_ratio=increase,"
                f"crop={RES_HD[0]}:{RES_HD[1]},"
                f"setsar=1,"
                f"zoompan=z={zoom_expr}:d={int(clip_dur * self.args.fps)}:s={RES_HD[0]}x{RES_HD[1]}:fps={self.args.fps}"
            )
        else:
            # Video → nur skalieren
            filters.append(
                f"scale={RES_HD[0]}:{RES_HD[1]}:force_original_aspect_ratio=increase,"
                f"crop={RES_HD[0]}:{RES_HD[1]},setsar=1,fps={self.args.fps}"
            )

        
        # HyperTrail (motion blur)
        if self.hypertrail:
            filters.append("tmix=frames=60:weights='1 1 1 1 1'")
        
        # Vignette for scenes (not intro/outro)
        if stype == 'scene' and self.vignette:
            filters.append(
                "split[vb1][vb2];"
                "[vb2]vignette=angle=0:mode=forward:eval=frame,eq=brightness=-0.20,gblur=sigma=8[vbmask];"
                "[vb1][vbmask]blend=all_expr='A*(1-0.25)+B*0.25'"
            )
        
        # Fade effects
        fade_in_start = self.args.fade_in_offset
        fade_out_start = clip_dur - self.args.fade_out + self.args.fade_out_offset
        
        if stype == 'intro':
            # Intro: no fade-in, only fade-out
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        elif stype == 'outro':
            # Outro: fade-in, fade-out at end
            filters.append(f"fade=t=in:st={fade_in_start}:d={self.args.fade_in}")
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        else:
            # Scene: both fades
            filters.append(f"fade=t=in:st={fade_in_start}:d={self.args.fade_in}")
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")

        
        # Format
        filters.append("format=yuv420p")
        
        filter_chain = ",".join(filters)
        
        # ───────────────────────────────────────────────────────────────────────
        # Text overlay for intro
        # ───────────────────────────────────────────────────────────────────────
        
        if stype == 'intro':
            title = self.meta.get('title', 'Untitled').replace("'", "\\'").replace(":", "\\:")
            author = self.meta.get('author', 'Unknown').replace("'", "\\'").replace(":", "\\:")
        
            fade_in_start = 2      # Sekunde, wann Text erscheinen soll
            fade_in_dur   = 1.5     # Dauer des Einblendens
            fade_out_start = 9     # Sekunde, wann Text ausblenden soll
            fade_out_dur   = 1     # Dauer des Ausblendens
        
            alpha_expr = (
                f"if(lt(t,{fade_in_start}),0,"
                f"if(lt(t,{fade_in_start + fade_in_dur}),(t-{fade_in_start})/{fade_in_dur},"
                f"if(lt(t,{fade_out_start}),1,"
                f"if(lt(t,{fade_out_start + fade_out_dur}),1-(t-{fade_out_start})/{fade_out_dur},0))))"
            )
        
            filter_chain += (
                f",drawtext=text='{title}':fontsize=72:fontcolor=white:alpha='{alpha_expr}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2-40:shadowcolor=black:shadowx=2:shadowy=2"
            )
            filter_chain += (
                f",drawtext=text='{author}':fontsize=36:fontcolor=white:alpha='{alpha_expr}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2+60:shadowcolor=black:shadowx=2:shadowy=2"
            )
            
        
        
        # ───────────────────────────────────────────────────────────────────────
        # FFmpeg command
        # ───────────────────────────────────────────────────────────────────────
        
        cmd = ['ffmpeg', '-y']

        # Nur bei Bildern -loop 1 setzen
        if src.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            cmd += ['-loop', '1']
        
        cmd += [
            '-i', str(src),
            '-vf', filter_chain,
            '-t', str(clip_dur),
            '-r', str(self.args.fps),
            *self._enc(),
            '-an',
            str(out_file)
        ]

        
        print(f"   Rendering {clip_dur:.2f}s...")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check if file is valid
            if not out_file.exists() or out_file.stat().st_size < 1000:
                print(f"❌ Scene {sid}: Generated file is empty or invalid")
                return None
            
            return out_file
        except subprocess.CalledProcessError as e:
            print(f"❌ Error rendering scene {sid}:")
            print(f"   {e.stderr}")
            return None
    
    def merge(self, clips):
        """
        Concatenate all clips and add audio.
        """
        print("\n🔗 Merging clips...")
        
        # Create concat file
        concat_file = self.tmp / 'concat.txt'
        with open(concat_file, 'w') as f:
            for clip in clips:
                if clip and clip.exists():
                    f.write(f"file '{clip.absolute()}'\n")
        
        merged_silent = self.tmp / 'merged_silent.mp4'
        
        # Concat video
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            str(merged_silent)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Add audio
        final = self.out / 'story_hd.mp4'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(merged_silent),
            '-i', str(self.audio),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', str(AUDIO_RATE),
            '-shortest',
            str(final)
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"✅ Final video: {final}")
        return final
    
    def extend_to_audio(self, video):
        """
        Extend video with black frames if audio is longer.
        """
        # Get durations
        v_dur = self._get_duration(video)
        a_dur = self._get_duration(self.audio)
        
        if a_dur <= v_dur:
            return video
        
        diff = a_dur - v_dur
        print(f"⏱️  Extending video by {diff:.2f}s to match audio")
        
        extended = self.out / 'story_hd_extended.mp4'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video),
            '-i', str(self.audio),
            '-vf', f"tpad=stop_mode=clone:stop_duration={diff}",
            *self._enc(),
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', str(AUDIO_RATE),
            str(extended)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Replace original
        extended.replace(video)
        return video
    
    def export_sd(self, hd_video):
        """
        Create SD version (640×360 @ 300kbps).
        """
        print("\n📱 Exporting SD version...")
        
        sd_video = self.out / 'story_sd.mp4'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(hd_video),
            '-vf', f"scale={RES_SD[0]}:{RES_SD[1]}:force_original_aspect_ratio=decrease,"
                   f"pad={RES_SD[0]}:{RES_SD[1]}:(ow-iw)/2:(oh-ih)/2",
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-b:v', '300k',
            '-c:a', 'aac',
            '-b:a', '96k',
            '-ar', str(AUDIO_RATE),
            str(sd_video)
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✅ SD video: {sd_video}")
    
    def _get_duration(self, file):
        """Get media duration in seconds."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def render_all(self):
        """
        Main rendering pipeline.
        """
        scenes = self.meta['scenes']
        
        print(f"\n🎬 Rendering {len(scenes)} scenes...")
        
        # Render all clips
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            clips = list(executor.map(self.render_unit, scenes))
        
        # Filter valid clips
        valid_clips = [c for c in clips if c and c.exists()]
        
        if not valid_clips:
            raise RuntimeError("❌ No valid clips generated")
        
        # Merge
        final = self.merge(valid_clips)
        
        # Extend if needed
        final = self.extend_to_audio(final)
        
        # Export SD
        self.export_sd(final)
        
        print("\n✨ Rendering complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Story Renderer v13.9 – 30fps • 48kHz • timing-correct')
    ap.add_argument('--path', required=True, help='Base path for project')
    ap.add_argument('--audiobook', default=None, help='Audio file path')
    ap.add_argument('--metadata', default=None, help='Metadata JSON path')
    ap.add_argument('--output', default=None, help='Output directory')
    ap.add_argument('--fps', type=int, default=FPS_LOCK, help='Frame rate')
    ap.add_argument('--fade-in', type=float, default=1.5, help='Fade-in duration (s)')
    ap.add_argument('--fade-out', type=float, default=2.0, help='Fade-out duration (s)')
    ap.add_argument('--fade-in-offset', type=float, default=1.0, help='Fade-in offset (s)')
    ap.add_argument('--fade-out-offset', type=float, default=0.0, help='Fade-out offset (s)')
    ap.add_argument('--kb-strength', type=float, default=0.06, help='Ken Burns zoom strength')
    ap.add_argument('--kb-direction', choices=['in','out'], default='in', help='Ken Burns direction')
    ap.add_argument('--kb-ease', action='store_true', help='Ken Burns ease-out')
    ap.add_argument('--threads', type=int, default=8, help='FFmpeg threads')
    ap.add_argument('--workers', type=int, default=4, help='Parallel workers')
    ap.add_argument('--no-hypertrail', action='store_true', help='Disable HyperTrail (tmix)')
    ap.add_argument('--no-vignette', action='store_true', help='Disable soft vignette')
    
    args = ap.parse_args()
    
    base = Path(args.path)
    audio = Path(args.audiobook) if args.audiobook else base / 'master.wav'
    meta = Path(args.metadata) if args.metadata else base / 'audiobook' / 'audiobook_metadata_test.json'
    out = Path(args.output) if args.output else base / 'story'
    
    renderer = StoryRenderer(base, audio, meta, out, args)
    renderer.render_all()


if __name__ == '__main__':
    main()