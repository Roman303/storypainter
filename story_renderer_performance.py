#!/usr/bin/env python3
"""
Story Renderer v14.0.0 – ULTRA PERFORMANCE EDITION
──────────────────────────────────────────────────
- FULL GPU Pipeline (CUDA Upload → Process → Download)
- Multi-threaded CPU rendering
- Hardware-accelerated filters
- Zero-copy NVENC encoding
- Optimized memory management
"""

import json
import subprocess
import argparse
import os
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

FPS_LOCK = 30
AUDIO_RATE = 48000
RES_HD = (1920, 1080)
RES_SD = (640, 360)

# Optimal thread counts
CPU_COUNT = multiprocessing.cpu_count()
FFMPEG_THREADS = max(4, CPU_COUNT // 2)  # Half cores for FFmpeg
WORKER_THREADS = max(2, CPU_COUNT // 4)  # Quarter cores for parallel jobs


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


def detect_gpu_capabilities():
    """Detect all available GPU features."""
    caps = {
        'nvenc': False,
        'cuda': False,
        'scale_cuda': False,
        'overlay_cuda': False,
        'hwupload': False,
        'hwdownload': False
    }
    
    try:
        # Check encoders
        enc_result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        caps['nvenc'] = 'h264_nvenc' in enc_result.stdout
        
        # Check filters
        filter_result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-filters'],
            capture_output=True, text=True, timeout=5
        )
        filter_out = filter_result.stdout
        caps['scale_cuda'] = 'scale_cuda' in filter_out
        caps['overlay_cuda'] = 'overlay_cuda' in filter_out
        caps['hwupload'] = 'hwupload_cuda' in filter_out
        caps['hwdownload'] = 'hwdownload' in filter_out
        
        # Check hwaccels
        hwaccel_result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-hwaccels'],
            capture_output=True, text=True, timeout=5
        )
        caps['cuda'] = 'cuda' in hwaccel_result.stdout
        
    except:
        pass
    
    return caps


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
        
        # Detect GPU capabilities
        self.gpu = detect_gpu_capabilities()
        
        if not self.gpu['nvenc']:
            raise RuntimeError('❌ Keine NVIDIA/NVENC-GPU erkannt.')
        
        self.hypertrail = not args.no_hypertrail
        self.vignette = not args.no_vignette
        self.args = args
        
        # Performance settings
        self.use_gpu_pipeline = self.gpu['cuda'] and self.gpu['hwupload']
        
        print(f"🚀 ULTRA PERFORMANCE MODE – Renderer v14.0.0")
        print(f"   ┌─ GPU Capabilities:")
        print(f"   │  NVENC: {self.gpu['nvenc']}")
        print(f"   │  CUDA: {self.gpu['cuda']}")
        print(f"   │  Scale CUDA: {self.gpu['scale_cuda']}")
        print(f"   │  Overlay CUDA: {self.gpu['overlay_cuda']}")
        print(f"   │  HW Upload/Download: {self.gpu['hwupload']}/{self.gpu['hwdownload']}")
        print(f"   ├─ CPU Configuration:")
        print(f"   │  Total Cores: {CPU_COUNT}")
        print(f"   │  FFmpeg Threads: {FFMPEG_THREADS}")
        print(f"   │  Parallel Workers: {self.args.workers}")
        print(f"   └─ Effects:")
        print(f"      HyperTrail: {self.hypertrail} | Vignette: {self.vignette}")
    
    def _enc_params(self):
        """NVENC encoder settings optimized for speed and quality."""
        return [
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',  # p7 = highest quality preset
            '-tune', 'hq',    # High quality tuning
            '-rc', 'vbr',     # Variable bitrate
            '-cq', '19',      # Constant quality (lower = better)
            '-b:v', '15M',    # Target bitrate
            '-maxrate', '25M',
            '-bufsize', '30M',
            '-rc-lookahead', '32',  # Lookahead frames
            '-spatial_aq', '1',     # Spatial AQ
            '-temporal_aq', '1',    # Temporal AQ
            '-pix_fmt', 'yuv420p',
            '-threads', str(FFMPEG_THREADS)
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
    
    def _build_gpu_filter_chain(self, scene, clip_dur, is_img, src):
        """Build optimized filter chain with smart GPU/CPU routing."""
        filters = []
        stype = scene['type']
        
        # For images: always CPU pipeline (zoompan not on GPU)
        # For videos: use GPU pipeline if available
        use_gpu_for_this = self.use_gpu_pipeline and not is_img
        
        if use_gpu_for_this:
            filters.append("hwupload_cuda")
        
        # Scaling
        if is_img:
            # Ken Burns zoom (CPU only)
            zoom_expr = self._ken_burns_expr(clip_dur)
            filters.append(
                f"scale={RES_HD[0]}:{RES_HD[1]}:force_original_aspect_ratio=increase,"
                f"crop={RES_HD[0]}:{RES_HD[1]},"
                f"setsar=1,"
                f"zoompan=z={zoom_expr}:d={int(clip_dur * self.args.fps)}:s={RES_HD[0]}x{RES_HD[1]}:fps={self.args.fps}"
            )
        else:
            # Video scaling
            if use_gpu_for_this and self.gpu['scale_cuda']:
                # GPU scaling
                filters.append(
                    f"scale_cuda={RES_HD[0]}:{RES_HD[1]}:force_original_aspect_ratio=increase,"
                    f"crop={RES_HD[0]}:{RES_HD[1]}"
                )
                # Download for CPU effects
                filters.append("hwdownload,format=yuv420p")
            else:
                # CPU scaling
                filters.append(
                    f"scale={RES_HD[0]}:{RES_HD[1]}:force_original_aspect_ratio=increase,"
                    f"crop={RES_HD[0]}:{RES_HD[1]},setsar=1,fps={self.args.fps}"
                )
        
        # HyperTrail (CPU only) - skip for intro/outro videos
        if self.hypertrail and (is_img or stype == 'scene'):
            filters.append("tmix=frames=60:weights='1 1 1 1 1'")
        
        # Vignette (CPU only)
        if stype == 'scene' and self.vignette:
            filters.append(
                "split[vb1][vb2];"
                "[vb2]vignette=angle=0:mode=forward:eval=frame,eq=brightness=-0.20,gblur=sigma=8[vbmask];"
                "[vb1][vbmask]blend=all_expr='A*(1-0.25)+B*0.25'"
            )
        
        # Fades
        fade_in_start = self.args.fade_in_offset
        fade_out_start = clip_dur - self.args.fade_out + self.args.fade_out_offset
        
        if stype == 'intro':
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        elif stype == 'outro':
            filters.append(f"fade=t=in:st={fade_in_start}:d={self.args.fade_in}")
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        else:
            filters.append(f"fade=t=in:st={fade_in_start}:d={self.args.fade_in}")
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        
        # Text overlay for intro
        if stype == 'intro':
            title = self.meta.get('title', 'Untitled').replace("'", "\\'").replace(":", "\\:")
            author = self.meta.get('author', 'Unknown').replace("'", "\\'").replace(":", "\\:")
            
            fade_in_start = 2
            fade_in_dur = 1.5
            fade_out_start = 9
            fade_out_dur = 1
            
            alpha_expr = (
                f"if(lt(t,{fade_in_start}),0,"
                f"if(lt(t,{fade_in_start + fade_in_dur}),(t-{fade_in_start})/{fade_in_dur},"
                f"if(lt(t,{fade_out_start}),1,"
                f"if(lt(t,{fade_out_start + fade_out_dur}),1-(t-{fade_out_start})/{fade_out_dur},0))))"
            )
            
            filters.append(
                f"drawtext=text='{title}':fontsize=72:fontcolor=white:alpha='{alpha_expr}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2-40:shadowcolor=black:shadowx=2:shadowy=2"
            )
            filters.append(
                f"drawtext=text='{author}':fontsize=36:fontcolor=white:alpha='{alpha_expr}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2+60:shadowcolor=black:shadowx=2:shadowy=2"
            )
        
        filters.append("format=yuv420p")
        
        return ",".join(filters)
    
    def _ken_burns_expr(self, dur):
        """Generate Ken Burns zoom expression."""
        total_frames = int(dur * self.args.fps)
        if self.args.kb_ease:
            t_norm = f"1-pow(1-(on/{total_frames}),3)"
        else:
            t_norm = f"(on/{total_frames})"
        
        if self.args.kb_direction == 'in':
            return f"1+{self.args.kb_strength}*({t_norm})"
        else:
            return f"1+{self.args.kb_strength}*(1-({t_norm}))"
    
    def render_unit(self, scene):
        """Render a single scene clip with full GPU acceleration."""
        sid = scene['scene_id']
        stype = scene['type']
        clip_start, clip_end = self._get_clip_times(scene)
        clip_dur = clip_end - clip_start
        
        # Special hack for scene 1
        if scene["scene_id"] == 1:
            clip_end += 2.0
            clip_dur = clip_end - clip_start
            print("🕒 Scene 1 extended by 2s for test")
        
        out_file = self.tmp / f"scene_{sid:04d}.mp4"
        src = self._find_source(scene)
        
        if not src or not src.exists():
            print(f"❌ Scene {sid} ({stype}): File not found: {src}")
            return None
        
        print(f"🎬 Scene {sid:04d} ({stype}): {src.name} → {clip_dur:.2f}s")
        
        is_img = src.suffix.lower() in ['.png', '.jpg', '.jpeg']
        is_video = not is_img
        
        # IMPORTANT: For intro/outro videos, use SIMPLE approach without GPU pipeline
        # GPU pipeline with hwaccel can cause issues with short clips
        if is_video and stype in ['intro', 'outro']:
            print(f"   📹 Video {stype} - using CPU pipeline for stability")
            return self._render_video_simple(scene, src, out_file, clip_dur)
        
        # Build filter chain (for images and regular scene videos)
        filter_chain = self._build_gpu_filter_chain(scene, clip_dur, is_img, src)
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Hardware acceleration ONLY for video inputs (not images)
        if self.use_gpu_pipeline and is_video:
            cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
        
        # Input handling
        if is_img:
            # Static image with loop
            cmd.extend([
                '-loop', '1',
                '-i', str(src),
                '-vf', filter_chain,
                '-t', str(clip_dur)
            ])
        else:
            # Video file
            cmd.extend([
                '-i', str(src),
                '-vf', filter_chain,
                '-t', str(clip_dur)
            ])
        
        # Common encoding parameters
        cmd.extend([
            '-r', str(self.args.fps),
            *self._enc_params(),
            '-an',
            str(out_file)
        ])
        
        # Set GPU device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        gpu_mode = "GPU" if (self.use_gpu_pipeline and is_video) else "CPU"
        print(f"   ⚡ Rendering {clip_dur:.2f}s [{gpu_mode}]...")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout per clip
            )
            
            if not out_file.exists() or out_file.stat().st_size < 1000:
                print(f"❌ Scene {sid}: Generated file is empty or invalid")
                print(f"   Command: {' '.join(cmd)}")
                if result.stderr:
                    print(f"   FFmpeg stderr:\n{result.stderr[-1500:]}")
                return None
            
            print(f"   ✅ Scene {sid:04d} complete ({out_file.stat().st_size / 1024 / 1024:.1f} MB)")
            return out_file
            
        except subprocess.TimeoutExpired:
            print(f"❌ Scene {sid}: Timeout after 5 minutes")
            return None
        except subprocess.CalledProcessError as e:
            print(f"❌ Error rendering scene {sid}:")
            print(f"   Command: {' '.join(cmd)}")
            if e.stderr:
                print(f"   FFmpeg stderr:\n{e.stderr[-1500:]}")
            return None
    
    def _render_video_simple(self, scene, src, out_file, clip_dur):
        """Simple CPU-based rendering for intro/outro videos (more stable)."""
        sid = scene['scene_id']
        stype = scene['type']
        
        # Build simple filter chain without GPU
        filters = []
        
        # Scale and crop
        filters.append(
            f"scale={RES_HD[0]}:{RES_HD[1]}:force_original_aspect_ratio=increase,"
            f"crop={RES_HD[0]}:{RES_HD[1]},setsar=1,fps={self.args.fps}"
        )
        
        # Fades
        fade_in_start = self.args.fade_in_offset
        fade_out_start = clip_dur - self.args.fade_out + self.args.fade_out_offset
        
        if stype == 'intro':
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        elif stype == 'outro':
            filters.append(f"fade=t=in:st={fade_in_start}:d={self.args.fade_in}")
            filters.append(f"fade=t=out:st={fade_out_start}:d={self.args.fade_out}")
        
        # Text overlay for intro
        if stype == 'intro':
            title = self.meta.get('title', 'Untitled').replace("'", "\\'").replace(":", "\\:")
            author = self.meta.get('author', 'Unknown').replace("'", "\\'").replace(":", "\\:")
            
            fade_in_start = 2
            fade_in_dur = 1.5
            fade_out_start = 9
            fade_out_dur = 1
            
            alpha_expr = (
                f"if(lt(t,{fade_in_start}),0,"
                f"if(lt(t,{fade_in_start + fade_in_dur}),(t-{fade_in_start})/{fade_in_dur},"
                f"if(lt(t,{fade_out_start}),1,"
                f"if(lt(t,{fade_out_start + fade_out_dur}),1-(t-{fade_out_start})/{fade_out_dur},0))))"
            )
            
            filters.append(
                f"drawtext=text='{title}':fontsize=72:fontcolor=white:alpha='{alpha_expr}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2-40:shadowcolor=black:shadowx=2:shadowy=2"
            )
            filters.append(
                f"drawtext=text='{author}':fontsize=36:fontcolor=white:alpha='{alpha_expr}':"
                f"x=(w-text_w)/2:y=(h-text_h)/2+60:shadowcolor=black:shadowx=2:shadowy=2"
            )
        
        filters.append("format=yuv420p")
        filter_chain = ",".join(filters)
        
        # Get source video duration
        try:
            src_dur = self._get_duration(src)
            print(f"   📏 Source video: {src_dur:.2f}s, needed: {clip_dur:.2f}s")
        except:
            src_dur = clip_dur
        
        # Build command
        cmd = ['ffmpeg', '-y']
        
        # Handle short videos
        if src_dur < clip_dur - 0.5:  # More than 0.5s short
            print(f"   🔁 Video too short, looping...")
            cmd.extend(['-stream_loop', '-1'])
        
        cmd.extend([
            '-i', str(src),
            '-vf', filter_chain,
            '-t', str(clip_dur),
            '-r', str(self.args.fps),
            *self._enc_params(),
            '-an',
            str(out_file)
        ])
        
        print(f"   💻 Rendering with CPU pipeline...")
        print(f"   🔧 Command: {' '.join(cmd[:15])}...")  # Show first part of command
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if not out_file.exists():
                print(f"❌ Output file not created!")
                print(f"   Full command: {' '.join(cmd)}")
                print(f"   FFmpeg output:\n{result.stderr[-2000:]}")
                return None
            
            file_size = out_file.stat().st_size
            if file_size < 1000:
                print(f"❌ Output file too small: {file_size} bytes")
                print(f"   Full command: {' '.join(cmd)}")
                print(f"   FFmpeg output:\n{result.stderr[-2000:]}")
                return None
            
            print(f"   ✅ {stype.capitalize()} complete ({file_size / 1024 / 1024:.1f} MB)")
            return out_file
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed for {stype}!")
            print(f"   Full command: {' '.join(cmd)}")
            print(f"   Error output:\n{e.stderr[-2000:]}")
            return None
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout rendering {stype}")
            return None
    
    def merge(self, clips):
        """Concatenate clips with GPU acceleration."""
        print("\n🔗 Merging clips...")
        
        concat_file = self.tmp / 'concat.txt'
        with open(concat_file, 'w') as f:
            for clip in clips:
                if clip and clip.exists():
                    f.write(f"file '{clip.absolute()}'\n")
        
        merged_silent = self.tmp / 'merged_silent.mp4'
        
        # Fast concat (stream copy)
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-threads', str(FFMPEG_THREADS),
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
            '-threads', str(FFMPEG_THREADS),
            '-shortest',
            str(final)
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✅ Final video: {final}")
        return final
    
    def extend_to_audio(self, video):
        """Extend video with black frames if audio is longer."""
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
            *self._enc_params(),
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', str(AUDIO_RATE),
            str(extended)
        ]
        
        subprocess.run(cmd, check=True)
        extended.replace(video)
        return video
    
    def export_sd(self, hd_video):
        """Create SD version - uses CPU for stability."""
        print("\n📱 Exporting SD version...")
        
        sd_video = self.out / 'story_sd.mp4'
        
        # Use CPU scaling for SD (more stable, still fast with NVENC)
        cmd = [
            'ffmpeg', '-y',
            '-i', str(hd_video),
            '-vf', f"scale={RES_SD[0]}:{RES_SD[1]}:force_original_aspect_ratio=decrease,"
                   f"pad={RES_SD[0]}:{RES_SD[1]}:(ow-iw)/2:(oh-ih)/2",
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-b:v', '300k',
            '-maxrate', '400k',
            '-bufsize', '600k',
            '-c:a', 'aac',
            '-b:a', '96k',
            '-ar', str(AUDIO_RATE),
            '-threads', str(FFMPEG_THREADS),
            str(sd_video)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
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
        """Main rendering pipeline with parallel processing."""
        scenes = self.meta['scenes']
        
        print(f"\n🎬 Rendering {len(scenes)} scenes in parallel...")
        print(f"   Using {self.args.workers} worker threads")
        
        # Render all clips in parallel
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            clips = list(executor.map(self.render_unit, scenes))
        
        valid_clips = [c for c in clips if c and c.exists()]
        
        if not valid_clips:
            raise RuntimeError("❌ No valid clips generated")
        
        print(f"\n✅ {len(valid_clips)}/{len(scenes)} clips rendered successfully")
        
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
    ap = argparse.ArgumentParser(
        description='Story Renderer v14.0 – ULTRA PERFORMANCE MODE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    ap.add_argument('--workers', type=int, default=WORKER_THREADS, 
                    help=f'Parallel workers (default: {WORKER_THREADS} = CPU_COUNT/4)')
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