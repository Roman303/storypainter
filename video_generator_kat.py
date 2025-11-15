#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Story Pipeline v7 – 3 Schritte, GPU-Zoom (Ken Burns), vereinfachte Fades,  
Intro abgedunkelt & geblurt beim Titel, SD/HD-Ausgabe.  
  
Schritt 1 (Scene-Clips bauen):  
- Für jede Szene wird ein Clip erzeugt, dessen Länge = Szene-Dauer + 1/2 Gap davor + 1/2 Gap danach.  
- Der eigentliche Inhalts-Fade-In beginnt relativ zum Clip-Start bei genau der Zeit,  
  an der die scene.start_time liegt (also bei 1/2 Gap davor). Der Fade-Out endet exakt bei scene.end_time.  
- Media-Handling:  
  * Bild: Ken Burns in GPU-Qualität (PyTorch, Bicubic), dann auf NVENC encodiert.  
  * Video: Skaliert/gecropt, Fades per ffmpeg. (NVENC, falls vorhanden)  
- Intro: kein HyperTrail mehr. Stattdessen Abdunkeln + GBlur während Titel-Phase, Titel-Text mit Alpha-Easing.  
  
Schritt 2 (Concat):  
- Alle Scene-Clips werden aneinandergehängt (gleiche Codec-Settings) und mit "-c copy" concateniert.  
  
Schritt 3 (Finalisieren):  
- Optionales Overlay-Video/Bild über gesamte Länge (transparenter Layer).  
- Vollständiges Audiobook muxen.  
- HD- und SD-Derivat erzeugen.  
  
Benötigt:  
- Python 3.10+  
- ffmpeg (mit NVENC, wenn GPU-Encoding gewünscht)  
- PyTorch + torchvision (für GPU-Ken-Burns)  
  pip install torch torchvision tqdm ffmpeg-python pillow  
"""  
  
from __future__ import annotations  
import argparse  
import json  
import logging  
import math  
import os  
from pathlib import Path  
import shutil  
import subprocess  
import sys  
from typing import List, Tuple, Optional  
from contextlib import contextmanager  
  
# Configure logging  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[  
        logging.StreamHandler(sys.stdout)  
    ]  
)  
logger = logging.getLogger(__name__)  
  
# ---- optional: only import torch when needed (image Ken Burns) ----  
try:  
    import torch  
    import torchvision.transforms.functional as TF  
    from torchvision.io import read_image  
    TORCH_OK = True  
    logger.info("✅ PyTorch successfully imported")  
except (ImportError, ModuleNotFoundError) as e:  
    TORCH_OK = False  
    logger.warning(f"⚠️  PyTorch not available: {e}")  
  
# ---------------- utils ----------------  
def run(cmd: List[str], quiet: bool = False) -> bool:  
    """Run subprocess command and return success status"""  
    try:  
        r = subprocess.run(cmd, capture_output=True, check=False)  
        if r.returncode != 0 and not quiet:  
            try:  
                logger.error(f"Command failed: {' '.join(cmd)}")  
                logger.error(f"Error: {r.stderr.decode('utf-8', 'ignore')}")  
            except Exception:  
                logger.error(f"Command failed with return code {r.returncode}")  
        return r.returncode == 0  
    except Exception as e:  
        if not quiet:  
            logger.error(f"Subprocess error: {e}")  
        return False  
  
def has_nvenc() -> bool:  
    """Check if NVENC is available"""  
    try:  
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],  
                           capture_output=True, text=True, check=True)  
        has_h264 = "h264_nvenc" in r.stdout  
        has_hevc = "hevc_nvenc" in r.stdout  
        return has_h264 or has_hevc  
    except Exception as e:  
        logger.debug(f"NVENC check failed: {e}")  
        return False  
  
def esc_txt(s: str) -> str:  
    """Escape text for FFmpeg drawtext"""  
    if not s:  
        return ""  
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")  
  
def clamp(x, lo, hi):  
    """Clamp value between lo and hi"""  
    return max(lo, min(hi, x))  
  
def ensure_dir(p: Path):  
    """Create directory if it doesn't exist"""  
    p.mkdir(parents=True, exist_ok=True)  
  
@contextmanager  
def temp_directory(prefix: str = "story_tmp_"):  
    """Context manager for temporary directories"""  
    import tempfile  
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))  
    try:  
        yield tmp_dir  
    finally:  
        try:  
            shutil.rmtree(tmp_dir, ignore_errors=True)  
        except Exception as e:  
            logger.warning(f"Failed to cleanup temp dir {tmp_dir}: {e}")  
  
# ------------- timing helpers -------------  
def compute_scene_windows(scenes: List[dict]) -> Tuple[List[float], List[float], List[float]]:  
    """  
    Liefert:  
    bases[i]    = end - start (Original-Szenenlänge)  
    half_prev[i]= 1/2 Gap zur vorherigen Szene (0 für i==0)  
    half_next[i]= 1/2 Gap zur nächsten Szene (0 für i==last)  
    """  
    n = len(scenes)  
    starts = [float(s["start_time"]) for s in scenes]  
    ends   = [float(s["end_time"])   for s in scenes]  
    bases  = [max(0.0, ends[i]-starts[i]) for i in range(n)]  
    half_prev = [0.0]*n  
    half_next = [0.0]*n  
    for i in range(n):  
        if i > 0:  
            gap = max(0.0, starts[i] - ends[i-1])  
            half_prev[i] = 0.5*gap  
        if i < n-1:  
            gap = max(0.0, starts[i+1] - ends[i])  
            half_next[i] = 0.5*gap  
    return bases, half_prev, half_next  
  
# ------------- GPU Ken Burns for images -------------  
def ken_burns_gpu_image(  
    img_path: Path,  
    out_path: Path,  
    width: int,  
    height: int,  
    fps: int,  
    clip_dur: float,  
    fi_start: float,  
    fi_dur: float,  
    fo_end_time: float,  
    fo_dur: float,  
    zoom_start: float,  
    zoom_end: float,  
    pan: str = "none",  
    ease: str = "linear",  
    use_fp16: bool = True,  
    nvenc: bool = True  
) -> Path:  
    """  
    Rendert Ken-Burns-Video (aus Einzelbild) auf GPU (PyTorch), schreibt PNG-Frames,  
    encodiert per ffmpeg (NVENC falls verfügbar) UND appliziert die Fades exakt.  
  
    fi_start: Zeitpunkt (relativ zu Clip-Start), an dem Fade-In beginnt.  
    fi_dur:   Dauer des Fade-In.  
    fo_end_time: Zeitpunkt (relativ zu Clip-Start), an dem der Fade-Out endet (== scene.end_time_rel).  
    fo_dur:   Dauer des Fade-Out.  
    """  
    if not TORCH_OK:  
        raise RuntimeError("PyTorch/torchvision nicht verfügbar – installiere torch/torchvision für GPU-Ken-Burns.")  
  
    from tqdm import tqdm  
    from PIL import Image  
  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    dtype = torch.float16 if (use_fp16 and device=="cuda") else torch.float32  
    logger.info(f"Using device: {device}, dtype: {dtype}")  
  
    num_frames = max(1, int(round(clip_dur * fps)))  
    logger.info(f"Rendering {num_frames} frames for Ken Burns effect")  
  
    with temp_directory("kb_frames_") as tmp_dir:  
        # Load image  
        img = read_image(str(img_path)).to(device=device, dtype=dtype) / 255.0  # [C,H,W]  
        C, H, W = img.shape  
  
        # helper: easing  
        def ease_fn(t: float) -> float:  
            if ease == "ease_in_out":  
                # smootherstep  
                return t*t*t*(t*(t*6 - 15) + 10)  
            elif ease == "ease_in":  
                return t*t  
            elif ease == "ease_out":  
                return 1 - (1-t)*(1-t)  
            return t  # linear  
  
        # pan direction vector (normalized)  
        pan_dx, pan_dy = 0.0, 0.0  
        if pan in ("left","right","up","down","diag_tl","diag_tr","diag_bl","diag_br"):  
            mapping = {  
                "left":(-1,0), "right":(1,0), "up":(0,-1), "down":(0,1),  
                "diag_tl":(-1,-1), "diag_tr":(1,-1), "diag_bl":(-1,1), "diag_br":(1,1)  
            }  
            pan_dx, pan_dy = mapping[pan]  
            # normalize  
            norm = math.sqrt(pan_dx*pan_dx + pan_dy*pan_dy)  
            if norm>0:   
                pan_dx, pan_dy = pan_dx/norm, pan_dy/norm  
  
        # render frames  
        for i in tqdm(range(num_frames), desc="Rendering Ken Burns frames"):  
            t = i/(num_frames-1) if num_frames>1 else 0.0  
            et = ease_fn(t)  
            scale = zoom_start + (zoom_end - zoom_start)*et  
  
            new_h, new_w = int(H*scale), int(W*scale)  
            zimg = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC)  
  
            # pan offset inside zoomed image so that crop moves along pan vector  
            max_off_x = max(0, new_w - width)  
            max_off_y = max(0, new_h - height)  
            off_x = int( (pan_dx * et * 0.5 + 0.5) * max_off_x - max_off_x*0.5 )  
            off_y = int( (pan_dy * et * 0.5 + 0.5) * max_off_y - max_off_y*0.5 )  
            # clamp  
            off_x = max(0, min(max_off_x, off_x + max_off_x//2))  
            off_y = max(0, min(max_off_y, off_y + max_off_y//2))  
  
            # crop  
            zimg = zimg[:, off_y:off_y+height, off_x:off_x+width]  
            if zimg.shape[1] != height or zimg.shape[2] != width:  
                zimg = TF.center_crop(zimg, [height, width])  
  
            # alpha for fades (0..1)  
            tt = i / fps  
            alpha = 1.0  
            if fi_dur > 0 and tt >= fi_start:  
                alpha = min(alpha, (tt - fi_start)/fi_dur)  
            if fo_dur > 0 and tt >= (fo_end_time - fo_dur):  
                alpha = min(alpha, max(0.0, (fo_end_time - tt)/fo_dur))  
            alpha = float(clamp(alpha, 0.0, 1.0))  
  
            # apply alpha to image against black (premultiply)  
            frame = (zimg.clamp(0,1) * alpha).to(dtype=torch.float32).cpu()  
            img_pil = TF.to_pil_image(frame)  
            img_pil.save(tmp_dir / f"f_{i:06d}.png")  
  
        # encode with ffmpeg (NVENC if available)  
        enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "12M"] if nvenc else ["-c:v","libx264","-crf","18","-preset","slow"]  
        cmd = [  
            "ffmpeg","-y",  
            "-framerate", str(fps),  
            "-i", str(tmp_dir / "f_%06d.png"),  
            "-pix_fmt","yuv420p",  
            *enc,  
            "-r", str(fps),  
            str(out_path)  
        ]  
        success = run(cmd, quiet=False)  
        if not success:  
            raise RuntimeError(f"Failed to encode Ken Burns video: {out_path}")  
      
    return out_path  
  
# ------------- ffmpeg-based renderers -------------  
def render_video_source_with_fades(  
    src: Path,  
    out_path: Path,  
    width:int, height:int, fps:int,  
    clip_dur: float,  
    fi_start: float, fi_dur: float,  
    fo_end_time: float, fo_dur: float,  
    nvenc: bool  
) -> Path:  
    """  
    Skaliert/letterboxt Videoquelle auf Zielauflösung und wendet Fades an,  
    sodass der Fade-In bei fi_start beginnt und der Fade-Out genau bei fo_end_time endet.  
    """  
    fo_start = max(0.0, fo_end_time - fo_dur)  
    base = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"  
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"  
            f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"  
            f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}[v]")  
  
    enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]  
           if nvenc else ["-c:v","libx264","-crf","18","-preset","slow","-pix_fmt","yuv420p"])  
  
    cmd = [  
        "ffmpeg","-y",  
        "-ss","0","-t",f"{clip_dur:.6f}",  
        "-i", str(src),  
        "-filter_complex", base,  
        "-map","[v]",  
        "-r", str(fps),  
        "-an",  
        *enc,  
        "-t", f"{clip_dur:.6f}",  
        str(out_path)  
    ]  
    success = run(cmd, quiet=False)  
    if not success:  
        raise RuntimeError(f"Failed to render video with fades: {out_path}")  
    return out_path  
  
def render_intro_with_title(  
    src: Optional[Path],  
    out_path: Path,  
    width:int, height:int, fps:int,  
    clip_dur: float,  
    title: str, author: str,  
    title_in: float = 2.0,  
    title_hold: float = 6.0,  
    title_out: float = 1.0,  
    darken: float = -0.25,  
    blur_sigma: float = 8.0,  
    nvenc: bool = True  
) -> Path:  
    """  
    Intro ohne HyperTrail:  
    - Quelle (Video/Bild oder schwarzer Background)  
    - während Titel-Phase: abdunkeln + gblur  
    - Titel/Autor mit Alpha-Kurve (ein/halten/aus)  
    """  
    t_total = clip_dur  
    t_in = title_in  
    t_full = title_in + title_hold  
    t_out = title_in + title_hold + title_out  
  
    # FIXED ALPHA EXPRESSION - Corrected fade-out logic  
    alpha_expr = (  
        f"if(lt(t,{t_in}), 0,"  
        f" if(lt(t,{t_full}), (t-{t_in})/{max(0.0001, title_hold):.6f},"  
        f"  if(lt(t,{t_out}), 1-((t-{t_full})/{max(0.0001, title_out):.6f}), 0)"  
        f" )"  
        f")"  
    )  
  
    logger.debug(f"Alpha expression: {alpha_expr}")  
    logger.info(f"Intro timing - in: {t_in}s, hold: {title_hold}s, out: {title_out}s, total: {t_out}s")  
  
    inputs = []  
    if src and src.exists():  
        if src.suffix.lower() in {".mp4",".mov",".mkv",".webm",".avi"}:  
            inputs = ["-i", str(src)]  
            base = "[0:v]"  
        else:  
            inputs = ["-loop","1","-t",f"{clip_dur:.6f}","-r",str(fps),"-i", str(src)]  
            base = "[0:v]"  
    else:  
        inputs = ["-f","lavfi","-t",f"{clip_dur:.6f}","-i", f"color=c=black:s={width}x{height}:r={fps}"]  
        base = "[0:v]"  
  
    txt_title = esc_txt(title)  
    txt_author = esc_txt(author)  
  
    # enable blur+darken only during title window [t_in, t_out]  
    # use vignette-like soft feel via gblur  
    flt = (  
        f"{base}scale={width}:{height}:force_original_aspect_ratio=decrease,"  
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"  
        f"eq=brightness={darken:.3f}:enable='between(t,{t_in},{t_out})',"  
        f"gblur=sigma={blur_sigma}:enable='between(t,{t_in},{t_out})'[b];"  
        f"[b]drawtext=text='{txt_title}':fontsize=72:fontcolor=white:"  
        f"alpha='{alpha_expr}':x=(w-text_w)/2:y=(h-text_h)/2-40:"  
        f"shadowcolor=black:shadowx=2:shadowy=2,"  
        f"drawtext=text='{txt_author}':fontsize=36:fontcolor=white:"  
        f"alpha='{alpha_expr}':x=(w-text_w)/2:y=(h-text_h)/2+60:"  
        f"shadowcolor=black:shadowx=2:shadowy=2[v]"  
    )  
  
    enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]  
           if nvenc else ["-c:v","libx264","-crf","18","-preset","slow","-pix_fmt","yuv420p"])  
  
    cmd = [  
        "ffmpeg","-y",  
        *inputs,  
        "-filter_complex", flt,  
        "-map","[v]",  
        "-r", str(fps),  
        "-an",  
        *enc,  
        "-t", f"{clip_dur:.6f}",  
        str(out_path)  
    ]  
    success = run(cmd, quiet=False)  
    if not success:  
        raise RuntimeError(f"Failed to render intro with title: {out_path}")  
    return out_path  
  
# ------------- main pipeline -------------  
class StoryV7:  
    def __init__(self, images_dir: Path, metadata_path: Path, output_dir: Path):  
        self.images_dir = Path(images_dir)  
        self.output_dir = Path(output_dir)  
        self.tmp_dir = self.output_dir / "temp_v7"  
        ensure_dir(self.output_dir)  
        ensure_dir(self.tmp_dir)  
  
        if not metadata_path.exists():  
            raise ValueError(f"Metadata file not found: {metadata_path}")  
          
        with open(metadata_path, "r", encoding="utf-8") as f:  
            self.meta = json.load(f)  
  
        self.nvenc = has_nvenc()  
        logger.info(f"🎞️ NVENC: {'aktiv' if self.nvenc else 'nicht gefunden (Fallback CPU)'}")  
  
    @staticmethod  
    def _is_video(p: Path) -> bool:  
        return p.suffix.lower() in {".mp4",".mov",".mkv",".webm",".avi"}  
  
    @staticmethod  
    def _is_image(p: Path) -> bool:  
        return p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}  
  
    def validate_scene(self, scene: dict, index: int) -> bool:  
        """Validate scene data"""  
        try:  
            start = float(scene["start_time"])  
            end = float(scene["end_time"])  
            if end <= start:  
                logger.warning(f"Scene {index}: Invalid timing {start} to {end}")  
                return False  
            return True  
        except (KeyError, ValueError) as e:  
            logger.warning(f"Scene {index}: Invalid scene data: {e}")  
            return False  
  
    def step1_build_scene_clips(  
        self,  
        images_prefix: str,  
        width:int, height:int, fps:int,  
        fade_in:float, fade_out:float,  
        kb_strength:float, kb_direction:str, kb_ease:str  
    ) -> Tuple[List[Path], List[float]]:  
        scenes = self.meta.get("scenes", [])  
        n = len(scenes)  
        if n==0:  
            raise RuntimeError("Keine Szenen im JSON.")  
  
        title = self.meta.get("title","")  
        author= self.meta.get("author","")  
  
        bases, half_prev, half_next = compute_scene_windows(scenes)  
  
        clips, durs = [], []  
        for i, s in enumerate(scenes):  
            if not self.validate_scene(s, i):  
                continue  
                  
            stype = s.get("type", "scene")  
            start = float(s["start_time"])  
            end   = float(s["end_time"])  
            base  = bases[i]  
            clip_dur = base + half_prev[i] + half_next[i]  
  
            # Fade-In beginnt bei relativer Zeit = half_prev[i]  
            fi_start = half_prev[i]  
            fi_dur   = clamp(fade_in, 0.0, clip_dur)  
            # Fade-Out endet am relativen Zeitpunkt = (half_prev + base)  
            fo_end   = half_prev[i] + base  
            fo_dur   = clamp(fade_out, 0.0, clip_dur)  
  
            outp = self.tmp_dir / f"scene_{i:04d}.mp4"  
  
            # choose source  
            try:  
                scene_id = int(s.get('scene_id', i))  
            except (ValueError, TypeError):  
                scene_id = i  
                logger.warning(f"Scene {i}: Invalid scene_id, using index {i}")  
              
            src_img = self.images_dir / f"{images_prefix}{scene_id:04d}.png"  
            intro_mp4 = self.output_dir.parent / "intro.mp4"  
            outro_mp4 = self.output_dir.parent / "outro.mp4"  
  
            if stype == "intro":  
                # Intro-Szene: Abdunkeln + Blur während Titel  
                src = intro_mp4 if intro_mp4.exists() else (src_img if src_img.exists() else None)  
                logger.info(f"🎬 Intro Szene {i}: {clip_dur:.2f}s")  
                render_intro_with_title(  
                    src=src,  
                    out_path=outp,  
                    width=width, height=height, fps=fps,  
                    clip_dur=clip_dur,  
                    title=title, author=author,  
                    title_in=2.0, title_hold=max(1.0, base-3.0), title_out=1.0,  
                    darken=-0.25, blur_sigma=8.0, nvenc=self.nvenc  
                )  
            elif stype == "outro":  
                # Einfach skalieren + Fades  
                src = None  
                if outro_mp4.exists(): src = outro_mp4  
                elif src_img.exists(): src = src_img  
                if src is None:  
                    # schwarzer BG  
                    src = Path("__BLACK__")  
  
                if src == Path("__BLACK__"):  
                    # build color background via ffmpeg  
                    fo_start = max(0.0, fo_end - fo_dur)  
                    flt = (f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"  
                           f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"  
                           f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}[v]")  
                    enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]  
                           if self.nvenc else ["-c:v","libx264","-crf","18","-preset","slow","-pix_fmt","yuv420p"])  
                    cmd = ["ffmpeg","-y","-f","lavfi","-t",f"{clip_dur:.6f}","-i",  
                           f"color=c=black:s={width}x{height}:r={fps}",  
                           "-filter_complex", flt, "-map","[v]","-r",str(fps),"-an",*enc,"-t",f"{clip_dur:.6f}",str(outp)]  
                    success = run(cmd, quiet=False)  
                    if not success:  
                        raise RuntimeError(f"Failed to create black background scene: {outp}")  
                elif self._is_image(src):  
                    ken_burns_gpu_image(  
                        img_path=src, out_path=outp,  
                        width=width, height=height, fps=fps,  
                        clip_dur=clip_dur,  
                        fi_start=fi_start, fi_dur=fi_dur,  
                        fo_end_time=fo_end, fo_dur=fo_dur,  
                        zoom_start=1.0, zoom_end=1.02+kb_strength*0.0,  # sehr leichter Zoom  
                        pan="none", ease=kb_ease, use_fp16=True, nvenc=self.nvenc  
                    )  
                else:  
                    render_video_source_with_fades(  
                        src=src, out_path=outp, width=width, height=height, fps=fps,  
                        clip_dur=clip_dur, fi_start=fi_start, fi_dur=fi_dur,  
                        fo_end_time=fo_end, fo_dur=fo_dur, nvenc=self.nvenc  
                    )  
            else:  
                # normale Szene  
                if src_img.exists() and self._is_image(src_img):  
                    logger.info(f"🖼️  Szene {i} (Bild) – KB GPU, {clip_dur:.2f}s, fades @in {fi_start:.2f}/{fi_dur:.2f}, @out_end {fo_end:.2f}/{fo_dur:.2f}")  
                    # derive zoom values from kb_strength (0..1 ~ 0..5%)  
                    z_start = 1.0  
                    z_end   = 1.0 + clamp(kb_strength,0.0,1.0)*0.05  
                    pan = kb_direction  # left/right/up/down/diag_*  
                    ken_burns_gpu_image(  
                        img_path=src_img, out_path=outp,  
                        width=width, height=height, fps=fps,  
                        clip_dur=clip_dur,  
                        fi_start=fi_start, fi_dur=fi_dur,  
                        fo_end_time=fo_end, fo_dur=fo_dur,  
                        zoom_start=z_start, zoom_end=z_end,  
                        pan=pan, ease=kb_ease, use_fp16=True, nvenc=self.nvenc  
                    )  
                else:  
                    # Fallback: schwarzer BG mit Fades  
                    logger.warning(f"⚠️  Szene {i}: Bild nicht gefunden → BLACK BG.")  
                    fo_start = max(0.0, fo_end - fo_dur)  
                    flt = (f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,"  
                           f"fade=t=in:st={fi_start:.6f}:d={fi_dur:.6f},"  
                           f"fade=t=out:st={fo_start:.6f}:d={fo_dur:.6f}[v]")  
                    enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]  
                           if self.nvenc else ["-c:v","libx264","-crf","18","-preset","slow","-pix_fmt","yuv420p"])  
                    cmd = ["ffmpeg","-y","-f","lavfi","-t",f"{clip_dur:.6f}","-i",  
                           f"color=c=black:s={width}x{height}:r={fps}",  
                           "-filter_complex", flt, "-map","[v]","-r",str(fps),"-an",*enc,"-t",f"{clip_dur:.6f}",str(outp)]  
                    success = run(cmd, quiet=False)  
                    if not success:  
                        raise RuntimeError(f"Failed to create black background scene: {outp}")  
  
            clips.append(outp); durs.append(clip_dur)  
  
        return clips, durs  
  
    def step2_concat(self, segs: List[Path], out_path: Path) -> Path:  
        if not segs:  
            raise ValueError("No segments to concatenate")  
              
        concat_file = out_path.parent / "concat_v7.txt"  
        with open(concat_file, "w", encoding="utf-8") as f:  
            for p in segs:  
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")  
          
        logger.info(f"🔗 Concat {len(segs)} Segmente … (copy)")  
        cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat_file),"-c","copy",str(out_path)]  
        success = run(cmd, quiet=False)  
        if not success:  
            raise RuntimeError("Concatenation failed")  
        return out_path  
  
    def step3_finalize(  
        self,  
        master_video: Path,  
        audiobook_file: Path,  
        overlay_file: Optional[Path],  
        overlay_opacity: float,  
        width:int, height:int, fps:int,  
        make_sd: bool  
    ) -> Tuple[Path, Optional[Path]]:  
        if not master_video.exists():  
            raise ValueError(f"Master video not found: {master_video}")  
        if not audiobook_file.exists():  
            raise ValueError(f"Audiobook file not found: {audiobook_file}")  
  
        visual = master_video  
        # Overlay optional  
        if overlay_file and overlay_file.exists():  
            logger.info("✨ Overlay anwenden (volle Länge) …")  
            ov_out = self.output_dir / "_overlay_master.mp4"  
            if overlay_file.suffix.lower() in {".mp4",".mov",".mkv",".webm",".avi"}:  
                ov_inputs = ["-stream_loop","-1","-i", str(overlay_file)]  
            else:  
                ov_inputs = ["-loop","1","-r",str(fps),"-i", str(overlay_file)]  
            enc = (["-c:v","h264_nvenc","-preset","p5","-b:v","12M","-pix_fmt","yuv420p"]  
                   if self.nvenc else ["-c:v","libx264","-crf","18","-preset","slow","-pix_fmt","yuv420p"])  
            cmd = [  
                "ffmpeg","-y",  
                "-i", str(master_video),  
                *ov_inputs,  
                "-filter_complex",  
                (f"[0:v]format=yuv420p[base];"  
                 f"[1:v]scale={width}:{height},format=rgba,"  
                 f"colorchannelmixer=aa={overlay_opacity:.3f}[ovr];"  
                 f"[base][ovr]overlay=0:0:shortest=1[out]"),  
                "-map","[out]","-an",*enc,str(ov_out)  
            ]  
            success = run(cmd, quiet=False)  
            if not success:  
                logger.warning("Overlay application failed, continuing without overlay")  
            else:  
                visual = ov_out  
  
        logger.info("🔊 Muxe Audio …")  
        final_hd = self.output_dir / "story_final_hd.mp4"  
        # Da visual ggf. schon NVENC-kodiert ist → copy Video  
        cmd_hd = [  
            "ffmpeg","-y",  
            "-fflags","+genpts",  
            "-i", str(visual),  
            "-i", str(audiobook_file),  
            "-map","0:v:0","-map","1:a:0",  
            "-c:v","copy",  
            "-c:a","aac","-b:a","192k",  
            "-movflags","+faststart",  
            "-shortest",  
            str(final_hd)  
        ]  
        success = run(cmd_hd, quiet=False)  
        if not success:  
            raise RuntimeError("Audio muxing failed")  
  
        final_sd = None  
        if make_sd:  
            logger.info("📦 Erzeuge SD-Derivat …")  
            final_sd = self.output_dir / "story_final_sd.mp4"  
            cmd_sd = [  
                "ffmpeg","-y",  
                "-i", str(final_hd),  
                "-vf", "scale=640:360:force_original_aspect_ratio=decrease,fps=30",  
                "-c:v","libx264","-b:v","600k",  
                "-c:a","aac","-b:a","96k",  
                "-movflags","+faststart",  
                str(final_sd)  
            ]  
            success = run(cmd_sd, quiet=False)  
            if not success:  
                logger.warning("SD generation failed")  
                final_sd = None  
  
        return final_hd, final_sd  
  
# ------------- CLI -------------  
def validate_paths(args) -> None:  
    """Validate input paths"""  
    base = Path(args.path)  
      
    # Check required files  
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")  
    audiobook = Path(args.audiobook) if args.audiobook else (base / "master.wav")  
      
    if not metadata.exists():  
        raise SystemExit(f"❌ Metadaten nicht gefunden: {metadata}")  
    if not audiobook.exists():  
        raise SystemExit(f"❌ Audio nicht gefunden: {audiobook}")  
      
    logger.info(f"✅ Metadaten: {metadata}")  
    logger.info(f"✅ Audio: {audiobook}")  
  
def main():  
    ap = argparse.ArgumentParser(description="Story Pipeline v7 – 3 Schritte, GPU-Ken-Burns, Intro-Blur, SD/HD")  
    ap.add_argument("--path", required=True, help="Projektbasis")  
    ap.add_argument("--images", default=None, help="Ordner mit Bildern")  
    ap.add_argument("--metadata", default=None, help="Pfad zur JSON-Metadatei")  
    ap.add_argument("--audiobook", default=None, help="Audio-Datei (volle Länge)")  
    ap.add_argument("--output", default=None, help="Ausgabeordner")  
    ap.add_argument("--fps", type=int, default=30)  
    ap.add_argument("--fade-in", type=float, default=1.0)  
    ap.add_argument("--fade-out", type=float, default=1.0)  
    ap.add_argument("--kb-strength", type=float, default=0.5, help="0..1 → ca. 0..5% Zoom")  
    ap.add_argument("--kb-direction", default="none",  
                    choices=["none","left","right","up","down","diag_tl","diag_tr","diag_bl","diag_br"])  
    ap.add_argument("--kb-ease", default="ease_in_out", choices=["linear","ease_in","ease_out","ease_in_out"])  
    ap.add_argument("--overlay", default=None, help="Overlay-Video/Bild")  
    ap.add_argument("--overlay-opacity", type=float, default=0.25)  
    ap.add_argument("--quality", choices=["hd","sd"], default="sd")  
    args = ap.parse_args()  
  
    # Validate inputs  
    validate_paths(args)  
      
    base = Path(args.path)  
    images_dir = Path(args.images) if args.images else (base / "images")  
    metadata = Path(args.metadata) if args.metadata else (base / "audiobook" / "audiobook_metadata.json")  
    audiobook= Path(args.audiobook) if args.audiobook else (base / "master.wav")  
    output = Path(args.output) if args.output else (base / "story_v7")  
  
    logger.info(f"📁 Bilder: {images_dir}")  
    logger.info(f"📁 Output: {output}")  
  
    try:  
        pipeline = StoryV7(images_dir, metadata, output)  
  
        # Schritt 1: Scene Clips  
        logger.info("🎬 Schritt 1: Scene Clips erzeugen")  
        clips, durs = pipeline.step1_build_scene_clips(  
            images_prefix="image_",  
            width=1920, height=1080, fps=args.fps,  
            fade_in=args.fade_in, fade_out=args.fade_out,  
            kb_strength=args.kb_strength, kb_direction=args.kb_direction, kb_ease=args.kb_ease  
        )  
  
        # Schritt 2: Concat  
        logger.info("🔗 Schritt 2: Segmente zusammenfügen")  
        merged = output / "_merged_master.mp4"  
        pipeline.step2_concat(clips, merged)  
  
        # Schritt 3: Finalisieren (Overlay + Audio, HD+SD)  
        logger.info("🎯 Schritt 3: Finalisieren")  
        overlay = Path(args.overlay) if args.overlay else None  
        hd, sd = pipeline.step3_finalize(  
            master_video=merged,  
            audiobook_file=audiobook,  
            overlay_file=overlay,  
            overlay_opacity=args.overlay_opacity,  
            width=1920, height=1080, fps=args.fps,  
            make_sd=(args.quality=="sd")  
        )  
  
        # Cleanup Temp  
        logger.info("🧹 Temporäre Dateien bereinigen")  
        try:  
            shutil.rmtree(pipeline.tmp_dir, ignore_errors=True)  
        except Exception as e:  
            logger.warning(f"Temp cleanup failed: {e}")  
  
        logger.info("✅ Fertig!")  
        logger.info(f"   HD: {hd}")  
        if sd: logger.info(f"   SD: {sd}")  
  
    except Exception as e:  
        logger.error(f"❌ Pipeline failed: {e}")  
        sys.exit(1)  
  
if __name__ == "__main__":  
    main()  