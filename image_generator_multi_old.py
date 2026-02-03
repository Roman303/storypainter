#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
import warnings

# Unterdrücke Warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.schedulers import DPMSolverMultistepScheduler

#############################################
# SDXL V15 - DreamShaper XL (ULTRA QUALITY)
#############################################

class UltraQualitySDXL:
    def __init__(
        self,
        model_base: str = "Lykon/dreamshaper-xl-1-0",  # Default
        model_refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_refiner: bool = False,
        output_width: int = None,
        output_height: int = None,
        steps: int = None,
        guidance: float = None,
        refiner_split: float = 0.75,
        scheduler: str = None,
    ):
        print("🚀 Initialisiere DreamShaper XL Pipeline (ULTRA QUALITY)")

        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine CUDA-GPU gefunden!")

        self.device = "cuda"
        
        # ✨ INTELLIGENTE DEFAULTS - nur setzen wenn nicht übergeben
        self.output_width = output_width if output_width is not None else 1920
        self.output_height = output_height if output_height is not None else 1080
        self.steps = steps if steps is not None else 42
        self.guidance = guidance if guidance is not None else 8.0
        self.scheduler_type = scheduler if scheduler is not None else "euler_a"
        self.use_refiner = bool(use_refiner)
        self.refiner_split = float(refiner_split)

        # ⚡ CUDA / SDPA Optimierung (ERSATZ für xformers)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-Tuning
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        # Qualitätsstufe - RTX 4090 = ULTRA
        if gpu_memory >= 20:
            quality_mode = "ULTRA"
            use_fp32_vae = True  # Kein Color Banding!
            enable_slicing = False
            enable_tiling = False
            compile_unet = False  # DEAKTIVIERT: kann hängen bleiben
        elif gpu_memory >= 16:
            quality_mode = "HIGH"
            use_fp32_vae = True
            enable_slicing = False
            enable_tiling = False
            compile_unet = False
        else:
            quality_mode = "MEDIUM"
            use_fp32_vae = False
            enable_slicing = True
            enable_tiling = True
            compile_unet = False

        print(f"🎯 Qualitätsmodus: {quality_mode}")

        # 📝 VERBESSERTER NEGATIVE PROMPT (gegen Splitting)
        self.default_negative = (
            # Verhindert Split-Komposition
            "split screen, multiple panels, divided image, collage, "
            "side by side, before and after, comparison, "
            "comic panels, storyboard, grid layout, tiled, "
            "dual view, two scenes, multiple views, "
            
            # Technische Artefakte
            "blurry, soft focus, out of focus, bokeh, depth of field, "
            "low detail, low resolution, low quality, worst quality, "
            "jpeg artifacts, compression artifacts, noisy, grainy, pixelated, "
            
            # Video-Artefakte
            "temporal noise, flickering, inconsistent lighting, motion blur, "
            "frame interpolation artifacts, interlaced, "
            
            # Wasserzeichen & UI
            "watermark, logo, signature, text, subtitles, UI elements, "
            "frame, border, letterbox, timestamp, "
            
            # Komposition
            "bad composition, cropped, cut off, out of frame, "
            "asymmetric, unbalanced, tilted horizon, "
            
            # Anatomie (wenn Personen)
            "bad anatomy, extra limbs, deformed hands, missing fingers, "
            "extra fingers, mutated hands, fused fingers, "
            "bad proportions, gross proportions, "
            
            # Stil-Ausschlüsse
            "cartoon, anime, illustration, painting, drawing, sketch, "
            "3d render, cgi, artificial, "
            
            # Belichtung & Farbe
            "oversaturated, undersaturated, overexposed, underexposed, "
            "harsh lighting, flat lighting, oversharpened, "
            "color banding, posterization, chromatic aberration"
        )

        # 🔥 BASE PIPELINE
        print(f"🔥 Lade DreamShaper XL ({scheduler})...")
        self.base = DiffusionPipeline.from_pretrained(
            model_base,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,  # Kein Wasserzeichen!
        ).to(self.device)

        # 📊 SCHEDULER WÄHLEN
        if self.scheduler_type == "euler_a":
            # Euler Ancestral - kreativere, detailreichere Ergebnisse
            self.base.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.base.scheduler.config
            )
            print("✅ Scheduler: Euler Ancestral (kreativ)")
        else:
            # DPM++ 2M Karras - konsistenter, glatter
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(
                self.base.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
                solver_order=2,
            )
            print("✅ Scheduler: DPM++ 2M Karras (konsistent)")

        # 💾 SPEICHER-OPTIMIERUNGEN
        if not enable_slicing:
            # Für 4090: maximale Qualität ohne Slicing
            pass
        else:
            self.base.enable_attention_slicing()
            self.base.enable_vae_slicing()

        # 🎨 VAE OPTIMIERUNG (kritisch für Qualität!)
        # WICHTIG: VAE FP32 funktioniert nur mit enable_model_cpu_offload()
        # Für normale Nutzung: VAE Tiling statt FP32
        if use_fp32_vae and gpu_memory >= 24:
            # Nur für sehr große GPUs mit CPU Offload
            try:
                self.base.enable_model_cpu_offload()
                self.base.vae.to(dtype=torch.float32)
                print("✅ VAE FP32 + CPU Offload (maximale Schärfe)")
            except Exception as e:
                print(f"⚠️ VAE FP32 fehlgeschlagen: {e}")
                print("   Verwende VAE Tiling stattdessen")
                self.base.enable_vae_tiling()
                self.base.enable_vae_slicing()
        else:
            # Standard: VAE Tiling (beste Kompatibilität)
            # Disable Slicing für schärfere Ergebnisse auf RTX 4090
            if gpu_memory >= 20:
                self.base.enable_vae_tiling()
                # Kein Slicing = schärfere Details
                print("✅ VAE FP16 + Tiling (optimiert für Schärfe)")
            else:
                self.base.enable_vae_tiling()
                self.base.enable_vae_slicing()
                print("✅ VAE FP16 + Tiling/Slicing")

        if enable_tiling and not use_fp32_vae:
            # Zusätzliches Tiling für kleinere GPUs
            pass  # Bereits aktiviert

        # ⚡ TORCH.COMPILE (PyTorch 2.0+, OPTIONAL)
        # HINWEIS: Deaktiviert, da es manchmal beim ersten Run hängen kann
        # Für erfahrene Nutzer: compile_unet=True aktivieren für 15-20% Speedup
        if compile_unet:
            try:
                print("⚡ Kompiliere UNet mit torch.compile (kann 2-5 Min dauern)...")
                self.base.unet = torch.compile(
                    self.base.unet, 
                    mode="reduce-overhead",
                    fullgraph=True
                )
                print("✅ UNet kompiliert (schneller ab 2. Bild)")
            except Exception as e:
                print(f"⚠️ torch.compile fehlgeschlagen: {e}")
        else:
            print("ℹ️ torch.compile deaktiviert (stabile Performance)")

        torch.cuda.empty_cache()

        # 🔥 OPTIONALER REFINER
        self.refiner = None
        if self.use_refiner:
            print("🔥 Lade SDXL Refiner...")
            self.refiner = DiffusionPipeline.from_pretrained(
                model_refiner,
                text_encoder_2=self.base.text_encoder_2,
                vae=self.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                add_watermarker=False,
            ).to(self.device)

            self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(
                self.refiner.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )

            if not enable_slicing:
                pass
            else:
                self.refiner.enable_attention_slicing()
                self.refiner.enable_vae_slicing()

        print(f"✨ Pipeline bereit: {self.output_width}x{self.output_height}, {self.steps} steps @ CFG {self.guidance}")

    @torch.inference_mode()
    def generate(self, prompt: str, negative_prompt: str = None, seed: int = 42):
        torch.cuda.empty_cache()

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        neg_prompt = negative_prompt if negative_prompt else self.default_negative

        start = time.time()

        # 🎨 BASE GENERATION
        if self.use_refiner:
            # Mit Refiner: Split bei refiner_split
            base_steps = int(self.steps * self.refiner_split)
            
            latents = self.base(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=self.output_width,
                height=self.output_height,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                denoising_end=self.refiner_split,
                generator=generator,
                output_type="latent",
            ).images

            # Refiner Pass
            output = self.refiner(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=latents,
                num_inference_steps=self.steps,
                denoising_start=self.refiner_split,
                generator=generator,
            )
        else:
            # Nur Base Model
            output = self.base(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=self.output_width,
                height=self.output_height,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                generator=generator,
            )

        elapsed = time.time() - start
        torch.cuda.empty_cache()
        
        return output.images[0], elapsed


def process_book(input_path: Path, pipeline: UltraQualitySDXL, force_regenerate: bool = False):
    """Verarbeitet ein Buch-Verzeichnis mit book_scenes.json"""
    
    json_file = input_path / "book_scenes.json"
    
    if not json_file.exists():
        print(f"❌ Keine book_scenes.json gefunden in: {input_path}")
        return
    
    # JSON laden
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Book Info extrahieren
    book_info = data.get("book_info", {})
    title = book_info.get("title", "Unbekannt")
    author = book_info.get("author", "Unbekannt")
    base_style = book_info.get("style", "")
    
    scenes = data.get("scenes", [])
    
    if not scenes:
        print(f"❌ Keine Szenen gefunden in JSON")
        return
    
    # Output-Verzeichnis
    output_dir = input_path / "renders"
    output_dir.mkdir(exist_ok=True)
    
    # Header
    print("\n" + "="*80)
    print(f"📚 BUCH: {title}")
    print(f"✍️  AUTOR: {author}")
    if base_style:
        print(f"🎨 BASE STYLE: {base_style}")
    print(f"📊 SZENEN: {len(scenes)}")
    print(f"⚙️  SETTINGS: {pipeline.output_width}x{pipeline.output_height} | {pipeline.steps} steps | CFG {pipeline.guidance} | {pipeline.scheduler_type}")
    if force_regenerate:
        print(f"🔄 MODUS: Überschreibe existierende Bilder")
    else:
        print(f"⏭️  MODUS: Überspringe existierende Bilder")
    print("="*80 + "\n")
    
    results = []
    errors = []
    skipped = []
    total_time = 0
    
    # Szenen rendern
    for i, scene in enumerate(scenes, 1):
        scene_id = scene.get("id", i)
        scene_prompt = scene.get("image_prompt", "")
        negative = scene.get("negative_prompt", None)
        seed = scene.get("seed", 42)
        
        # Dateiname bestimmen
        filename = output_dir / f"scene_{int(scene_id):04d}.png"
        
        # ✅ CHECK: Existiert das Bild bereits?
        if filename.exists() and not force_regenerate:
            file_size = filename.stat().st_size / (1024 * 1024)
            print("="*80)
            print(f"⏭️  SZENE {i}/{len(scenes)} (ID: {scene_id}) - ÜBERSPRINGE")
            print(f"   {filename.name} existiert bereits ({file_size:.2f} MB)")
            print("="*80 + "\n")
            skipped.append(str(filename))
            continue
        
        # ✨ BASE STYLE MIT SCENE PROMPT KOMBINIEREN
        if base_style and base_style.strip():
            full_prompt = f"{base_style}, {scene_prompt}"
        else:
            full_prompt = scene_prompt
        
        # Szenen-Header
        print("="*80)
        print(f"🖼️  SZENE {i}/{len(scenes)} (ID: {scene_id})")
        print("-"*80)
        print(f"📝 VOLLSTÄNDIGER PROMPT:")
        print(f"   {full_prompt}")
        print(f"🎲 SEED: {seed}")
        print("="*80)
        
        try:
            img, elapsed = pipeline.generate(full_prompt, negative, seed)
            total_time += elapsed
            
            img.save(filename, quality=95, optimize=True)
            
            file_size = filename.stat().st_size / (1024 * 1024)
            avg_time = total_time / (len(results) + 1)
            remaining = len(scenes) - i - len(skipped)
            eta = avg_time * remaining
            
            print(f"✅ GESPEICHERT: {filename.name}")
            print(f"   Größe: {file_size:.2f} MB | Zeit: {elapsed:.1f}s")
            print(f"   Verbleibend: {remaining} Bilder | ETA: {eta/60:.1f} min | Ø {avg_time:.1f}s/Bild")
            print()
            
            results.append(str(filename))
            
        except Exception as e:
            error_msg = f"Szene {scene_id}: {str(e)}"
            print(f"❌ FEHLER: {error_msg}\n")
            errors.append(error_msg)
            continue
    
    # Zusammenfassung
    print("\n" + "="*80)
    print("🎉 RENDERING ABGESCHLOSSEN")
    print("="*80)
    print(f"✅ Neu generiert: {len(results)} Bilder")
    if skipped:
        print(f"⏭️  Übersprungen: {len(skipped)} Bilder (bereits vorhanden)")
    print(f"⏱️  Gesamtzeit: {total_time/60:.1f} min" if total_time > 0 else "")
    if results:
        print(f"   Durchschnitt: {total_time/len(results):.1f}s/Bild")
    if errors:
        print(f"❌ Fehler: {len(errors)}")
        for err in errors:
            print(f"   • {err}")
    print(f"📁 Bilder in: {output_dir.absolute()}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DreamShaper XL Ultra Quality")
    parser.add_argument("--path", type=str, required=True, help="Pfad zum Buch-Ordner")
    parser.add_argument("--model", type=str, default="Lykon/dreamshaper-xl-1-0", 
                        help="HuggingFace Model ID (z.B. RunDiffusion/Juggernaut-XL-v9)")
    parser.add_argument("--width", type=int, default=None, help="Bildbreite (default: 1920)")
    parser.add_argument("--height", type=int, default=None, help="Bildhöhe (default: 1080)")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion Steps (default: 42)")
    parser.add_argument("--guidance", type=float, default=None, help="CFG Scale (default: 8.0)")
    parser.add_argument("--scheduler", type=str, default=None, choices=["dpm++", "euler_a"], help="Scheduler (default: euler_a)")
    parser.add_argument("--refiner", action="store_true", help="SDXL Refiner aktivieren")
    parser.add_argument("--compile", action="store_true", help="torch.compile aktivieren (experimentell)")
    parser.add_argument("--force", action="store_true", help="Existierende Bilder überschreiben")
    args = parser.parse_args()

    pipeline = UltraQualitySDXL(
        model_base=args.model,
        output_width=args.width,
        output_height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        scheduler=args.scheduler,
        use_refiner=args.refiner,
    )

    process_book(Path(args.path), pipeline, force_regenerate=args.force)


if __name__ == "__main__":
    main()