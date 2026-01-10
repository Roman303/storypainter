#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
import warnings

# Unterdrücke FutureWarnings von Transformers/PyTorch
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

#############################################
# SDXL V14 - ULTRA QUALITY PIPELINE
# CLI Tool für Batch-Rendering von Büchern
#############################################

class UltraQualitySDXL:
    def __init__(
        self,
        model_base: str = "Lykon/DreamShaper-XL",
        model_refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_refiner: bool = True,
        output_width: int = 2304,
        output_height: int = 1296,
        steps: int = 50,
        guidance: float = 5.0,
        refiner_split: float = 0.8,
    ):
        print("🚀 Initialisiere SDXL V14 Ultra Quality Pipeline...")

        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine CUDA-GPU gefunden!")

        self.device = "cuda"
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.use_refiner = bool(use_refiner)
        self.refiner_split = float(refiner_split)

        # Performance Optimierungen
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Intelligente Qualitätsstufe basierend auf VRAM
        if gpu_memory >= 24:
            quality_mode = "ULTRA"
            use_fp32_vae = True
            enable_slicing = False
            enable_tiling = False
        elif gpu_memory >= 20:
            quality_mode = "HIGH"
            use_fp32_vae = True
            enable_slicing = False
            enable_tiling = False
        elif gpu_memory >= 16:
            quality_mode = "MEDIUM"
            use_fp32_vae = False
            enable_slicing = False
            enable_tiling = False
        else:
            quality_mode = "LOW"
            use_fp32_vae = False
            enable_slicing = True
            enable_tiling = True
        
        print(f"🎯 Qualitätsmodus: {quality_mode}")
        self.quality_mode = quality_mode

        # Optimierter Negative Prompt
        self.default_negative = (
            "blurry, soft focus, out of focus, low detail, low resolution, "
            "jpeg artifacts, compression artifacts, noisy, grainy, "
            "temporal noise, flickering, inconsistent lighting, "
            "watermark, logo, signature, text, subtitles, UI elements, "
            "bad composition, cropped, cut off, out of frame, "
            "bad anatomy, extra limbs, deformed hands, "
            "cartoon, anime, illustration, painting, drawing, "
            "oversaturated, undersaturated, overexposed, underexposed, "
            "harsh lighting, flat lighting, oversharpened, color banding"
        )

        # BASE PIPELINE
        print("🔥 Lade SDXL Base Model...")
        self.base = DiffusionPipeline.from_pretrained(
            model_base,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        try:
            self.base.enable_xformers_memory_efficient_attention()
            print("✅ xFormers aktiviert")
        except:
            print("⚠️ xFormers nicht verfügbar")

        # VAE Qualität basierend auf VRAM
        if use_fp32_vae:
            self.base.vae.to(dtype=torch.float32)
            print("✅ VAE in FP32 (maximale Qualität, kein Banding)")
        else:
            print("ℹ️ VAE in FP16 (VRAM-optimiert)")
        
        # Memory-Optimierungen für niedrigere VRAM
        if enable_slicing:
            self.base.enable_attention_slicing(slice_size=1)
            print("⚙️ Attention Slicing aktiviert (VRAM-Spar-Modus)")
        
        if enable_tiling:
            self.base.enable_vae_tiling()
            print("⚙️ VAE Tiling aktiviert (VRAM-Spar-Modus)")
        
        # Memory Management
        torch.cuda.empty_cache()

        # REFINER
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
            ).to(self.device)

            self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(
                self.refiner.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )

            try:
                self.refiner.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            # Gleiche Memory-Optimierungen für Refiner
            if enable_slicing:
                self.refiner.enable_attention_slicing(slice_size=1)
            
            if enable_tiling:
                self.refiner.enable_vae_tiling()

        print(f"✨ Pipeline bereit: {self.output_width}x{self.output_height}, {self.steps} steps")

    @torch.inference_mode()
    def generate(self, prompt: str, negative_prompt: str = None, seed: int = 42):
        # Memory cleanup vor jedem Generate
        torch.cuda.empty_cache()
        
        w, h = self.output_width, self.output_height
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        neg_prompt = negative_prompt if negative_prompt else self.default_negative
        split = self.refiner_split

        start_time = time.time()

        if self.use_refiner and self.refiner is not None:
            # Base Generation
            base_output = self.base(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=w,
                height=h,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                denoising_end=split,
                output_type="latent",
                generator=generator,
            )
            
            # Memory cleanup zwischen Base und Refiner
            torch.cuda.empty_cache()
            
            # Refiner
            refined_output = self.refiner(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=self.steps,
                denoising_start=split,
                image=base_output.images,
                generator=generator,
            )
            
            img = refined_output.images[0]
        else:
            # Nur Base
            output = self.base(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=w,
                height=h,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                generator=generator,
            )
            img = output.images[0]
        
        # Memory cleanup nach Generate
        torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        return img, elapsed


def process_book(input_path: Path, pipeline: UltraQualitySDXL):
    """Verarbeitet ein Buch-Verzeichnis mit book_scenes.json"""
    
    json_file = input_path / "book_scenes.json"
    
    if not json_file.exists():
        print(f"❌ Keine book_scenes.json gefunden in: {input_path}")
        return
    
    # JSON laden
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Book Info
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
    print("\n" + "="*70)
    print(f"📚 {title}")
    print(f"✍️  {author}")
    print(f"🎨 {base_style}")
    print(f"📊 {len(scenes)} Szenen")
    print("="*70 + "\n")
    
    results = []
    errors = []
    
    # Szenen rendern
    for i, scene in enumerate(scenes, 1):
        scene_id = scene.get("id", i)
        prompt = scene.get("image_prompt", "")
        negative = scene.get("negative_prompt", None)
        seed = scene.get("seed", 42)
        
        # Prompt mit base_style kombinieren
        if base_style and base_style not in prompt:
            full_prompt = f"{base_style}, {prompt}"
        else:
            full_prompt = prompt
        
        prompt_preview = full_prompt[:80] + "..." if len(full_prompt) > 80 else full_prompt
        
        print(f"{'='*70}")
        print(f"🖼️  Szene {i}/{len(scenes)} (ID: {scene_id})")
        print(f"📝 {prompt_preview}")
        print(f"🎲 Seed: {seed}")
        print(f"{'='*70}")
        
        try:
            img, elapsed = pipeline.generate(full_prompt, negative, seed)
            
            filename = output_dir / f"scene_{int(scene_id):04d}.png"
            img.save(filename, quality=95, optimize=False)
            
            file_size = filename.stat().st_size / (1024 * 1024)
            print(f"✅ Gespeichert: {filename.name} ({file_size:.2f} MB, {elapsed:.1f}s)\n")
            
            results.append(str(filename))
            
        except Exception as e:
            error_msg = f"Szene {scene_id}: {str(e)}"
            print(f"❌ FEHLER: {error_msg}\n")
            errors.append(error_msg)
            continue
    
    # Zusammenfassung
    print("="*70)
    print("🎉 RENDERING ABGESCHLOSSEN")
    print("="*70)
    print(f"✅ Erfolgreich: {len(results)}/{len(scenes)} Bilder")
    if errors:
        print(f"❌ Fehler: {len(errors)}")
        for err in errors:
            print(f"   • {err}")
    print(f"📁 Bilder in: {output_dir.absolute()}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="SDXL V14 - Ultra Quality Image Generator für Bücher"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Pfad zum Buch-Verzeichnis mit book_scenes.json"
    )
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--height", type=int, default=1728)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--guidance", type=float, default=4.2)
    parser.add_argument("--no-refiner", action="store_true", help="Refiner deaktivieren")
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    
    if not input_path.exists():
        print(f"❌ Pfad existiert nicht: {input_path}")
        return
    
    # Pipeline initialisieren
    pipeline = UltraQualitySDXL(
        use_refiner=False,
        output_width=args.width,
        output_height=args.height,
        steps=args.steps,
        guidance=args.guidance,
    )
    
    # Buch verarbeiten
    process_book(input_path, pipeline)


if __name__ == "__main__":
    main()