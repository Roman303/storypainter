import os
import json
import time
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLRefinerPipeline,
)
from diffusers.schedulers import DPMSolverMultistepScheduler

#############################################
# SDXL ULTRA QUALITY PIPELINE (RTX 4090)
# - Native 2304x1296 (oder beliebig, aber durch 64 teilbar empfohlen)
# - SDXL Refiner korrekt via denoising_end/start (80/20)
# - DPM++ (dpmsolver++) + Karras Sigmas
# - Keine "Qualitäts-kostenden" Speichertricks auf 4090 (slicing/tiling aus)
#############################################

class UltraQualitySDXL:
    def __init__(
        self,
        model_base: str = "stabilityai/stable-diffusion-xl-base-1.0",
        model_refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_refiner: bool = True,
        output_width: int = 2304,
        output_height: int = 1296,
        steps: int = 50,
        guidance: float = 5.0,
        refiner_split: float = 0.8,  # 0.8 = 80% base, 20% refiner
    ):
        print("🚀 Initialisiere Ultra-Quality SDXL (refactored)...")

        if not torch.cuda.is_available():
            raise RuntimeError("Keine CUDA-GPU gefunden. Für SDXL ist eine GPU nötig.")

        self.device = "cuda"
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.use_refiner = bool(use_refiner)
        self.refiner_split = float(refiner_split)

        # Kleiner Quality/Perf Win auf RTX 30/40:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        #############################################
        # Negative Prompt (dein Original, leicht gesäubert)
        #############################################
        self.negative_prompt = (
    		"blurry, soft focus, low detail, low resolution, jpeg artifacts, noisy, "
    		"watermark, logo, signature, text, subtitles, UI elements, "
    		"bad composition, cropped, out of frame, "
   	 	"bad anatomy, extra limbs, extra fingers, deformed hands, "
    		"cartoon, anime, chibi, illustration, painterly, "
    		"oversharpened, harsh edges, color banding"
	)

        #############################################
        # BASE PIPELINE
        #############################################
        print("📥 Lade SDXL Base...")
        self.base = StableDiffusionXLPipeline.from_pretrained(
            model_base,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        # ✅ Richtiger Scheduler: DPM++ (dpmsolver++) + Karras
        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        # ✅ xFormers okay (normalerweise kein Qualitätsverlust, aber je nach Setup: testbar)
        # Wenn du maximal reproduzierbare Ergebnisse willst: auskommentieren.
        self.base.enable_xformers_memory_efficient_attention()

        # ❌ Auf 4090 für beste Qualität meist aus:
        # self.base.enable_vae_tiling()
        # self.base.enable_attention_slicing()

        # Optional: VAE in fp16 lassen (Standard). Wenn du Banding siehst:
        # self.base.vae.to(dtype=torch.float32)

        #############################################
        # REFINER
        #############################################
        self.refiner = None
        if self.use_refiner:
            print("📥 Lade SDXL Refiner...")
            self.refiner = StableDiffusionXLRefinerPipeline.from_pretrained(
                model_refiner,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.device)

            self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(
                self.refiner.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )

            self.refiner.enable_xformers_memory_efficient_attention()
            # ❌ für Qualität aus lassen:
            # self.refiner.enable_vae_tiling()
            # self.refiner.enable_attention_slicing()

        print("✨ Ultra-Quality SDXL bereit!")

    def _validate_dims(self, w: int, h: int) -> None:
        # SDXL arbeitet am saubersten mit Dimensionen, die durch 64 teilbar sind.
        if (w % 64) != 0 or (h % 64) != 0:
            print(
                f"⚠️ Hinweis: {w}x{h} ist nicht durch 64 teilbar. "
                "Das kann leicht Qualität/Speed beeinflussen."
            )

    #############################################
    # IMAGE GENERATION
    #############################################
    @torch.inference_mode()
    def generate(self, prompt: str, seed: int = 42):
        print("🎨 STARTE ULTRA-SDXL RENDERING...")

        w, h = self.output_width, self.output_height
        self._validate_dims(w, h)

        # Einheitlicher Generator (reproduzierbar)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # Refiner Split: Base endet bei z.B. 0.8, Refiner startet bei 0.8
        split = max(0.0, min(1.0, self.refiner_split))

        if self.use_refiner and self.refiner is not None:
            print(f"➡ Base denoising_end={split:.2f} | Refiner denoising_start={split:.2f}")

            # ✅ Base in Latents bis zum Split
            base_latents = self.base(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                width=w,
                height=h,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                denoising_end=split,
                output_type="latent",
                generator=generator,
            ).images

            print("✨ REFINER WIRD ANGEWENDET...")

            # ✅ Refiner übernimmt ab Split
            refined = self.refiner(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                denoising_start=split,
                image=base_latents,
                generator=generator,
            ).images[0]

            return refined

        # Kein Refiner: direkt Base rendern
        img = self.base(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=w,
            height=h,
            guidance_scale=self.guidance,
            num_inference_steps=self.steps,
            generator=generator,
        ).images[0]
        return img


#############################################
# BATCH RENDER / JSON WORKFLOW
#############################################

class UltraBatchRenderer:
    def __init__(self, pipeline: UltraQualitySDXL):
        self.pipeline = pipeline

    def render_from_scenes(self, json_file: str, output_dir: str = "output_ultra"):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenes = data.get("scenes", [])
        print(f"📘 Lade {len(scenes)} Szenen aus JSON...")

        results = []
        for scene in scenes:
            scene_id = scene.get("id", 0)
            prompt = scene.get("image_prompt", "")

            # Optional: Seed pro Szene (falls in JSON vorhanden)
            seed = scene.get("seed", 42)

            print(f"🖼 Generiere Szene {scene_id} (seed={seed})...")

            t0 = time.time()
            img = self.pipeline.generate(prompt, seed=seed)
            dt = time.time() - t0

            filename = output / f"scene_{int(scene_id):04d}.png"
            img.save(filename)

            print(f"✅ Gespeichert: {filename} ({dt:.1f}s)")
            results.append(str(filename))

        return results
