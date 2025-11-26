import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLRefinerPipeline,
)
from diffusers.schedulers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
import time
from pathlib import Path
import json
import os

#############################################
# SDXL ULTRA QUALITY PIPELINE (RTX 4090)
# - Native 2304x1296
# - Refiner 80/20 workflow
# - High-end negative prompts
# - Optimal scheduler (DPM++ 2M Karras)
# - Latent upscaling optional
#############################################

class UltraQualitySDXL:
    def __init__(self,
                 model_base="stabilityai/stable-diffusion-xl-base-1.0",
                 model_refiner="stabilityai/stable-diffusion-xl-refiner-1.0",
                 use_refiner=True,
                 output_width=2304,
                 output_height=1296,
                 steps=60,
                 guidance=6.0):

        print("🚀 Initialisiere Ultra-Quality SDXL...")

        if not torch.cuda.is_available():
            raise RuntimeError("Keine GPU gefunden – 4090 nötig!")

        self.device = "cuda"
        self.output_width = output_width
        self.output_height = output_height
        self.steps = steps
        self.guidance = guidance
        self.use_refiner = use_refiner

        print(f"GPU: {torch.cuda.get_device_name()}")

        #############################################
        # BASE PIPELINE
        #############################################
        print("📥 Lade SDXL Base...")
        self.base = StableDiffusionXLPipeline.from_pretrained(
            model_base,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)

        # Optimaler Scheduler
        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config
        )

        # Speicheroptimiert
        self.base.enable_xformers_memory_efficient_attention()
        self.base.enable_vae_tiling()
        self.base.enable_attention_slicing()

        #############################################
        # REFINER
        #############################################
        if use_refiner:
            print("📥 Lade SDXL Refiner...")
            self.refiner = StableDiffusionXLRefinerPipeline.from_pretrained(
                model_refiner,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)

            self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(
                self.refiner.scheduler.config
            )

            self.refiner.enable_xformers_memory_efficient_attention()
            self.refiner.enable_vae_tiling()
            self.refiner.enable_attention_slicing()

        #############################################
        # Negative Prompt (HIGH-END)
        #############################################
        self.negative_prompt = (
            "(worst quality:1.4), (low quality:1.4), (jpeg artifacts:1.3), "
            "nsfw, watermark, text, logo, blurry, distorted, mutated hands, extra limbs, "
            "overexposed, underexposed, bad composition, bad anatomy, deformed fingers, "
            "glitch, noisy, artifacts, cartoon, anime"
        )

        print("✨ Ultra-Quality SDXL bereit!")

    #############################################
    # IMAGE GENERATION
    #############################################
    def generate(self, prompt, seed=42):
        print("🎨 STARTE ULTRA-SDXL RENDERING...")

        generator = torch.Generator(self.device).manual_seed(seed)

        # 80% Base
        base_steps = int(self.steps * 0.8)
        # 20% Refiner
        refiner_steps = self.steps - base_steps

        print(f"➡ Base: {base_steps} Steps | Refiner: {refiner_steps} Steps")

        #############################################
        # BASE PHASE
        #############################################
        base_image = self.base(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=self.output_width,
            height=self.output_height,
            guidance_scale=self.guidance,
            num_inference_steps=base_steps,
            output_type="latent",
            generator=generator,
        ).images

        #############################################
        # REFINER PHASE
        #############################################
        if self.use_refiner:
            print("✨ REFINER WIRD ANGEWENDET...")
            refined = self.refiner(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=refiner_steps,
                image=base_image,
                generator=generator,
            ).images[0]
            return refined
        else:
            return self.base.decode_latents(base_image)[0]


#############################################
# BATCH RENDER / JSON WORKFLOW
#############################################

class UltraBatchRenderer:
    def __init__(self, pipeline: UltraQualitySDXL):
        self.pipeline = pipeline

    def render_from_scenes(self, json_file, output_dir="output_ultra"):
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

            print(f"🖼 Generiere Szene {scene_id}...")

            img = self.pipeline.generate(prompt)

            filename = output / f"scene_{scene_id:04d}.png"
            img.save(filename)
            results.append(str(filename))

        return results
