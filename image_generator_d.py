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
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

#############################################
# SDXL V14 - DreamShaper XL (HD Video optimiert)
#############################################

class UltraQualitySDXL:
    def __init__(
        self,
        model_base: str = "Lykon/dreamshaper-xl-1-0",
        model_refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_refiner: bool = False,
        output_width: int = 3072,
        output_height: int = 1728,
        steps: int = 36,
        guidance: float = 4.2,
        refiner_split: float = 0.65,
    ):
        print("🚀 Initialisiere DreamShaper XL Pipeline (ohne xformers)")

        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine CUDA-GPU gefunden!")

        self.device = "cuda"
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.use_refiner = bool(use_refiner)
        self.refiner_split = float(refiner_split)

        # CUDA / SDPA Optimierung (ERSATZ für xformers)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        # Qualitätsstufe
        if gpu_memory >= 24:
            quality_mode = "ULTRA"
            use_fp32_vae = True
            enable_slicing = False
            enable_tiling = False
        elif gpu_memory >= 16:
            quality_mode = "HIGH"
            use_fp32_vae = False
            enable_slicing = False
            enable_tiling = False
        else:
            quality_mode = "LOW"
            use_fp32_vae = False
            enable_slicing = True
            enable_tiling = True

        print(f"🎯 Qualitätsmodus: {quality_mode}")

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
        print("🔥 Lade DreamShaper XL...")
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

        # Speicher-Optimierungen
        self.base.enable_attention_slicing()
        self.base.enable_vae_slicing()

        if use_fp32_vae:
            self.base.vae.to(dtype=torch.float32)
            print("✅ VAE FP32 (kein Banding)")
        else:
            print("ℹ️ VAE FP16")

        if enable_tiling:
            self.base.enable_vae_tiling()

        torch.cuda.empty_cache()

        # OPTIONALER REFINER (standard AUS)
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

            self.refiner.enable_attention_slicing()
            self.refiner.enable_vae_slicing()

        print(f"✨ Pipeline bereit: {self.output_width}x{self.output_height}, {self.steps} steps")

    @torch.inference_mode()
    def generate(self, prompt: str, negative_prompt: str = None, seed: int = 42):
        torch.cuda.empty_cache()

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        neg_prompt = negative_prompt if negative_prompt else self.default_negative

        start = time.time()

        output = self.base(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=self.output_width,
            height=self.output_height,
            guidance_scale=self.guidance,
            num_inference_steps=self.steps,
            generator=generator,
        )

        torch.cuda.empty_cache()
        return output.images[0], time.time() - start


def process_book(input_path: Path, pipeline: UltraQualitySDXL):
    json_file = input_path / "book_scenes.json"
    if not json_file.exists():
        print("❌ book_scenes.json fehlt")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    output_dir = input_path / "renders"
    output_dir.mkdir(exist_ok=True)

    for i, scene in enumerate(scenes, 1):
        prompt = scene.get("image_prompt", "")
        seed = scene.get("seed", 42)

        print(f"🖼️ Szene {i}/{len(scenes)}")
        img, t = pipeline.generate(prompt, seed=seed)

        fn = output_dir / f"scene_{i:04d}.png"
        img.save(fn)
        print(f"✅ {fn.name} ({t:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--height", type=int, default=1728)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--guidance", type=float, default=4.2)
    args = parser.parse_args()

    pipeline = UltraQualitySDXL(
        output_width=args.width,
        output_height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        use_refiner=False,
    )

    process_book(Path(args.path), pipeline)


if __name__ == "__main__":
    main()
