#!/usr/bin/env python3
"""
DreamShaper XL - Schnelltest
Generiert ein Testbild zur Verifikation der Installation
"""

import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
import time

def main():
    print("🔍 Prüfe CUDA...")
    if not torch.cuda.is_available():
        print("❌ CUDA nicht verfügbar!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    print("\n🔥 Lade DreamShaper XL...")
    start_load = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-xl-1-0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        add_watermarker=False,
    ).to("cuda")
    
    # Scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    
    # SDPA aktivieren
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # VAE Optimierung (entweder FP32 ODER Tiling, nicht beides)
    # Option 1: VAE FP32 für beste Qualität (benötigt mehr VRAM)
    # pipe.vae.to(dtype=torch.float32)
    
    # Option 2: VAE Tiling für weniger VRAM (Standard)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    
    load_time = time.time() - start_load
    print(f"✅ Geladen in {load_time:.1f}s")
    
    print("\n🎨 Generiere Testbild...")
    prompt = (
        "a majestic mountain landscape at golden hour, "
        "snow-capped peaks, crystal clear alpine lake, "
        "dramatic clouds, cinematic lighting, ultra detailed, "
        "professional photography, 8k uhd, dslr"
    )
    
    negative = (
        "blurry, low quality, cartoon, anime, painting, "
        "oversaturated, watermark, text"
    )
    
    start_gen = time.time()
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=1536,
        height=1024,
        guidance_scale=7.0,
        num_inference_steps=35,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    
    gen_time = time.time() - start_gen
    
    output_path = "/workspace/test_dreamshaper.png"
    image.save(output_path, optimize=True, quality=95)
    
    print(f"✅ Generiert in {gen_time:.1f}s")
    print(f"💾 Gespeichert: {output_path}")
    print("\n🎉 Installation erfolgreich!")
    print(f"📊 Performance: {gen_time:.1f}s für 1536x1024 @ 35 steps")

if __name__ == "__main__":
    main()