#!/usr/bin/env python3
"""
Stable Diffusion XL Bildgenerator - MIT STYLE aus JSON
"""

import torch
from diffusers import DiffusionPipeline
from pathlib import Path
import json
import time
import argparse
import os

from fx_system import apply_fx

class AudiobookImageGenerator:
    def __init__(self, 
                 model_name="stabilityai/stable-diffusion-xl-base-1.0",
                 use_refiner=False,
                 output_format="16:9"):
        """
        Initialisiert SDXL Pipeline
        """
        self.global_negative_prompt = "lowres, blurry, deformed, extra limbs, text, watermark, logo, cgi, 3d render, cartoon, anime, digital art, modern photography, sharp digital edges, overprocessed, unrealistic skin, oversaturated"


        print(f"🎨 Lade Stable Diffusion XL...")
        
        # GPU-Check
        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine GPU gefunden! SDXL braucht CUDA.")
        
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Bildformat
        if output_format == "16:9":
            self.width = 1344
            self.height = 768
        else:
            self.width = 1024
            self.height = 1024
        
        try:
            # Base Pipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Memory-Optimierungen für A4000
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()
            
            print("✅ SDXL Pipeline bereit!\n")
            
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            print("💡 Tipp: libGL Problem? Führe aus: apt install -y libgl1-mesa-glx")
            raise

    def generate_image(self, 
                      prompt, 
                      negative_prompt="text, watermark, low quality, blurry, distorted",
                      num_inference_steps=30,
                      guidance_scale=7.5,
                      seed=None):
        """
        Generiert ein Bild aus Prompt
        """
        # Seed setzen falls gewünscht
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        try:
            # Image generieren
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            return image
            
        except Exception as e:
            print(f"❌ Fehler bei Bildgenerierung: {e}")
            raise

    def generate_from_json(self, 
                          metadata_file,
                          output_dir="output/images",
                          quality_preset="balanced"):
        """
        Generiert alle Bilder aus Mistral-Metadaten MIT STYLE
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Quality Presets
        quality_settings = {
            "fast": {"steps": 20, "guidance": 7.0},
            "balanced": {"steps": 30, "guidance": 7.5},
            "high": {"steps": 50, "guidance": 8.0}
        }
        settings = quality_settings.get(quality_preset, quality_settings["balanced"])
        
        # Lade Metadaten
        print(f"📖 Lade Metadaten: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extrahiere Style aus book_info
        book_info = metadata.get("book_info", {})
        style = book_info.get("style", "cinematic")
        total_scenes = book_info.get("total_scenes", 0)
        fx_type = book_info.get("fx", "none")
        
        print(f"🎨 Style aus JSON: {style}")
        print(f"📊 Szenen total: {total_scenes}")
        
        # KORRIGIERT: Handle both list and dict formats
        scenes = metadata.get("scenes", [])
        if isinstance(scenes, dict):  # Falls es doch ein Dict ist
            scenes = list(scenes.values())
        
        print(f"🎨 Generiere {len(scenes)} Bilder ({quality_preset} quality)")
        print(f"   Steps: {settings['steps']}, Guidance: {settings['guidance']}")
        print(f"   Format: {self.width}x{self.height}")
        print(f"   Style: {style}\n")
        
        generated_images = []
        total_time = 0
        
        for i, scene in enumerate(scenes, 1):
            scene_id = scene.get("id", i)
            prompt = scene.get("image_prompt", "")
            scene_negative = scene.get("negative_prompt", "")
            # kombiniere globalen und szenenspezifischen Negativprompt
            negative = f"{self.global_negative_prompt}, {scene_negative}".strip(", ")

            
            # ✅ STYLE ZUM PROMPT HINZUFÜGEN
            if style and style.lower() not in prompt.lower():
                enhanced_prompt = f"{prompt},{style}"
                print(f"   🎨 Style hinzugefügt: {style[:50]}...")
            else:
                enhanced_prompt = prompt
            
            # Dateiname
            image_file = output_path / f"image_{scene_id:04d}.png"
            
            # Skip wenn bereits existiert
            if image_file.exists():
                print(f"[{i:04d}] ⏭️ Bereits vorhanden: {image_file.name}")
                generated_images.append(str(image_file))
                continue
            
            # Generiere Bild
            print(f"[{i:04d}/{len(scenes)}] 🎨 Generiere Bild...")
            print(f"         Prompt: {enhanced_prompt[:800]}...")
            
            start = time.time()
            try:
                image = self.generate_image(
                    prompt=enhanced_prompt,
                    negative_prompt=negative,
                    num_inference_steps=settings['steps'],
                    guidance_scale=settings['guidance'],
                    seed=42 + scene_id  # Konsistente Seeds
                )

                if fx_type and fx_type != "none":
                    image = apply_fx(image, fx_type)
                    print(f"🎞️ FX angewendet: {fx_type}")
                
                # Speichere Bild
                image.save(image_file, quality=95)
                duration = time.time() - start
                total_time += duration
                
                print(f"         ✅ Gespeichert ({duration:.1f}s): {image_file.name}")
                generated_images.append(str(image_file))
                
            except Exception as e:
                print(f"         ❌ Fehler: {e}")
                continue
        
        # Zusammenfassung
        print(f"\n{'='*60}")
        print(f"✅ Fertig! {len(generated_images)}/{len(scenes)} Bilder generiert")
        print(f"⏱️ Gesamtzeit: {total_time/60:.1f} Minuten")
        print(f"📁 Ausgabe: {output_path}")
        print(f"🎨 Verwendeter Style: {style}")
        
        # Speichere Bild-Metadaten
        image_metadata = {
            "total_images": len(generated_images),
            "resolution": f"{self.width}x{self.height}",
            "quality_preset": quality_preset,
            "style_used": style,
            "images": [
                {
                    "scene_id": scenes[i].get("id", i+1),
                    "file": img,
                    "prompt": scenes[i].get("image_prompt", "")[:100],
                    "enhanced_prompt": enhanced_prompt[:100] if 'enhanced_prompt' in locals() else ""
                }
                for i, img in enumerate(generated_images)
            ]
        }
        
        meta_file = output_path / "images_metadata.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(image_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Metadaten: {meta_file}")
        
        return generated_images

def main():
    # --- CLI ---
    parser = argparse.ArgumentParser(description="Audiobook Image Generator (Pfad-basiert)")
    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")
    parser.add_argument("--quality", default="high", help="Qualitätspreset (z. B. low, medium, high)")
    parser.add_argument("--format", default="16:9", help="Bildformat (z. B. 1:1, 16:9, 9:16)")
    parser.add_argument("--use-refiner", action="store_false", help="WICHTIG: Kein Refiner für A4000 (default: an)")
    args = parser.parse_args()

    base_path = args.path

    # --- CONFIG ---
    CONFIG = {
        # Eingabe / Ausgabe (aus --path zusammengesetzt)
        "metadata_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "images"),

        # Generator-Einstellungen
        "use_refiner": args.use_refiner,
        "output_format": args.format,
        "quality_preset": args.quality,
    }

    # --- Validierung & Vorbereitung ---
    if not os.path.isfile(CONFIG["metadata_file"]):
        raise FileNotFoundError(f"Metadata-Datei nicht gefunden: {CONFIG['metadata_file']}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # --- Debug-Ausgabe ---
    print("✅ Using configuration:")
    for k, v in CONFIG.items():
        print(f"{k:18}: {v}")

    # --- Hauptlogik ---
    try:
        generator = AudiobookImageGenerator(
            use_refiner=CONFIG["use_refiner"],
            output_format=CONFIG["output_format"],
        )

        images = generator.generate_from_json(
            metadata_file=CONFIG["metadata_file"],
            output_dir=CONFIG["output_dir"],
            quality_preset=CONFIG["quality_preset"],
        )

        print(f"\n🎉 {len(images)} Bilder bereit für Video-Rendering!")

    except Exception as e:
        print(f"\n💥 KRITISCHER FEHLER: {e}")
        print("💡 Mögliche Lösungen:")
        print("   1. apt install -y libgl1-mesa-glx libglib2.0-0")
        print("   2. Environment prüfen: source /workspace/sdxl_env/bin/activate")
        print("   3. JSON Format prüfen")

if __name__ == "__main__":
    main()