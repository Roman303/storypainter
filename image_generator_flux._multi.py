#!/usr/bin/env python3
"""
FLUX.1 Bildgenerator – MIT STYLE aus JSON
(Version 2025.10 – erweitert mit Sub-Image-Unterstützung)
"""

import torch
from diffusers import FluxPipeline
from pathlib import Path
import json
import time


class FluxImageGenerator:
    def __init__(self,
                 model_name="black-forest-labs/FLUX.1-dev",
                 output_format="16:9"):
        """
        Initialisiert FLUX.1 Pipeline
        """
        print("🎨 Lade FLUX.1 Modell...")

        # GPU Check
        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine GPU gefunden! FLUX.1 benötigt CUDA.")

        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Format (Aspect Ratio)
        if output_format == "16:9":
            self.width = 1920
            self.height = 1080
        else:
            self.width = 1024
            self.height = 1024

        try:
            # FLUX Pipeline laden
            self.pipe = FluxPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )

            # Optimierungen
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()

            print("✅ FLUX.1 Pipeline bereit!\n")

        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            print("💡 Tipp: Prüfe, ob das Modell installiert ist.")
            raise

    def generate_image(self,
                       prompt,
                       negative_prompt="text, watermark, low quality, blurry, distorted",
                       num_inference_steps=25,
                       guidance_scale=4.0,
                       seed=None):
        """
        Generiert ein einzelnes Bild aus einem Prompt mit FLUX.1
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        try:
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
        Generiert alle Bilder (Szenen + Sub-Szenen) aus JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Quality Presets
        quality_settings = {
            "fast": {"steps": 15, "guidance": 3.5},
            "balanced": {"steps": 25, "guidance": 4.0},
            "high": {"steps": 35, "guidance": 4.5}
        }
        settings = quality_settings.get(quality_preset, quality_settings["balanced"])

        # JSON laden
        print(f"📖 Lade Metadaten: {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        book_info = metadata.get("book_info", {})
        style = book_info.get("style", "cinematic")
        total_scenes = book_info.get("total_scenes", 0)

        scenes = metadata.get("scenes", [])
        if isinstance(scenes, dict):
            scenes = list(scenes.values())

        print(f"🎬 Generiere {len(scenes)} Hauptszenen ({quality_preset})")
        print(f"   Steps: {settings['steps']}, Guidance: {settings['guidance']}")
        print(f"   Auflösung: {self.width}x{self.height}")
        print(f"   Style: {style}\n")

        generated_images = []
        total_time = 0

        for i, scene in enumerate(scenes, 1):
            scene_id = scene.get("id", i)
            prompt = scene.get("image_prompt", "")
            negative = scene.get("negative_prompt", "text, watermark, low quality, blurry")

            # Style hinzufügen
            if style and style.lower() not in prompt.lower():
                enhanced_prompt = f"{style}, {prompt}"
            else:
                enhanced_prompt = prompt

            image_file = output_path / f"image_{scene_id:04d}.png"

            # Hauptbild
            if not image_file.exists():
                print(f"[{i:04d}/{len(scenes)}] 🎨 Generiere Hauptbild ...")
                print(f"         Prompt: {enhanced_prompt[:200]}...")
                start = time.time()
                try:
                    image = self.generate_image(
                        prompt=enhanced_prompt,
                        negative_prompt=negative,
                        num_inference_steps=settings["steps"],
                        guidance_scale=settings["guidance"],
                        seed=42 + scene_id
                    )
                    image.save(image_file, quality=95)
                    duration = time.time() - start
                    total_time += duration
                    print(f"         ✅ Gespeichert ({duration:.1f}s): {image_file.name}")
                except Exception as e:
                    print(f"         ❌ Fehler bei Hauptbild: {e}")
                    continue
            else:
                print(f"[{i:04d}] ⏭️ Hauptbild vorhanden: {image_file.name}")

            generated_images.append(str(image_file))

            # === SUB-IMAGES ===
            sub_prompts = scene.get("sub_img_prompts", [])
            if sub_prompts:
                print(f"         ➕ {len(sub_prompts)} Folgebilder für Szene {scene_id} ...")
                for j, sub_prompt in enumerate(sub_prompts, 1):
                    sub_id = f"{scene_id:04d}_{j:02d}"
                    sub_file = output_path / f"image_{sub_id}.png"
                    if sub_file.exists():
                        print(f"            ⏭️ Folgebild existiert: {sub_file.name}")
                        continue

                    if style and style.lower() not in sub_prompt.lower():
                        enhanced_sub = f"{style}, {sub_prompt}"
                    else:
                        enhanced_sub = sub_prompt

                    try:
                        sub_img = self.generate_image(
                            prompt=enhanced_sub,
                            negative_prompt=negative,
                            num_inference_steps=settings["steps"],
                            guidance_scale=settings["guidance"],
                            seed=1000 + scene_id * 10 + j
                        )
                        sub_img.save(sub_file, quality=95)
                        print(f"            ✅ Gespeichert: {sub_file.name}")
                        generated_images.append(str(sub_file))
                    except Exception as e:
                        print(f"            ❌ Fehler bei Folgebild: {e}")

        # === Zusammenfassung ===
        print("\n" + "=" * 60)
        print(f"✅ Fertig! {len(generated_images)} Bilder erzeugt")
        print(f"⏱️ Gesamtzeit: {total_time/60:.1f} Minuten")
        print(f"📁 Ausgabe: {output_path}")
        print(f"🎨 Style: {style}")

        # Metadaten speichern
        image_metadata = {
            "total_images": len(generated_images),
            "resolution": f"{self.width}x{self.height}",
            "quality_preset": quality_preset,
            "style_used": style,
            "images": generated_images
        }
        meta_file = output_path / "images_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(image_metadata, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadaten gespeichert: {meta_file}")

        return generated_images


# Beispiel-Nutzung
def main():
    try:
        generator = FluxImageGenerator(output_format="16:9")

        images = generator.generate_from_json(
            metadata_file="/workspace/code/book_scenes.json",
            output_dir="/workspace/code/output_flux/images",
            quality_preset="balanced"
        )

        print(f"\n🎉 {len(images)} FLUX.1 Bilder bereit für Weiterverarbeitung!")

    except Exception as e:
        print(f"\n💥 Kritischer Fehler: {e}")
        print("💡 Prüfe mögliche Ursachen:")
        print("   1. GPU verfügbar?")
        print("   2. Modell installiert? (black-forest-labs/FLUX.1-dev)")
        print("   3. JSON-Format korrekt?")


if __name__ == "__main__":
    main()
