#!/usr/bin/env python3
"""
Optimierte FLUX.1 Bildgenerator-Version
(Platzsparend + Stabil)
"""

import os
import torch
from diffusers import FluxPipeline
from pathlib import Path
import json
import time

# ----------------------------------------------------
# 🧹 Globale Optimierungen
# ----------------------------------------------------
# Hugging Face Cache zentral definieren
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Optional: beschleunigt CUDA auf stabilen Systemen
torch.backends.cudnn.benchmark = True


class FluxImageGenerator:
    def __init__(self,
                 model_name="black-forest-labs/FLUX.1-schnell",
                 output_format="16:9"):
        """
        Initialisiert FLUX.1 Pipeline (optimiert)
        """
        print("🎨 Initialisiere FLUX.1 Generator...")

        # GPU Check
        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine GPU gefunden! FLUX.1 benötigt CUDA.")

        print(f"✅ GPU erkannt: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

        # Auflösung wählen
        if output_format == "16:9":
            self.width, self.height = 1600, 900
        else:
            self.width, self.height = 1024, 1024

        # Modell laden
        try:
            print(f"📦 Lade Modell: {model_name}")
            self.pipe = FluxPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True
            )

            # Optimierungen
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

            print("✅ Modell erfolgreich geladen und optimiert!\n")

        except Exception as e:
            print(f"❌ Fehler beim Laden des Modells: {e}")
            print("💡 Prüfe Internet, Token und GPU-Kompatibilität.")
            raise

    # ----------------------------------------------------
    # 🔮 Einzelbild generieren
    # ----------------------------------------------------
    def generate_image(self,
                       prompt,
                       negative_prompt="text, watermark, low quality, blurry, distorted",
                       num_inference_steps=25,
                       guidance_scale=4.0,
                       seed=None):
        """
        Generiert ein einzelnes Bild mit FLUX.1
        """
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            return result

        except Exception as e:
            print(f"❌ Fehler bei der Bildgenerierung: {e}")
            raise

    # ----------------------------------------------------
    # 📖 JSON Batch-Verarbeitung
    # ----------------------------------------------------
    def generate_from_json(self,
                           metadata_file,
                           output_dir="output/images",
                           quality_preset="balanced"):
        """
        Generiert alle Bilder basierend auf einem JSON-File
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        quality_settings = {
            "fast": {"steps": 15, "guidance": 3.5},
            "balanced": {"steps": 25, "guidance": 4.0},
            "high": {"steps": 35, "guidance": 4.5}
        }
        settings = quality_settings.get(quality_preset, quality_settings["balanced"])

        print(f"📖 Lade JSON-Metadaten: {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        style = metadata.get("book_info", {}).get("style", "cinematic")
        scenes = metadata.get("scenes", [])
        if isinstance(scenes, dict):
            scenes = list(scenes.values())

        print(f"\n🎨 Style: {style}")
        print(f"📊 Szenen: {len(scenes)} | Qualität: {quality_preset}")
        print(f"🖼️ Auflösung: {self.width}x{self.height}")
        print("=" * 60)

        generated_images, total_time = [], 0

        for i, scene in enumerate(scenes, 1):
            prompt = scene.get("image_prompt", "")
            negative = scene.get("negative_prompt", "text, watermark, low quality, blurry")

            if style.lower() not in prompt.lower():
                prompt = f"{style}, {prompt}"

            image_file = output_path / f"image_{i:04d}.png"
            if image_file.exists():
                print(f"[{i:04d}] ⏭️ Überspringe (bereits vorhanden): {image_file.name}")
                generated_images.append(str(image_file))
                continue

            print(f"[{i:04d}/{len(scenes)}] 🎨 Generiere Bild...")
            print(f"         Prompt: {prompt[:150]}...")

            start = time.time()
            try:
                image = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=settings["steps"],
                    guidance_scale=settings["guidance"],
                    seed=42 + i
                )
                image.save(image_file, quality=95)
                duration = time.time() - start
                total_time += duration
                print(f"         ✅ {image_file.name} ({duration:.1f}s)")
                generated_images.append(str(image_file))

            except Exception as e:
                print(f"         ❌ Fehler bei Szene {i}: {e}")

        print("\n" + "=" * 60)
        print(f"✅ Fertig! {len(generated_images)}/{len(scenes)} Bilder generiert")
        print(f"⏱️ Gesamtzeit: {total_time/60:.1f} Minuten")
        print(f"📁 Ausgabe: {output_path}")

        # Metadaten speichern
        meta = {
            "total_images": len(generated_images),
            "resolution": f"{self.width}x{self.height}",
            "quality_preset": quality_preset,
            "style_used": style,
            "images": [
                {"id": i + 1, "file": img, "prompt": scenes[i].get("image_prompt", "")[:100]}
                for i, img in enumerate(generated_images)
            ]
        }

        meta_file = output_path / "images_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadaten gespeichert: {meta_file}")
        return generated_images


# ----------------------------------------------------
# 🧠 Hauptprogramm
# ----------------------------------------------------
def main():
    try:
        generator = FluxImageGenerator(output_format="16:9")
        images = generator.generate_from_json(
            metadata_file="root/workspace/storypainter/input/gegendaswelt/book_scenes (1).json",
            output_dir="root/workspace/storypainter/input/gegendaswelt/images",
            quality_preset="balanced"
        )
        print(f"\n🎉 {len(images)} FLUX.1 Bilder fertig!")
    except Exception as e:
        print(f"\n💥 Kritischer Fehler: {e}")
        print("   ➤ GPU prüfen")
        print("   ➤ HF-Token & Internet prüfen")
        print("   ➤ JSON-Struktur prüfen")


if __name__ == "__main__":
    main()
