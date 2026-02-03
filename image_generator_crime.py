#!/usr/bin/env python3
"""
SDXL V14 - CRIME/NOIR CINEMATIC IMAGE GENERATOR - 16GB OPTIMIERT
Kinoreife Crime-Szenen mit cinematic LoRAs
Optimiert für RTX 4000 / A4000 / ähnliche 16GB GPUs
"""

import os
import json
import time
import argparse
import random
import hashlib
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image, PngImagePlugin

#############################################
# AGGRESSIVE MEMORY MANAGEMENT
#############################################

def aggressive_memory_cleanup():
    """Aggressives Memory-Cleanup für 16GB GPUs"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def get_memory_stats():
    """Zeigt aktuelle VRAM-Nutzung"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        return f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Free: {free:.2f}GB"
    return "N/A"

#############################################
# LoRA MANAGER - MEMORY-OPTIMIERT
#############################################

class LoRAManager:
    """Verwaltet LoRAs mit aggressivem Memory Management"""
    
    DEFAULT_LORA_PATHS = {
        "cinematic_lighting": {
            "path": "./loras/sdxl/cinematic_lighting_xl.safetensors",
            "scale": 0.75,
            "trigger": "cinematic lighting, dramatic lighting, film noir",
            "description": "Professionelle Kino-Beleuchtung"
        },
        "film_grain": {
            "path": "./loras/sdxl/film_grain_xl.safetensors",
            "scale": 0.6,
            "trigger": "film grain, 35mm film, analog photography",
            "description": "Authentisches Filmkorn"
        },
        "realistic_vision": {
            "path": "./loras/sdxl/realistic_vision_xl.safetensors",
            "scale": 0.65,
            "trigger": "photorealistic, professional photography",
            "description": "Fotografischer Realismus"
        },
        "detail_tweaker": {
            "path": "./loras/sdxl/detail_tweaker_xl.safetensors",
            "scale": 0.6,
            "trigger": "hyperdetailed, intricate details",
            "description": "Verbessert feine Details"
        },
        "noir_style": {
            "path": "./loras/sdxl/noir_style_xl.safetensors",
            "scale": 0.7,
            "trigger": "film noir style, high contrast, chiaroscuro",
            "description": "Film Noir Ästhetik"
        }
    }
    
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        self.loaded_loras = {}
        self.active_adapters = []
        self.weights = {}
    
    def load_lora(self, lora_name: str, custom_path: str = None, scale: float = None):
        """Lädt eine einzelne LoRA"""
        lora_config = self.DEFAULT_LORA_PATHS.get(lora_name)
        
        if not lora_config and not custom_path:
            print(f"⚠️  LoRA '{lora_name}' nicht in Defaults, benötige --lora-path")
            return False
        
        lora_path = custom_path or lora_config["path"]
        lora_scale = scale or lora_config.get("scale", 0.6)
        
        if not os.path.exists(lora_path):
            print(f"⚠️  LoRA Datei nicht gefunden: {lora_path}")
            return False
        
        try:
            aggressive_memory_cleanup()
            
            self.base_pipeline.load_lora_weights(
                lora_path,
                adapter_name=lora_name
            )
            
            self.loaded_loras[lora_name] = {
                "path": lora_path,
                "scale": lora_scale,
                "config": lora_config
            }
            
            print(f"✅ LoRA geladen: {lora_name} (Scale: {lora_scale})")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden von LoRA {lora_name}: {e}")
            return False
    
    def activate_loras(self, lora_list: List[str], weights: List[float] = None):
        """Aktiviert LoRAs mit Memory-Check"""
        if not lora_list:
            return
        
        active_loras = []
        active_weights = []
        
        for i, lora_name in enumerate(lora_list):
            if lora_name in self.loaded_loras:
                active_loras.append(lora_name)
                weight = weights[i] if weights and i < len(weights) else self.loaded_loras[lora_name]["scale"]
                active_weights.append(weight)
        
        if active_loras:
            self.base_pipeline.set_adapters(active_loras, adapter_weights=active_weights)
            self.active_adapters = active_loras
            self.weights = dict(zip(active_loras, active_weights))
            
            print("🎛️  Aktive LoRAs:")
            for lora, weight in zip(active_loras, active_weights):
                print(f"   • {lora}: {weight}")
            print(f"   💾 {get_memory_stats()}")
    
    def deactivate_loras(self):
        """Deaktiviert alle LoRAs mit Memory-Cleanup"""
        if self.active_adapters:
            self.base_pipeline.disable_lora()
            self.active_adapters = []
            self.weights = {}
            aggressive_memory_cleanup()
            print(f"⚙️  LoRAs deaktiviert | 💾 {get_memory_stats()}")
    
    def get_lora_triggers(self, lora_list: List[str]) -> str:
        """Gibt Trigger-Wörter für aktive LoRAs zurück"""
        triggers = []
        for lora_name in lora_list:
            if lora_name in self.loaded_loras:
                config = self.loaded_loras[lora_name].get("config", {})
                trigger = config.get("trigger", "")
                if trigger:
                    triggers.append(trigger)
        return ", ".join(triggers)
    
    def create_lora_signature(self, lora_list: List[str], weights: List[float]) -> str:
        """Erstellt eindeutige Signatur für LoRA-Kombination"""
        if not lora_list:
            return "no_lora"
        
        signature_parts = []
        for lora, weight in zip(lora_list, weights):
            signature_parts.append(f"{lora}_{weight:.2f}")
        
        signature = "_".join(signature_parts)
        return hashlib.md5(signature.encode()).hexdigest()[:8]

#############################################
# BOTANICAL SDXL - 16GB OPTIMIERT
#############################################

class BotanicalSDXLLoRA:
    def __init__(
        self,
        model_base: str = "SG161222/RealVisXL_V4.0",
        use_refiner: bool = False,
        output_width: int = 1536,
        output_height: int = 864,
        steps: int = 35,
        guidance: float = 5.5,
    ):
        print("🎬 Initialisiere CRIME CINEMATIC SDXL (16GB optimiert)...")
        
        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine CUDA-GPU gefunden!")
        
        self.device = "cuda"
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.use_refiner = bool(use_refiner)
        
        # Performance Optimierungen
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 15:
            print(f"⚠️  WARNUNG: GPU hat nur {gpu_memory:.1f}GB VRAM - Fehler möglich!")
        
        self.quality_mode = "MEDIUM_16GB"
        print(f"🎯 Qualitätsmodus: {self.quality_mode}")
        
        # CRIME/NOIR CINEMATIC PROMPT TEMPLATES
        self.cinematic_templates = {
            "noir_detective": {
                "base": "{scene}, film noir style, detective investigating",
                "enhancers": [
                    "high contrast lighting", "dramatic shadows", "chiaroscuro",
                    "1940s noir aesthetic", "venetian blind shadows", "moody atmosphere",
                    "cinematic composition", "shallow depth of field", "atmospheric fog"
                ],
                "recommended_loras": ["cinematic_lighting", "noir_style", "realistic_vision"]
            },
            "crime_scene": {
                "base": "{scene}, crime scene investigation",
                "enhancers": [
                    "forensic photography", "evidence markers", "dramatic lighting",
                    "gritty realism", "shallow focus", "cinematic framing",
                    "police tape visible", "urban decay", "ambient light"
                ],
                "recommended_loras": ["realistic_vision", "detail_tweaker"]
            },
            "urban_noir": {
                "base": "{scene}, urban noir cityscape",
                "enhancers": [
                    "rain-slicked streets", "neon reflections", "night photography",
                    "cinematic color grading", "blue and orange tones", "atmospheric haze",
                    "street lights", "urban grit", "film noir cinematography"
                ],
                "recommended_loras": ["cinematic_lighting", "film_grain", "realistic_vision"]
            },
            "interrogation": {
                "base": "{scene}, interrogation room scene",
                "enhancers": [
                    "single overhead light", "stark contrast", "claustrophobic framing",
                    "dramatic shadows on face", "institutional setting", "tense atmosphere",
                    "cinematic lighting", "cold color temperature", "high contrast"
                ],
                "recommended_loras": ["cinematic_lighting", "noir_style"]
            },
            "chase_action": {
                "base": "{scene}, action chase sequence",
                "enhancers": [
                    "motion blur", "dynamic angle", "handheld camera style",
                    "gritty realism", "urban environment", "dramatic lighting",
                    "cinematic action", "shallow depth of field", "film grain"
                ],
                "recommended_loras": ["cinematic_lighting", "film_grain", "realistic_vision"]
            },
            "mobster_meeting": {
                "base": "{scene}, organized crime meeting",
                "enhancers": [
                    "smoky atmosphere", "dim lighting", "1970s aesthetic",
                    "godfather style", "rich colors", "cinematic composition",
                    "dramatic shadows", "luxurious interior", "tension"
                ],
                "recommended_loras": ["cinematic_lighting", "realistic_vision"]
            },
            "alley_confrontation": {
                "base": "{scene}, dark alley confrontation",
                "enhancers": [
                    "harsh street lighting", "wet pavement", "urban decay",
                    "dramatic shadows", "film noir style", "atmospheric fog",
                    "cinematic framing", "high contrast", "gritty texture"
                ],
                "recommended_loras": ["cinematic_lighting", "noir_style", "film_grain"]
            }
        }
        
        self.mood_options = {
            "tense": ["high tension", "suspenseful atmosphere", "dramatic intensity"],
            "dark": ["dark mood", "ominous atmosphere", "foreboding"],
            "gritty": ["gritty realism", "raw texture", "urban decay"],
            "atmospheric": ["atmospheric haze", "moody lighting", "cinematic fog"],
            "violent": ["aftermath of violence", "crime scene markers", "forensic details"]
        }
        
        self.default_negative = (
            "cartoon, anime, illustration, painting, drawing, "
            "bright colors, oversaturated, cheerful, happy, clean, pristine, "
            "CGI, 3D render, video game graphics, "
            "blurry, out of focus, poorly drawn, bad anatomy, deformed, "
            "watermark, logo, text, subtitles, UI elements, "
            "low quality, jpeg artifacts, compression, amateur, "
            "modern smartphone aesthetic, selfie, social media filter"
        )
        
        # PIPELINE INITIALISIERUNG
        self._initialize_pipelines(model_base, gpu_memory)
        
        # LoRA MANAGER
        self.lora_manager = LoRAManager(self.base)
        
        print(f"✨ Pipeline bereit | 💾 {get_memory_stats()}")
    
    def _initialize_pipelines(self, model_base: str, gpu_memory: float):
        """Initialisiert Pipeline mit aggressiven Memory-Optimierungen"""
        
        print("🔥 Lade RealVisXL (16GB optimiert)...")
        
        aggressive_memory_cleanup()
        
        try:
            self.base = DiffusionPipeline.from_pretrained(
                model_base,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.device)
            print("✅ RealVisXL V4.0 geladen")
        except Exception as e:
            print(f"⚠️  RealVisXL nicht verfügbar, verwende Standard SDXL: {e}")
            self.base = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.device)
        
        # Optimierter Scheduler
        self.base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        
        # xFormers aktivieren
        try:
            self.base.enable_xformers_memory_efficient_attention()
            print("✅ xFormers aktiviert")
        except:
            print("ℹ️  xFormers nicht verfügbar")
        
        # KRITISCH: VAE in FP16 für 16GB GPU
        self.base.vae.to(dtype=torch.float16)
        print("⚙️  VAE in FP16 (Memory-optimiert)")
        
        # KRITISCH: Alle Memory-Optimierungen
        self.base.enable_attention_slicing(slice_size=1)
        self.base.enable_vae_tiling()
        print("✅ Attention Slicing aktiviert")
        print("✅ VAE Tiling aktiviert")
        
        # CPU Offloading für maximale VRAM-Ersparnis
        self.base.enable_model_cpu_offload()
        print("✅ CPU Offloading aktiviert")
        
        aggressive_memory_cleanup()
        
        # Refiner wird bei 16GB nicht empfohlen
        self.refiner = None
        if self.use_refiner:
            print("⚠️  WARNUNG: Refiner bei 16GB GPU nicht empfohlen!")
            print("⚠️  Dies kann zu CUDA Out-of-Memory Fehlern führen!")
            print("⚠️  Verwende --no-refiner für stabile Generierung")
    
    def create_crime_prompt(self, scene_description: str, style: str = "noir_detective", 
                           mood: str = "tense", custom_detail: str = None) -> Tuple[str, List[str]]:
        """Erstellt optimierte Crime/Noir Cinematic Prompts mit LoRA-Empfehlungen"""
        template = self.cinematic_templates.get(style, self.cinematic_templates["noir_detective"])
        
        # Mood-Details hinzufügen
        if custom_detail:
            mood_detail = custom_detail
        else:
            mood_options = self.mood_options.get(mood, self.mood_options["tense"])
            mood_detail = random.choice(mood_options)
        
        # Basis-Prompt
        base_prompt = template["base"].format(scene=scene_description)
        
        # Enhancer auswählen
        enhancers = random.sample(template["enhancers"], min(5, len(template["enhancers"])))
        
        # Kamera-Setup für Crime/Noir
        camera_specs = [
            "shot on Arri Alexa", "anamorphic lens", "Panavision camera",
            "35mm film", "Cooke anamorphic", "cinematic bokeh",
            "Roger Deakins cinematography", "Emmanuel Lubezki style"
        ]
        camera = random.choice(camera_specs)
        
        # Lighting-Specs
        lighting_specs = [
            "Rembrandt lighting", "low key lighting", "motivated lighting",
            "practical lights only", "natural window light", "streetlight ambience"
        ]
        lighting = random.choice(lighting_specs)
        
        prompt_parts = [base_prompt, mood_detail] + enhancers + [camera, lighting]
        
        # LoRA-Trigger hinzufügen
        lora_triggers = self.lora_manager.get_lora_triggers(template["recommended_loras"])
        if lora_triggers:
            prompt_parts.append(lora_triggers)
        
        final_prompt = ", ".join(prompt_parts)
        
        # Länge optimieren
        if len(final_prompt) > 350:
            words = final_prompt.split(", ")
            final_prompt = ", ".join(words[:16])
        
        return final_prompt, template["recommended_loras"]
    
    def load_loras(self, lora_names: List[str], custom_paths: Dict[str, str] = None, 
                  custom_scales: Dict[str, float] = None):
        """Lädt und konfiguriert LoRAs"""
        print(f"\n📦 Lade {len(lora_names)} LoRAs...")
        
        for lora_name in lora_names:
            custom_path = custom_paths.get(lora_name) if custom_paths else None
            custom_scale = custom_scales.get(lora_name) if custom_scales else None
            
            self.lora_manager.load_lora(lora_name, custom_path, custom_scale)
    
    def generate_with_loras(
        self,
        prompt: str,
        lora_names: List[str] = None,
        lora_weights: List[float] = None,
        negative_prompt: str = None,
        seed: int = None,
        disable_loras_after: bool = True
    ) -> Tuple[Image.Image, float, Dict]:
        """Generiert Bild mit LoRAs und aggressivem Memory Management"""
        
        # KRITISCH: Aggressive Memory cleanup VOR Generation
        aggressive_memory_cleanup()
        print(f"💾 Vor Generation: {get_memory_stats()}")
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        
        neg_prompt = negative_prompt or self.default_negative
        
        lora_signature = ""
        if lora_names:
            self.lora_manager.activate_loras(lora_names, lora_weights)
            lora_signature = self.lora_manager.create_lora_signature(lora_names, lora_weights or [])
        
        start_time = time.time()
        
        try:
            # Nur Base Generation (kein Refiner bei 16GB)
            output = self.base(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=self.output_width,
                height=self.output_height,
                guidance_scale=self.guidance,
                num_inference_steps=self.steps,
                generator=generator,
            )
            img = output.images[0]
            
            elapsed = time.time() - start_time
            
            metadata = {
                "prompt": prompt,
                "negative_prompt": neg_prompt[:500],
                "seed": seed,
                "steps": self.steps,
                "guidance": self.guidance,
                "size": f"{self.output_width}x{self.output_height}",
                "model": "RealVisXL_V4.0_16GB",
                "refiner_used": False,
                "generation_time": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "lora_signature": lora_signature,
                "active_loras": lora_names if lora_names else [],
                "lora_weights": lora_weights if lora_weights else []
            }
            
            # KRITISCH: Cleanup NACH Generation
            if disable_loras_after and lora_names:
                self.lora_manager.deactivate_loras()
            
            aggressive_memory_cleanup()
            print(f"💾 Nach Generation: {get_memory_stats()}")
            
            return img, elapsed, metadata
            
        except Exception as e:
            if lora_names:
                self.lora_manager.deactivate_loras()
            aggressive_memory_cleanup()
            raise RuntimeError(f"Generation fehlgeschlagen: {str(e)}")
    
    def save_with_metadata(self, image: Image.Image, filepath: Path, metadata: Dict):
        """Speichert Bild mit umfangreichen Metadaten"""
        png_info = PngImagePlugin.PngInfo()
        
        # Alle Metadaten als JSON
        png_info.add_text("generation_data", json.dumps(metadata, ensure_ascii=False))
        
        # Einzelne Felder für Kompatibilität
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                png_info.add_text(key, str(value))
        
        # EXIF Daten
        exif = image.getexif()
        exif[270] = metadata.get("prompt", "")[:100]
        exif[305] = "BotanicalSDXL-LoRA-16GB"
        exif[306] = metadata.get("timestamp", "")
        
        image.save(filepath, "PNG", pnginfo=png_info, exif=exif, quality=95, optimize=True)

#############################################
# HAUPTPROZESSOR
#############################################

def process_crime_scenes_with_loras(input_path: Path, pipeline: BotanicalSDXLLoRA, args):
    """Verarbeitet Crime-Szenen mit LoRA-Optimierung"""
    
    json_file = input_path / "book_scenes.json"
    
    if not json_file.exists():
        print(f"❌ Keine book_scenes.json gefunden in: {input_path}")
        return [], [], None
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    book_info = data.get("book_info", {})
    title = book_info.get("title", "Botanisches Hörbuch")
    scenes = data.get("scenes", [])
    
    if not scenes:
        print(f"❌ Keine Szenen gefunden")
        return [], [], None
    
    # Szenen-Filter wenn angegeben
    start_scene = args.start_scene if hasattr(args, 'start_scene') and args.start_scene else 1
    end_scene = args.end_scene if hasattr(args, 'end_scene') and args.end_scene else len(scenes)
    
    scenes = scenes[start_scene-1:end_scene]
    
    # Output-Verzeichnis
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = input_path / f"botanical_loras_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # Custom LoRA Handling
    custom_lora_paths = {}
    custom_lora_scales = {}
    
    if args.lora_paths:
        for lora_spec in args.lora_paths:
            if ":" in lora_spec:
                name, path = lora_spec.split(":", 1)
                custom_lora_paths[name] = path
                print(f"📂 Custom LoRA: {name} -> {path}")
    
    if args.lora_scales:
        for scale_spec in args.lora_scales:
            if ":" in scale_spec:
                name, scale = scale_spec.split(":", 1)
                custom_lora_scales[name] = float(scale)
                print(f"⚖️  Custom Scale: {name} -> {scale}")
    
    # LoRAs laden (maximal 2-3 für 16GB) - Crime optimiert
    lora_names_to_load = args.loras.split(",") if args.loras else ["cinematic_lighting", "realistic_vision"]
    lora_names_to_load = lora_names_to_load[:3]  # Limit für 16GB
    
    pipeline.load_loras(lora_names_to_load, custom_lora_paths, custom_lora_scales)
    
    # Header
    print("\n" + "="*80)
    print("🎬 CRIME CINEMATIC SDXL MIT LoRA-INTEGRATION (16GB OPTIMIERT)")
    print("="*80)
    print(f"📚 Projekt: {title}")
    print(f"📊 Szenen: {len(scenes)} (von {start_scene} bis {end_scene})")
    print(f"🎛️  Aktive LoRAs: {', '.join(lora_names_to_load)}")
    print(f"📏 Auflösung: {pipeline.output_width}x{pipeline.output_height}")
    print(f"🔢 Steps: {pipeline.steps}")
    print(f"💾 {get_memory_stats()}")
    print(f"📁 Ausgabe: {output_dir}")
    print("="*80 + "\n")
    
    results = []
    errors = []
    scene_metadata_list = []
    total_start = time.time()
    
    # Szenen rendern
    for i, scene in enumerate(scenes, 1):
        scene_id = scene.get("id", i)
        base_prompt = scene.get("image_prompt", "")
        custom_negative = scene.get("negative_prompt", "")
        seed = scene.get("seed", random.randint(0, 2**32 - 1))
        
        scene_style = scene.get("style", "noir_detective")
        mood = scene.get("mood", "tense")
        custom_detail = scene.get("detail", None)
        
        # Prompt mit LoRA-Optimierung erstellen
        enhanced_prompt, recommended_loras = pipeline.create_crime_prompt(
            base_prompt, scene_style, mood, custom_detail
        )
        
        # Für diese Szene spezifische LoRAs (max 2 für 16GB)
        scene_loras = scene.get("loras", recommended_loras)[:2]
        scene_lora_weights = scene.get("lora_weights", None)
        
        # Negativ-Prompt kombinieren
        final_negative = f"{pipeline.default_negative}, {custom_negative}" if custom_negative else pipeline.default_negative
        
        print(f"{'='*80}")
        print(f"🎬 SZENE {i}/{len(scenes)} (ID: {scene_id})")
        print(f"🎭 Szene: {base_prompt[:60]}...")
        print(f"✨ Enhanced: {enhanced_prompt[:80]}...")
        print(f"🎛️  LoRAs: {', '.join(scene_loras)}")
        if scene_lora_weights:
            print(f"⚖️  Weights: {scene_lora_weights}")
        print(f"🎲 Seed: {seed}")
        print(f"💾 Start: {get_memory_stats()}")
        print(f"{'='*80}")
        
        try:
            # KRITISCH: Memory cleanup vor jeder Generation
            aggressive_memory_cleanup()
            time.sleep(2)
            
            # Bild mit LoRAs generieren
            img, elapsed, metadata = pipeline.generate_with_loras(
                prompt=enhanced_prompt,
                lora_names=scene_loras,
                lora_weights=scene_lora_weights,
                negative_prompt=final_negative,
                seed=seed,
                disable_loras_after=True
            )
            
            # Metadaten erweitern
            metadata.update({
                "scene_id": scene_id,
                "original_prompt": base_prompt,
                "style": scene_style,
                "mood": mood,
                "enhanced_prompt": enhanced_prompt,
                "scene_loras": scene_loras,
                "scene_lora_weights": scene_lora_weights,
                "generation_time_seconds": round(elapsed, 2)
            })
            
            # Einfacher Dateiname: image_0001.png
            filename = output_dir / f"image_{int(scene_id):04d}.png"
            
            # Speichern
            pipeline.save_with_metadata(img, filename, metadata)
            
            # Thumbnail
            thumb = img.copy()
            thumb.thumbnail((384, 384), Image.Resampling.LANCZOS)
            thumb.save(output_dir / f"thumb_{int(scene_id):04d}.jpg", "JPEG", quality=85)
            
            file_size = filename.stat().st_size / (1024 * 1024)
            print(f"✅ GESPEICHERT: {filename.name}")
            print(f"   📁 {file_size:.2f} MB | ⏱️  {elapsed:.1f}s | 💾 {get_memory_stats()}\n")
            
            # Für Log speichern
            scene_metadata = metadata.copy()
            scene_metadata["filename"] = filename.name
            scene_metadata["file_size_mb"] = round(file_size, 2)
            scene_metadata_list.append(scene_metadata)
            
            results.append(str(filename))
            
            # KRITISCH: Cleanup nach jeder Szene
            del img
            del output
            aggressive_memory_cleanup()
            
        except Exception as e:
            error_msg = f"Szene {scene_id}: {str(e)}"
            print(f"❌ FEHLER: {error_msg}\n")
            errors.append(error_msg)
            
            # Bei CUDA-Fehler aggressiv cleanen und warten
            if "CUDA" in str(e).upper() or "memory" in str(e).lower():
                print("⚠️  CUDA Memory Fehler - aggressive Cleanup und Pause...")
                aggressive_memory_cleanup()
                time.sleep(10)
            else:
                time.sleep(3)
            continue
        
        # Pause zwischen Szenen
        if i < len(scenes):
            time.sleep(3)
    
    # Finale Cleanup
    pipeline.lora_manager.deactivate_loras()
    aggressive_memory_cleanup()
    
    total_time = time.time() - total_start
    
    # Log-Datei speichern
    log_file = output_dir / "generation_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "book_info": book_info,
            "pipeline_config": {
                "model": "RealVisXL_V4.0_16GB_optimized",
                "size": f"{pipeline.output_width}x{pipeline.output_height}",
                "steps": pipeline.steps,
                "guidance": pipeline.guidance,
                "refiner": False,
                "quality_mode": pipeline.quality_mode,
                "loaded_loras": list(pipeline.lora_manager.loaded_loras.keys()),
                "gpu": torch.cuda.get_device_name(0)
            },
            "scenes": scene_metadata_list,
            "summary": {
                "total_scenes": len(scenes),
                "successful": len(results),
                "failed": len(errors),
                "total_time_seconds": round(total_time, 2),
                "avg_time_per_scene": round(total_time / len(scenes), 2) if scenes else 0
            }
        }, f, indent=2, ensure_ascii=False)
    
    # Zusammenfassung
    print("\n" + "="*80)
    print("🎉 CRIME CINEMATIC RENDERING ABGESCHLOSSEN")
    print("="*80)
    print(f"✅ Erfolgreich: {len(results)}/{len(scenes)} Bilder")
    print(f"❌ Fehler: {len(errors)}")
    
    if results:
        avg_size = sum(Path(f).stat().st_size for f in results) / len(results) / (1024*1024)
        avg_time = total_time / len(results)
        print(f"📊 Durchschnittliche Größe: {avg_size:.2f} MB")
        print(f"⏱️  Durchschnittliche Zeit: {avg_time:.1f}s pro Bild")
        print(f"⏱️  Gesamtzeit: {total_time/60:.1f} Minuten")
    
    print(f"📁 Ausgabeverzeichnis: {output_dir.absolute()}")
    print(f"📋 Log-Datei: {log_file}")
    print(f"💾 Final: {get_memory_stats()}")
    
    if errors:
        print("\n⚠️  FEHLERLISTE (letzte 5):")
        for err in errors[-5:]:
            print(f"   • {err}")
    
    print("="*80)
    
    return results, errors, output_dir

def main():
    parser = argparse.ArgumentParser(
        description="SDXL V14 - Crime/Noir Cinematic Bildgenerierung (16GB GPU optimiert)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele für 16GB GPU (Crime/Noir):
  %(prog)s --path /pfad/zum/projekt --loras cinematic_lighting,noir_style
  %(prog)s --path /pfad --width 1792 --height 1024 --steps 40
  %(prog)s --path /pfad --loras cinematic_lighting,film_grain,realistic_vision
  %(prog)s --path /pfad --start-scene 1 --end-scene 10
        """
    )
    
    # Hauptargumente
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Pfad zum Projekt-Verzeichnis mit book_scenes.json"
    )
    
    # LoRA-Einstellungen (Crime optimiert)
    parser.add_argument(
        "--loras",
        type=str,
        default="cinematic_lighting,realistic_vision",
        help="Komma-getrennte LoRA-Namen (max 2-3 für 16GB, empfohlen: cinematic_lighting,noir_style)"
    )
    
    parser.add_argument(
        "--lora-paths",
        type=str,
        nargs="+",
        help="Custom LoRA-Pfade im Format name:pfad"
    )
    
    parser.add_argument(
        "--lora-scales",
        type=str,
        nargs="+",
        help="Custom LoRA-Weights im Format name:weight"
    )
    
    # Qualitätseinstellungen (16GB optimiert)
    parser.add_argument(
        "--width",
        type=int,
        default=1536,
        help="Bildbreite (default: 1536, max empfohlen: 1792)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=864,
        help="Bildhöhe (default: 864, max empfohlen: 1024)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="Denoising steps (default: 35, max empfohlen: 45)"
    )
    
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.5,
        help="Guidance scale (default: 5.5)"
    )
    
    parser.add_argument(
        "--no-refiner",
        action="store_true",
        help="Refiner deaktivieren (empfohlen für 16GB)"
    )
    
    # Szenen-Bereich
    parser.add_argument(
        "--start-scene",
        type=int,
        help="Erste Szene zum Rendern (default: 1)"
    )
    
    parser.add_argument(
        "--end-scene",
        type=int,
        help="Letzte Szene zum Rendern (default: alle)"
    )
    
    # Performance
    parser.add_argument(
        "--seed",
        type=int,
        help="Globaler Seed für alle Generationen"
    )
    
    args = parser.parse_args()
    
    # Validierung für 16GB
    if args.width > 1792 or args.height > 1024:
        print("⚠️  WARNUNG: Auflösung > 1792x1024 kann bei 16GB GPU zu Fehlern führen!")
        response = input("Fortfahren? (j/n): ")
        if response.lower() != 'j':
            return 0
    
    if not args.no_refiner:
        print("⚠️  WARNUNG: Refiner bei 16GB GPU NICHT empfohlen!")
        print("⚠️  Verwende --no-refiner für stabile Generierung")
        response = input("Trotzdem fortfahren? (j/n): ")
        if response.lower() != 'j':
            return 0
    
    # Globale Seed setzen
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"🌱 Globaler Seed: {args.seed}")
    
    # Pipeline initialisieren
    try:
        pipeline = BotanicalSDXLLoRA(
            use_refiner=not args.no_refiner,
            output_width=args.width,
            output_height=args.height,
            steps=args.steps,
            guidance=args.guidance,
        )
    except Exception as e:
        print(f"❌ Fehler bei Pipeline-Initialisierung: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Buch verarbeiten
    input_path = Path(args.path)
    
    if not input_path.exists():
        print(f"❌ Pfad existiert nicht: {input_path}")
        return 1
    
    results, errors, output_dir = process_crime_scenes_with_loras(input_path, pipeline, args)
    
    # Finaler Memory-Cleanup
    aggressive_memory_cleanup()
    
    if errors and len(errors) > len(results) / 2:
        print("\n⚠️  KRITISCH: Mehr als 50% Fehler!")
        print("⚠️  Empfehlungen:")
        print("   • Reduziere Auflösung: --width 1280 --height 720")
        print("   • Verwende nur 1 LoRA: --loras detail_tweaker")
        print("   • Reduziere Steps: --steps 30")
        return 1
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\nℹ️  Abbruch durch Benutzer")
        aggressive_memory_cleanup()
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        aggressive_memory_cleanup()
        exit_code = 1
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Gesamtzeit: {total_time/60:.1f} Minuten")
    
    exit(exit_code)