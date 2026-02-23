#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
import warnings
import re
import requests
import shutil
import tempfile
from urllib.parse import urlparse

# Unterdrücke Warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.schedulers import DPMSolverMultistepScheduler

#############################################
# SDXL Multi-Model Generator mit LoRA Support
# VERSION 2.0 - Erweitert mit echtem LoRA-Support
#############################################

def check_disk_space(min_gb: float = 5.0):
    """Überprüft verfügbaren Speicherplatz"""
    try:
        stat = shutil.disk_usage('/')
        free_gb = stat.free / (1024**3)
        print(f"💾 Verfügbarer Speicherplatz: {free_gb:.1f} GB")
        
        if free_gb < min_gb:
            print(f"⚠️  WARNUNG: Weniger als {min_gb} GB frei!")
            return False
        return True
    except:
        return True  # Falls Prüfung fehlschlägt, weiter machen


def cleanup_temp_files():
    """Räumt temporäre Dateien auf"""
    try:
        # Lösche alte temporäre Dateien
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                # Lösche Dateien älter als 1 Tag
                if os.path.isfile(item_path):
                    if time.time() - os.path.getmtime(item_path) > 86400:
                        os.remove(item_path)
                        print(f"🗑️  Alte Temp-Datei gelöscht: {item}")
            except:
                pass
    except:
        pass


def parse_civitai_url(url: str) -> dict:
    """Extrahiert Model Info aus CivitAI URL"""
    # Pattern 1: Mit modelVersionId
    version_match = re.search(r'modelVersionId=(\d+)', url)
    if version_match:
        version_id = version_match.group(1)
        return {
            'type': 'civitai',
            'version_id': version_id,
            'url': url
        }
    
    # Pattern 2: Nur Model ID - hol die neueste Version via API
    model_match = re.search(r'/models/(\d+)', url)
    if model_match:
        model_id = model_match.group(1)
        
        # API aufrufen um Model-Info zu holen und neueste Version zu finden
        try:
            print(f"📡 Hole Model-Info für ID {model_id} von CivitAI...")
            api_url = f"https://civitai.com/api/v1/models/{model_id}"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Finde die neueste Version
            model_versions = data.get('modelVersions', [])
            if model_versions:
                # Sortiere nach createdAt, neueste zuerst
                sorted_versions = sorted(
                    model_versions,
                    key=lambda x: x.get('createdAt', ''),
                    reverse=True
                )
                latest_version_id = sorted_versions[0].get('id')
                
                print(f"✅ Neueste Version ID gefunden: {latest_version_id}")
                
                return {
                    'type': 'civitai',
                    'version_id': str(latest_version_id),
                    'model_id': model_id,
                    'url': url,
                    'auto_detected': True
                }
            else:
                raise ValueError(f"Keine Model-Versionen für Model ID {model_id} gefunden")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Fehler beim Abrufen der Model-Info: {e}")
            raise ValueError(f"Konnte Model-Versionen von CivitAI nicht abrufen: {e}")
    
    return None


def download_civitai_model(version_id: str, model_dir: Path) -> Path:
    """Lädt Model von CivitAI herunter"""
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📡 Hole Model-Info von CivitAI (Version ID: {version_id})...")
    
    try:
        # Versuche erst die Model-Info ohne Auth (oft funktioniert das noch)
        api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 401:
            print("⚠️  CivitAI benötigt jetzt API-Authentifizierung")
            print("ℹ️  Versuche alternative Methode...")
            
            # Versuche den direkten Download-Link (manchmal funktioniert das noch)
            download_url = f"https://civitai.com/api/download/models/{version_id}"
            
            # Setze User-Agent Header
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Versuche Download mit Headers
            response = requests.get(download_url, headers=headers, timeout=30, stream=True)
            
            if response.status_code == 401:
                raise Exception("CivitAI API benötigt Authentifizierung. Bitte besuche die Website und lade das Model manuell herunter.")
        
        response.raise_for_status()
        data = response.json()
        
        model_name = data.get('model', {}).get('name', 'civitai_model')
        model_name = re.sub(r'[^\w\-_.]', '_', model_name)
        version_name = data.get('name', 'v1')
        version_name = re.sub(r'[^\w\-_.]', '_', version_name)
        
        filename = f"{model_name}_{version_name}.safetensors"
        output_path = model_dir / filename
        
        # Check ob bereits vorhanden
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024**3)
            print(f"✅ Model bereits vorhanden: {output_path.name} ({file_size:.2f} GB)")
            return output_path
        
        # Download URL
        download_url = f"https://civitai.com/api/download/models/{version_id}"
        
        print(f"📥 Lade Model herunter: {model_name}")
        print(f"   Speichere als: {filename}")
        print(f"   Dies kann einige Minuten dauern...")
        
        # Setze Headers für Download
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': f'https://civitai.com/models/{data.get("model", {}).get("id", "")}',
        }
        
        # Versuche Download
        response = requests.get(download_url, headers=headers, timeout=30, stream=True)
        
        if response.status_code == 401:
            # Alternative: Versuche die direkte Download-URL aus den Files-Info
            files = data.get('files', [])
            if files:
                for file_info in files:
                    if file_info.get('primary', False):
                        direct_url = file_info.get('downloadUrl')
                        if direct_url:
                            print(f"ℹ️  Verwende alternative Download-URL...")
                            response = requests.get(direct_url, headers=headers, timeout=30, stream=True)
                            break
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if downloaded % (100 * 1024 * 1024) < chunk_size:
                            progress = (downloaded / total_size) * 100
                            downloaded_mb = downloaded / (1024**2)
                            total_mb = total_size / (1024**2)
                            print(f"   Progress: {progress:.1f}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)", end='\r')
        
        print()
        file_size = output_path.stat().st_size / (1024**3)
        print(f"✅ Download abgeschlossen: {output_path.name} ({file_size:.2f} GB)")
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Fehler beim Download: {e}")
        print(f"ℹ️  CivitAI benötigt möglicherweise eine API-Authentifizierung.")
        print(f"📋 Bitte lade das Model manuell herunter:")
        print(f"   1. Gehe zu: https://civitai.com/models/565243")
        print(f"   2. Klicke auf 'Download' für die Version")
        print(f"   3. Speichere die .safetensors Datei in: /workspace/models/civitai/")
        print(f"   4. Starte das Script erneut mit dem lokalen Pfad")
        
        # Frage nach manuellem Download
        manual_path = model_dir / f"NeoNoirCinema_SinCityStyle_SDXL.safetensors"
        if manual_path.exists():
            print(f"✅ Gefunden: {manual_path}")
            return manual_path
        
        raise
    
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        if 'output_path' in locals() and output_path.exists():
            output_path.unlink()
        raise


def parse_lora_input(lora_string: str) -> dict:
    """
    Parst LoRA Input String
    Format: "URL_oder_Pfad:Weight" oder nur "URL_oder_Pfad"
    
    Returns:
        dict mit 'input' (Original-String) und 'weight'
    """
    parts = lora_string.rsplit(':', 1)
    
    if len(parts) == 2:
        # Prüfe ob der letzte Teil ein valides Gewicht ist
        try:
            weight = float(parts[1])
            # Valides Gewicht - verwende ersten Teil als Pfad
            return {
                'input': parts[0],
                'weight': weight
            }
        except ValueError:
            # Kein valides Weight, behandle gesamten String als Pfad
            pass
    
    # Kein Weight angegeben oder kein valides Weight
    return {
        'input': lora_string,
        'weight': 1.0
    }


def resolve_model_path(model_input: str, force_type: str = None) -> dict:
    """
    Löst Model-Input auf und gibt Dict mit Typ und Pfad zurück
    
    Args:
        model_input: URL, Pfad oder HuggingFace ID
        force_type: Optional - erzwinge einen bestimmten Typ ('lora', 'checkpoint_sdxl', etc.)
    """
    
    # Check 1: CivitAI URL
    if 'civitai.com' in model_input:
        civitai_info = parse_civitai_url(model_input)
        
        if not civitai_info:
            raise ValueError(f"Konnte CivitAI URL nicht parsen: {model_input}")
        
        if civitai_info.get('needs_version'):
            raise ValueError(
                f"CivitAI URL enthält keine modelVersionId!\n"
                f"Gehe zu {model_input} und kopiere den Link mit '?modelVersionId=...'"
            )
        
        # Speicherplatz prüfen
        check_disk_space(15)  # Mindestens 15 GB für CivitAI Models
        
        model_dir = Path("/workspace/models/civitai")
        local_path = download_civitai_model(civitai_info['version_id'], model_dir)
        
        # Model-Typ erkennen (force_type hat Priorität)
        if force_type:
            model_type = force_type
        else:
            filename = str(local_path).lower()
            if 'lora' in filename:
                model_type = 'lora'
            elif 'embedding' in filename or '.pt' in filename:
                model_type = 'embedding'
            elif '.safetensors' in filename or '.ckpt' in filename:
                model_type = 'checkpoint_sdxl'
            else:
                model_type = 'checkpoint_sdxl'
        
        return {
            'type': model_type,
            'path': str(local_path),
            'source': 'civitai',
            'filename': local_path.name
        }
    
    # Check 2: Lokaler Pfad
    if Path(model_input).exists():
        if force_type:
            model_type = force_type
        else:
            filename = model_input.lower()
            if 'lora' in filename:
                model_type = 'lora'
            elif 'embedding' in filename or '.pt' in filename:
                model_type = 'embedding'
            elif '.safetensors' in filename or '.ckpt' in filename:
                model_type = 'checkpoint_sdxl'
            else:
                model_type = 'huggingface'
            
        return {
            'type': model_type,
            'path': model_input,
            'source': 'local',
            'filename': Path(model_input).name
        }
    
    # Check 3: HuggingFace ID
    if '/' in model_input:
        # Speicherplatz prüfen für HF Cache
        check_disk_space(5)
        return {
            'type': 'huggingface',
            'path': model_input,
            'source': 'huggingface',
            'filename': model_input.split('/')[-1]
        }
    
    # Default
    return {
        'type': 'huggingface',
        'path': model_input,
        'source': 'huggingface',
        'filename': model_input
    }


class UltraQualitySDXL:
    def __init__(
        self,
        model_info: dict = None,
        lora_list: list = None,  # ✨ NEU: Liste von LoRA-Dicts
        model_refiner: str = None,
        use_refiner: bool = False,
        output_width: int = None,
        output_height: int = None,
        steps: int = None,
        guidance: float = None,
        refiner_split: float = 0.75,
        scheduler: str = None,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ):
        # Model Info verarbeiten
        if model_info is None:
            model_info = {'type': 'huggingface', 'path': base_model}
        
        self.lora_list = lora_list or []
        
        print(f"🚀 Initialisiere Pipeline")
        print(f"   Model: {model_info.get('filename', model_info['path'])}")
        print(f"   Typ: {model_info['type']}")
        
        if self.lora_list:
            print(f"   LoRAs: {len(self.lora_list)}")
            for i, lora in enumerate(self.lora_list, 1):
                print(f"      {i}. {lora.get('filename', 'unknown')} @ weight {lora.get('weight', 1.0)}")

        if not torch.cuda.is_available():
            raise RuntimeError("❌ Keine CUDA-GPU gefunden!")

        self.device = "cuda"
        self.model_info = model_info
        
        # ✨ OPTIMIERTE DEFAULTS FÜR SPEICHER
        self.output_width = output_width if output_width is not None else 1920
        self.output_height = output_height if output_height is not None else 1080
        self.steps = steps if steps is not None else 35
        self.guidance = guidance if guidance is not None else 7.0
        self.scheduler_type = scheduler if scheduler is not None else "euler_a"
        self.use_refiner = bool(use_refiner)
        self.refiner_split = float(refiner_split)
        self.base_model = base_model

        # Kein Refiner default für Speicher
        if model_refiner is None:
            model_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

        # ⚡ CUDA Optimierungen
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        # Qualitätsstufe basierend auf GPU Memory
        if gpu_memory >= 20:
            quality_mode = "ULTRA"
            enable_slicing = False
            enable_tiling = False
            self.output_width = self.output_width if output_width is not None else 1920
            self.output_height = self.output_height if output_height is not None else 1080
            self.steps = self.steps if steps is not None else 35
        elif gpu_memory >= 16:
            quality_mode = "HIGH"
            enable_slicing = False
            enable_tiling = False
        elif gpu_memory >= 8:
            quality_mode = "MEDIUM"
            enable_slicing = True
            enable_tiling = True
        else:
            quality_mode = "LOW"
            enable_slicing = True
            enable_tiling = True
            self.output_width = 896
            self.output_height = 512
            self.steps = 20

        print(f"🎯 Qualitätsmodus: {quality_mode}")
        print(f"   Auflösung: {self.output_width}x{self.output_height}")
        print(f"   Steps: {self.steps} @ CFG {self.guidance}")

        # 📝 NEGATIVE PROMPT
        self.default_negative = (
            "blurry, soft focus, out of focus, bokeh, "
            "low detail, low resolution, low quality, worst quality, "
            "jpeg artifacts, compression artifacts, noisy, grainy, pixelated, "
            "watermark, logo, signature, text, subtitles, UI elements, "
            "frame, border, letterbox, timestamp, "
            "bad anatomy, extra limbs, deformed hands, missing fingers, "
            "extra fingers, mutated hands, fused fingers, "
            "bad proportions, "
            "oversaturated, undersaturated, overexposed, underexposed, "
            "harsh lighting, flat lighting"
        )

        # 🔥 MODELLADEN - SPEICHEROPTIMIERT
        print(f"🔥 Lade Model...")
        
        # CUDA Cache leeren
        torch.cuda.empty_cache()
        
        try:
            if model_info['type'] == 'huggingface':
                # HF Model mit minimalen Optionen
                self.base = DiffusionPipeline.from_pretrained(
                    model_info['path'],
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    add_watermarker=False,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                print(f"✅ HuggingFace Model geladen")
            
            elif model_info['type'] in ['checkpoint_sdxl', 'checkpoint']:
                # Checkpoint mit from_single_file
                try:
                    print(f"📂 Lade Checkpoint...")
                    self.base = StableDiffusionXLPipeline.from_single_file(
                        model_info['path'],
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                        add_watermarker=False,
                    ).to(self.device)
                    print(f"✅ Checkpoint geladen")
                except Exception as e:
                    print(f"❌ Checkpoint konnte nicht geladen werden: {e}")
                    print(f"🔄 Verwende Basis SDXL...")
                    self.base = DiffusionPipeline.from_pretrained(
                        self.base_model,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16",
                        add_watermarker=False,
                        low_cpu_mem_usage=True,
                    ).to(self.device)
            
            elif model_info['type'] == 'lora':
                # Einzelnes LoRA als --model übergeben (Legacy Support)
                print(f"⚠️  LoRA als --model übergeben - lade auf Basis-SDXL")
                self.base = DiffusionPipeline.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    add_watermarker=False,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                
                # Füge es zur LoRA-Liste hinzu
                if not self.lora_list:
                    self.lora_list = []
                self.lora_list.insert(0, {
                    'path': model_info['path'],
                    'weight': 1.0,
                    'filename': model_info.get('filename', Path(model_info['path']).name)
                })
                print(f"✅ Basis-Model geladen, LoRA wird gleich geladen...")
            
            else:
                # Fallback
                self.base = DiffusionPipeline.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    add_watermarker=False,
                    low_cpu_mem_usage=True,
                ).to(self.device)
        
        except Exception as e:
            print(f"❌ Kritischer Fehler beim Modelladen: {e}")
            raise

        # ✨ LORAS LADEN (NEU!)
        if self.lora_list:
            print(f"🎨 Lade {len(self.lora_list)} LoRA(s)...")
            
            loaded_loras = []
            
            for i, lora_info in enumerate(self.lora_list, 1):
                try:
                    lora_path = lora_info['path']
                    lora_weight = lora_info.get('weight', 1.0)
                    lora_name = lora_info.get('filename', Path(lora_path).name)
                    
                    print(f"   [{i}/{len(self.lora_list)}] Lade: {lora_name}")
                    print(f"       Gewicht: {lora_weight}")
                    
                    # Adapter-Name für dieses LoRA
                    adapter_name = f"lora_{i}"
                    
                    # LoRA laden
                    self.base.load_lora_weights(
                        lora_path,
                        adapter_name=adapter_name
                    )
                    
                    loaded_loras.append({
                        'adapter_name': adapter_name,
                        'weight': lora_weight,
                        'filename': lora_name
                    })
                    
                    print(f"   ✅ LoRA {i} geladen als '{adapter_name}'")
                    
                except Exception as e:
                    print(f"   ❌ LoRA {i} konnte nicht geladen werden: {e}")
                    print(f"      Überspringe dieses LoRA...")
                    continue
            
            # Wenn mehrere LoRAs geladen wurden, alle aktivieren mit Gewichten
            if len(loaded_loras) > 0:
                if len(loaded_loras) > 1:
                    adapter_names = [lora['adapter_name'] for lora in loaded_loras]
                    adapter_weights = [lora['weight'] for lora in loaded_loras]
                    
                    try:
                        self.base.set_adapters(adapter_names, adapter_weights)
                        print(f"✅ Alle {len(loaded_loras)} LoRAs aktiviert mit Gewichten:")
                        for lora in loaded_loras:
                            print(f"   • {lora['filename']}: {lora['weight']}")
                    except Exception as e:
                        print(f"⚠️  Multi-LoRA Gewichtung fehlgeschlagen: {e}")
                        print(f"   Verwende Standard-Aktivierung...")
                        # Fallback: Nur erstes LoRA aktivieren
                        try:
                            self.base.set_adapters([loaded_loras[0]['adapter_name']])
                            print(f"✅ Verwende nur erstes LoRA: {loaded_loras[0]['filename']}")
                        except:
                            pass
                else:
                    # Nur ein LoRA
                    try:
                        # Bei einzelnem LoRA können wir auch fuse_lora() verwenden für Performance
                        # Aber das ist optional - set_adapters funktioniert auch
                        self.base.set_adapters([loaded_loras[0]['adapter_name']])
                        print(f"✅ LoRA aktiviert: {loaded_loras[0]['filename']} @ {loaded_loras[0]['weight']}")
                        
                        # Optional: LoRA für bessere Performance einbrennen
                        # self.base.fuse_lora(lora_scale=loaded_loras[0]['weight'])
                        # print(f"   ℹ️  LoRA eingebrannt für optimale Performance")
                    except Exception as e:
                        print(f"⚠️  LoRA-Aktivierung fehlgeschlagen: {e}")

        # 📊 SCHEDULER
        if self.scheduler_type == "euler_a":
            self.base.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.base.scheduler.config
            )
            print("✅ Scheduler: Euler Ancestral (schnell)")
        else:
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(
                self.base.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
                solver_order=2,
            )
            print("✅ Scheduler: DPM++ 2M Karras")

        # 💾 SPEICHER-OPTIMIERUNGEN (immer aktiv für Stabilität)
        self.base.enable_attention_slicing()
        self.base.enable_vae_slicing()
        self.base.enable_vae_tiling()
        print("✅ Alle Speicher-Optimierungen aktiviert")

        # Cache leeren
        torch.cuda.empty_cache()

        # 🔥 REFINER NUR WENN EXPLIZIT GEWÜNSCHT
        self.refiner = None
        if self.use_refiner:
            print("🔥 Lade SDXL Refiner...")
            try:
                self.refiner = DiffusionPipeline.from_pretrained(
                    model_refiner,
                    text_encoder_2=self.base.text_encoder_2,
                    vae=self.base.vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    add_watermarker=False,
                    low_cpu_mem_usage=True,
                ).to(self.device)

                self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.refiner.scheduler.config,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=True,
                )

                self.refiner.enable_attention_slicing()
                self.refiner.enable_vae_slicing()
                    
                print("✅ Refiner geladen")
            except Exception as e:
                print(f"❌ Refiner konnte nicht geladen werden: {e}")
                self.use_refiner = False

        print(f"✨ Pipeline bereit")
        print(f"   VRAM belegt: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    @torch.inference_mode()
    def generate(self, prompt: str, negative_prompt: str = None, seed: int = 42):
        torch.cuda.empty_cache()

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        neg_prompt = negative_prompt if negative_prompt else self.default_negative

        start = time.time()

        try:
            if self.use_refiner and self.refiner is not None:
                # Mit Refiner
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

                output = self.refiner(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    image=latents,
                    num_inference_steps=self.steps,
                    denoising_start=self.refiner_split,
                    generator=generator,
                )
            else:
                # Ohne Refiner
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
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️  CUDA Out of Memory - reduziere weiter...")
            torch.cuda.empty_cache()
            
            # Noch kleinere Auflösung
            reduced_width = max(512, self.output_width // 2)
            reduced_height = max(512, self.output_height // 2)
            
            output = self.base(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=reduced_width,
                height=reduced_height,
                guidance_scale=self.guidance,
                num_inference_steps=max(15, self.steps // 2),
                generator=generator,
            )
            elapsed = time.time() - start
            
            print(f"⚠️  Bild mit stark reduzierter Auflösung: {reduced_width}x{reduced_height}")
        
        except Exception as e:
            print(f"❌ Generierungsfehler: {e}")
            raise
        
        torch.cuda.empty_cache()
        
        return output.images[0], elapsed


def process_book(input_path: Path, pipeline: UltraQualitySDXL, force_regenerate: bool = False):
    """Verarbeitet ein Buch-Verzeichnis mit book_scenes.json"""
    
    json_file = input_path / "book_scenes.json"
    
    if not json_file.exists():
        print(f"❌ Keine book_scenes.json gefunden in: {input_path}")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    book_info = data.get("book_info", {})
    title = book_info.get("title", "Unbekannt")
    author = book_info.get("author", "Unbekannt")
    base_style = book_info.get("style", "")
    
    scenes = data.get("scenes", [])
    
    if not scenes:
        print(f"❌ Keine Szenen gefunden in JSON")
        return
    
    output_dir = input_path / "renders"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print(f"📚 BUCH: {title}")
    print(f"✍️  AUTOR: {author}")
    if base_style:
        print(f"🎨 BASE STYLE: {base_style}")
    print(f"📊 SZENEN: {len(scenes)}")
    print(f"⚙️  SETTINGS: {pipeline.output_width}x{pipeline.output_height} | {pipeline.steps} steps | CFG {pipeline.guidance}")
    print(f"🎭 MODEL: {pipeline.model_info.get('filename', pipeline.model_info['path'])}")
    if pipeline.lora_list:
        print(f"🎨 LORAs: {len(pipeline.lora_list)}")
        for i, lora in enumerate(pipeline.lora_list, 1):
            print(f"   {i}. {lora.get('filename', 'unknown')} @ {lora.get('weight', 1.0)}")
    print(f"💾 VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    if force_regenerate:
        print(f"🔄 MODUS: Überschreibe existierende Bilder")
    else:
        print(f"⏭️  MODUS: Überspringe existierende Bilder")
    print("="*80 + "\n")
    
    results = []
    errors = []
    skipped = []
    total_time = 0
    
    for i, scene in enumerate(scenes, 1):
        scene_id = scene.get("id", i)
        scene_prompt = scene.get("image_prompt", "")
        negative = scene.get("negative_prompt", None)
        seed = scene.get("seed", 42)
        
        filename = output_dir / f"image_{int(scene_id):04d}.png"
        
        if filename.exists() and not force_regenerate:
            file_size = filename.stat().st_size / (1024 * 1024)
            print("="*80)
            print(f"⏭️  SZENE {i}/{len(scenes)} (ID: {scene_id}) - ÜBERSPRINGE")
            print(f"   {filename.name} existiert bereits ({file_size:.2f} MB)")
            print("="*80 + "\n")
            skipped.append(str(filename))
            continue
        
        if base_style and base_style.strip():
            full_prompt = f"{base_style}, {scene_prompt}"
        else:
            full_prompt = scene_prompt
        
        print("="*80)
        print(f"🖼️  SZENE {i}/{len(scenes)} (ID: {scene_id})")
        print("-"*80)
        if len(full_prompt) > 180:
            print(f"📝 PROMPT: {full_prompt[:180]}...")
        else:
            print(f"📝 PROMPT: {full_prompt}")
        print(f"🎲 SEED: {seed}")
        print("="*80)
        
        try:
            # Speicher vor jedem Bild leeren
            torch.cuda.empty_cache()
            
            img, elapsed = pipeline.generate(full_prompt, negative, seed)
            total_time += elapsed
            
            img.save(filename, quality=95, optimize=True)
            
            file_size = filename.stat().st_size / (1024 * 1024)
            avg_time = total_time / (len(results) + 1)
            remaining = len(scenes) - i - len(skipped)
            eta = avg_time * remaining
            
            print(f"✅ GESPEICHERT: {filename.name}")
            print(f"   Größe: {file_size:.2f} MB | Zeit: {elapsed:.1f}s")
            if remaining > 0:
                print(f"   Verbleibend: {remaining} Bilder | ETA: {eta/60:.1f} min")
            print()
            
            results.append(str(filename))
            
        except Exception as e:
            error_msg = f"Szene {scene_id}: {str(e)}"
            print(f"❌ FEHLER: {error_msg}\n")
            errors.append(error_msg)
            
            # Nach Fehler mehr Speicher freigeben
            torch.cuda.empty_cache()
            time.sleep(1)
            continue
    
    print("\n" + "="*80)
    print("🎉 RENDERING ABGESCHLOSSEN")
    print("="*80)
    print(f"✅ Neu generiert: {len(results)} Bilder")
    if skipped:
        print(f"⏭️  Übersprungen: {len(skipped)} Bilder")
    print(f"⏱️  Gesamtzeit: {total_time/60:.1f} min")
    if results:
        print(f"   Durchschnitt: {total_time/len(results):.1f}s pro Bild")
    if errors:
        print(f"❌ Fehler: {len(errors)}")
        for err in errors[:3]:
            print(f"   • {err}")
    print(f"📁 Bilder in: {output_dir.absolute()}")
    
    # Finale Aufräumarbeiten
    cleanup_temp_files()
    torch.cuda.empty_cache()
    print("="*80 + "\n")


def cleanup_old_models(max_models: int = 10):
    """Löscht alte Modelle um Speicherplatz freizugeben"""
    try:
        model_dir = Path("/workspace/models/civitai")
        if model_dir.exists():
            models = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.ckpt"))
            models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(models) > max_models:
                print(f"🗑️  Lösche alte Modelle (behalte nur {max_models})...")
                for model in models[max_models:]:
                    size_gb = model.stat().st_size / (1024**3)
                    print(f"   Lösche: {model.name} ({size_gb:.2f} GB)")
                    model.unlink()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="SDXL Generator mit LoRA-Support v2.0",
        epilog="""
BEISPIELE:
  
  Nur Checkpoint:
    %(prog)s --path ./input --model "Lykon/dreamshaper-xl-1-0"
  
  Checkpoint + LoRA:
    %(prog)s --path ./input --model "Lykon/dreamshaper-xl-1-0" \\
             --lora "https://civitai.com/models/123?modelVersionId=456:0.8"
  
  Mehrere LoRAs:
    %(prog)s --path ./input --model "Lykon/dreamshaper-xl-1-0" \\
             --lora "url1:1.0" --lora "url2:0.7"
  
  Basis-SDXL + LoRA:
    %(prog)s --path ./input --lora "/path/to/lora.safetensors:0.9"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--path", type=str, required=True, 
                       help="Pfad zum Buch-Ordner")
    parser.add_argument("--model", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0", 
                       help="HuggingFace Model ID oder CivitAI URL (Checkpoint)")
    
    # ✨ NEU: LoRA Parameter
    parser.add_argument("--lora", type=str, action='append', dest='loras',
                       help="LoRA URL oder Pfad, optional mit :WEIGHT (z.B. 'url:0.8'). "
                            "Kann mehrfach verwendet werden für multiple LoRAs.")
    
    parser.add_argument("--width", type=int, default=None, 
                       help="Bildbreite (default: auto)")
    parser.add_argument("--height", type=int, default=None, 
                       help="Bildhöhe (default: auto)")
    parser.add_argument("--steps", type=int, default=None, 
                       help="Diffusion Steps (default: 35)")
    parser.add_argument("--guidance", type=float, default=None, 
                       help="CFG Scale (default: 7.0)")
    parser.add_argument("--scheduler", type=str, default=None, 
                       choices=["dpm++", "euler_a"], 
                       help="Scheduler (default: euler_a)")
    parser.add_argument("--refiner", action="store_true", 
                       help="SDXL Refiner aktivieren")
    parser.add_argument("--force", action="store_true", 
                       help="Existierende Bilder überschreiben")
    parser.add_argument("--base-model", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0", 
                       help="Basis Model für LoRAs")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Alte Modelle aufräumen vor Start")
    parser.add_argument("--low-memory", action="store_true", 
                       help="Aktiviere Low-Memory Mode")
    
    args = parser.parse_args()

    # Aufräumen falls gewünscht
    if args.cleanup:
        cleanup_old_models(5)
    
    # Speicherplatz prüfen
    check_disk_space(5)
    
    # Model Info auflösen
    model_info = resolve_model_path(args.model)
    print(f"📁 Model Info: Typ={model_info['type']}, Source={model_info.get('source', 'unknown')}")
    
    # ✨ LoRAs verarbeiten (NEU!)
    lora_list = []
    if args.loras:
        print(f"\n🎨 Verarbeite {len(args.loras)} LoRA Input(s)...")
        
        for lora_string in args.loras:
            # Parse Input (extrahiert Pfad und Weight)
            lora_parsed = parse_lora_input(lora_string)
            
            # LoRA-Pfad auflösen (Download von CivitAI falls nötig)
            try:
                lora_info = resolve_model_path(lora_parsed['input'], force_type='lora')
                lora_info['weight'] = lora_parsed['weight']
                lora_list.append(lora_info)
                print(f"   ✅ {lora_info.get('filename', 'unknown')} @ weight {lora_info['weight']}")
            except Exception as e:
                print(f"   ❌ LoRA konnte nicht aufgelöst werden: {e}")
                print(f"      Überspringe: {lora_string}")
        
        if not lora_list:
            print(f"⚠️  Keine LoRAs konnten geladen werden!")
    
    # Low-Memory Mode
    if args.low_memory:
        print("\n🔧 Low-Memory Mode aktiviert")
        if args.width is None:
            args.width = 896
        if args.height is None:
            args.height = 512
        if args.steps is None:
            args.steps = 20
        if not args.refiner:
            args.refiner = False
    
    # Pipeline erstellen
    pipeline = UltraQualitySDXL(
        model_info=model_info,
        lora_list=lora_list,  # ✨ NEU!
        base_model=args.base_model,
        output_width=args.width,
        output_height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        scheduler=args.scheduler,
        use_refiner=args.refiner,
    )

    # Bilder generieren
    process_book(Path(args.path), pipeline, force_regenerate=args.force)


if __name__ == "__main__":
    # Temporäre Dateien aufräumen
    try:
        cleanup_temp_files()
    except:
        pass
    
    main()