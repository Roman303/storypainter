#!/usr/bin/env python3
"""
Hörbuch-Generator für Szenen-basierte Audiobooks
- Liest Szenen aus book_metadata.json
- Generiert Audio pro Szene in Chunks
- Benennt Dateien nach Szenen-Schema: scene_0001_chunk_001.wav
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"


class SceneBasedAudiobookGenerator:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        
    def load_progress(self):
        """Lädt Fortschritt"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                # Stelle sicher dass alle Keys existieren
                if "completed_scenes" not in progress:
                    progress["completed_scenes"] = []
                if "completed_chunks" not in progress:
                    progress["completed_chunks"] = []
                return progress
        return {"completed_scenes": [], "completed_chunks": []}
    
    def save_progress(self, scene_id, chunk_id):
        """Speichert Fortschritt"""
        progress = self.load_progress()
        chunk_key = f"scene_{scene_id:04d}_chunk_{chunk_id:03d}"
        
        if chunk_key not in progress["completed_chunks"]:
            progress["completed_chunks"].append(chunk_key)
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def mark_scene_complete(self, scene_id):
        """Markiert Szene als komplett"""
        progress = self.load_progress()
        if scene_id not in progress["completed_scenes"]:
            progress["completed_scenes"].append(scene_id)
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def split_scene_into_chunks(self, scene_text, max_chunk_length=350):
        """
        Teilt Szenentext in TTS-optimierte Chunks.
        Trennt an Satzgrenzen, achtet auf Lesbarkeit für XTTS.
        Keine künstlichen Pausen eingefügt (das übernimmst du beim Zusammenfügen).
        """
        import re

        # Vorreinigung: Steuerzeichen raus, Whitespaces vereinheitlichen
        text = scene_text.replace('_', ' ')
        text = re.sub(r'\s+', ' ', text).strip()

        # Satzweise splitten – unterstützt ., !, ?, … 
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        chunks = []
        current_chunk = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            # Unicode-Säuberung
            s = s.replace('\u00A0', ' ').replace('\u202f', ' ').strip()

            # Falls Satz zu lang, bei Kommas oder Doppelpunkten teilen
            if len(s) > max_chunk_length:
                parts = re.split(r'(?<=[,;:—–])\s+', s)
            else:
                parts = [s]

            for part in parts:
                if len(current_chunk) + len(part) > max_chunk_length and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = part + " "
                else:
                    current_chunk += part + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


    def prepare_text_for_xtts(self, raw_text: str) -> str:
        """
        Bereitet Text für XTTS v2 vor:
        - entfernt technische Zeichen
        - ersetzt typografische Sonderzeichen
        - schützt Zahlen mit Punkt (17. -> 17-tes)
        - ersetzt Guillemets («») und Gedankenstriche
        - normalisiert Anführungen und Leerzeichen
        """
        import re

        t = raw_text.strip()

        # Steuerzeichen entfernen
        t = t.replace('_', '').replace('*', '').replace('#', '').replace('|', '')

        # Guillemets und typografische Quotes
        t = t.replace("«", '"').replace("»", '"').replace("„", '"').replace("“", '"').replace("‚", "'").replace("‘", "'")

        # Gedankenstriche / lange Bindestriche zu Komma-Pausen
        t = re.sub(r'\s*[–—]\s*', ', ', t)

        # Punkt hinter Zahl → schützen
        t = re.sub(r'(\d+)\.', r'\1-tes', t)

        # Drei Punkte normalisieren (XTTS liest sonst zu lange Pausen)
        t = re.sub(r'\.{3,}', '...', t)

        # Zeilenumbrüche vereinheitlichen
        t = re.sub(r'\s+', ' ', t)

        # Typische Anführungszeichen fixen
        t = t.replace('“', '"').replace('”', '"').replace('„', '"')

        # Nach Satzzeichen ein Leerzeichen erzwingen
        t = re.sub(r'([.!?])([A-ZÄÖÜ])', r'\1 \2', t)

        # Letzter Feinschliff
        t = t.strip()

        return t

    
    def generate_chunk_audio(self, tts, chunk_text, scene_id, chunk_id):
        """Generiert Audio für einen Chunk"""
        text = self.prepare_text_for_xtts(chunk_text)
        output_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_id:03d}.wav"
        
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=self.config["speaker_wav"],
                language=self.config["language"],
                file_path=str(output_file),
                temperature=self.config.get("temperature", 0.75),
                repetition_penalty=self.config.get("repetition_penalty", 11.0),
                speed=1.0
            )
            return str(output_file)
        except Exception as e:
            print(f"    ❌ Fehler: {e}")
            return None

    def generate_audiobook_from_scenes(self):
        """Hauptfunktion: Generiert Audio aus book_metadata.json"""
        from TTS.api import TTS
        import torch
        
        print("\n🎧 SZENEN-BASIERTER HÖRBUCH-GENERATOR")
        print("="*60)
        
        # GPU-Status
        print(f"\n🔥 Hardware-Info:")
        print(f"   CUDA verfügbar: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Modell laden - MEHRERE METHODEN PROBIEREN
        print("\n📥 Lade XTTS-Modell...")
        tts = None
        
        # METHODE 1: Mit explizitem model_path und config_path
        if "model_path" in self.config and "config_path" in self.config:
            print(f"   Versuche mit model_path: {self.config['model_path']}")
            try:
                tts = TTS(
                    model_path=self.config["model_path"],
                    config_path=self.config["config_path"]
                )
                print("   ✅ Modell mit model_path geladen")
            except Exception as e:
                print(f"   ⚠️ model_path Fehler: {e}")
                tts = None
        
        # METHODE 2: Mit Modell-Namen (lädt von HuggingFace/lokal)
        if tts is None:
            print("   Versuche mit Modell-Namen: tts_models/multilingual/multi-dataset/xtts_v2")
            try:
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                print("   ✅ Modell von HuggingFace geladen")
            except Exception as e:
                print(f"   ⚠️ HuggingFace Fehler: {e}")
                tts = None
        
        # METHODE 3: Fallback auf einfaches Modell
        if tts is None:
            print("   Versuche Fallback-Modell: tts_models/de/thorsten/tacotron2-DDC")
            try:
                tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC")
                print("   ✅ Fallback-Modell geladen")
            except Exception as e:
                print(f"   ❌ Alle Modell-Lade-Versuche fehlgeschlagen!")
                print(f"   Letzter Fehler: {e}")
                print("\n💡 Lösungen:")
                print("   1. Prüfe model_path in CONFIG")
                print("   2. Lösche korrupte Modelle: rm -rf ~/.local/share/tts")
                print("   3. Installiere TTS neu: pip install --force-reinstall TTS")
                return False
        
        # Auf GPU verschieben falls verfügbar
        if torch.cuda.is_available():
            tts = tts.cuda()
            print("   ✅ Modell auf GPU")
        else:
            print("   ✅ Modell auf CPU")

        # Szenen laden
        print(f"\n📖 Lade Szenen aus: {self.config['scenes_file']}")
        with open(self.config["scenes_file"], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        scenes = metadata.get("scenes", [])
        print(f"✅ {len(scenes)} Szenen geladen")
        
        # Fortschritt laden
        progress = self.load_progress()
        completed_scenes = set(progress.get("completed_scenes", []))
        completed_chunks = set(progress.get("completed_chunks", []))

        # Szenen durchgehen
        print(f"\n🎙️ Generiere Audio für {len(scenes)} Szenen...")
        print("="*60)
        
        total_chunks = 0
        successful_chunks = 0
        failed_chunks = 0

        for scene_idx, scene in enumerate(scenes, 1):
            scene_id = scene_idx
            scene_text = scene.get("text", "")
            
            if not scene_text:
                print(f"\n[Szene {scene_id:04d}] ⚠️ Kein Text vorhanden, überspringe...")
                continue
            
            print(f"\n{'─'*60}")
            print(f"[Szene {scene_id:04d}/{len(scenes)}]")
            print(f"   Position: {scene.get('start_pos', 0)} - {scene.get('end_pos', 0)}")
            print(f"   Text-Länge: {len(scene_text)} Zeichen")
            
            # Prüfe ob Szene bereits komplett
            if scene_id in completed_scenes:
                print(f"   ⏭️ Szene bereits komplett generiert")
                continue
            
            # Teile in Chunks
            chunks = self.split_scene_into_chunks(
                scene_text, 
                self.config.get("max_chunk_length", 250)
            )
            print(f"   📝 {len(chunks)} Chunks erstellt")
            
            # Generiere Chunks
            scene_successful = 0
            for chunk_idx, chunk_text in enumerate(chunks, 1):
                chunk_key = f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}"
                
                # Skip wenn bereits generiert
                if chunk_key in completed_chunks:
                    print(f"   [{chunk_idx:03d}] ⏭️ Bereits vorhanden")
                    scene_successful += 1
                    total_chunks += 1
                    continue
                
                # Generiere Audio
                preview = chunk_text[:60] + ("..." if len(chunk_text) > 60 else "")
                print(f"   [{chunk_idx:03d}] 🎤 {preview}")
                
                start = time.time()
                output_file = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_idx)
                duration = time.time() - start
                
                if output_file:
                    print(f"           ✅ Erstellt in {duration:.1f}s")
                    self.save_progress(scene_id, chunk_idx)
                    scene_successful += 1
                    successful_chunks += 1
                else:
                    print(f"           ❌ Fehlgeschlagen")
                    failed_chunks += 1
                
                total_chunks += 1
            
            # Markiere Szene als komplett wenn alle Chunks erfolgreich
            if scene_successful == len(chunks):
                self.mark_scene_complete(scene_id)
                print(f"   ✅ Szene {scene_id:04d} komplett!")
            else:
                print(f"   ⚠️ Szene {scene_id:04d} unvollständig ({scene_successful}/{len(chunks)} Chunks)")

        # Zusammenfassung
        print(f"\n{'='*60}")
        print(f"✅ FERTIG!")
        print(f"📊 Statistik:")
        print(f"   Szenen gesamt: {len(scenes)}")
        print(f"   Chunks gesamt: {total_chunks}")
        print(f"   Erfolgreich: {successful_chunks}")
        print(f"   Fehlgeschlagen: {failed_chunks}")
        print(f"\n📁 Ausgabe: {self.output_dir}")
        print(f"📄 Fortschritt: {self.progress_file}")
        
        return True

def main():
    # --- Argumente aus Kommandozeile einlesen ---
    ap = argparse.ArgumentParser(description="Text-to-Speech Pipeline")
    ap.add_argument("--path", required=True, help="Basis-path imput- and output")
    args = ap.parse_args()

    base_path = args.path

    # --- CONFIG ---
    CONFIG = {
        # Stimmen / Modell (bleibt unverändert)
        "model_path": "/workspace/storypainter/voices/tom",
        "config_path": "/workspace/storypainter/voices/tom/config.json",
        "speaker_wav": "/workspace/storypainter/voices/tom/norma.wav",

        # Eingabe / Ausgabe (werden dynamisch kombiniert)
        "scenes_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "tts"),

        # TTS-Einstellungen
        "max_chunk_length": 440,
        "language": "de",
        "temperature": 0.60,
        "repetition_penalty": 2.0,
    }
    
    # Pfad-Validierung
    print("🔍 Prüfe Pfade...")
    required_paths = {
        "model_path": CONFIG["model_path"],
        "config_path": CONFIG["config_path"],
        "speaker_wav": CONFIG["speaker_wav"],
        "scenes_file": CONFIG["scenes_file"]
    }
    
    for path_key, path in required_paths.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {path_key}: {path}")
        if not exists:
            print(f"\n❌ Pfad existiert nicht: {path}")
            print(f"   Bitte korrigiere den Pfad in CONFIG['{path_key}']")
            sys.exit(1)
    
    print("\n✅ Alle Pfade validiert\n")
    
    # Generator starten
    generator = SceneBasedAudiobookGenerator(CONFIG)
    success = generator.generate_audiobook_from_scenes()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()