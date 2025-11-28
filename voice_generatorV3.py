#!/usr/bin/env python3
"""
Hörbuch-Generator für Szenen-basierte Audiobooks
- Liest Szenen aus book_metadata.json
- Generiert Audio pro Szene in Chunks
- Benennt Dateien nach Szenen-Schema: scene_0001_chunk_001.wav
- NEU: Whisper-QC für jeden Chunk + Retry-Logik
"""

import os
import sys
import json
import time
import argparse
import difflib
import re
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"


class SceneBasedAudiobookGenerator:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"

        # QC-Logging
        self.qc_problems_file = self.output_dir / "qc_problems.json"

        # Whisper-Modell (lazy load)
        self.whisper = None

    # ----------------------------
    # Fortschritt (aktuell optional)
    # ----------------------------
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

    # ----------------------------
    # Text-Chunking & Vorbereitung
    # ----------------------------
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

    # ----------------------------
    # Whisper / QC-Helfer
    # ----------------------------
    def ensure_whisper_loaded(self):
        """Lädt das Whisper QC-Modell nur einmal (lazy loading)."""
        if self.whisper is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("   ❌ faster-whisper nicht installiert. QC wird deaktiviert.")
            self.whisper = None
            return

        # Device bestimmen
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

        device = "cuda" if use_cuda else "cpu"

        model_name = self.config.get("whisper_model_name", "medium")
        compute_type = self.config.get(
            "whisper_compute_type",
            "int8_float16" if device == "cuda" else "int8"
        )

        print(f"\n📥 Lade Whisper QC-Modell ({model_name}, device={device}, compute_type={compute_type})...")
        try:
            self.whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
            print("   ✅ Whisper QC-Modell geladen")
        except Exception as e:
            print(f"   ❌ Konnte Whisper QC-Modell nicht laden: {e}")
            self.whisper = None

    def transcribe_with_whisper(self, wav_path: str) -> str:
        """Transkribiert eine WAV-Datei mit Whisper."""
        if self.whisper is None:
            return ""

        segments, _ = self.whisper.transcribe(wav_path, language="de")
        return " ".join([s.text for s in segments])

    def normalize_text_for_eval(self, text: str) -> str:
        """Einfaches Normalisieren für CER-Berechnung."""
        if not text:
            return ""
        t = text.lower()
        # alles außer Buchstaben/Zahlen/äöüß zu Leerzeichen
        t = re.sub(r"[^0-9a-zäöüß]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def compute_cer(self, ref: str, hyp: str) -> float:
        """
        Approximierte Character Error Rate (1 - Ähnlichkeitsindex).
        ref und hyp sollten vorher normalisiert werden.
        """
        ref = ref or ""
        hyp = hyp or ""
        if not ref and not hyp:
            return 0.0
        if not ref and hyp:
            return 1.0
        matcher = difflib.SequenceMatcher(None, ref, hyp)
        return 1.0 - matcher.ratio()

    def log_qc_problem(self, scene_id, chunk_id, text, transcript, cer_value, attempts):
        """Schreibt problematische Chunks in qc_problems.json."""
        entry = {
            "scene_id": scene_id,
            "chunk_id": chunk_id,
            "attempts": attempts,
            "cer": cer_value,
            "text": text,
            "transcript": transcript,
            "status": "failed"
        }

        data = []
        if self.qc_problems_file.exists():
            try:
                with open(self.qc_problems_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []

        data.append(entry)

        with open(self.qc_problems_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ----------------------------
    # TTS + QC
    # ----------------------------
    def generate_chunk_audio(self, tts, chunk_text, scene_id, chunk_id, temperature):
        """Generiert Audio für einen Chunk (ein Versuch mit gegebener Temperatur)."""
        text = self.prepare_text_for_xtts(chunk_text)
        output_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_id:03d}.wav"

        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=self.config["speaker_wav"],
                language=self.config["language"],
                file_path=str(output_file),
                temperature=temperature,
                repetition_penalty=self.config.get("repetition_penalty", 1.45),
                speed=1.0
            )
            return str(output_file)
        except Exception as e:
            print(f"    ❌ Fehler bei TTS: {e}")
            return None

    def generate_chunk_with_qc(self, tts, chunk_text, scene_id, chunk_id):
        """
        Generiert einen Chunk mit bis zu N Versuchen und Whisper-QC.

        - nutzt eine Temperatur-Liste (z.B. [0.70, 0.55, 0.35])
        - bricht ab, sobald CER unter Schwelle fällt
        - loggt problematische Chunks in qc_problems.json
        """
        # QC-Parameter aus Config
        base_temp = self.config.get("temperature", 0.70)
        temp_schedule = self.config.get("qc_temperature_schedule", [base_temp, 0.55, 0.35])
        cer_threshold = self.config.get("qc_cer_threshold", 0.08)  # 8%

        # Whisper-Modell laden (lazy)
        self.ensure_whisper_loaded()

        ref_norm = self.normalize_text_for_eval(chunk_text)
        last_cer = 1.0
        last_transcript = ""
        attempts = 0

        # Falls Whisper nicht verfügbar ist → TTS ohne QC, aber kein Fehler
        if self.whisper is None:
            print("           ⚠️ QC deaktiviert (kein Whisper-Modell verfügbar) – rendere einmal ohne Prüfung")
            path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, base_temp)
            return True, {"cer": None, "attempts": 1, "transcript": None}

        for temp in temp_schedule:
            attempts += 1
            print(f"           🔁 QC-Versuch {attempts} mit Temperatur {temp:.2f}")

            path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, temp)
            if not path:
                continue

            transcript = self.transcribe_with_whisper(path)
            hyp_norm = self.normalize_text_for_eval(transcript)
            cer_value = self.compute_cer(ref_norm, hyp_norm)
            last_cer = cer_value
            last_transcript = transcript

            print(f"               🔍 CER={cer_value:.3f} (Schwelle {cer_threshold:.3f})")

            if cer_value <= cer_threshold:
                # QC bestanden
                return True, {
                    "cer": cer_value,
                    "attempts": attempts,
                    "transcript": transcript
                }

        # Wenn wir hier landen: alle Versuche über Schwelle → Problem loggen
        print(f"           ⚠️ QC fehlgeschlagen nach {attempts} Versuchen (CER={last_cer:.3f})")
        self.log_qc_problem(scene_id, chunk_id, chunk_text, last_transcript, last_cer, attempts)

        return False, {
            "cer": last_cer,
            "attempts": attempts,
            "transcript": last_transcript
        }

    # ----------------------------
    # Hauptpipeline
    # ----------------------------
    def generate_audiobook_from_scenes(self):
        """Generiert Audio und erzeugt nur WAVs, die im Ordner noch fehlen, inkl. Whisper-QC."""
        from TTS.api import TTS
        import torch

        print("\n🎧 SZENEN-BASIERTER HÖRBUCH-GENERATOR mit QC")
        print("=" * 60)

        # GPU-Status
        print(f"\n🔥 Hardware-Info:")
        print(f"   CUDA verfügbar: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # Modell laden
        print("\n📥 Lade XTTS-Modell...")
        tts = None

        # METHODE 1: expliziter Pfad
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

        # METHODE 2: Modellname
        if tts is None:
            print("   Versuche mit Modell-Namen: tts_models/multilingual/multi-dataset/xtts_v2")
            try:
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                print("   ✅ Modell von HuggingFace geladen")
            except Exception as e:
                print(f"   ⚠️ HuggingFace Fehler: {e}")
                tts = None

        # METHODE 3: Fallback
        if tts is None:
            print("   Versuche Fallback-Modell: tts_models/de/thorsten/tacotron2-DDC")
            try:
                tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC")
                print("   ✅ Fallback-Modell geladen")
            except Exception as e:
                print("   ❌ Alle Modell-Lade-Versuche fehlgeschlagen!")
                print(f"   Letzter Fehler: {e}")
                return False

        # Auf GPU verschieben falls verfügbar
        if torch.cuda.is_available():
            tts = tts.cuda()
            print("   ✅ XTTS-Modell auf GPU")
        else:
            print("   ✅ XTTS-Modell auf CPU")

        # Szenen laden
        print(f"\n📖 Lade Szenen aus: {self.config['scenes_file']}")
        with open(self.config["scenes_file"], 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        scenes = metadata.get("scenes", [])
        print(f"✅ {len(scenes)} Szenen geladen")

        print(f"\n🎙️ Generiere Audio (nur fehlende Dateien, mit QC)...")
        print("=" * 60)

        total_chunks = 0
        newly_generated = 0
        skipped_existing = 0
        failed_chunks = 0

        for scene_idx, scene in enumerate(scenes, 1):
            scene_id = scene_idx
            scene_text = scene.get("text", "")

            if not scene_text:
                print(f"\n[Szene {scene_id:04d}] ⚠️ Kein Text vorhanden, überspringe...")
                continue

            print(f"\n{'─' * 60}")
            print(f"[Szene {scene_id:04d}/{len(scenes)}]")
            print(f"   Position: {scene.get('start_pos', 0)} - {scene.get('end_pos', 0)}")
            print(f"   Text-Länge: {len(scene_text)} Zeichen")

            # Szene in Chunks aufteilen
            chunks = self.split_scene_into_chunks(
                scene_text,
                self.config.get("max_chunk_length", 250)
            )
            print(f"   📝 {len(chunks)} Chunks erstellt")

            for chunk_idx, chunk_text in enumerate(chunks, 1):
                total_chunks += 1
                output_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}.wav"

                # nur Dateiexistenz checken
                if output_file.exists():
                    print(f"   [{chunk_idx:03d}] ⏭️ {output_file.name} existiert bereits")
                    skipped_existing += 1
                    continue

                preview = chunk_text[:60] + ("..." if len(chunk_text) > 60 else "")
                print(f"   [{chunk_idx:03d}] 🎤 Erzeuge {output_file.name} – {preview}")

                start = time.time()
                success, qc_info = self.generate_chunk_with_qc(tts, chunk_text, scene_id, chunk_idx)
                duration = time.time() - start

                if success:
                    cer_val = qc_info["cer"]
                    cer_str = f"{cer_val:.3f}" if cer_val is not None else "n/a"
                    print(f"           ✅ Erstellt in {duration:.1f}s (CER={cer_str}, Versuche={qc_info['attempts']})")
                    newly_generated += 1
                else:
                    print(f"           ⚠️ Audio erstellt, aber QC nicht bestanden (CER={qc_info['cer']:.3f})")
                    newly_generated += 1
                    failed_chunks += 1

                # optional: Fortschritt pro Chunk speichern
                self.save_progress(scene_id, chunk_idx)

            # Szene als komplett markieren
            self.mark_scene_complete(scene_id)

        # Zusammenfassung
        print(f"\n{'=' * 60}")
        print(f"✅ FERTIG!")
        print(f"📊 Statistik:")
        print(f"   Chunks gesamt (Szenen × Chunks): {total_chunks}")
        print(f"   Neu generiert: {newly_generated}")
        print(f"   Übersprungen (Datei existiert): {skipped_existing}")
        print(f"   Chunks mit QC-Problemen: {failed_chunks}")
        print(f"\n📁 Ausgabe: {self.output_dir}")
        if self.qc_problems_file.exists():
            print(f"   🔎 Details zu problematischen Chunks: {self.qc_problems_file}")

        return failed_chunks == 0


def main():
    # --- Argumente aus Kommandozeile einlesen ---
    ap = argparse.ArgumentParser(description="Text-to-Speech Pipeline")
    ap.add_argument("--path", required=True, help="Basis-path input- and output")
    args = ap.parse_args()

    base_path = args.path

    # --- CONFIG ---
    CONFIG = {
        # Stimmen / Modell (bleibt unverändert)
        "model_path": "/workspace/storypainter/voices",
        "config_path": "/workspace/storypainter/voices/config.json",
        "speaker_wav": "/workspace/storypainter/voices/2.wav",

        # Eingabe / Ausgabe (werden dynamisch kombiniert)
        "scenes_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "tts"),

        # TTS-Einstellungen
        "max_chunk_length": 320,
        "language": "de",
        "temperature": 0.70,
        "top_p": 0.93,
        "top_k": 35,
        "repetition_penalty": 1.45,

        # QC-Settings
        # Whisper-Modell & Compute-Type
        "whisper_model_name": "medium",          # z.B. "small", "medium", "large-v3"
        "whisper_compute_type": "int8_float16",  # für GPU, für CPU evtl. "int8"

        # Temperatur-Versuche für QC (Variante A: jeder Chunk wird geprüft)
        "qc_temperature_schedule": [0.70, 0.55, 0.35],

        # CER-Schwelle für "gut genug"
        "qc_cer_threshold": 0.08
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
