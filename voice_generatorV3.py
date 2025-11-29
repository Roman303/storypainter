#!/usr/bin/env python3
"""
Hörbuch-Generator für Szenen-basierte Audiobooks
- Liest Szenen aus book_metadata.json
- Generiert Audio pro Szene in Chunks
- Benennt Dateien nach Szenen-Schema: scene_0001_chunk_001.wav
- Whisper-QC für jeden Chunk + Retry-Logik
- V4: Whisper fest auf CPU + Re-Splitting von Problem-Chunks
"""

# ---- WICHTIG: ONNX-GPU & Whisper-Backend konfigurieren, bevor irgendwas importiert wird ----
import os
os.environ["ORT_DISABLE_ALL_GPU"] = "1"       # blockt alle ONNX-GPU-Provider
os.environ["ORT_BACKEND"] = "CPU"            # erzwingt ORT CPU-Backend
os.environ["ORT_PROVIDER"] = "CPU"           # kein CUDA/TensorRT Provider
os.environ["FWHISPER_BACKEND"] = "ct2"       # faster-whisper soll CTranslate2 benutzen

import sys
import json
import time
import argparse
import difflib
import re
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
import soundfile as sf
import numpy as np
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
    
    def split_problematic_chunk(self, text, max_len=None):
        """
        Splittet einen Problem-Chunk neu – Satzweise, dann Kommas, dann halbieren.
        Wird nur verwendet, wenn der ursprüngliche Chunk nach allen QC-Versuchen
        immer noch durchfällt.
        """
        if max_len is None:
            max_len = self.config.get("retry_chunk_length", 180)

        # Zuerst aufräumen
        t = text.strip()

        # 1) Satzweise splitten
        sentences = re.split(r'(?<=[.!?…])\s+', t)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Wenn mehrere Sätze und alle <= max_len: nimm sie
        if len(sentences) > 1 and all(len(s) <= max_len for s in sentences):
            return sentences

        # 2) Falls nur 1 langer Satz → nach Kommas splitten
        if len(sentences) == 1 or any(len(s) > max_len for s in sentences):
            parts = re.split(r',\s*', t)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1 and all(len(p) <= max_len for p in parts):
                return parts

        # 3) Fallback → hart in ca. gleich lange Teile schneiden
        if len(t) > max_len:
            num_sub = max(2, len(t) // max_len + 1)
            chunk_size = len(t) // num_sub
            subs = []
            start = 0
            for i in range(num_sub - 1):
                subs.append(t[start:start + chunk_size].strip())
                start += chunk_size
            subs.append(t[start:].strip())
            subs = [s for s in subs if s]
            if subs:
                return subs

        # wenn alles schiefgeht: Original zurück
        return [t]

    def prepare_text_for_xtts(self, raw_text: str) -> str:
        """
        Vollständig gehärtete Textvorbereitung für XTTS:
        - entfernt technische Zeichen / PDF-Artefakte
        - vereinheitlicht ALLE Anführungszeichen
        - neutralisiert lange Dashes (– — ― − ‒ etc.)
        - entfernt gefährliche Unicode-Symbole
        - schützt Punkt hinter Zahl
        - normalisiert Mehrfach-Leerzeichen
        - sorgt für stabile XTTS-Aussprache ohne lange Pausen
        """
        import re
    
        t = raw_text.strip()
    
        # ------------------------------
        # 1. Steuerzeichen entfernen
        # ------------------------------
        remove_chars = ['_', '*', '#', '|', '·', '•', '●', '►', '◄', '~']
        for c in remove_chars:
            t = t.replace(c, '')
    
        # Unsichtbare PDF-Zeichen & Zero-Width
        zero_width = ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]
        for z in zero_width:
            t = t.replace(z, "")
    
        # ------------------------------
        # 2. Typografische Quotes normalisieren
        # ------------------------------
        quote_map = {
            "«": '"', "»": '"',
            "„": '"', "“": '"', "”": '"',
            "‚": "'", "‘": "'", "ʼ": "'",
            "´": "'", "˝": '"', "‹": '"', "›": '"',
            "❝": '"', "❞": '"'
        }
        for bad, good in quote_map.items():
            t = t.replace(bad, good)
    
        # Hängende Quotes entfernen
        t = re.sub(r'(^"|"$)', '', t)              # am Anfang/Ende kompletter Text
        t = re.sub(r'\s"(\s|$)', ' ', t)           # isoliertes "
        t = re.sub(r"\s'(\s|$)", ' ', t)           # isoliertes '
    
        # ------------------------------
        # 3. Lange Dashes & Sonder-Dashes neutralisieren
        # ------------------------------
        dash_variants = [
            "–",  # EN DASH
            "—",  # EM DASH
            "―",  # HORIZONTAL BAR
            "−",  # MINUS SIGN (Mathe)
            "‒",  # FIGURE DASH
            "⁃",  # BULLET DASH
            "﹘", "﹣", "－", "ｰ"  # CJK Varianten
        ]
    
        for d in dash_variants:
            t = t.replace(d, ", ")
    
        # Doppelte und dreifache Dashes → Komma
        t = re.sub(r'[\-–—]{2,}', ', ', t)
    
        # ------------------------------
        # 4. Zahlen mit Punkt schützen
        #    12. → 12-tes
        # ------------------------------
        t = re.sub(r'(\d+)\.(\s|$)', r'\1-tes ', t)
    
        # ------------------------------
        # 5. Mehrfach-Punkte normalisieren
        # ------------------------------
        t = re.sub(r'\.{3,}', '...', t)
    
        # ------------------------------
        # 6. Whitespaces normalisieren
        # ------------------------------
        t = t.replace("\u00A0", " ")  # Non-breaking space
        t = t.replace("\u202F", " ")  # Narrow no-break space
        t = re.sub(r'\s+', ' ', t).strip()
    
        # ------------------------------
        # 7. Leerzeichen nach Satzzeichen sicherstellen
        # ------------------------------
        t = re.sub(r'([.!?])([A-ZÄÖÜ])', r'\1 \2', t)
    
        # ------------------------------
        # 8. Finale Säuberung
        # ------------------------------
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

        # Device explizit aus CONFIG lesen, nicht automatisch CUDA nehmen
        device = self.config.get("whisper_device", "cpu")  # "cpu" oder "cuda"

        model_name = self.config.get("whisper_model_name", "medium")

        if device == "cpu":
            default_compute = "int8"
        else:
            default_compute = "int8_float16"

        compute_type = self.config.get("whisper_compute_type", default_compute)

        print(f"\n📥 Lade Whisper QC-Modell ({model_name}, device={device}, compute_type={compute_type})...")
        try:
            self.whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
            backend = getattr(self.whisper, "_model_type", "unknown")
            print(f"   ✅ Whisper QC-Modell geladen (Backend: {backend})")
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

    
    def remove_long_silences(self, wav_path, max_silence_sec=1.0):
        """
        Entfernt Stille > max_silence_sec automatisch, egal ob echtes silent() oder leises Rauschen.
        Nutzt Librósa-Energy-Detection → sehr zuverlässig.
        """
        y, sr = librosa.load(wav_path, sr=None)
    
        # Frame-Größe für Erkennung
        frame_length = int(0.03 * sr)   # 30ms
        hop_length = int(0.01 * sr)     # 10ms
    
        # Energie pro Frame berechnen
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
        # Threshold bestimmen (adaptiv)
        silence_thresh = np.percentile(rms, 20) * 0.6   # sehr sicherer Schwellenwert
    
        silent_frames = rms < silence_thresh
    
        # Frames in Zeiten umrechnen
        sil_times = librosa.frames_to_time(np.where(silent_frames)[0], sr=sr, hop_length=hop_length)
    
        # Lange Stille finden
        min_sil_dur = max_silence_sec
        chunks = []
        current_start = None
    
        for i, t in enumerate(sil_times):
            if current_start is None:
                current_start = t
                prev_t = t
                continue
    
            if t - prev_t > 0.05:  # 50ms -> neue Stille beginnt
                if prev_t - current_start >= min_sil_dur:
                    chunks.append((current_start, prev_t))
                current_start = t
            prev_t = t
    
        # letzten Block prüfen
        if current_start is not None and prev_t - current_start >= min_sil_dur:
            chunks.append((current_start, prev_t))
    
        if not chunks:
            return False
    
        print(f"      ✂️ Librósa: {len(chunks)} lange Stillen gefunden")
    
        # Audio neu zusammenbauen (Stillen entfernen)
        keep_segments = []
        last_end = 0.0
    
        for (start, end) in chunks:
            # vorheriges Audio behalten
            keep_segments.append(y[int(last_end * sr):int(start * sr)])
            last_end = end
    
        # letzten Teil behalten
        keep_segments.append(y[int(last_end * sr):])
    
        if len(keep_segments) == 1:
            return False
    
        new_audio = np.concatenate(keep_segments)
    
        sf.write(wav_path, new_audio, sr)
        return True

    # ----------------------------
    # TTS + QC
    # ----------------------------
    def generate_chunk_audio(self, tts, chunk_text, scene_id, chunk_id, temperature, part_idx=None):
        """
        Generiert Audio für einen Chunk (ein Versuch mit gegebener Temperatur).
        part_idx: optional für Subchunks (Re-Splitting), z.B. _part_01
        """
        text = self.prepare_text_for_xtts(chunk_text)

        base_name = f"scene_{scene_id:04d}_chunk_{chunk_id:03d}"
        if part_idx is not None:
            base_name += f"_part_{part_idx:02d}"

        output_file = self.output_dir / f"{base_name}.wav"

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

    def generate_chunk_with_qc(self, tts, chunk_text, scene_id, chunk_id, part_idx=None):
        """
        Generiert einen Chunk (oder Subchunk) mit bis zu N Versuchen und Whisper-QC.

        - nutzt eine Temperatur-Liste (z.B. [0.70, 0.55, 0.35])
        - bricht ab, sobald CER unter Schwelle fällt
        - loggt problematische Chunks in qc_problems.json
        - part_idx: None = normaler Chunk, sonst Subchunk-ID
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
            path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, base_temp, part_idx=part_idx)
            if path:
                trimmed = self.remove_long_silences(
                    path,
                    max_silence_sec=self.config.get("max_silence_sec", 1.5)
                )
                if trimmed:
                    print("               ✂️ Lange Stille (librosa) entfernt")
            return True, {"cer": None, "attempts": 1, "transcript": None}

        for temp in temp_schedule:
            attempts += 1
            label = f"{chunk_id:03d}" if part_idx is None else f"{chunk_id:03d}_part_{part_idx:02d}"
            print(f"           🔁 QC-Versuch {attempts} für Chunk {label} mit Temperatur {temp:.2f}")
        
            path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, temp, part_idx=part_idx)
            if not path:
                continue
        
            # --- Silence fix direkt nach jedem Render ---
            trimmed = self.remove_long_silences(
                path,
                max_silence_sec=self.config.get("max_silence_sec", 1.5)
            )
            if trimmed:
                print("               ✂️ Lange Stille (librosa) entfernt")
            # --------------------------------------------
        
            transcript = self.transcribe_with_whisper(path)
            hyp_norm = self.normalize_text_for_eval(transcript)
            cer_value = self.compute_cer(ref_norm, hyp_norm)


            print(f"               🔍 CER={cer_value:.3f} (Schwelle {cer_threshold:.3f})")

            if cer_value <= cer_threshold:
                # QC bestanden
                return True, {
                    "cer": cer_value,
                    "attempts": attempts,
                    "transcript": transcript
                }

        # Wenn wir hier landen: alle Versuche über Schwelle → Problem loggen
        log_chunk_id = f"{chunk_id:03d}" if part_idx is None else f"{chunk_id:03d}_part_{part_idx:02d}"
        print(f"           ⚠️ QC fehlgeschlagen nach {attempts} Versuchen (CER={last_cer:.3f}) für Chunk {log_chunk_id}")
        self.log_qc_problem(scene_id, log_chunk_id, chunk_text, last_transcript, last_cer, attempts)

        return False, {
            "cer": last_cer,
            "attempts": attempts,
            "transcript": last_transcript
        }
        


    def trim_long_silences(self, wav_path, max_silence_ms=1500, target_silence_ms=500):
        """
        Entfernt zu lange Stille im Audio:
        - max_silence_ms: maximale Stille, die erlaubt ist (z.B. 1500 ms)
        - target_silence_ms: auf wie viel Millisekunden gekürzt werden soll
        """
        try:
            audio = AudioSegment.from_wav(wav_path)
        except Exception as e:
            print(f"      ⚠️ Konnte {wav_path} nicht laden: {e}")
            return False
    
        # Stille finden (Unter -45 dBFS gilt als Stille)
        silent_ranges = detect_silence(
            audio,
            min_silence_len=400,   # ab 0.4s definieren wir Stille
            silence_thresh=-45     # Pegel
        )
    
        if not silent_ranges:
            return False
    
        modified = False
        new_audio = audio
        offset = 0
    
        for start, end in silent_ranges:
            silence_len = end - start
    
            if silence_len > max_silence_ms:
                print(f"      ✂️ Lange Stille gefunden: {silence_len} ms → kürze auf {target_silence_ms} ms")
    
                # erstellen neue reduzierte Stille
                reduced = AudioSegment.silent(duration=target_silence_ms)
    
                # Audio neu zusammensetzen
                new_audio = (
                    new_audio[:start-offset] +
                    reduced +
                    new_audio[end-offset:]
                )
    
                offset += (silence_len - target_silence_ms)
                modified = True
    
        if modified:
            new_audio.export(wav_path, format="wav")
            return True
    
        return False
    

    def merge_subchunks(self, scene_id, chunk_id):
        """
        Fügt alle Subchunks (part_xx) eines Chunks wieder zu einer Datei zusammen.
        Beispiel:
            scene_0001_chunk_010_part_01.wav
            scene_0001_chunk_010_part_02.wav
        → erzeugt:
            scene_0001_chunk_010.wav
        """
        base_pattern = f"scene_{scene_id:04d}_chunk_{chunk_id:03d}_part_"
        part_files = sorted(self.output_dir.glob(f"{base_pattern}*.wav"))
    
        if not part_files:
            return False
    
        output_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_id:03d}.wav"
        print(f"           🔗 Fasse {len(part_files)} Subchunks zusammen → {output_file.name}")
    
        combined = AudioSegment.silent(duration=0)
        for wav in part_files:
            combined += AudioSegment.from_wav(wav)
    
        combined.export(output_file, format="wav")
    
        # Optional: Teile löschen
        for wav in part_files:
            try:
                wav.unlink()
            except:
                pass
    
        return True

    
    # ----------------------------
    # Hauptpipeline
    # ----------------------------
    def generate_audiobook_from_scenes(self):
        """Generiert Audio und erzeugt nur WAVs, die im Ordner noch fehlen, inkl. Whisper-QC."""
        from TTS.api import TTS
        import torch

        print("\n🎧 SZENEN-BASIERTER HÖRBUCH-GENERATOR mit QC (V4)")
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

        print(f"\n🎙️ Generiere Audio (nur fehlende Dateien, mit QC + Re-Splitting)...")
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
                base_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}.wav"

                # nur Dateiexistenz checken
                if base_file.exists():
                    print(f"   [{chunk_idx:03d}] ⏭️ {base_file.name} existiert bereits")
                    skipped_existing += 1
                    continue

                preview = chunk_text[:60] + ("..." if len(chunk_text) > 60 else "")
                print(f"   [{chunk_idx:03d}] 🎤 Erzeuge {base_file.name} – {preview}")

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
                    # Ursprünglichen (schlechten) Chunk löschen, bevor wir Subchunks erzeugen
                    if base_file.exists():
                        try:
                            base_file.unlink()
                            print(f"           🗑️ Lösche problematische Datei {base_file.name}")
                        except Exception as e:
                            print(f"           ⚠️ Konnte {base_file.name} nicht löschen: {e}")

                    # Re-Splitting des Chunks versuchen
                    subchunks = self.split_problematic_chunk(
                        chunk_text,
                        self.config.get("retry_chunk_length", 180)
                    )

                    if len(subchunks) > 1:
                        print(f"           ✂️ Chunk wird neu geteilt in {len(subchunks)} Subchunks")
                        for sub_i, sub_text in enumerate(subchunks, 1):
                            label = f"{chunk_idx:03d}_part_{sub_i:02d}"
                            print(f"           🔄 Sub-Chunk {label}")
                            sub_start = time.time()
                            sub_success, sub_qc = self.generate_chunk_with_qc(
                                tts, sub_text, scene_id, chunk_idx, part_idx=sub_i
                            )
                            sub_duration = time.time() - sub_start
                            if sub_success:
                                cer_val = sub_qc["cer"]
                                cer_str = f"{cer_val:.3f}" if cer_val is not None else "n/a"
                                print(f"               ✅ Sub-Chunk OK in {sub_duration:.1f}s (CER={cer_str}, Versuche={sub_qc['attempts']})")
                            else:
                                print(f"               ⚠️ Sub-Chunk QC failed (CER={sub_qc['cer']:.3f})")
                                failed_chunks += 1

                        # Nach Abschluss aller Subchunks: wieder zusammenfügen
                        merged = self.merge_subchunks(scene_id, chunk_idx)
                        if merged:
                            print(f"           🔗 Subchunks zusammengeführt → Chunk {chunk_idx:03d} wiederhergestellt")


                        newly_generated += 1
                    else:
                        # selbst nach Re-Splitting keine sinnvolle Teilung möglich
                        failed_chunks += 1
                        newly_generated += 1

                # optional: Fortschritt pro Chunk speichern
                self.save_progress(scene_id, chunk_idx)

            # Szene als komplett markieren
            self.mark_scene_complete(scene_id)

        # Zusammenfassung
        print(f"\n{'=' * 60}")
        print(f"✅ FERTIG!")
        print(f"📊 Statistik:")
        print(f"   Chunks gesamt (Szenen × Chunks): {total_chunks}")
        print(f"   Neu generiert (inkl. Subchunks): {newly_generated}")
        print(f"   Übersprungen (Datei existiert): {skipped_existing}")
        print(f"   Chunks/Subchunks mit QC-Problemen: {failed_chunks}")
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
        # Stimmen / Modell
        "model_path": "/workspace/storypainter/voices/teo",
        "config_path": "/workspace/storypainter/voices/teo/config.json",
        "speaker_wav": "/workspace/storypainter/voices/teo/2.wav",

        # Eingabe / Ausgabe
        "scenes_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "tts"),

        # TTS-Einstellungen
        "max_chunk_length": 300,
        "language": "de",
        "temperature": 0.70,
        "top_p": 0.93,
        "top_k": 35,
        "repetition_penalty": 1.45,

        # QC-Settings
        # Whisper-Modell & Compute-Type
        "whisper_model_name": "medium",     # z.B. "small", "medium"
        "whisper_device": "cpu",            # CPU erzwingen (stabil, kein cuDNN)
        "whisper_compute_type": "int8",     # für CPU

        # Temperatur-Versuche für QC (Variante A: jeder Chunk wird geprüft)
        "qc_temperature_schedule": [0.70, 0.55, 0.35],
        "max_silence_sec": 0.9,
        # CER-Schwelle für "gut genug"
        "qc_cer_threshold": 0.12,           # etwas weniger strikt als 0.08

        # Re-Splitting: Ziel-Länge für Problem-Chunks
        "retry_chunk_length": 180
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
