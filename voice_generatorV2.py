#!/usr/bin/env python3
"""
Hörbuch-Generator für Szenen-basierte Audiobooks - OPTIMIERT
- GPU-optimiert für RTX 4090
- Originale werden als _a.wav gesichert
- Kompaktes QC-Logging
- Zuverlässiges Silence Removal
"""

import os
os.environ["ORT_DISABLE_ALL_GPU"] = "1"
os.environ["ORT_BACKEND"] = "CPU"
os.environ["ORT_PROVIDER"] = "CPU"
os.environ["FWHISPER_BACKEND"] = "ct2"
os.environ["COQUI_TOS_AGREED"] = "1"

import sys
import json
import time
import argparse
import difflib
import re
from pathlib import Path
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np


class SceneBasedAudiobookGenerator:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        self.qc_problems_file = self.output_dir / "qc_problems.json"
        self.whisper = None

    # ========== PROGRESS TRACKING ==========
    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                if "completed_scenes" not in progress:
                    progress["completed_scenes"] = []
                if "completed_chunks" not in progress:
                    progress["completed_chunks"] = []
                return progress
        return {"completed_scenes": [], "completed_chunks": []}

    def save_progress(self, scene_id, chunk_id):
        progress = self.load_progress()
        chunk_key = f"scene_{scene_id:04d}_chunk_{chunk_id:03d}"
        if chunk_key not in progress["completed_chunks"]:
            progress["completed_chunks"].append(chunk_key)
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def mark_scene_complete(self, scene_id):
        progress = self.load_progress()
        if scene_id not in progress["completed_scenes"]:
            progress["completed_scenes"].append(scene_id)
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    # ========== TEXT PROCESSING ==========
    def split_scene_into_chunks(self, scene_text, max_chunk_length=350):
        text = scene_text.replace('_', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        chunks = []
        current_chunk = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s = s.replace('\u00A0', ' ').replace('\u202f', ' ').strip()

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
        if max_len is None:
            max_len = self.config.get("retry_chunk_length", 180)

        t = text.strip()
        sentences = re.split(r'(?<=[.!?…])\s+', t)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 1 and all(len(s) <= max_len for s in sentences):
            return sentences

        if len(sentences) == 1 or any(len(s) > max_len for s in sentences):
            parts = re.split(r',\s*', t)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1 and all(len(p) <= max_len for p in parts):
                return parts

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

        return [t]

    def prepare_text_for_xtts(self, raw_text: str) -> str:
        t = raw_text.strip()
        
        # Steuerzeichen
        remove_chars = ['_', '*', '#', '|', '·', '•', '●', '►', '◄', '~']
        for c in remove_chars:
            t = t.replace(c, '')
        
        zero_width = ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]
        for z in zero_width:
            t = t.replace(z, "")
        
        # Quotes normalisieren
        quote_map = {
            "«": '"', "»": '"', "„": '"', """: '"', """: '"',
            "‚": "'", "'": "'", "ʼ": "'", "´": "'", "ˈ": '"',
            "‹": '"', "›": '"', "〝": '"', "〞": '"'
        }
        for bad, good in quote_map.items():
            t = t.replace(bad, good)
        
        t = re.sub(r'(^"|"$)', '', t)
        t = re.sub(r'\s"(\s|$)', ' ', t)
        t = re.sub(r"\s'(\s|$)", ' ', t)
        
        # Dashes
        dash_variants = ["–", "—", "―", "−", "‑", "⁃", "﹘", "﹣", "－", "ｰ"]
        for d in dash_variants:
            t = t.replace(d, ", ")
        t = re.sub(r'[\-–—]{2,}', ', ', t)
        
        # Zahlen mit Punkt
        t = re.sub(r'(\d+)\.(\s|$)', r'\1-tes ', t)
        
        # Mehrfach-Punkte
        t = re.sub(r'\.{3,}', '...', t)
        
        # Whitespaces
        t = t.replace("\u00A0", " ")
        t = t.replace("\u202F", " ")
        t = re.sub(r'\s+', ' ', t).strip()
        
        # Leerzeichen nach Satzzeichen
        t = re.sub(r'([.!?])([A-ZÄÖÜ])', r'\1 \2', t)
        
        return t.strip()

    # ========== WHISPER QC ==========
    def ensure_whisper_loaded(self):
        if self.whisper is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("   ⚠️ faster-whisper nicht installiert. QC wird deaktiviert.")
            self.whisper = None
            return

        device = self.config.get("whisper_device", "cpu")
        model_name = self.config.get("whisper_model_name", "medium")
        compute_type = self.config.get("whisper_compute_type", "int8")

        print(f"\n🔥 Lade Whisper QC-Modell ({model_name}, device={device}, compute_type={compute_type})...")
        try:
            self.whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f"   ✅ Whisper QC-Modell geladen")
        except Exception as e:
            print(f"   ⚠️ Konnte Whisper QC-Modell nicht laden: {e}")
            self.whisper = None

    def transcribe_with_whisper(self, wav_path: str) -> str:
        if self.whisper is None:
            return ""
        segments, _ = self.whisper.transcribe(wav_path, language="de")
        return " ".join([s.text for s in segments])

    def normalize_text_for_eval(self, text: str) -> str:
        if not text:
            return ""
        t = text.lower()
        t = re.sub(r"[^0-9a-zäöüß]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def compute_cer(self, ref: str, hyp: str) -> float:
        ref = ref or ""
        hyp = hyp or ""
        if not ref and not hyp:
            return 0.0
        if not ref and hyp:
            return 1.0
        matcher = difflib.SequenceMatcher(None, ref, hyp)
        return 1.0 - matcher.ratio()

    def log_qc_problem(self, scene_id, chunk_id, cer_value, attempts):
        """KOMPAKTES QC-Logging - nur Dateiname + CER"""
        entry = {
            "file": f"scene_{scene_id:04d}_chunk_{chunk_id:03d}.wav",
            "cer": round(cer_value, 3),
            "attempts": attempts
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

    # ========== SILENCE REMOVAL (VERBESSERT) ==========
    def remove_long_silences(self, wav_path, max_silence_sec=1.0):
        """
        ZUVERLÄSSIGE Silence-Entfernung mit Librosa Energy Detection.
        Läuft IMMER, nicht nur bei QC.
        """
        try:
            y, sr = librosa.load(wav_path, sr=None)
        except Exception as e:
            print(f"      ⚠️ Konnte {wav_path} nicht laden: {e}")
            return False

        frame_length = int(0.03 * sr)
        hop_length = int(0.01 * sr)
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        silence_thresh = np.percentile(rms, 20) * 0.6
        silent_frames = rms < silence_thresh
        
        sil_times = librosa.frames_to_time(np.where(silent_frames)[0], sr=sr, hop_length=hop_length)
        
        min_sil_dur = max_silence_sec
        chunks = []
        current_start = None
        prev_t = None
        
        for t in sil_times:
            if current_start is None:
                current_start = t
                prev_t = t
                continue
            
            if t - prev_t > 0.05:
                if prev_t - current_start >= min_sil_dur:
                    chunks.append((current_start, prev_t))
                current_start = t
            prev_t = t
        
        if current_start is not None and prev_t is not None and prev_t - current_start >= min_sil_dur:
            chunks.append((current_start, prev_t))
        
        if not chunks:
            return False
        
        print(f"      ✂️ {len(chunks)} lange Stillen gefunden & entfernt")
        
        keep_segments = []
        last_end = 0.0
        
        for (start, end) in chunks:
            keep_segments.append(y[int(last_end * sr):int(start * sr)])
            last_end = end
        
        keep_segments.append(y[int(last_end * sr):])
        
        if len(keep_segments) == 1:
            return False
        
        new_audio = np.concatenate(keep_segments)
        sf.write(wav_path, new_audio, sr)
        return True

    # ========== TTS + QC MIT BACKUP ==========
    def generate_chunk_audio(self, tts, chunk_text, scene_id, chunk_id, temperature, part_idx=None):
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
            print(f"    ⚠️ Fehler bei TTS: {e}")
            return None

    def backup_original(self, wav_path):
        """Sichert Original als _a.wav"""
        p = Path(wav_path)
        backup_path = p.parent / (p.stem + "_a.wav")
        try:
            import shutil
            shutil.copy2(wav_path, backup_path)
            return True
        except Exception as e:
            print(f"      ⚠️ Backup fehlgeschlagen: {e}")
            return False

    def generate_chunk_with_qc(self, tts, chunk_text, scene_id, chunk_id, part_idx=None):
        """
        Generiert Chunk mit QC.
        - Original wird IMMER als _a.wav gesichert
        - Silence Removal läuft IMMER
        - Ab 3. Versuch: Log in qc_problems.json
        """
        base_temp = self.config.get("temperature", 0.70)
        temp_schedule = self.config.get("qc_temperature_schedule", [base_temp, 0.55, 0.35])
        cer_threshold = self.config.get("qc_cer_threshold", 0.08)

        self.ensure_whisper_loaded()

        ref_norm = self.normalize_text_for_eval(chunk_text)
        last_cer = 1.0
        attempts = 0

        # Ohne Whisper: einmal rendern + Silence Fix
        if self.whisper is None:
            print("           ⚠️ QC deaktiviert (kein Whisper) – rendere ohne Prüfung")
            path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, base_temp, part_idx=part_idx)
            if path:
                self.remove_long_silences(path, max_silence_sec=self.config.get("max_silence_sec", 1.0))
            return True, {"cer": None, "attempts": 1, "transcript": None}

        for temp in temp_schedule:
            attempts += 1
            label = f"{chunk_id:03d}" if part_idx is None else f"{chunk_id:03d}_part_{part_idx:02d}"
            print(f"           🔍 QC-Versuch {attempts} für Chunk {label} (Temp {temp:.2f})")
        
            path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, temp, part_idx=part_idx)
            if not path:
                continue
            
            # Original-Backup bei erstem Versuch
            if attempts == 1:
                self.backup_original(path)
            
            # Silence Fix (IMMER!)
            self.remove_long_silences(path, max_silence_sec=self.config.get("max_silence_sec", 1.0))
            
            # Whisper QC
            transcript = self.transcribe_with_whisper(path)
            hyp_norm = self.normalize_text_for_eval(transcript)
            cer_value = self.compute_cer(ref_norm, hyp_norm)
            last_cer = cer_value

            print(f"               📊 CER={cer_value:.3f} (Schwelle {cer_threshold:.3f})")

            if cer_value <= cer_threshold:
                return True, {"cer": cer_value, "attempts": attempts, "transcript": transcript}

        # Nach allen Versuchen: Problem loggen
        log_chunk_id = f"{chunk_id:03d}" if part_idx is None else f"{chunk_id:03d}_part_{part_idx:02d}"
        print(f"           ⚠️ QC fehlgeschlagen nach {attempts} Versuchen (CER={last_cer:.3f})")
        self.log_qc_problem(scene_id, log_chunk_id, last_cer, attempts)

        return False, {"cer": last_cer, "attempts": attempts, "transcript": ""}

    def merge_subchunks(self, scene_id, chunk_id):
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
    
        for wav in part_files:
            try:
                wav.unlink()
            except:
                pass
    
        return True

    # ========== MAIN PIPELINE ==========
    def generate_audiobook_from_scenes(self):
        from TTS.api import TTS
        import torch

        print("\n🎧 SZENEN-BASIERTER HÖRBUCH-GENERATOR V4 - OPTIMIERT")
        print("=" * 60)

        # GPU-Check
        print(f"\n🔥 Hardware-Info:")
        print(f"   CUDA verfügbar: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # XTTS laden
        print("\n🔥 Lade XTTS-Modell...")
        tts = None

        if "model_path" in self.config and "config_path" in self.config:
            try:
                tts = TTS(
                    model_path=self.config["model_path"],
                    config_path=self.config["config_path"]
                )
                print("   ✅ Custom-Modell geladen")
            except Exception as e:
                print(f"   ⚠️ Custom-Modell Fehler: {e}")

        if tts is None:
            try:
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                print("   ✅ HuggingFace-Modell geladen")
            except Exception as e:
                print(f"   ⚠️ HuggingFace Fehler: {e}")
                return False

        # GPU aktivieren (Auto-Detect, funktioniert mit 4070/4090/etc.)
        if torch.cuda.is_available():
            # Optional: Spezifische GPU wählen (0, 1, 2...)
            gpu_id = self.config.get("gpu_id", 0)
            torch.cuda.set_device(gpu_id)
            
            tts = tts.cuda()
            torch.backends.cudnn.benchmark = True  # Performance-Boost
            
            gpu_name = torch.cuda.get_device_name(gpu_id)
            vram = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"   ✅ XTTS auf GPU {gpu_id}: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("   ✅ XTTS auf CPU")

        # Szenen laden
        print(f"\n📖 Lade Szenen aus: {self.config['scenes_file']}")
        with open(self.config["scenes_file"], 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        scenes = metadata.get("scenes", [])
        print(f"✅ {len(scenes)} Szenen geladen")

        print(f"\n🎙️ Generiere Audio (nur fehlende Dateien)...")
        print("=" * 60)

        total_chunks = 0
        newly_generated = 0
        skipped_existing = 0
        failed_chunks = 0

        for scene_idx, scene in enumerate(scenes, 1):
            scene_id = scene_idx
            scene_text = scene.get("text", "")

            if not scene_text:
                print(f"\n[Szene {scene_id:04d}] ⚠️ Kein Text, überspringe...")
                continue

            print(f"\n{'─' * 60}")
            print(f"[Szene {scene_id:04d}/{len(scenes)}]")
            print(f"   Text-Länge: {len(scene_text)} Zeichen")

            chunks = self.split_scene_into_chunks(
                scene_text,
                self.config.get("max_chunk_length", 250)
            )
            print(f"   📝 {len(chunks)} Chunks erstellt")

            for chunk_idx, chunk_text in enumerate(chunks, 1):
                total_chunks += 1
                base_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}.wav"

                if base_file.exists():
                    print(f"   [{chunk_idx:03d}] ⏭️ {base_file.name} existiert")
                    skipped_existing += 1
                    continue

                preview = chunk_text[:60] + ("..." if len(chunk_text) > 60 else "")
                print(f"   [{chunk_idx:03d}] 🎤 {preview}")

                start = time.time()
                success, qc_info = self.generate_chunk_with_qc(tts, chunk_text, scene_id, chunk_idx)
                duration = time.time() - start

                if success:
                    cer_str = f"{qc_info['cer']:.3f}" if qc_info['cer'] else "n/a"
                    print(f"           ✅ Fertig in {duration:.1f}s (CER={cer_str})")
                    newly_generated += 1
                else:
                    print(f"           ⚠️ QC nicht bestanden (CER={qc_info['cer']:.3f})")
                    
                    # Original löschen vor Re-Split
                    if base_file.exists():
                        base_file.unlink()

                    # Re-Splitting
                    subchunks = self.split_problematic_chunk(
                        chunk_text,
                        self.config.get("retry_chunk_length", 180)
                    )

                    if len(subchunks) > 1:
                        print(f"           ✂️ Neu aufgeteilt in {len(subchunks)} Subchunks")
                        for sub_i, sub_text in enumerate(subchunks, 1):
                            sub_success, sub_qc = self.generate_chunk_with_qc(
                                tts, sub_text, scene_id, chunk_idx, part_idx=sub_i
                            )
                            if sub_success:
                                print(f"               ✅ Sub-Chunk {sub_i} OK")
                            else:
                                print(f"               ⚠️ Sub-Chunk {sub_i} fehlgeschlagen")
                                failed_chunks += 1

                        self.merge_subchunks(scene_id, chunk_idx)
                        newly_generated += 1
                    else:
                        failed_chunks += 1
                        newly_generated += 1

                self.save_progress(scene_id, chunk_idx)

            self.mark_scene_complete(scene_id)

        print(f"\n{'=' * 60}")
        print(f"✅ FERTIG!")
        print(f"📊 Statistik:")
        print(f"   Chunks gesamt: {total_chunks}")
        print(f"   Neu generiert: {newly_generated}")
        print(f"   Übersprungen: {skipped_existing}")
        print(f"   Fehlerhafte: {failed_chunks}")
        print(f"\n📁 Ausgabe: {self.output_dir}")
        if self.qc_problems_file.exists():
            print(f"   🔎 QC-Probleme: {self.qc_problems_file}")

        return failed_chunks == 0


def main():
    ap = argparse.ArgumentParser(description="XTTS Hörbuch Generator")
    ap.add_argument("--path", required=True, help="Basis-Pfad für Input/Output")
    args = ap.parse_args()

    base_path = args.path

    CONFIG = {
        # Custom Voice Model
        "model_path": "/workspace/storypainter/voices/teo",
        "config_path": "/workspace/storypainter/voices/teo/config.json",
        "speaker_wav": "/workspace/storypainter/voices/teo/2.wav",

        # Dateien
        "scenes_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "tts"),

        # TTS-Einstellungen
        "max_chunk_length": 300,
        "language": "de",
        "temperature": 0.70,
        "repetition_penalty": 1.45,

        # QC (Whisper auf CPU für Stabilität)
        "whisper_model_name": "medium",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "qc_temperature_schedule": [0.70, 0.55, 0.35],
        "qc_cer_threshold": 0.12,
        "max_silence_sec": 0.9,
        "retry_chunk_length": 180
    }

    # Pfad-Validierung
    print("🔍 Prüfe Pfade...")
    required = {
        "model_path": CONFIG["model_path"],
        "config_path": CONFIG["config_path"],
        "speaker_wav": CONFIG["speaker_wav"],
        "scenes_file": CONFIG["scenes_file"]
    }

    for name, path in required.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        if not exists:
            print(f"\n❌ Pfad nicht gefunden: {path}")
            sys.exit(1)

    print("\n✅ Alle Pfade OK\n")

    generator = SceneBasedAudiobookGenerator(CONFIG)
    success = generator.generate_audiobook_from_scenes()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()