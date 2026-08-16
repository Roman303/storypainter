#!/usr/bin/env python3
"""
Hörbuch-Generator V7 - Production Ready
- Multi-Sample Support (4-6 Samples für natürliche Stimme)
- RTX 4070/4090 optimiert
- 3 Quality Modes: --low 0/1/2
- Zahlen-Normalisierung für weniger Artefakte
- Latent-Cache: Speaker-Embeddings einmal vorberechnet → kein Re-Encoding pro Chunk
- Kein Whisper/QC - maximale Geschwindigkeit
- RTF-Anzeige pro Chunk
- \n\n Absatz-Splitting
- Low Mode 2: chunk_split statt max_chunk_length ändern
"""

import os
os.environ["COQUI_TOS_AGREED"] = "1"

import sys
import json
import time
import argparse
import re
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence


class SceneBasedAudiobookGenerator:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        
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

    def split_scene_into_chunks(self, scene_text, max_chunk_length=350):
        # ERST bei \n\n splitten (Absätze) → jeder Absatz wird eigener Chunk-Block
        paragraphs = re.split(r'\n\n+', scene_text)
        all_chunks = []

        for paragraph in paragraphs:
            text = paragraph.replace('_', ' ')
            text = re.sub(r'[ \t]+', ' ', text).strip()
            if not text:
                continue

            sentences = re.split(r'(?<=[.!?…])\s+', text)
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
                        all_chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                    else:
                        current_chunk += part + " "

            if current_chunk.strip():
                all_chunks.append(current_chunk.strip())

        return all_chunks
    
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

    def _number_to_german(self, num):
        """Hilfsfunktion: Zahl → deutsches Wort"""
        ones = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
        teens = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", 
                 "siebzehn", "achtzehn", "neunzehn"]
        tens = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
        
        if num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            digit = num % 10
            ten = num // 10
            if digit == 0:
                return tens[ten]
            return ones[digit] + "und" + tens[ten]
        elif num < 1000:
            hundred = num // 100
            rest = num % 100
            result = ones[hundred] + "hundert"
            if rest > 0:
                result += self._number_to_german(rest)
            return result
        elif num < 1000000:
            thousands = num // 1000
            rest = num % 1000
            result = self._number_to_german(thousands) + "tausend"
            if rest > 0:
                result += self._number_to_german(rest)
            return result
        elif num < 1000000000:
            millions = num // 1000000
            rest = num % 1000000
            if millions == 1:
                result = "eine Million"
            else:
                result = self._number_to_german(millions) + " Millionen"
            if rest > 0:
                result += self._number_to_german(rest)
            return result
        else:
            return str(num)  # Fallback für sehr große Zahlen

    def prepare_text_for_xtts(self, raw_text: str) -> str:
        t = raw_text.strip()
        
        # 1. Formatierungszeichen entfernen (unerwünschte)
        remove_chars = ['_', '*', '#', '|', '·', '•', '◆', '►', '◄', '~']
        for c in remove_chars:
            t = t.replace(c, '')
        
        # 2. Unsichtbare/Zero-Width Zeichen entfernen
        zero_width = ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]
        for z in zero_width:
            t = t.replace(z, "")
        
        # 3. Anführungszeichen vereinheitlichen
        quote_map = {
            "«": '"', "»": '"', "„": '"', """: '"', """: '"',
            "‚": "'", "'": "'", "'": "'", "ʼ": "'", "´": "'", "ˈ": "'",
            "‹": '"', "›": '"', "〈": '"', "〉": '"'
        }
        for bad, good in quote_map.items():
            t = t.replace(bad, good)
        
        # 4. Einsame Anführungszeichen entfernen
        t = re.sub(r'(^"|"$)', '', t)
        t = re.sub(r'\s"(\s|$)', ' ', t)
        t = re.sub(r"\s'(\s|$)", ' ', t)
        
        # 5. Bindestriche und Gedankenstriche zu Kommas
        dash_variants = ["–", "—", "―", "−", "‐", "‑", "﹘", "﹣", "－", "ｰ"]
        for d in dash_variants:
            t = t.replace(d, ", ")
        t = re.sub(r'[\-–—]{2,}', ', ', t)
        
        # === 6. ZAHLEN-NORMALISIERUNG (NEU!) ===
        
        # Jahreszahlen: 1984 → neunzehnhundertvierundachtzig
        def year_to_words(match):
            year = int(match.group(1))
            if 1000 <= year <= 2100:
                return self._number_to_german(year)
            return match.group(0)
        
        t = re.sub(r'\b(1\d{3}|20\d{2})\b', year_to_words, t)
        
        # Große Zahlen mit Punkten: 1.000.000 → eine Million
        def large_number_to_words(match):
            num_str = match.group(0).replace('.', '')
            try:
                num = int(num_str)
                if num >= 1000:
                    return self._number_to_german(num)
            except:
                pass
            return match.group(0)
        
        t = re.sub(r'\b\d{1,3}(?:\.\d{3})+\b', large_number_to_words, t)
        
        # Ordnungszahlen: 1. → erste, 2. → zweite (nur häufige)
        ordinals = {
            "1.": "erste", "2.": "zweite", "3.": "dritte", "4.": "vierte", 
            "5.": "fünfte", "6.": "sechste", "7.": "siebte", "8.": "achte",
            "9.": "neunte", "10.": "zehnte", "20.": "zwanzigste", "100.": "hundertste"
        }
        for num, word in ordinals.items():
            t = re.sub(rf'\b{re.escape(num)}\s', f'{word} ', t)
        
        # Uhrzeiten: 14:30 → vierzehn Uhr dreißig
        def time_to_words(match):
            h, m = int(match.group(1)), int(match.group(2))
            h_word = self._number_to_german(h)
            if m == 0:
                return f"{h_word} Uhr"
            m_word = self._number_to_german(m)
            return f"{h_word} Uhr {m_word}"
        
        t = re.sub(r'\b(\d{1,2}):(\d{2})\b', time_to_words, t)
        
        # Prozent und Währungen
        t = re.sub(r'(\d+)\s*%', lambda m: self._number_to_german(int(m.group(1))) + ' Prozent', t)
        t = re.sub(r'(\d+)\s*€', lambda m: self._number_to_german(int(m.group(1))) + ' Euro', t)
        t = re.sub(r'(\d+)\s*\$', lambda m: self._number_to_german(int(m.group(1))) + ' Dollar', t)
        
        # 7. Ellipsen vereinheitlichen
        t = re.sub(r'\.{3,}', '...', t)
        
        # 8. Nicht-Breaking Spaces zu normalen Leerzeichen
        t = t.replace("\u00A0", " ")
        t = t.replace("\u202F", " ")
        
        # 9. Zeilenumbrüche BEHALTEN, aber normalisieren
        lines = t.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip():
                cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
                cleaned_lines.append(cleaned_line)
        
        t = '\n'.join(cleaned_lines)
        
        # 10. Fehlende Leerzeichen nach Satzzeichen hinzufügen
        t = re.sub(r'([.!?…])(?![ \n])', r'\1 ', t)
        
        # 11. Optional: SSML-Pausen
        if "[PAUSE_LONG]" in t:
            t = t.replace("[PAUSE_LONG]", '<break time="800ms"/>')
        if "[PAUSE_MEDIUM]" in t:
            t = t.replace("[PAUSE_MEDIUM]", '<break time="400ms"/>')
        
        return t

    def remove_long_silences(self, wav_path, max_silence_sec=1.0):
        try:
            audio = AudioSegment.from_wav(wav_path)
        except Exception as e:
            print(f"      ⚠️ Konnte {wav_path} nicht laden: {e}")
            return False
    
        min_silence_len = int(max_silence_sec * 1000)
        silence_thresh = max(audio.dBFS - 25, -55)
    
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=350
        )
    
        if len(chunks) <= 1:
            return False
    
        print(f"      ✂️ Entferne {len(chunks)-1} lange Stillen (> {max_silence_sec}s)")
    
        new_audio = AudioSegment.empty()
        for c in chunks:
            new_audio += c
    
        new_audio.export(wav_path, format="wav")
        return True
    
    def generate_chunk_audio(self, tts, chunk_text, scene_id, chunk_id, temperature, part_idx=None):
        text = self.prepare_text_for_xtts(chunk_text)

        base_name = f"scene_{scene_id:04d}_chunk_{chunk_id:03d}"
        if part_idx is not None:
            base_name += f"_part_{part_idx:02d}"

        output_file = self.output_dir / f"{base_name}.wav"

        try:
            if self.gpt_cond_latent is not None and self.speaker_embedding is not None:
                model = tts.synthesizer.tts_model
                out = model.inference(
                    text=text,
                    language=self.config["language"],
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    temperature=temperature,
                    repetition_penalty=self.config.get("repetition_penalty", 1.45),
                    top_p=self.config.get("top_p", 0.93),
                    top_k=self.config.get("top_k", 35),
                    speed=self.config.get("speed", 1.0),
                )
                sf.write(str(output_file), out["wav"], 24000)
            else:
                tts.tts_to_file(
                    text=text,
                    speaker_wav=self.config["speaker_wav"],
                    language=self.config["language"],
                    file_path=str(output_file),
                    temperature=temperature,
                    repetition_penalty=self.config.get("repetition_penalty", 1.45),
                    top_p=self.config.get("top_p", 0.93),
                    top_k=self.config.get("top_k", 35),
                    speed=self.config.get("speed", 1.0),
                )
            return str(output_file)
        except Exception as e:
            print(f"    ⚠️ Fehler bei TTS: {e}")
            return None

    def generate_chunk_with_qc(self, tts, chunk_text, scene_id, chunk_id, part_idx=None):
        temperature = self.config.get("temperature", 0.60)
        path = self.generate_chunk_audio(tts, chunk_text, scene_id, chunk_id, temperature, part_idx=part_idx)
        if path:
            self.remove_long_silences(path, max_silence_sec=self.config.get("max_silence_sec", 1.0))
            return True, {"cer": None, "attempts": 1}
        return False, {"cer": None, "attempts": 1}

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

    def generate_audiobook_from_scenes(self):
        from TTS.api import TTS
        import torch

        print("\n🎧 SZENEN-BASIERTER HÖRBUCH-GENERATOR V7 - PRODUCTION")
        print("=" * 60)
        
        low_mode = self.config.get("low_mode", 0)
        if low_mode > 0:
            print(f"\n⚙️ LOW MODE {low_mode} AKTIV")
            print(f"   Temperature: {self.config['temperature']}")
            print(f"   Top-P: {self.config['top_p']}")
            print(f"   Top-K: {self.config['top_k']}")
            print(f"   Repetition Penalty: {self.config['repetition_penalty']}")
            print(f"   Max Chunk Length: {self.config['max_chunk_length']}")
            print(f"   Speed: {self.config.get('speed', 1.0)}")

        print(f"\n🔥 Hardware-Info:")
        print(f"   CUDA verfügbar: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

        if torch.cuda.is_available():
            gpu_id = self.config.get("gpu_id", 0)
            torch.cuda.set_device(gpu_id)
            tts = tts.cuda()
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(gpu_id)
            vram = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"   ✅ XTTS auf GPU {gpu_id}: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            print("   ✅ XTTS auf CPU")

        # Latents einmal vorberechnen → kein Re-Encoding bei jedem Chunk
        print("\n🔧 Berechne Speaker-Latents (einmalig)...")
        try:
            speaker_wav = self.config["speaker_wav"]
            gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            )
            self.gpt_cond_latent = gpt_cond_latent
            self.speaker_embedding = speaker_embedding
            print("   ✅ Latents gecacht - kein Re-Encoding pro Chunk mehr!")
        except Exception as e:
            print(f"   ⚠️ Latent-Cache fehlgeschlagen, nutze Fallback: {e}")
            self.gpt_cond_latent = None
            self.speaker_embedding = None

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
            print(f"   📝 {len(chunks)} Chunks erstellt (max {self.config['max_chunk_length']} Zeichen)")

            for chunk_idx, chunk_text in enumerate(chunks, 1):
                total_chunks += 1
                base_file = self.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}.wav"

                if base_file.exists():
                    print(f"   [{chunk_idx:03d}] ⭐️ {base_file.name} existiert")
                    skipped_existing += 1
                    continue

                # === LOW MODE 2: chunk_split → Parts rendern → mergen ===
                if low_mode == 2:
                    chunk_split = self.config.get("chunk_split", 2)
                    sub_chunks = []
                    remaining = chunk_text
                    part_len = max(1, len(chunk_text) // chunk_split)

                    for i in range(chunk_split - 1):
                        split_pos = part_len
                        for j in range(split_pos, min(split_pos + 80, len(remaining))):
                            if remaining[j] in '.!?,;':
                                split_pos = j + 1
                                break
                        part = remaining[:split_pos].strip()
                        if part:
                            sub_chunks.append(part)
                        remaining = remaining[split_pos:].strip()

                    if remaining:
                        sub_chunks.append(remaining)
                    sub_chunks = [s for s in sub_chunks if s]

                    print(f"   [{chunk_idx:03d}] ✂️ LOW MODE 2: {len(sub_chunks)} Parts (je ~{part_len} Zeichen)")

                    sub_failed = False
                    for sub_i, sub_text in enumerate(sub_chunks, 1):
                        preview = sub_text[:50] + ("..." if len(sub_text) > 50 else "")
                        print(f"       [{chunk_idx:03d}.{sub_i}] 🎤 {preview}")

                        start = time.time()
                        success, _ = self.generate_chunk_with_qc(
                            tts, sub_text, scene_id, chunk_idx, part_idx=sub_i
                        )
                        duration = time.time() - start

                        if success:
                            print(f"           ✅ Fertig in {duration:.1f}s")
                        else:
                            print(f"           ⚠️ Fehlgeschlagen")
                            sub_failed = True

                    if not sub_failed:
                        self.merge_subchunks(scene_id, chunk_idx)
                        newly_generated += 1
                    else:
                        failed_chunks += 1

                    self.save_progress(scene_id, chunk_idx)
                
                else:
                    # === NORMAL MODE / LOW MODE 1: Direkt generieren ===
                    preview = chunk_text[:60] + ("..." if len(chunk_text) > 60 else "")
                    print(f"   [{chunk_idx:03d}] 🎤 {preview}")

                    start = time.time()
                    success, _ = self.generate_chunk_with_qc(tts, chunk_text, scene_id, chunk_idx)
                    duration = time.time() - start

                    if success:
                        try:
                            audio_dur = sf.info(str(base_file)).duration
                            rtf = duration / audio_dur if audio_dur > 0 else 0
                            print(f"           ✅ Fertig in {duration:.1f}s | Audio: {audio_dur:.1f}s | RTF: {rtf:.2f}")
                        except Exception:
                            print(f"           ✅ Fertig in {duration:.1f}s")
                        newly_generated += 1
                    else:
                        print(f"           ⚠️ Generierung fehlgeschlagen")

                        if base_file.exists():
                            base_file.unlink()

                        subchunks = self.split_problematic_chunk(
                            chunk_text,
                            self.config.get("retry_chunk_length", 180)
                        )

                        if len(subchunks) > 1:
                            print(f"           ✂️ Neu aufgeteilt in {len(subchunks)} Subchunks")
                            for sub_i, sub_text in enumerate(subchunks, 1):
                                sub_success, _ = self.generate_chunk_with_qc(
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
        
        return failed_chunks == 0


def main():
    ap = argparse.ArgumentParser(description="XTTS Hörbuch Generator V7")
    ap.add_argument("--path", required=True, help="Basis-Pfad für Input/Output")
    ap.add_argument("--low", type=int, choices=[0, 1, 2], default=0,
                    help="Qualitätsstufe: 0=Original, 1=Stabil, 2=Extra-Stabil (halbe Chunks)")
    args = ap.parse_args()

    base_path = args.path
    low_mode = args.low
    
    # === BASIS-CONFIG (Original - Mode 0) ===
    CONFIG = {
        # Custom Voice Model
        "model_path": "/workspace/storypainter/voices/tomhq",
        "config_path": "/workspace/storypainter/voices/tomhq/config.json",
        
        # Multi-Sample Reference (4 Samples für natürliche Stimme)
        "speaker_wav": [
            "/workspace/storypainter/voices/tomhq/neutral.wav",
            "/workspace/storypainter/voices/tomhq/question.wav",
            "/workspace/storypainter/voices/tomhq/excited.wav",
            "/workspace/storypainter/voices/tomhq/sad.wav"
        ],
        
        # Dateien
        "scenes_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "tts"),
    
        # TTS-Einstellungen (Original)
        "max_chunk_length": 240,
        "language": "de",
        "temperature": 0.60,
        "top_p": 0.93,
        "top_k": 35,
        "repetition_penalty": 1.45,
        "speed": 1.0,
        "max_silence_sec": 1.0,
        "retry_chunk_length": 180,
        
        # GPU
        "gpu_id": 0,
        
        # Low-Mode Flag
        "low_mode": low_mode
    }
    
    # === ÜBERSCHREIBE SETTINGS BASIEREND AUF --low ===
    if low_mode == 1:
        print("\n🔧 LOW MODE 1: Stabile Einstellungen aktiviert")
        CONFIG.update({
            "temperature": 0.50,
            "top_p": 0.88,
            "top_k": 30,
            "repetition_penalty": 1.55,
        })
    
    elif low_mode == 2:
        print("\n🔧 LOW MODE 2: Extra-Stabil (chunk_split=2) aktiviert")
        CONFIG.update({
            "temperature": 0.50,
            "top_p": 0.88,
            "top_k": 30,
            "repetition_penalty": 1.55,
            "chunk_split": 2,  # Jeden Chunk in 2 Parts → gleiche Dateinummerierung!
        })
    
    # Pfad-Validierung
    print("\n📂 Prüfe Pfade...")
    required_paths = {
        "model_path": CONFIG["model_path"],
        "config_path": CONFIG["config_path"],
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
    
    # Speaker WAV(s) validieren
    speaker_wavs = CONFIG["speaker_wav"]
    if isinstance(speaker_wavs, str):
        speaker_wavs = [speaker_wavs]
    
    print(f"\n🎤 Validiere Speaker-Samples ({len(speaker_wavs)} Dateien)...")
    missing_samples = []
    for wav in speaker_wavs:
        exists = os.path.exists(wav)
        status = "✅" if exists else "❌"
        wav_name = os.path.basename(wav)
        print(f"   {status} {wav_name}")
        if not exists:
            missing_samples.append(wav)
    
    if missing_samples:
        print(f"\n⚠️ Fehlende Speaker-Samples:")
        for wav in missing_samples:
            print(f"   - {wav}")
        print("\n💡 OPTIONEN:")
        print("   A) Erstelle die 4 Samples (neutral/question/excited/sad)")
        print("   B) Nutze vorübergehend nur 1 Sample:")
        print("      Ändere CONFIG['speaker_wav'] zu:")
        print('      "speaker_wav": "/workspace/storypainter/voices/tomhq/neutral.wav"')
        sys.exit(1)
    
    print("\n✅ Alle Pfade OK\n")
    
    generator = SceneBasedAudiobookGenerator(CONFIG)
    success = generator.generate_audiobook_from_scenes()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()