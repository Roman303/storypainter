#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_generator.py
──────────────────────────────────────────────────────────────────────────────
XTTS-v2 (z. B. 2.0.3) Sprachgenerator – NUR METHODE 1 (expliziter model_path + config_path)

Features
- Lädt XTTS-v2 direkt über Pfade (ohne HF-Modelnamen)
- Liest Szenen-JSON (mit scene_id, text)
- Teilt Szenentext in Chunks (konfigurierbar) und erzeugt pro Chunk eine WAV
- Speichert WAVs exakt mit der vom Modell gelieferten Sample-Rate (kein Resampling)
- Prüft/warnte pro Datei, falls Rate ≠ erwarteter 24000 Hz
- Fortschritt (progress.json) – Wiederaufnahme möglich
- Temperatur & Repetition-Penalty (weitergereicht, falls vom Backend unterstützt)

Benötigt:
  pip install TTS soundfile tqdm

Ablauf (danach wie gehabt):
  python combine_audio.py
  python combine_audio_scene.py
  ...
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from TTS.tts.configs.xtts_config import XttsConfig

import soundfile as sf
from tqdm import tqdm

# Coqui TTS – XTTS v2
try:
    from TTS.tts.models.xtts import Xtts
except Exception as e:
    raise RuntimeError(
        "Coqui TTS ist nicht installiert oder inkompatibel. "
        "Installiere z.B.: pip install TTS"
    ) from e


# ────────────────────────────────────────────────────────────────────────────
# Hilfen
# ────────────────────────────────────────────────────────────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def split_text_into_chunks(text: str, max_len: int) -> List[str]:
    """
    Teilt Text in sinnvolle Chunks bis max_len (Bevorzugung: Absatz, dann Satzende).
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    pos = 0
    n = len(text)

    # einfache Satz-/Absatz-Suche
    sentence_end = re.compile(r"([.!?])(\"|\”|\)|\s|$)")
    while pos < n:
        target = min(pos + max_len, n)

        # 1) Versuche, bis zum nächsten Absatzende (zwei \n)
        para_idx = text.rfind("\n\n", pos, target)
        if para_idx != -1 and para_idx > pos + max_len * 0.6:
            end = para_idx + 2
        else:
            # 2) Satzende suchen im Fenster [pos+0.6L, target]
            search_start = int(pos + max_len * 0.6)
            m = None
            for mm in sentence_end.finditer(text, search_start, target):
                m = mm
            if m:
                end = m.end()
            else:
                # 3) Hart schneiden
                end = target

        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos = end

    return chunks


# ────────────────────────────────────────────────────────────────────────────
# Datenmodell
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    model_path: Path
    config_path: Path
    speaker_wav: Path | None
    scenes_file: Path
    output_dir: Path
    progress_file: Path
    language: str = "de"
    max_chunk_length: int = 220
    temperature: float = 0.65
    repetition_penalty: float = 12.0
    expected_rate: int = 24000  # 24 kHz


# ────────────────────────────────────────────────────────────────────────────
# Hauptklasse
# ────────────────────────────────────────────────────────────────────────────

class VoiceGeneratorXTTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts = self._load_model()
        self.output_rate = self._detect_output_rate()
        print(f"🎚️  Modell-Ausgabesamplerate laut Config: {self.output_rate} Hz")

    # ─────────────────────────────
 
    
    def _load_model(self):
        print("🎙️  Lade XTTS-v2 Modell (METHODE 1: model_path + config_path)…")
    
        # ---- Pfadprüfung ----
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(f"❌ model_path nicht gefunden: {self.cfg.model_path}")
        if not self.cfg.config_path.exists():
            raise FileNotFoundError(f"❌ config_path nicht gefunden: {self.cfg.config_path}")
    
        # ---- Config laden ----
        cfg = XttsConfig()
        cfg.load_json(str(self.cfg.config_path))
    
        # ---- Modell initialisieren ----
        tts = Xtts.init_from_config(cfg)
    
        # ---- Checkpoint laden ----
        checkpoint_dir = str(Path(self.cfg.model_path).parent)
    
        print("🎤 Lade Sprecher-Embedding …")
        self.speaker_latents = None
        if self.cfg.speaker_wav:
            lat = self.tts.get_conditioning_latents(audio_path=str(self.cfg.speaker_wav))
            # 🔧 Kompatibilität: Coqui-Versionen geben Dict ODER Tuple zurück
            if isinstance(lat, dict):
                self.speaker_latents = {
                    "gpt_cond_latent": lat.get("gpt_cond_latent"),
                    "speaker_embedding": lat.get("speaker_embedding"),
                }
            elif isinstance(lat, (list, tuple)) and len(lat) >= 2:
                self.speaker_latents = {
                    "gpt_cond_latent": lat[0],
                    "speaker_embedding": lat[1],
                }
            else:
                raise RuntimeError(f"Unbekanntes Format von speaker_latents: {type(lat)}")
            print("✅ Speaker-Embedding geladen")
        else:
            print("⚠️ Keine Referenzstimme angegeben – generiere Default-Latents")
            self.speaker_latents = {
                "gpt_cond_latent": None,
                "speaker_embedding": None,
            }

    
        return tts

    def _detect_output_rate(self) -> int:
        # XTTS hat audio_config.output_sample_rate; fallback auf erwarteten Wert
        try:
            return int(getattr(self.tts.audio_config, "output_sample_rate", self.cfg.expected_rate))
        except Exception:
            return self.cfg.expected_rate

    # ─────────────────────────────
    # Fortschritt
    # ─────────────────────────────

    def load_progress(self) -> Dict:
        if not self.cfg.progress_file.exists():
            return {"completed_scenes": [], "completed_chunks": []}
        try:
            with open(self.cfg.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"completed_scenes": [], "completed_chunks": []}

    def save_progress(self, scene_id: int, chunk_idx: int):
        prog = self.load_progress()
        key = f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}"
        if key not in prog.get("completed_chunks", []):
            prog.setdefault("completed_chunks", []).append(key)
        with open(self.cfg.progress_file, "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2, ensure_ascii=False)

    def mark_scene_complete(self, scene_id: int):
        prog = self.load_progress()
        if scene_id not in prog.get("completed_scenes", []):
            prog.setdefault("completed_scenes", []).append(scene_id)
        with open(self.cfg.progress_file, "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────
    # Synthese
    # ─────────────────────────────

    def _tts_call(self, text: str) -> tuple[list[float], int]:
        """
        Führt die XTTS-v2-Inference aus (kompatibel mit Dict- und Tuple-Rückgaben).
        """
        if self.speaker_latents is None:
            raise RuntimeError(
                "Speaker-Latents fehlen! Bitte --speaker-wav korrekt angeben."
            )
    
        try:
            result = self.tts.inference(
                text=text,
                gpt_cond_latent=self.speaker_latents["gpt_cond_latent"],
                speaker_embedding=self.speaker_latents["speaker_embedding"],
                language=self.cfg.language,
            )
    
            # 🔍 Rückgabetyp prüfen
            wav = None
            rate = getattr(self.tts.audio_config, "output_sample_rate", self.cfg.expected_rate)
    
            if isinstance(result, dict):
                # ältere Coqui-Version (<=0.22)
                wav = result.get("wav", None)
            elif isinstance(result, (list, tuple)):
                # neuere Coqui-Version (>=0.23) -> (wav,) oder (wav, rate)
                if len(result) >= 1:
                    wav = result[0]
                if len(result) >= 2 and isinstance(result[1], (int, float)):
                    rate = int(result[1])
            else:
                raise TypeError(f"Unbekannter Rückgabetyp: {type(result)}")
    
            if wav is None:
                raise ValueError("Keine Waveform in Rückgabe gefunden")
    
            return wav, rate
    
        except Exception as e:
            print(f"❌ Fehler beim XTTS-Inference: {e}")
            raise



    def generate_chunk_audio(self, scene_id: int, chunk_idx: int, text: str) -> Path | None:
        """
        Erzeugt eine WAV für einen einzelnen Chunk – speichert mit *exakter* Modellrate.
        """
        out_file = self.cfg.output_dir / f"scene_{scene_id:04d}_chunk_{chunk_idx:03d}.wav"
        try:
            wav, rate = self._tts_call(text)
            # Erwartung prüfen
            if rate != self.cfg.expected_rate:
                print(f"⚠️  Modell meldet Samplerate {rate} Hz (erwartet {self.cfg.expected_rate} Hz) — speichere mit {rate} Hz.")

            # WAV schreiben (kein Resample!)
            sf.write(out_file, wav, rate, subtype="PCM_16")

            # Verifizieren
            sr_check = sf.info(out_file).samplerate
            if sr_check != self.cfg.expected_rate:
                print(f"⚠️  Datei {out_file.name}: {sr_check} Hz (≠ {self.cfg.expected_rate} Hz)")
            else:
                print(f"   ✓ Gespeichert @ {sr_check} Hz")

            return out_file
        except Exception as e:
            print(f"   ❌ Fehler bei Chunk {chunk_idx:03d}: {e}")
            return None

    # ─────────────────────────────
    # Orchestrierung
    # ─────────────────────────────

    def run(self) -> bool:
        # Szenen lesen
        if not self.cfg.scenes_file.exists():
            raise FileNotFoundError(f"❌ Szenen-JSON nicht gefunden: {self.cfg.scenes_file}")

        with open(self.cfg.scenes_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenes = data.get("scenes", [])
        print(f"📖 {len(scenes)} Szenen aus {self.cfg.scenes_file.name} geladen")
        if not scenes:
            print("❌ Keine Szenen gefunden.")
            return False

        progress = self.load_progress()
        completed_chunks = set(progress.get("completed_chunks", []))
        completed_scenes = set(progress.get("completed_scenes", []))

        total_chunks = 0
        success_chunks = 0
        failed_chunks = 0

        print("\n🎙️  Starte TTS-Generierung …")
        print(f"   Sprache: {self.cfg.language} | max_chunk_len: {self.cfg.max_chunk_length}")
        print(f"   Temperatur: {self.cfg.temperature} | Repetition Penalty: {self.cfg.repetition_penalty}")
        print(f"   Erwartete Rate: {self.cfg.expected_rate} Hz\n")

        for sidx, scene in enumerate(scenes, 1):
            scene_id = int(scene.get("scene_id", sidx))
            text = (scene.get("text", "") or "").strip()
            if not text:
                print(f"⚠️  Szene {scene_id:04d} leer – übersprungen")
                continue

            print("=" * 60)
            print(f"🎬 Szene {scene_id:04d} – Textlänge: {len(text)}")
            chunks = split_text_into_chunks(text, self.cfg.max_chunk_length)
            print(f"   🧩 {len(chunks)} Chunks")

            scene_ok = 0
            for cidx, chunk_text in enumerate(chunks, 1):
                key = f"scene_{scene_id:04d}_chunk_{cidx:03d}"
                total_chunks += 1

                if key in completed_chunks:
                    print(f"   [{cidx:03d}] ⏭️  Bereits vorhanden")
                    scene_ok += 1
                    continue

                # Vorschau-Log
                preview = chunk_text[:70].replace("\n", " ")
                if len(chunk_text) > 70:
                    preview += "…"
                print(f"   [{cidx:03d}] 🎤 {preview}")

                out = self.generate_chunk_audio(scene_id, cidx, chunk_text)
                if out:
                    self.save_progress(scene_id, cidx)
                    scene_ok += 1
                    success_chunks += 1
                else:
                    failed_chunks += 1

            if scene_ok == len(chunks) and len(chunks) > 0:
                self.mark_scene_complete(scene_id)
                print(f"   ✅ Szene {scene_id:04d} komplett ({scene_ok}/{len(chunks)})")
            else:
                print(f"   ⚠️ Szene {scene_id:04d} unvollständig ({scene_ok}/{len(chunks)})")

        print("\n" + "=" * 60)
        print("✅ Fertig")
        print(f"   Chunks gesamt:    {total_chunks}")
        print(f"   Erfolgreich:      {success_chunks}")
        print(f"   Fehlgeschlagen:   {failed_chunks}")
        print(f"📂 Ausgabe:          {self.cfg.output_dir}")
        print(f"📄 Fortschritt:      {self.cfg.progress_file}")
        print("=" * 60 + "\n")

        return failed_chunks == 0

def main():
    # --- Argument einlesen ---
    ap = argparse.ArgumentParser(description="XTTS-v2 Voice Generator (Pfad-basiert, CONFIG-Stil)")
    ap.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")
    args = ap.parse_args()

    base_path = args.path

    # --- CONFIG ---
    CONFIG = {
        # Stimmen / Modell (bleibt unverändert)
        "model_path": "/workspace/voices/franziska300/model.pth",
        "config_path": "/workspace/voices/franziska300/config.json",
        "speaker_wav": "/workspace/voices/franziska300/dataset/wavs/die-faelle-des-prof-machata_00000141.wav",

        # Eingabe / Ausgabe (werden dynamisch kombiniert)
        "scenes_file": os.path.join(base_path, "book_scenes.json"),
        "output_dir": os.path.join(base_path, "tts"),
        "progress_file": os.path.join(base_path, "tts", "progress.json"),

        # TTS-Einstellungen
        "max_chunk_length": 200,
        "language": "de",
        "temperature": 0.65,
        "repetition_penalty": 2.0,
        "expected_rate": 24000,
    }

    # --- Pfad-Validierung ---
    if not os.path.isfile(CONFIG["scenes_file"]):
        raise FileNotFoundError(f"Scenes file not found: {CONFIG['scenes_file']}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # --- Debug-Ausgabe zum Testen ---
    print("✅ Using configuration:")
    for k, v in CONFIG.items():
        print(f"{k:15}: {v}")

    gen = VoiceGeneratorXTTS(cfg)
    ok = gen.run()
    if not ok:
        # Non-zero exit wäre auch möglich; wir lassen informative Ausgabe stehen.
        pass


if __name__ == "__main__":
    main()
