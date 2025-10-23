#!/usr/bin/env python3
"""
Kombiniert Chunk-WAVs zu Szenen-WAVs
- Liest scene_XXXX_chunk_YYY.wav
- Kombiniert zu scene_XXXX.wav
- Fügt natürliche Pausen zwischen Chunks ein
"""

import os
import re
import argparse
from pathlib import Path
from pydub import AudioSegment
from collections import defaultdict


class SceneWavCombiner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_chunk_files(self):
        """Findet alle Chunk-Dateien und gruppiert nach Szene"""
        pattern = re.compile(r'scene_(\d{4})_chunk_(\d{3})\.wav')
        scenes = defaultdict(list)
        
        for wav_file in sorted(self.input_dir.glob("scene_*.wav")):
            match = pattern.match(wav_file.name)
            if match:
                scene_id = int(match.group(1))
                chunk_id = int(match.group(2))
                scenes[scene_id].append((chunk_id, wav_file))
        
        # Sortiere Chunks pro Szene
        for scene_id in scenes:
            scenes[scene_id].sort(key=lambda x: x[0])
        
        return scenes
    
    def combine_scene(self, scene_id, chunks, pause_ms=300):
        """
        Kombiniert Chunks einer Szene
        
        Args:
            scene_id: Szenen-Nummer
            chunks: Liste von (chunk_id, filepath) Tupeln
            pause_ms: Pause zwischen Chunks in Millisekunden
        """
        print(f"🔧 Szene {scene_id:04d}: Kombiniere {len(chunks)} Chunks...")
        
        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=pause_ms)
        
        for i, (chunk_id, chunk_file) in enumerate(chunks, 1):
            try:
                # Lade Chunk
                audio = AudioSegment.from_wav(chunk_file)
                
                # Füge Audio hinzu
                combined += audio
                
                # Füge Pause hinzu (außer beim letzten Chunk)
                if i < len(chunks):
                    combined += pause
                
                print(f"   ✓ Chunk {chunk_id:03d} ({len(audio)/1000:.1f}s)")
                
            except Exception as e:
                print(f"   ❌ Fehler bei Chunk {chunk_id:03d}: {e}")
                continue
        
        # Speichere kombinierte Szene
        output_file = self.output_dir / f"scene_{scene_id:04d}.wav"
        combined.export(output_file, format="wav")
        
        duration = len(combined) / 1000
        print(f"   ✅ Szene {scene_id:04d}.wav gespeichert ({duration:.1f}s)\n")
        
        return output_file, duration
    
    def combine_all_scenes(self, pause_ms=300):
        """Kombiniert alle Szenen"""
        print("="*60)
        print("🎵 SZENEN-WAV-COMBINER")
        print("="*60)
        print(f"\n📁 Input:  {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"⏱️ Pause zwischen Chunks: {pause_ms}ms\n")
        
        # Finde Chunk-Dateien
        scenes = self.find_chunk_files()
        
        if not scenes:
            print("❌ Keine Chunk-Dateien gefunden!")
            print(f"   Erwartet: scene_XXXX_chunk_YYY.wav in {self.input_dir}")
            return []
        
        print(f"✅ {len(scenes)} Szenen mit Chunks gefunden\n")
        
        # Kombiniere jede Szene
        combined_files = []
        total_duration = 0
        
        for scene_id in sorted(scenes.keys()):
            chunks = scenes[scene_id]
            output_file, duration = self.combine_scene(scene_id, chunks, pause_ms)
            combined_files.append(output_file)
            total_duration += duration
        
        # Zusammenfassung
        print("="*60)
        print(f"✅ FERTIG!")
        print(f"📊 {len(combined_files)} Szenen kombiniert")
        print(f"⏱️ Gesamtdauer: {total_duration/60:.1f} Minuten")
        print(f"📁 Ausgabe: {self.output_dir}")
        
        return combined_files


def main():
    parser = argparse.ArgumentParser(description="Kombiniert Chunk-WAVs zu Szenen-WAVs")

    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")
    parser.add_argument("--input", default=None, help="Input-Verzeichnis mit Chunks")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis für Szenen")
    parser.add_argument("--pause", type=int, default=300, help="Pause zwischen Chunks in ms (Standard: 300)")

    args = parser.parse_args()
    base_path = args.path

    CONFIG = {
        "input": args.input or os.path.join(base_path, "tts"),
        "output": args.output or os.path.join(base_path, "scenes"),
        "pause": args.pause,
    }
    
    combiner = SceneWavCombiner(CONFIG["input"], CONFIG["output"])
    combiner.combine_all_scenes(pause_ms=CONFIG["pause"])


if __name__ == "__main__":
    main()
