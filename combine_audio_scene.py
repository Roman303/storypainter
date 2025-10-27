#!/usr/bin/env python3
"""
Audio-Renderer (Fixed v2)
- Fügt definierte Pausen, Intro- und Outro-Stille korrekt ein
- Szene 0 = Intro-Stille, letzte Szene + 1 = Outro-Stille
- Alle Szenen, Pausen und Zeiten werden im JSON exakt protokolliert
"""

import os
import argparse
from pathlib import Path
import json
from pydub import AudioSegment


class AudiobookRendererFixedV2:
    def __init__(self, scenes_dir, output_dir, book_info=None):
        self.scenes_dir = Path(scenes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.book_info = book_info or {}

    def find_scene_audio_files(self):
        audio_files = sorted(self.scenes_dir.glob("scene_*.wav"))
        scenes = []
        for audio_file in audio_files:
            try:
                scene_id = int(audio_file.stem.split('_')[1])
            except Exception:
                continue
            scenes.append({"scene_id": scene_id, "audio_file": audio_file})
        scenes.sort(key=lambda s: s["scene_id"])
        return scenes

    def create_complete_audiobook(self, pause_between_scenes: float = 2.0, intro_length: float = 10.0, outro_length: float = 10.0, mp3_bitrate: str = "192k"):
        print("=" * 60)
        print("🎧 AUDIOBOOK-RENDERER (FIXED v2)")
        print("=" * 60)

        scenes = self.find_scene_audio_files()
        if not scenes:
            print("❌ Keine Szenen gefunden!")
            return None

        title = self.book_info.get("title", "Complete Audiobook")
        author = self.book_info.get("author", "TTS Generator")

        print(f"📘 Titel: {title}")
        print(f"✍️  Autor: {author}")
        print(f"✅ {len(scenes)} Szenen gefunden")
        print(f"🔈 Intro-Länge: {intro_length:.3f}s")
        print(f"🔈 Outro-Länge: {outro_length:.3f}s")
        print(f"⏸️  Pause zwischen Szenen: {pause_between_scenes:.3f}s\n")

        pause_ms = int(round(pause_between_scenes * 1000))
        intro_ms = int(round(intro_length * 1000))
        outro_ms = int(round(outro_length * 1000))

        combined_audio = AudioSegment.silent(duration=0)
        metadata = {
            "title": title,
            "author": author,
            "total_scenes": len(scenes) + 2,  # inkl. Intro & Outro
            "pause_duration": pause_between_scenes,
            "intro_length": intro_length,
            "outro_length": outro_length,
            "timestamps_base": "absolute",
            "scenes": [],
        }

        current_ms = 0

        # --- Intro ---
        print(f"[000] Intro-Stille: {intro_length:.3f}s  @ 0.000s → {intro_length:.3f}s")
        combined_audio += AudioSegment.silent(duration=intro_ms)
        metadata["scenes"].append({
            "scene_id": 0,
            "type": "intro",
            "start_time": 0.0,
            "end_time": round(intro_length, 3),
            "duration": round(intro_length, 3),
        })
        current_ms += intro_ms

        # Pause nach Intro
        if pause_ms > 0:
            combined_audio += AudioSegment.silent(duration=pause_ms)
            current_ms += pause_ms

        # --- Hauptszenen ---
        for i, scene in enumerate(scenes):
            scene_id = scene["scene_id"]
            audio_file = scene["audio_file"]

            try:
                audio = AudioSegment.from_wav(audio_file)
            except Exception as e:
                print(f"⚠️  Szene {scene_id} konnte nicht geladen werden: {e}")
                continue

            duration_ms = len(audio)
            duration_s = round(duration_ms / 1000.0, 3)

            start_s = round(current_ms / 1000.0, 3)
            end_s = round((current_ms + duration_ms) / 1000.0, 3)

            print(f"[{i+1:03d}] Szene {scene_id:04d}: {duration_s:.3f}s  @ {start_s:.3f}s → {end_s:.3f}s")

            combined_audio += audio

            metadata["scenes"].append({
                "scene_id": scene_id,
                "type": "scene",
                "start_time": start_s,
                "end_time": end_s,
                "duration": duration_s,
            })

            current_ms += duration_ms

            # Pause nach Szene (außer letzter, dann vor Outro extra)
            if pause_ms > 0:
                combined_audio += AudioSegment.silent(duration=pause_ms)
                current_ms += pause_ms

        # --- Outro ---
        start_s = round(current_ms / 1000.0, 3)
        end_s = round((current_ms + outro_ms) / 1000.0, 3)
        print(f"[{len(scenes)+1:03d}] Outro-Stille: {outro_length:.3f}s  @ {start_s:.3f}s → {end_s:.3f}s")

        combined_audio += AudioSegment.silent(duration=outro_ms)

        metadata["scenes"].append({
            "scene_id": len(scenes) + 1,
            "type": "outro",
            "start_time": start_s,
            "end_time": end_s,
            "duration": round(outro_length, 3),
        })

        total_ms = len(combined_audio)
        total_s = round(total_ms / 1000.0, 3)

        # --- Dateien schreiben ---
        wav_output = self.output_dir / "complete_audiobook.wav"
        mp3_output = self.output_dir / "complete_audiobook.mp3"

        print(f"\n💾 Speichere WAV...")
        combined_audio.export(wav_output, format="wav")

        print(f"💾 Speichere MP3...")
        combined_audio.export(mp3_output, format="mp3", bitrate=mp3_bitrate,
                              tags={"title": title, "artist": author, "album": title})

        metadata["total_duration"] = total_s
        metadata["total_duration_formatted"] = f"{int(total_s // 60)}:{int(total_s % 60):02d}"

        metadata_file = self.output_dir / "audiobook_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"📄 Metadaten: {metadata_file}")
        print(f"✅ Gesamtzeit: {total_s/60:.1f} Minuten")
        print(f"🎵 WAV: {wav_output}")
        print(f"🎵 MP3: {mp3_output}")

        return {
            "wav": wav_output,
            "mp3": mp3_output,
            "metadata": metadata_file,
            "duration": total_s,
        }


def load_book_info(base_path: str):
    info_path = Path(base_path) / "book_scenes.json"
    if not info_path.exists():
        return {}
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("book_info", {})
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Audio-Renderer (Fixed v2)")
    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")
    parser.add_argument("--scenes", default=None, help="Verzeichnis mit Szenen-Audio (scene_*.wav)")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis")
    parser.add_argument("--pause", type=float, default=4.0, help="Pause zwischen Szenen (Sekunden)")
    parser.add_argument("--intro-length", type=float, default=10.0, help="Intro-Stille (Sekunden)")
    parser.add_argument("--outro-length", type=float, default=10.0, help="Outro-Stille (Sekunden)")
    parser.add_argument("--mp3-bitrate", default="192k", help="MP3 Bitrate")

    args = parser.parse_args()

    base_path = args.path
    book_info = load_book_info(base_path)

    CONFIG = {
        "scenes": args.scenes or os.path.join(base_path, "scenes"),
        "output": args.output or os.path.join(base_path, "audiobook"),
        "pause": args.pause,
        "intro": args.intro_length,
        "outro": args.outro_length,
        "mp3_bitrate": args.mp3_bitrate,
    }

    renderer = AudiobookRendererFixedV2(
        scenes_dir=CONFIG["scenes"],
        output_dir=CONFIG["output"],
        book_info=book_info,
    )

    renderer.create_complete_audiobook(
        pause_between_scenes=CONFIG["pause"],
        intro_length=CONFIG["intro"],
        outro_length=CONFIG["outro"],
        mp3_bitrate=CONFIG["mp3_bitrate"],
    )


if __name__ == "__main__":
    main()
