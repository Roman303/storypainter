#!/usr/bin/env python3
"""
Audio-Renderer - Kombiniert Szenen-Audio zu komplettem Hörbuch
- Fügt Pausen zwischen Szenen ein
- Fügt optionale Kopf-/Schluss-Stille (Intro/Outro-Padding) hinzu
- Erstellt WAV und MP3
- Schreibt Metadaten mit exakten Timestamps inkl. Head-/Tail-Silence-Offset
"""

import os
import argparse
from pathlib import Path
import json
from pydub import AudioSegment


class AudiobookRenderer:
    def __init__(self, scenes_dir, output_dir, book_info=None):
        self.scenes_dir = Path(scenes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.book_info = book_info or {}

    def find_scene_audio_files(self):
        """Findet alle Szenen-Audio-Dateien und sortiert numerisch nach scene_id"""
        audio_files = sorted(self.scenes_dir.glob("scene_*.wav"))
        scenes = []
        for audio_file in audio_files:
            try:
                # erlaubt z.B. scene_0001.wav oder scene_1.wav
                scene_id_str = audio_file.stem.split('_')[1]
                scene_id = int(scene_id_str)
            except Exception:
                continue
            scenes.append({"scene_id": scene_id, "audio_file": audio_file})

        scenes.sort(key=lambda s: s["scene_id"])
        return scenes

    def create_complete_audiobook(
        self,
        pause_between_scenes: float = 2.0,
        head_silence: float = 10.0,
        tail_silence: float = 10.0,
        mp3_bitrate: str = "192k",
    ):
        """
        Erstellt komplettes Hörbuch aus Szenen
        """
        print("=" * 60)
        print("🎧 AUDIOBOOK-RENDERER")
        print("=" * 60)

        scenes = self.find_scene_audio_files()
        if not scenes:
            print("❌ Keine Szenen-Audio-Dateien gefunden!")
            return None

        title = self.book_info.get("title", "Complete Audiobook")
        author = self.book_info.get("author", "TTS Generator")

        print(f"\n📘 Titel: {title}")
        print(f"✍️  Autor: {author}")
        print(f"📁 Szenen-Audio: {self.scenes_dir}")
        print(f"📁 Output:       {self.output_dir}")
        print(f"✅ {len(scenes)} Szenen gefunden")
        print(f"⏸️  Pause zw. Szenen: {pause_between_scenes:.3f}s")
        print(f"🔕  Head-Silence:     {head_silence:.3f}s")
        print(f"🔕  Tail-Silence:     {tail_silence:.3f}s\n")

        # Zeiten in Millisekunden
        head_ms = int(round(head_silence * 1000))
        tail_ms = int(round(tail_silence * 1000))
        pause_ms = int(round(pause_between_scenes * 1000))

        combined_audio = AudioSegment.silent(duration=head_ms)

        metadata = {
            "title": title,
            "author": author,
            "total_scenes": len(scenes),
            "pause_duration": pause_between_scenes,
            "head_silence": head_silence,
            "tail_silence": tail_silence,
            "timestamps_base": "absolute_with_head_silence",
            "scenes": [],
        }

        current_ms = head_ms  # Start nach Head-Silence

        # --- Szenen zusammenfügen ---
        for i, scene in enumerate(scenes, 1):
            scene_id = scene["scene_id"]
            audio_file = scene["audio_file"]

            try:
                audio = AudioSegment.from_wav(audio_file)
            except Exception as e:
                print(f"   ⚠️  Szene {scene_id:04d} konnte nicht geladen werden: {e}")
                continue

            duration_ms = len(audio)
            duration_s = round(duration_ms / 1000.0, 3)

            # Start/Endzeiten exakt im finalen Audiobook
            start_s = round(current_ms / 1000.0, 3)
            end_s = round((current_ms + duration_ms) / 1000.0, 3)

            # Ausgabe mit Pause
            if i < len(scenes):
                end_with_pause_s = round((end_s + pause_between_scenes), 3)
            else:
                end_with_pause_s = end_s

            print(f"   [{i:03d}] Szene {scene_id:04d}: {duration_s:.3f}s @ {start_s:.3f}s → {end_s:.3f}s (inkl. Pause bis {end_with_pause_s:.3f}s)")

            combined_audio += audio

            metadata["scenes"].append({
                "scene_id": scene_id,
                "start_time": start_s,
                "end_time": end_s,
                "duration": duration_s,
            })

            # Zeit für nächste Szene fortschreiben
            current_ms += duration_ms
            if i < len(scenes) and pause_ms > 0:
                combined_audio += AudioSegment.silent(duration=pause_ms)
                current_ms += pause_ms

        # Tail-Silence ans Ende
        if tail_ms > 0:
            combined_audio += AudioSegment.silent(duration=tail_ms)

        total_ms = len(combined_audio)
        total_duration = round(total_ms / 1000.0, 3)

        # --- Dateien schreiben ---
        wav_output = self.output_dir / "complete_audiobook.wav"
        mp3_output = self.output_dir / "complete_audiobook.mp3"

        print(f"\n💾 Speichere WAV...")
        combined_audio.export(wav_output, format="wav")
        wav_size_mb = wav_output.stat().st_size / (1024 ** 2)
        print(f"   ✅ WAV: {wav_size_mb:.1f} MB ({total_duration/60:.1f} Min)")

        print(f"💾 Speichere MP3...")
        combined_audio.export(
            mp3_output,
            format="mp3",
            bitrate=mp3_bitrate,
            tags={"title": title, "artist": author, "album": title},
        )
        mp3_size_mb = mp3_output.stat().st_size / (1024 ** 2)
        print(f"   ✅ MP3: {mp3_size_mb:.1f} MB ({total_duration/60:.1f} Min)")

        # --- Metadaten schreiben ---
        metadata["total_duration"] = total_duration
        metadata["total_duration_formatted"] = f"{int(total_duration // 60)}:{int(total_duration % 60):02d}"

        metadata_file = self.output_dir / "audiobook_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadaten: {metadata_file}")

        print(f"\n{'=' * 60}")
        print("✅ HÖRBUCH FERTIG!")
        print(f"📊 Szenen: {len(scenes)}")
        print(f"⏱️ Dauer:  {total_duration/60:.1f} Minuten")
        print(f"🎵 WAV:    {wav_output}")
        print(f"🎵 MP3:    {mp3_output}")
        print(f"{'=' * 60}")

        return {
            "wav": wav_output,
            "mp3": mp3_output,
            "metadata": metadata_file,
            "duration": total_duration,
        }


def load_book_info(base_path: str):
    """Lädt Titel und Autor aus book_scenes.json, falls vorhanden"""
    info_path = Path(base_path) / "book_scenes.json"
    if not info_path.exists():
        return {}
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("book_info", {})
    except Exception as e:
        print(f"⚠️  Konnte book_scenes.json nicht lesen: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Audio-Renderer für Hörbücher")
    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")
    parser.add_argument("--scenes", default=None, help="Verzeichnis mit Szenen-Audio (scene_*.wav)")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis für das Hörbuch")
    parser.add_argument("--pause", type=float, default=2.6, help="Pause zwischen Szenen in Sekunden (Standard: 2.0)")
    parser.add_argument("--head-silence", type=float, default=10.0, help="Stille am Anfang in Sekunden (Standard: 10.0)")
    parser.add_argument("--tail-silence", type=float, default=10.0, help="Stille am Ende in Sekunden (Standard: 10.0)")
    parser.add_argument("--mp3-bitrate", default="192k", help="MP3 Bitrate")

    args = parser.parse_args()
    base_path = args.path

    book_info = load_book_info(base_path)

    CONFIG = {
        "scenes": args.scenes or os.path.join(base_path, "scenes"),
        "output": args.output or os.path.join(base_path, "audiobook"),
        "pause": args.pause,
        "head_silence": args.head_silence,
        "tail_silence": args.tail_silence,
        "mp3_bitrate": args.mp3_bitrate,
    }

    renderer = AudiobookRenderer(
        scenes_dir=CONFIG["scenes"],
        output_dir=CONFIG["output"],
        book_info=book_info,
    )

    result = renderer.create_complete_audiobook(
        pause_between_scenes=CONFIG["pause"],
        head_silence=CONFIG["head_silence"],
        tail_silence=CONFIG["tail_silence"],
        mp3_bitrate=CONFIG["mp3_bitrate"],
    )

    if result:
        print("\n🎉 Audio-Rendering abgeschlossen!")
    else:
        print("\n❌ Audio-Rendering fehlgeschlagen!")


if __name__ == "__main__":
    main()
