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
    def __init__(self, scenes_dir, output_dir):
        self.scenes_dir = Path(scenes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_scene_audio_files(self):
        """Findet alle Szenen-Audio-Dateien"""
        audio_files = sorted(self.scenes_dir.glob("scene_*.wav"))
        scenes = []
        for audio_file in audio_files:
            try:
                scene_id = int(audio_file.stem.split('_')[1])
            except Exception:
                continue
            scenes.append({"scene_id": scene_id, "audio_file": audio_file})
        return scenes

    def create_complete_audiobook(
        self,
        pause_between_scenes: float = 2.0,
        head_silence: float = 10.0,
        tail_silence: float = 10.0,
        mp3_bitrate: str = "192k",
        title: str = "Complete Audiobook",
        album: str = "Generated Audiobook",
        artist: str = "TTS Generator",
    ):
        """
        Erstellt komplettes Hörbuch aus Szenen

        Args:
            pause_between_scenes: Sekunden Pause zwischen Szenen
            head_silence: Sekunden Stille am Anfang (für Intro-Video)
            tail_silence: Sekunden Stille am Ende (für Outro-Video)
        """
        print("=" * 60)
        print("🎧 AUDIOBOOK-RENDERER")
        print("=" * 60)
        print(f"\n📁 Szenen-Audio: {self.scenes_dir}")
        print(f"📁 Output:       {self.output_dir}")
        print(f"⏸️  Pause zw. Szenen: {pause_between_scenes:.3f}s")
        print(f"🔕  Head-Silence:     {head_silence:.3f}s")
        print(f"🔕  Tail-Silence:     {tail_silence:.3f}s\n")

        scenes = self.find_scene_audio_files()
        if not scenes:
            print("❌ Keine Szenen-Audio-Dateien gefunden!")
            return None

        print(f"✅ {len(scenes)} Szenen gefunden\n")

        combined_audio = AudioSegment.silent(duration=int(head_silence * 1000))
        pause = AudioSegment.silent(duration=int(pause_between_scenes * 1000))

        metadata = {
            "total_scenes": len(scenes),
            "pause_duration": pause_between_scenes,
            "head_silence": head_silence,
            "tail_silence": tail_silence,
            "scenes": [],
        }

        current_time = head_silence  # Sekunden – Start nach Head-Silence

        # Szenen zusammenfügen
        for i, scene in enumerate(scenes, 1):
            scene_id = scene["scene_id"]
            audio_file = scene["audio_file"]
            try:
                audio = AudioSegment.from_wav(audio_file)
            except Exception as e:
                print(f"   ⚠️ Szene {scene_id:04d} konnte nicht geladen werden: {e}")
                continue

            duration = len(audio) / 1000.0
            print(f"   [{i:03d}] Szene {scene_id:04d}: {duration:.3f}s @ {current_time:.3f}s")

            # zu kombiniertes Audio
            combined_audio += audio

            # Metadaten sichern (Start/Ende inkl. Head-Silence-Offset)
            metadata["scenes"].append({
                "scene_id": scene_id,
                "start_time": current_time,
                "end_time": current_time + duration,
                "duration": duration,
            })
            current_time += duration

            # Pause hinterher (außer letzte Szene)
            if i < len(scenes):
                combined_audio += pause
                current_time += pause_between_scenes

        # Tail-Silence ans Ende
        combined_audio += AudioSegment.silent(duration=int(tail_silence * 1000))
        total_duration = len(combined_audio) / 1000.0

        # Dateien schreiben
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
            tags={"title": title, "album": album, "artist": artist},
        )
        mp3_size_mb = mp3_output.stat().st_size / (1024 ** 2)
        print(f"   ✅ MP3: {mp3_size_mb:.1f} MB ({total_duration/60:.1f} Min)")

        # Metadaten schreiben
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


def main():
    parser = argparse.ArgumentParser(description="Audio-Renderer für Hörbücher")

    # Nur path ist Pflicht
    parser.add_argument("--path", required=True, help="Basis-Pfad für Eingabe- und Ausgabedateien")

    # Alle anderen Parameter sind flexibel (überschreibbar)
    parser.add_argument("--scenes", default=None,
                        help="Verzeichnis mit Szenen-Audio (scene_*.wav)")
    parser.add_argument("--output", default=None,
                        help="Output-Verzeichnis für das Hörbuch")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Pause zwischen Szenen in Sekunden (Standard: 2.0)")
    parser.add_argument("--head-silence", type=float, default=10.0,
                        help="Stille am Anfang in Sekunden (Standard: 10.0)")
    parser.add_argument("--tail-silence", type=float, default=10.0,
                        help="Stille am Ende in Sekunden (Standard: 10.0)")
    parser.add_argument("--mp3-bitrate", default="192k", help="MP3 Bitrate")

    args = parser.parse_args()
    base_path = args.path

    # --- CONFIG ---
    CONFIG = {
        "scenes": args.scenes or os.path.join(base_path, "scenes"),
        "output": args.output or os.path.join(base_path, "audiobook"),
        "pause": args.pause,
        "head_silence": args.head_silence,
        "tail_silence": args.tail_silence,
        "mp3_bitrate": args.mp3_bitrate,
    }


    renderer = AudiobookRenderer(scenes_dir=args.scenes, output_dir=args.output)
    result = renderer.create_complete_audiobook(
        pause_between_scenes=args.pause,
        head_silence=args.head_silence,
        tail_silence=args.tail_silence,
        mp3_bitrate=args.mp3_bitrate
    )

    if result:
        print("\n🎉 Audio-Rendering abgeschlossen!")
    else:
        print("\n❌ Audio-Rendering fehlgeschlagen!")


if __name__ == "__main__":
    main()
