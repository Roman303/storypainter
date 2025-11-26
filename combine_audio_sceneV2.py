#!/usr/bin/env python3
"""
Audio-Renderer (Extended with Music)
- Fügt definierte Pausen, Intro- und Outro-Stille korrekt ein
- Unterstützt Intro-Musik (music_0000.wav)
- Unterstützt Musik-Mix pro Szene (music_0001.wav → rollierend)
- Unterstützt letzte Musik vor Outro (music_1111.wav)
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

        # Musik-Verwaltung
        self.music_dir = Path(scenes_dir).parent / "music"
        self.music_index = 1
        self.intro_music = None
        self.last_music = None

    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # MUSIK MANAGEMENT
    # -------------------------------------------------------------

    def load_next_music(self):
        """Lädt den nächsten Musik-WAV aus music_0001.wav usw."""
        music_path = self.music_dir / f"music_{self.music_index:04d}.wav"

        if not music_path.exists():
            # loop back
            self.music_index = 1
            music_path = self.music_dir / f"music_{self.music_index:04d}.wav"

        try:
            music = AudioSegment.from_wav(music_path)
            print(f"🎶 Lade Szenen-Musik: {music_path.name}")
        except Exception as e:
            print(f"⚠️ Musik konnte nicht geladen werden: {e}")
            return None

        self.music_index += 1
        return music

    # -------------------------------------------------------------

    def mix_music_into_scene(self, scene_audio: AudioSegment, music_audio: AudioSegment):
        """Mischt Musik 1 Sekunde vor Szenenende ein."""
        if music_audio is None:
            return scene_audio

        scene_len = len(scene_audio)
        music_len = len(music_audio)

        start_ms = max(scene_len - 1000, 0)

        if start_ms + music_len > scene_len:
            overhang = start_ms + music_len - scene_len
            fade_ms = min(1200, overhang + 400)
            music_audio = music_audio.fade_out(fade_ms)
            music_audio = music_audio[:scene_len - start_ms]

        music_audio = music_audio - 6
        mixed = scene_audio.overlay(music_audio, position=start_ms)

        print(f"🎵 Musik für Szene eingemischt (Start: {start_ms/1000:.2f}s)")
        return mixed

    # -------------------------------------------------------------

    def create_complete_audiobook(
        self,
        pause_between_scenes: float = 2.0,
        intro_length: float = 10.0,
        outro_length: float = 10.0,
        mp3_bitrate: str = "192k"
    ):

        print("=" * 60)
        print("🎧 AUDIOBOOK-RENDERER (MUSIC EXTENDED)")
        print("=" * 60)

        scenes = self.find_scene_audio_files()
        if not scenes:
            print("❌ Keine Szenen gefunden!")
            return None

        title = self.book_info.get("title", "Complete Audiobook")
        author = self.book_info.get("author", "TTS Generator")

        pause_ms = int(round(pause_between_scenes * 1000))
        intro_ms = int(round(intro_length * 1000))
        outro_ms = int(round(outro_length * 1000))

        print(f"📘 Titel: {title}")
        print(f"✍️  Autor: {author}")
        print(f"🎬 Szenen: {len(scenes)}")
        print(f"⏸️  Pausen: {pause_between_scenes}s")
        print(f"🎵 Intro-Länge: {intro_length}s")
        print(f"🎵 Outro-Länge: {outro_length}s")
        print("─────────────────────────────────────────────\n")

        combined_audio = AudioSegment.silent(duration=0)

        # -------------------------------------------------------------
        # INTRO MUSIK LADEN
        # -------------------------------------------------------------

        intro_music_path = self.music_dir / "music_0000.wav"
        if intro_music_path.exists():
            try:
                self.intro_music = AudioSegment.from_wav(intro_music_path)
                print("🎶 Intro-Musik erkannt: music_0000.wav")
            except Exception as e:
                print(f"⚠️ Intro-Musik Fehler: {e}")

        # WENN vorhanden → direkt am Masteranfang abspielen
        if self.intro_music:
            combined_audio = combined_audio.overlay(self.intro_music, position=0)
            print("🎵 Intro-Musik wird am Anfang abgespielt.\n")

        metadata = {
            "title": title,
            "author": author,
            "total_scenes": len(scenes) + 2,
            "pause_duration": pause_between_scenes,
            "intro_length": intro_length,
            "outro_length": outro_length,
            "timestamps_base": "absolute",
            "scenes": [],
        }

        current_ms = 0

        # -------------------------------------------------------------
        # INTRO STILLE
        # -------------------------------------------------------------

        print(f"[000] Intro-Stille: {intro_length}s @ 0 → {intro_length}s")
        combined_audio += AudioSegment.silent(duration=intro_ms)

        metadata["scenes"].append({
            "scene_id": 0,
            "type": "intro",
            "start_time": 0.0,
            "end_time": round(intro_length, 3),
            "duration": round(intro_length, 3),
        })

        current_ms += intro_ms

        if pause_ms > 0:
            combined_audio += AudioSegment.silent(duration=pause_ms)
            print(f"⏸️ Intro-Pause: {pause_between_scenes}s")
            current_ms += pause_ms

        # -------------------------------------------------------------
        # SZENEN LOOP
        # -------------------------------------------------------------

        for i, scene in enumerate(scenes):
            scene_id = scene["scene_id"]
            audio_file = scene["audio_file"]

            try:
                audio = AudioSegment.from_wav(audio_file)
            except Exception as e:
                print(f"⚠️ Szene {scene_id} Fehler: {e}")
                continue

            print(f"\n🎙️ Szene {scene_id} laden: {len(audio)/1000:.3f}s")

            # SZENEN-MUSIK LADEN
            music = None
            if i == 0:
                # Erste Szene → Musik erst NACH Intro benutzen → also 0001
                self.music_index = 1
                music = self.load_next_music()
            else:
                music = self.load_next_music()

            # Musik einmischen
            audio = self.mix_music_into_scene(audio, music)

            duration_ms = len(audio)
            start_s = round(current_ms / 1000.0, 3)
            end_s = round((current_ms + duration_ms) / 1000.0, 3)

            print(f"⏺️ Szene {scene_id} @ {start_s}s → {end_s}s")

            combined_audio += audio

            metadata["scenes"].append({
                "scene_id": scene_id,
                "type": "scene",
                "start_time": start_s,
                "end_time": end_s,
                "duration": duration_ms / 1000.0,
            })

            current_ms += duration_ms

            # Pause nach Szene
            if pause_ms > 0:
                combined_audio += AudioSegment.silent(duration=pause_ms)
                print(f"⏸️ Pause nach Szene: {pause_between_scenes}s")
                current_ms += pause_ms

        # -------------------------------------------------------------
        # LAST-SCENE SPECIAL MUSIC (music_1111.wav)
        # -------------------------------------------------------------

        last_music_path = self.music_dir / "music_1111.wav"
        if last_music_path.exists():
            try:
                self.last_music = AudioSegment.from_wav(last_music_path)
                print("\n🔔 Spezialmusik erkannt: music_1111.wav")
            except Exception as e:
                print(f"⚠️ Spezialmusik Fehler: {e}")
                self.last_music = None

        if self.last_music:
            print("🎵 Letzte-Musik wird vor dem Outro abgespielt.")
            combined_audio = combined_audio.overlay(self.last_music, position=current_ms)
            combined_audio += AudioSegment.silent(duration=len(self.last_music))
            current_ms += len(self.last_music)

            print("🔻 Fading out Endmusik (0.4s)")
            combined_audio = combined_audio.fade_out(400)

        # -------------------------------------------------------------
        # OUTRO-STILLE
        # -------------------------------------------------------------

        print(f"\n[OUTRO] Outro-Stille {outro_length}s")
        combined_audio += AudioSegment.silent(duration=outro_ms)
        current_ms += outro_ms

        # -------------------------------------------------------------
        # WRITE FILES
        # -------------------------------------------------------------

        wav_out = self.output_dir / "complete_audiobook.wav"
        mp3_out = self.output_dir / "complete_audiobook.mp3"
        metadata_file = self.output_dir / "audiobook_metadata.json"

        print("\n💾 Schreibe WAV…")
        combined_audio.export(wav_out, format="wav")

        print("💾 Schreibe MP3…")
        combined_audio.export(mp3_out, format="mp3", bitrate=mp3_bitrate,
                              tags={"title": title, "artist": author, "album": title})

        metadata["total_duration"] = round(len(combined_audio) / 1000.0, 3)
        metadata["total_duration_formatted"] = f"{int(metadata['total_duration']//60)}:{int(metadata['total_duration']%60):02d}"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n📄 Metadaten:", metadata_file)
        print("🎉 Rendering abgeschlossen!\n")

        return {
            "wav": wav_out,
            "mp3": mp3_out,
            "metadata": metadata_file,
            "duration": metadata["total_duration"],
        }


# -------------------------------------------------------------
# METADATA LOADER
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audio-Renderer (Music Extended)")
    parser.add_argument("--path", required=True)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pause", type=float, default=4.0)
    parser.add_argument("--intro-length", type=float, default=10.0)
    parser.add_argument("--outro-length", type=float, default=10.0)
    parser.add_argument("--mp3-bitrate", default="192k")

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
