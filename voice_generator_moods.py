#!/usr/bin/env python3
"""
XTTS2 Voice Generator - Mood / Reference Edition

Konzept:
- EIN Sprecher pro Lauf
- EIN Fine-Tuned XTTS2-Modell im VRAM
- JSON enthält bereits fertige Chunks
- KEIN Chunk-Splitting
- Jede Dialogzeile besitzt:
    speaker
    mood
    text

- Für jede Stimmung werden automatisch ALLE passenden WAVs gefunden:

    neutral.wav
    neutral_2.wav
    neutral_3.wav

  bzw.

    sad.wav
    sad_2.wav
    sad_3.wav

- Die gefundenen WAVs werden gemeinsam als speaker_wav
  an XTTS2 übergeben.

Beispiel:

voices/tomhq/
    model.pth
    config.json
    neutral.wav
    neutral_2.wav
    sad.wav
    sad_2.wav
    excited.wav

Ausgabe:

scene_0001_chunk_001_TomHQ_neutral.wav
scene_0001_chunk_002_TomHQ_sad.wav
scene_0001_chunk_003_TomHQ_excited.wav
"""

import os

os.environ["COQUI_TOS_AGREED"] = "1"

import sys
import json
import time
import argparse
import re
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence


class VoiceGenerator:
    def __init__(self, config):
        self.config = config

        self.output_dir = Path(
            config["output_dir"]
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.progress_file = (
            self.output_dir / "progress.json"
        )

        self.current_speaker = config["speaker"]

        self.voice_dir = Path(
            config["voice_dir"]
        )

    # =========================================================
    # PROGRESS
    # =========================================================

    def load_progress(self):
        if self.progress_file.exists():
            try:
                with open(
                    self.progress_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    progress = json.load(f)

                if "completed" not in progress:
                    progress["completed"] = []

                return progress

            except Exception:
                pass

        return {
            "completed": []
        }

    def save_progress(self, item_id):
        progress = self.load_progress()

        if item_id not in progress["completed"]:
            progress["completed"].append(item_id)

        with open(
            self.progress_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                progress,
                f,
                indent=2,
                ensure_ascii=False
            )

    # =========================================================
    # MOOD → WAVS
    # =========================================================

    def get_mood_references(self, mood):
        """
        Findet automatisch ALLE WAV-Dateien für eine Stimmung.

        Beispiele:

            sad.wav
            sad_2.wav
            sad_3.wav

        werden alle gefunden.

        Die Reihenfolge ist:
            sad.wav
            sad_2.wav
            sad_3.wav
            ...

        """

        if not mood:
            mood = self.config.get(
                "default_mood",
                "neutral"
            )

        mood = str(mood).strip().lower()

        # -----------------------------------------------------
        # Erst die normale Basisdatei
        # -----------------------------------------------------

        references = []

        base_file = (
            Path(self.config["ref_dir"]) / f"{mood}.wav"
        )

        if base_file.exists():
            references.append(base_file)

        # -----------------------------------------------------
        # Danach _2, _3, _4 ...
        # -----------------------------------------------------

        pattern = re.compile(
            rf"^{re.escape(mood)}_(\d+)\.wav$",
            re.IGNORECASE
        )

        numbered = []

        for wav in Path(self.config["ref_dir"]).glob("*.wav"):

            match = pattern.match(
                wav.name
            )

            if match:
                number = int(
                    match.group(1)
                )

                numbered.append(
                    (number, wav)
                )

        numbered.sort(
            key=lambda x: x[0]
        )

        for _, wav in numbered:
            references.append(wav)

        # -----------------------------------------------------
        # Keine Referenz gefunden
        # -----------------------------------------------------

        if not references:

            raise FileNotFoundError(
                f"Keine Referenz-WAV für "
                f"Emotion '{mood}' gefunden.\n"
                f"Gesucht in: {self.voice_dir}\n"
                f"Erwartet z.B.:\n"
                f"  {mood}.wav\n"
                f"  {mood}_2.wav\n"
                f"  {mood}_3.wav"
            )

        return [
            str(path)
            for path in references
        ]

    # =========================================================
    # TEXT
    # =========================================================

    def _number_to_german(self, num):
        ones = [
            "null",
            "eins",
            "zwei",
            "drei",
            "vier",
            "fünf",
            "sechs",
            "sieben",
            "acht",
            "neun"
        ]

        teens = [
            "zehn",
            "elf",
            "zwölf",
            "dreizehn",
            "vierzehn",
            "fünfzehn",
            "sechzehn",
            "siebzehn",
            "achtzehn",
            "neunzehn"
        ]

        tens = [
            "",
            "",
            "zwanzig",
            "dreißig",
            "vierzig",
            "fünfzig",
            "sechzig",
            "siebzig",
            "achtzig",
            "neunzig"
        ]

        if num < 10:
            return ones[num]

        if num < 20:
            return teens[num - 10]

        if num < 100:
            digit = num % 10
            ten = num // 10

            if digit == 0:
                return tens[ten]

            return (
                ones[digit]
                + "und"
                + tens[ten]
            )

        if num < 1000:
            hundred = num // 100
            rest = num % 100

            result = (
                ones[hundred]
                + "hundert"
            )

            if rest:
                result += (
                    self._number_to_german(
                        rest
                    )
                )

            return result

        if num < 1000000:
            thousands = num // 1000
            rest = num % 1000

            result = (
                self._number_to_german(
                    thousands
                )
                + "tausend"
            )

            if rest:
                result += (
                    self._number_to_german(
                        rest
                    )
                )

            return result

        return str(num)

    def prepare_text_for_xtts(
        self,
        raw_text: str
    ) -> str:

        t = raw_text.strip()

        # Formatierungszeichen
        remove_chars = [
            "_",
            "*",
            "#",
            "|",
            "·",
            "•",
            "◆",
            "►",
            "◄",
            "~"
        ]

        for c in remove_chars:
            t = t.replace(c, "")

        # Zero width
        zero_width = [
            "\u200B",
            "\u200C",
            "\u200D",
            "\u2060",
            "\uFEFF"
        ]

        for z in zero_width:
            t = t.replace(z, "")

        # Anführungszeichen
        quote_map = {
            "«": '"',
            "»": '"',
            "„": '"',
            "“": '"',
            "”": '"',
            "‚": "'",
            "‘": "'",
            "’": "'",
            "ʼ": "'",
            "´": "'",
            "ˈ": "'",
            "‹": '"',
            "›": '"'
        }

        for bad, good in quote_map.items():
            t = t.replace(
                bad,
                good
            )

        # Gedankenstriche
        dash_variants = [
            "–",
            "—",
            "―",
            "−",
            "‐",
            "-",
            "﹘",
            "﹣",
            "－",
            "ｰ"
        ]

        for d in dash_variants:
            t = t.replace(
                d,
                ", "
            )

        t = re.sub(
            r"[\-–—]{2,}",
            ", ",
            t
        )

        # Jahreszahlen
        def year_to_words(match):
            year = int(
                match.group(1)
            )

            if 1000 <= year <= 2100:
                return self._number_to_german(
                    year
                )

            return match.group(0)

        t = re.sub(
            r"\b(1\d{3}|20\d{2})\b",
            year_to_words,
            t
        )

        # Große Zahlen
        def large_number_to_words(match):
            try:
                num = int(
                    match.group(0).replace(
                        ".",
                        ""
                    )
                )

                if num >= 1000:
                    return (
                        self._number_to_german(
                            num
                        )
                    )

            except Exception:
                pass

            return match.group(0)

        t = re.sub(
            r"\b\d{1,3}(?:\.\d{3})+\b",
            large_number_to_words,
            t
        )

        # Prozent
        t = re.sub(
            r"(\d+)\s*%",
            lambda m:
                self._number_to_german(
                    int(m.group(1))
                )
                + " Prozent",
            t
        )

        # Euro
        t = re.sub(
            r"(\d+)\s*€",
            lambda m:
                self._number_to_german(
                    int(m.group(1))
                )
                + " Euro",
            t
        )

        # Dollar
        t = re.sub(
            r"(\d+)\s*\$",
            lambda m:
                self._number_to_german(
                    int(m.group(1))
                )
                + " Dollar",
            t
        )

        # Ellipsen
        t = re.sub(
            r"\.{3,}",
            "...",
            t
        )

        # Spaces
        t = t.replace(
            "\u00A0",
            " "
        )

        t = t.replace(
            "\u202F",
            " "
        )

        # Mehrere Spaces
        t = re.sub(
            r"[ \t]+",
            " ",
            t
        )

        # Fehlende Spaces
        t = re.sub(
            r"([.!?…])(?![ \n])",
            r"\1 ",
            t
        )

        return t.strip()

    # =========================================================
    # SILENCE CLEANUP
    # =========================================================

    def remove_long_silences(
        self,
        wav_path,
        max_silence_sec=1.0
    ):

        try:

            audio = AudioSegment.from_wav(
                wav_path
            )

        except Exception as e:

            print(
                f"      ⚠️ Konnte "
                f"{wav_path} nicht laden: {e}"
            )

            return False

        if len(audio) == 0:
            return False

        min_silence_len = int(
            max_silence_sec * 1000
        )

        silence_thresh = max(
            audio.dBFS - 25,
            -55
        )

        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=350
        )

        if len(chunks) <= 1:
            return False

        print(
            f"      ✂️ Entferne "
            f"{len(chunks)-1} lange Stillen"
        )

        new_audio = AudioSegment.empty()

        for chunk in chunks:
            new_audio += chunk

        new_audio.export(
            wav_path,
            format="wav"
        )

        return True

    # =========================================================
    # OUTPUT NAME
    # =========================================================

    def make_output_filename(
        self,
        item,
        index
    ):

        scene = item.get(
            "scene",
            item.get(
                "scene_id",
                1
            )
        )

        chunk = item.get(
            "chunk",
            item.get(
                "chunk_id",
                index
            )
        )

        speaker = item.get(
            "speaker",
            self.current_speaker
        )

        mood = item.get(
            "mood",
            self.config.get(
                "default_mood",
                "neutral"
            )
        )

        # Sichere Dateinamen
        speaker = re.sub(
            r"[^a-zA-Z0-9äöüÄÖÜß_-]+",
            "_",
            str(speaker)
        )

        mood = re.sub(
            r"[^a-zA-Z0-9äöüÄÖÜß_-]+",
            "_",
            str(mood)
        )

        return (
            f"scene_{int(scene):04d}"
            f"_chunk_{int(chunk):04d}"
            f"_{speaker}"
            f"_{mood}.wav"
        )

    # =========================================================
    # GENERATE
    # =========================================================

    def generate_one(
        self,
        tts,
        item,
        index
    ):

        text = item.get(
            "text",
            ""
        ).strip()

        if not text:
            print(
                "   ⚠️ Kein Text → übersprungen"
            )

            return True

        speaker = item.get(
            "speaker",
            self.current_speaker
        )

        mood = item.get(
            "mood",
            self.config.get(
                "default_mood",
                "neutral"
            )
        )

        # -----------------------------------------------------
        # Prüfen: richtiger Sprecher
        # -----------------------------------------------------

        if speaker != self.current_speaker:

            print(
                f"   ⚠️ Sprecher '{speaker}' "
                f"übersprungen."
            )

            print(
                f"      Aktueller Testsprecher: "
                f"{self.current_speaker}"
            )

            return True

        # -----------------------------------------------------
        # Referenzen
        # -----------------------------------------------------

        references = self.get_mood_references(
            mood
        )

        output_name = (
            self.make_output_filename(
                item,
                index
            )
        )

        output_file = (
            self.output_dir / output_name
        )

        # -----------------------------------------------------
        # Progress / vorhandene Datei
        # -----------------------------------------------------

        item_id = item.get(
            "id",
            output_name
        )

        if output_file.exists():

            print(
                f"   ⭐ {output_name} "
                f"existiert bereits"
            )

            self.save_progress(
                item_id
            )

            return True

        # -----------------------------------------------------
        # Anzeige
        # -----------------------------------------------------

        preview = text[:80]

        if len(text) > 80:
            preview += "..."

        print()
        print(
            f"   🎤 {speaker}"
            f" | 🎭 {mood}"
        )

        print(
            f"   📝 {preview}"
        )

        print(
            f"   🎙️ Referenzen "
            f"({len(references)}):"
        )

        for ref in references:
            print(
                f"      - "
                f"{os.path.basename(ref)}"
            )

        # -----------------------------------------------------
        # Text vorbereiten
        # -----------------------------------------------------

        prepared_text = (
            self.prepare_text_for_xtts(
                text
            )
        )

        # -----------------------------------------------------
        # XTTS
        # -----------------------------------------------------

        start = time.time()

        try:

            tts.tts_to_file(

                text=prepared_text,

                # EIN oder MEHRERE Mood-Samples
                speaker_wav=references,

                language=self.config[
                    "language"
                ],

                file_path=str(
                    output_file
                ),

                temperature=self.config[
                    "temperature"
                ],

                repetition_penalty=self.config[
                    "repetition_penalty"
                ],

                top_p=self.config[
                    "top_p"
                ],

                top_k=self.config[
                    "top_k"
                ],

                speed=self.config[
                    "speed"
                ]
            )

        except Exception as e:

            print(
                f"   ❌ TTS-Fehler: {e}"
            )

            return False

        duration = (
            time.time() - start
        )

        # -----------------------------------------------------
        # Silence Cleanup
        # -----------------------------------------------------

        if self.config.get(
            "remove_long_silences",
            True
        ):

            self.remove_long_silences(
                output_file,
                self.config.get(
                    "max_silence_sec",
                    1.0
                )
            )

        # -----------------------------------------------------
        # Audio-Dauer / RTF
        # -----------------------------------------------------

        try:

            import soundfile as sf

            audio_duration = (
                sf.info(
                    str(output_file)
                ).duration
            )

            rtf = (
                duration / audio_duration
                if audio_duration > 0
                else 0
            )

            print(
                f"   ✅ Fertig "
                f"in {duration:.1f}s"
                f" | Audio "
                f"{audio_duration:.1f}s"
                f" | RTF {rtf:.2f}"
            )

        except Exception:

            print(
                f"   ✅ Fertig "
                f"in {duration:.1f}s"
            )

        self.save_progress(
            item_id
        )

        return True

    # =========================================================
    # JSON LADEN
    # =========================================================

    def load_items(self):
        print(
            f"\n📖 Lade JSON:"
            f"\n   {self.config['scenes_file']}"
        )

        with open(
            self.config["scenes_file"],
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        scenes = data.get("scenes", [])

        items = []

        for scene_index, scene in enumerate(scenes, 1):
            chunks = scene.get("chunks", [])

            for chunk_index, chunk in enumerate(chunks, 1):
                text = chunk.get("text", "").strip()
                mood = chunk.get("mood", "neutral").strip().lower()

                if not text:
                    print(
                        f"⚠️ Szene {scene_index}, "
                        f"Chunk {chunk_index}: kein Text"
                    )
                    continue

                items.append({
                    "scene": scene_index,
                    "chunk": chunk_index,
                    "mood": mood,
                    "text": text
                })

        return items

    # =========================================================
    # HAUPTLAUF
    # =========================================================

    def run(self):

        import torch
        from TTS.api import TTS

        print()
        print("=" * 65)
        print(
            "🎧 XTTS2 VOICE GENERATOR"
        )
        print(
            "   Single Speaker / Mood References"
        )
        print("=" * 65)

        print(
            f"\n👤 Sprecher: "
            f"{self.current_speaker}"
        )

        print(
            f"🎙️ Voice Directory:"
            f"\n   {self.voice_dir}"
        )

        # -----------------------------------------------------
        # GPU
        # -----------------------------------------------------

        print("\n🔥 Hardware")

        print(
            f"   CUDA: "
            f"{torch.cuda.is_available()}"
        )

        if torch.cuda.is_available():

            gpu_id = self.config.get(
                "gpu_id",
                0
            )

            torch.cuda.set_device(
                gpu_id
            )

            print(
                f"   GPU: "
                f"{torch.cuda.get_device_name(gpu_id)}"
            )

            print(
                f"   VRAM: "
                f"{torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB"
            )

        # -----------------------------------------------------
        # Modell laden
        # -----------------------------------------------------

        print(
            "\n🔥 Lade Fine-Tune-Modell..."
        )

        try:

            tts = TTS(
                model_path=self.config[
                    "model_path"
                ],
                config_path=self.config[
                    "config_path"
                ]
            )

        except Exception as e:

            print(
                f"❌ Modell konnte nicht "
                f"geladen werden:\n{e}"
            )

            return False

        if torch.cuda.is_available():

            tts = tts.cuda()

            torch.backends.cudnn.benchmark = True

            print(
                "   ✅ Modell auf GPU"
            )

        else:

            print(
                "   ✅ Modell auf CPU"
            )

        # -----------------------------------------------------
        # Referenz-Dateien anzeigen
        # -----------------------------------------------------

        print(
            "\n🎙️ Gefundene Mood-Referenzen:"
        )

        moods = set()

        for wav in self.voice_dir.glob(
            "*.wav"
        ):

            name = wav.stem

            # neutral_2 → neutral
            match = re.match(
                r"^(.+?)_\d+$",
                name
            )

            if match:
                mood = match.group(1)
            else:
                mood = name

            moods.add(
                mood
            )

        for mood in sorted(moods):

            try:

                refs = (
                    self.get_mood_references(
                        mood
                    )
                )

                print(
                    f"   🎭 {mood}: "
                    f"{len(refs)} Sample(s)"
                )

            except Exception:
                pass

        # -----------------------------------------------------
        # JSON
        # -----------------------------------------------------

        items = self.load_items()

        print(
            f"\n✅ {len(items)} "
            f"Chunks/Dialogzeilen geladen"
        )

        # -----------------------------------------------------
        # Generieren
        # -----------------------------------------------------

        success_count = 0
        failed_count = 0
        skipped_count = 0

        print()
        print("=" * 65)
        print(
            "🎤 GENERIERUNG"
        )
        print("=" * 65)

        for index, item in enumerate(
            items,
            1
        ):

            speaker = item.get(
                "speaker",
                self.current_speaker
            )

            # In diesem Testlauf
            # nur EIN Sprecher
            if speaker != self.current_speaker:

                skipped_count += 1

                print(
                    f"\n[{index:04d}] "
                    f"⏭️ {speaker} "
                    f"→ anderer Sprecher, "
                    f"übersprungen"
                )

                continue

            ok = self.generate_one(
                tts,
                item,
                index
            )

            if ok:
                success_count += 1
            else:
                failed_count += 1

        # -----------------------------------------------------
        # Statistik
        # -----------------------------------------------------

        print()
        print("=" * 65)
        print(
            "✅ FERTIG"
        )
        print("=" * 65)

        print(
            f"   Gesamt:      {len(items)}"
        )

        print(
            f"   Erfolgreich: {success_count}"
        )

        print(
            f"   Übersprungen:{skipped_count}"
        )

        print(
            f"   Fehler:      {failed_count}"
        )

        print(
            f"\n📁 Ausgabe:"
            f"\n   {self.output_dir}"
        )

        return failed_count == 0


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "XTTS2 Single-Speaker "
            "Mood Voice Generator"
        )
    )

    parser.add_argument(
        "--path",
        required=True,
        help=(
            "Basisverzeichnis für "
            "JSON und Ausgabe"
        )
    )

    parser.add_argument(
        "--speaker",
        default="TomHQ",
        help=(
            "Sprecher für diesen Lauf"
        )
    )

    args = parser.parse_args()

    base_path = Path(
        args.path
    )

    speaker = args.speaker

    # =========================================================
    # HIER DEIN FINE-TUNE
    # =========================================================

    voice_dir = Path(
        f"/workspace/storypainter/voices/{speaker.lower()}"
    )

    ref_dir = voice_dir / "ref"

    # Falls deine Ordner Groß-/Kleinschreibung
    # anders verwenden, einfach hier anpassen.

    config = {

        # -----------------------------------------------------
        # Sprecher
        # -----------------------------------------------------

        "speaker": speaker,

        "voice_dir": str(voice_dir),
        "ref_dir": str(ref_dir),    

        # -----------------------------------------------------
        # Fine-Tuned XTTS2
        # -----------------------------------------------------

        "model_path": str(
            voice_dir
        ),

        "config_path": str(
            voice_dir / "config.json"
        ),

        # -----------------------------------------------------
        # JSON
        # -----------------------------------------------------

        "scenes_file": str(
            base_path / "book_scenes.json"
        ),

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        "output_dir": str(
            base_path / "tts"
        ),

        # -----------------------------------------------------
        # TTS
        # -----------------------------------------------------

        "language": "de",

        "temperature": 0.60,

        "top_p": 0.93,

        "top_k": 35,

        "repetition_penalty": 1.45,

        "speed": 1.0,

        # -----------------------------------------------------
        # Standard-Mood
        # -----------------------------------------------------

        "default_mood": "neutral",

        # -----------------------------------------------------
        # Silence
        # -----------------------------------------------------

        "remove_long_silences": True,

        "max_silence_sec": 1.0,

        # -----------------------------------------------------
        # GPU
        # -----------------------------------------------------

        "gpu_id": 0
    }

    # =========================================================
    # VALIDIERUNG
    # =========================================================

    print(
        "\n📂 Prüfe Pfade..."
    )

    required = {

        "voice_dir":
            config["voice_dir"],

        "ref_dir":
            config["ref_dir"],

        "model_path":
            config["model_path"],

        "config_path":
            config["config_path"],

        "scenes_file":
            config["scenes_file"]
    }

    for name, path in required.items():

        exists = os.path.exists(path)

        status = (
            "✅"
            if exists
            else "❌"
        )

        print(
            f"   {status} "
            f"{name}: "
            f"{path}"
        )

        if not exists:

            print(
                f"\n❌ Fehlender Pfad: "
                f"{path}"
            )

            sys.exit(1)

    # ---------------------------------------------------------
    # WAVs
    # ---------------------------------------------------------

    wavs = sorted(
        ref_dir.glob(
            "*.wav"
        )
    )

    if not wavs:

        print(
            "\n❌ Keine WAV-Dateien "
            "im Referenz-Verzeichnis."
        )

        sys.exit(1)

    print(
        f"\n🎙️ {len(wavs)} "
        f"Referenz-WAV-Dateien in {ref_dir} gefunden"
    )

    for wav in wavs:

        print(
            f"   - {wav.name}"
        )

    # =========================================================
    # START
    # =========================================================

    generator = VoiceGenerator(
        config
    )

    success = generator.run()

    sys.exit(
        0
        if success
        else 1
    )


if __name__ == "__main__":
    main()
