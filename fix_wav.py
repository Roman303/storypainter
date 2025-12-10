from pydub import AudioSegment, silence
import os
from pathlib import Path

INPUT_DIR = Path("tts")            # dein Ausgabeordner
MAX_SILENCE = 1200                # 1.2 Sekunden (in Millisekunden)

def remove_silence_from_file(wav_path):
    print(f"Bearbeite: {wav_path.name}")

    audio = AudioSegment.from_wav(wav_path)

    # Stille-Parameter:
    # silence_thresh: alles 30 dB unter Durchschnitt = Stille
    silence_thresh = audio.dBFS - 30

    # Liste aus Segmenten, die NICHT Stille sind
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=MAX_SILENCE,   # Stille länger als 1.2s → Split
        silence_thresh=silence_thresh,
        keep_silence=200               # lässt 0.2s Stille, damit es natürlich klingt
    )

    if len(chunks) <= 1:
        print("   → keine langen Stillen gefunden")
        return

    print(f"   → {len(chunks)-1} lange Stillen entfernt")

    output_audio = AudioSegment.empty()
    for c in chunks:
        output_audio += c

    output_audio.export(wav_path, format="wav")


def process_all():
    wav_files = sorted(INPUT_DIR.glob("*.wav"))

    print(f"{len(wav_files)} Dateien gefunden.\n")

    for wav in wav_files:
        remove_silence_from_file(wav)


if __name__ == "__main__":
    process_all()
