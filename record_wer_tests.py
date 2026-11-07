import sounddevice as sd
import wave
import numpy as np
import os
import time


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_FOLDER = "audio"

SAMPLE_RATE = 16000

CHANNELS = 1

DURATION = 10


# ============================================================
# TEST SENTENCES
# ============================================================

TEST_SENTENCES = [
    "My name is Navya.",

    "I am a computer science student.",

    "I am working on a speech to text project.",

    "Water is clear, tasteless, and odorless.",

    "Water is a chemical compound with the formula H2O.",

    "Each molecule of water contains two hydrogen atoms and one oxygen atom.",

    "Machine learning is a branch of artificial intelligence.",

    "Natural language processing helps computers understand human language.",

    "The next important part of our project is question answering.",

    "What is water?"
]


# ============================================================
# CREATE AUDIO FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SPEECH-TO-TEXT WER TEST RECORDING")
print("=" * 70)

print(
    "\nWe will record 10 test sentences."
)

print(
    "Each recording will be 10 seconds long."
)

print(
    "Speak clearly and naturally."
)

print("=" * 70)


# ============================================================
# RECORD EACH SENTENCE
# ============================================================

for number, sentence in enumerate(
    TEST_SENTENCES,
    start=1
):

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"test{number}.wav"
    )


    # --------------------------------------------------------
    # DISPLAY SENTENCE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        f"TEST {number} / {len(TEST_SENTENCES)}"
    )

    print("=" * 70)

    print("\nSpeak this sentence:")

    print(
        f"\n\"{sentence}\"\n"
    )


    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    input(
        "Press ENTER when you are ready..."
    )


    print("\nRecording starts in:")

    print("3")

    time.sleep(1)

    print("2")

    time.sleep(1)

    print("1")

    time.sleep(1)


    print(
        "\n🎙️ RECORDING NOW!"
    )

    print(
        "Speak clearly...\n"
    )


    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    audio = sd.rec(
        int(
            DURATION *
            SAMPLE_RATE
        ),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )

    sd.wait()


    print(
        "\nRecording completed!"
    )


    # --------------------------------------------------------
    # NORMALIZE AUDIO
    # --------------------------------------------------------

    audio_float = audio.astype(
        np.float32
    )

    maximum = np.max(
        np.abs(audio_float)
    )


    if maximum > 0:

        target = (
            0.85 *
            32767
        )

        audio_float = (
            audio_float *
            (target / maximum)
        )

        audio_float = np.clip(
            audio_float,
            -32768,
            32767
        )


    audio = audio_float.astype(
        np.int16
    )


    # --------------------------------------------------------
    # SAVE WAV
    # --------------------------------------------------------

    with wave.open(
        output_file,
        "wb"
    ) as wav_file:

        wav_file.setnchannels(
            CHANNELS
        )

        wav_file.setsampwidth(
            2
        )

        wav_file.setframerate(
            SAMPLE_RATE
        )

        wav_file.writeframes(
            audio.tobytes()
        )


    print(
        f"Saved: {output_file}"
    )


    # --------------------------------------------------------
    # SHORT BREAK
    # --------------------------------------------------------

    if number < len(TEST_SENTENCES):

        print(
            "\nGet ready for the next sentence..."
        )

        time.sleep(2)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)

print(
    "ALL 10 RECORDINGS COMPLETED!"
)

print("=" * 70)

print(
    "\nYour files are stored in:"
)

print(
    "audio/"
)

for number in range(
    1,
    len(TEST_SENTENCES) + 1
):

    print(
        f"  test{number}.wav"
    )

print("=" * 70)