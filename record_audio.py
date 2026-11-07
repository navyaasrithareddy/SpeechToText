import sounddevice as sd
import wave
import numpy as np

# ============================================================
# Settings
# ============================================================

OUTPUT_FILE = r"D:\nlplabproject\SpeechToText\audio\test.wav"

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 10


# ============================================================
# Start recording
# ============================================================

print("=" * 60)
print("SPEECH RECORDING")
print("=" * 60)

print(f"\nRecording for {DURATION} seconds...")
print("Speak clearly and normally.")
print("Keep the microphone close to you.")
print("Start speaking now!\n")


audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16"
)

sd.wait()


# ============================================================
# Normalize recording
# ============================================================

audio_float = audio.astype(np.float32)

maximum = np.max(np.abs(audio_float))

if maximum > 0:

    # Increase volume without clipping

    target = 0.85 * 32767

    audio_float = audio_float * (target / maximum)

    audio_float = np.clip(
        audio_float,
        -32768,
        32767
    )

audio = audio_float.astype(np.int16)


# ============================================================
# Save WAV file
# ============================================================

with wave.open(OUTPUT_FILE, "wb") as wav_file:

    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SAMPLE_RATE)

    wav_file.writeframes(
        audio.tobytes()
    )


# ============================================================
# Finished
# ============================================================

print("\nRecording completed!")
print("Audio saved to:")
print(OUTPUT_FILE)

print("=" * 60)