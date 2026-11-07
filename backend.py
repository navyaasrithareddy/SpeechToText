import wave
import re
import numpy as np
import torch

from scipy.signal import (
    resample_poly,
    butter,
    sosfiltfilt
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_FILE = r"C:\Users\nandra navya asritha\.cache\huggingface\hub\models--facebook--seamless-m4t-unity-small-s2t\snapshots\06c5c9a98d60b9bd44a0956fbd9a1a3851dccd22\unity_on_device_s2t.ptl"


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("SPEECH-TO-TEXT BACKEND")
print("=" * 60)

print("\nLoading SeamlessM4T Small S2T model...")

model = torch.jit.load(
    MODEL_FILE,
    map_location=device
)

model.eval()

print("Model loaded successfully!")
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 60)


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_np, sample_rate):
    audio_np = audio_np.astype(np.float32)

    if len(audio_np) == 0:
        return audio_np

    maximum_possible = 32768.0

    if np.max(np.abs(audio_np)) > 1.5:
        audio_np = audio_np / maximum_possible

    audio_np = audio_np - np.mean(audio_np)

    if sample_rate >= 16000 and len(audio_np) > 100:
        try:
            low_cutoff = 80
            high_cutoff = min(7500, (sample_rate / 2) - 100)

            sos = butter(
                4,
                [low_cutoff, high_cutoff],
                btype="bandpass",
                fs=sample_rate,
                output="sos"
            )

            audio_np = sosfiltfilt(sos, audio_np).astype(np.float32)

        except Exception as e:
            print("Filtering skipped:", e)

    if len(audio_np) > 0:
        absolute_audio = np.abs(audio_np)
        peak = np.max(absolute_audio)

        if peak > 0:
            threshold = peak * 0.02
            speech_indices = np.where(absolute_audio > threshold)[0]

            if len(speech_indices) > 0:
                start = speech_indices[0]
                end = speech_indices[-1] + 1
                margin = int(sample_rate * 0.15)

                start = max(0, start - margin)
                end = min(len(audio_np), end + margin)

                audio_np = audio_np[start:end]

    peak = np.max(np.abs(audio_np))

    if peak > 0:
        target_peak = 0.85
        gain = target_peak / peak
        gain = min(gain, 4.0)
        audio_np = audio_np * gain

    audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
    return audio_np


# ============================================================
# SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_file, tgt_lang="eng"):
    try:
        with wave.open(audio_file, "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            number_of_frames = wav.getnframes()
            audio_bytes = wav.readframes(number_of_frames)
    except Exception as e:
        raise ValueError(f"Could not read WAV audio: {e}")

    if sample_width != 2:
        raise ValueError("Audio must be 16-bit PCM WAV.")

    if number_of_frames == 0:
        raise ValueError("Audio file is empty.")

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    if channels > 1:
        audio_np = audio_np.reshape(-1, channels)
        audio_np = np.mean(audio_np, axis=1)

    print("\nPreprocessing audio...")
    audio_np = preprocess_audio(audio_np, sample_rate)

    if len(audio_np) == 0:
        raise ValueError("No usable speech was found in the audio.")

    TARGET_SAMPLE_RATE = 16000

    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"Resampling audio: {sample_rate} Hz → {TARGET_SAMPLE_RATE} Hz")
        audio_np = resample_poly(audio_np, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32)

    audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
    audio_np = audio_np.reshape(1, -1)
    
    audio = torch.from_numpy(audio_np.copy()).float().to(device)

    print(f"\nRunning speech-to-text with target language: {tgt_lang}...")

    with torch.no_grad():
        # Pass tgt_lang to satisfy the compiled TorchScript wrapper
        result = model(audio, tgt_lang=tgt_lang)

    if isinstance(result, str):
        transcription = result.strip()
    else:
        transcription = str(result).strip()

    # Strip any model tag artifacts
    transcription = re.sub(r"<lang:[a-z]+>", "", transcription).strip()

    if not transcription:
        raise ValueError("No speech was detected.")

    print("Transcription:", transcription)
    return transcription