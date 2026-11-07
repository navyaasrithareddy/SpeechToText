import wave
import numpy as np
import torch

# ============================================================
# 1. File paths
# ============================================================

MODEL_FILE = r"C:\Users\nandra navya asritha\.cache\huggingface\hub\models--facebook--seamless-m4t-unity-small-s2t\snapshots\06c5c9a98d60b9bd44a0956fbd9a1a3851dccd22\unity_on_device_s2t.ptl"

AUDIO_FILE = r"audio\test.wav"


# ============================================================
# 2. Select device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SPEECH-TO-TEXT USING SEAMLESSM4T")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. Load model
# ============================================================

print("\nLoading SeamlessM4T Small S2T model...")
print("Please wait...")

model = torch.jit.load(
    MODEL_FILE,
    map_location=device
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# 4. Load WAV audio
# ============================================================

print("\nLoading audio...")

with wave.open(AUDIO_FILE, "rb") as wav:

    sample_rate = wav.getframerate()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    number_of_frames = wav.getnframes()

    audio_bytes = wav.readframes(number_of_frames)


print("Original sample rate:", sample_rate)
print("Channels:", channels)
print("Sample width:", sample_width, "bytes")
print("Number of frames:", number_of_frames)


# ============================================================
# 5. Validate audio
# ============================================================

if sample_width != 2:
    raise ValueError(
        "Audio must be 16-bit PCM WAV."
    )

if sample_rate != 16000:
    raise ValueError(
        f"Audio must be 16000 Hz. "
        f"Current audio is {sample_rate} Hz."
    )

if number_of_frames == 0:
    raise ValueError(
        "Audio file is empty."
    )


# ============================================================
# 6. Convert WAV to NumPy
# ============================================================

audio_np = np.frombuffer(
    audio_bytes,
    dtype=np.int16
).astype(np.float32)


# ============================================================
# 7. Normalize audio
# ============================================================

audio_np = audio_np / 32768.0


# ============================================================
# 8. Convert stereo to mono
# ============================================================

if channels == 1:

    audio_np = audio_np.reshape(
        1,
        -1
    )

else:

    audio_np = audio_np.reshape(
        -1,
        channels
    )

    audio_np = np.mean(
        audio_np,
        axis=1
    )

    audio_np = audio_np.reshape(
        1,
        -1
    )


# ============================================================
# 9. Create PyTorch tensor
# ============================================================

audio = torch.from_numpy(
    audio_np.copy()
).float()


print("\nFinal audio information:")
print("Tensor shape:", audio.shape)
print("Tensor dtype:", audio.dtype)
print("Sample rate:", sample_rate)


# ============================================================
# 10. Check audio level
# ============================================================

audio_min = audio.min().item()
audio_max = audio.max().item()
audio_mean = audio.abs().mean().item()

print("\nAudio signal:")
print("Minimum:", audio_min)
print("Maximum:", audio_max)
print("Average absolute level:", audio_mean)


if audio_mean < 0.001:

    print("\nWARNING:")
    print("The recording is extremely quiet.")
    print("Please speak closer to the microphone.")


# ============================================================
# 11. Move audio to GPU / CPU
# ============================================================

audio = audio.to(device)


# ============================================================
# 12. Speech-to-text
# ============================================================

print("\n" + "=" * 60)
print("TRANSCRIBING AUDIO")
print("=" * 60)

with torch.no_grad():

    result = model(
        audio,
        tgt_lang="eng"
    )


# ============================================================
# 13. Convert result to text
# ============================================================

if isinstance(result, str):

    transcription = result.strip()

else:

    transcription = str(result).strip()


# ============================================================
# 14. Display result
# ============================================================

print("\n" + "=" * 60)
print("TRANSCRIPTION RESULT")
print("=" * 60)

if transcription:

    print(transcription)

else:

    print("No speech detected.")


print("=" * 60)
print("Speech-to-text completed successfully!")
print("=" * 60)