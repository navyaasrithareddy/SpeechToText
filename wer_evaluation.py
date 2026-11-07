import os
from jiwer import wer
from backend import transcribe_audio


# ============================================================
# WER EVALUATION
# ============================================================

TEST_DATA = [
    {
        "audio": "audio/test1.wav",
        "reference": "My name is Navya."
    },
    {
        "audio": "audio/test2.wav",
        "reference": "I am a computer science student."
    },
    {
        "audio": "audio/test3.wav",
        "reference": "I am working on a speech to text project."
    },
    {
        "audio": "audio/test4.wav",
        "reference": "Water is clear, tasteless, and odorless."
    },
    {
        "audio": "audio/test5.wav",
        "reference": "Water is a chemical compound with the formula H2O."
    },
    {
        "audio": "audio/test6.wav",
        "reference": "Each molecule of water contains two hydrogen atoms and one oxygen atom."
    },
    {
        "audio": "audio/test7.wav",
        "reference": "Machine learning is a branch of artificial intelligence."
    },
    {
        "audio": "audio/test8.wav",
        "reference": "Natural language processing helps computers understand human language."
    },
    {
        "audio": "audio/test9.wav",
        "reference": "The next important part of our project is question answering."
    },
    {
        "audio": "audio/test10.wav",
        "reference": "What is water?"
    }
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SEAMLESSM4T SPEECH-TO-TEXT WER EVALUATION")
print("=" * 70)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH AUDIO FILE
# ============================================================

for number, item in enumerate(
    TEST_DATA,
    start=1
):

    audio_file = item["audio"]
    reference = item["reference"]

    print("\n" + "-" * 70)

    print(
        f"TEST {number}"
    )

    print(
        "Reference:",
        reference
    )


    # --------------------------------------------------------
    # CHECK AUDIO FILE
    # --------------------------------------------------------

    if not os.path.exists(
        audio_file
    ):

        print(
            f"Audio file not found: {audio_file}"
        )

        print(
            "Skipping this test."
        )

        continue


    # --------------------------------------------------------
    # TRANSCRIBE
    # --------------------------------------------------------

    try:

        prediction = transcribe_audio(
            audio_file
        )

    except Exception as e:

        print(
            f"Transcription error: {e}"
        )

        continue


    # --------------------------------------------------------
    # CALCULATE WER
    # --------------------------------------------------------

    error_rate = wer(
        reference,
        prediction
    )


    accuracy = (
        max(
            0,
            (1 - error_rate) * 100
        )
    )


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print(
        "Prediction:",
        prediction
    )

    print(
        f"WER: {error_rate * 100:.2f}%"
    )

    print(
        f"Approximate word accuracy: "
        f"{accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "test": number,
            "reference": reference,
            "prediction": prediction,
            "wer": error_rate,
            "accuracy": accuracy
        }
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("WER EVALUATION SUMMARY")
print("=" * 70)


if results:

    total_wer = sum(
        result["wer"]
        for result in results
    )

    average_wer = (
        total_wer /
        len(results)
    )


    average_accuracy = (
        max(
            0,
            (1 - average_wer) * 100
        )
    )


    print(
        f"\nTests completed: "
        f"{len(results)}"
    )

    print(
        f"Average WER: "
        f"{average_wer * 100:.2f}%"
    )

    print(
        f"Approximate average word accuracy: "
        f"{average_accuracy:.2f}%"
    )


else:

    print(
        "\nNo audio files were available for evaluation."
    )


print("\n" + "=" * 70)