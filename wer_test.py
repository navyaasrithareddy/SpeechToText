from backend import transcribe_audio


# ============================================================
# WER TEST SENTENCES
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
# DISPLAY TEST SENTENCES
# ============================================================

print("=" * 70)
print("SPEECH-TO-TEXT ACCURACY TEST")
print("=" * 70)

print("\nWe will use these sentences for testing:\n")


for number, sentence in enumerate(
    TEST_SENTENCES,
    start=1
):

    print(
        f"{number}. {sentence}"
    )


print("\n" + "=" * 70)

print(
    "These sentences will be recorded and compared "
    "with the model's transcription."
)

print("=" * 70)