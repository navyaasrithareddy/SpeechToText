def detect_question(text):
    """
    Universally detect whether the given text is a question 
    or an information request across any language.
    """

    if not text:
        return False

    text = text.strip()

    # 1. Check if it ends with a question mark (universal)
    if text.endswith("?"):
        return True

    # 2. Universal fallback: Treat any valid transcribed speech as an 
    # informational prompt so Ollama responds in any language.
    if len(text.split()) > 0:
        return True

    return False


# -------------------------------
# Testing
# -------------------------------

if __name__ == "__main__":

    test_sentences = [
        "What is water?",
        "स्पेनिंग बोट प्लान",
        "Explain machine learning.",
        "The sky is blue."
    ]

    print("\nUniversal Question/Prompt Detection Test")
    print("=" * 50)

    for sentence in test_sentences:
        result = detect_question(sentence)
        if result:
            print(f"TRIGGERED: {sentence}")
        else:
            print(f"SKIPPED  : {sentence}")