# SpeechToText – Speech Recognition and Information Answering System

## Project Overview

SpeechToText is a speech recognition application that converts spoken English into text using the SeamlessM4T Small S2T model. It also identifies questions from the transcribed text and provides information-based answers using a locally running AI model through Ollama.

The application provides a simple interface built with Streamlit.

## Features

- **Speech-to-Text Conversion:** Converts spoken English audio into text.
- **Audio Recording:** Supports microphone-based audio recording.
- **Question Detection:** Identifies questions in the transcribed text.
- **AI-Based Information Answers:** Uses a locally running Ollama model to generate answers.
- **Audio Processing:** Processes audio at the required sample rate.
- **Transcription History:** Maintains transcription records and outputs.

## Technologies Used

- Python
- Streamlit
- SeamlessM4T Small S2T
- PyTorch
- Ollama
- HTML and CSS
- Git and GitHub

## Project Structure

```text
SpeechToText/
│
├── app.py
├── backend.py
├── speech_to_text.py
├── record_audio.py
├── information_answer.py
├── question_detector.py
├── record_wer_tests.py
├── wer_evaluation.py
├── wer_test.py
├── style.css
├── .gitignore
└── README.md
```

## How It Works

1. The user records or provides spoken English audio.
2. The audio is processed and passed to the SeamlessM4T Small S2T model.
3. The model generates a text transcription.
4. The application checks whether the transcribed text contains a question.
5. If a question is detected, the application uses a local Ollama model to generate an information-based answer.
6. The transcription and relevant outputs are displayed in the Streamlit interface.

## Installation and Execution

### Prerequisites

- Python 3.11
- Ollama installed locally
- Required SeamlessM4T Small S2T model files
- Required Python packages

### Run the Application

```bash
streamlit run app.py
```

Make sure the required model files and local Ollama model are configured before running the application.

## Project Goal

The goal of this project is to develop a local speech recognition and information-answering system that combines automatic speech transcription with question detection and locally generated AI responses.

## Author

Navya Asritha Reddy

