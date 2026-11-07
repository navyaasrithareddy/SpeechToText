import streamlit as st
import os
import re
import urllib.parse
import webbrowser
from datetime import datetime

from audio_recorder_streamlit import audio_recorder
from backend import transcribe_audio
from question_detector import detect_question
from information_answer import generate_information


# ============================================================
# PAGE CONFIGURATION & CUSTOM CSS LOADING
# ============================================================

st.set_page_config(
    page_title="Speech-to-Text AI Assistant",
    page_icon="🎙️",
    layout="wide"
)

css_file_path = "style.css"
if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================
# OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# SAVE HELPERS
# ============================================================

def save_audio(audio_bytes, timestamp):
    filename = f"audio_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    with open(filepath, "wb") as file:
        file.write(audio_bytes)
    return filepath


def save_transcription(text, timestamp):
    filename = f"transcription_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text)
    return filepath


def get_transcription_history():
    files = []
    if os.path.exists(OUTPUT_FOLDER):
        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.startswith("transcription_") and filename.endswith(".txt"):
                files.append(filename)
    files.sort(reverse=True)
    return files


def get_audio_for_transcription(transcription_filename):
    timestamp = transcription_filename.replace("transcription_", "").replace(".txt", "")
    audio_filename = f"audio_{timestamp}.wav"
    audio_filepath = os.path.join(OUTPUT_FOLDER, audio_filename)
    if os.path.exists(audio_filepath):
        return audio_filepath
    return None


# ============================================================
# SPEECH-TO-ACTION HANDLER (DYNAMIC PLATFORM ROUTER)
# ============================================================

def handle_speech_action(text):
    text_lower = text.lower()

    # Dictionary of popular platforms mapped to their official URLs
    platforms = {
        "youtube": "https://www.youtube.com",
        "you tube": "https://www.youtube.com",
        "amazon": "https://www.amazon.com",
        "flipkart": "https://www.flipkart.com",
        "myntra": "https://www.myntra.com",
        "google": "https://www.google.com",
        "github": "https://github.com",
        "chatgpt": "https://chatgpt.com",
        "netflix": "https://www.netflix.com",
        "wikipedia": "https://www.wikipedia.org"
    }

    # Check if any known platform keyword is in the spoken text
    for keyword, url in platforms.items():
        if keyword in text_lower:
            webbrowser.open(url)
            platform_name = keyword.capitalize()
            return f"🌐 **Action Executed**: Opening {platform_name}!\n\n🔗 **Direct Link**: [Open {platform_name}]({url})"

    # Handle weather searches specifically
    if "weather" in text_lower:
        query = urllib.parse.quote(text)
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        return f"🌤 **Action Executed**: Opening live weather forecast search!\n\n🔗 **Direct Link**: [View Weather Results]({url})"
    
    # Handle explicit search queries
    elif "search for" in text_lower or "look up" in text_lower:
        query_text = text_lower.replace("search for", "").replace("look up", "").strip()
        query = urllib.parse.quote(query_text)
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        return f"🔍 **Action Executed**: Searching Google for '{query_text}'!\n\n🔗 **Direct Link**: [Google Search Results]({url})"
    
    # General fallback search
    else:
        query = urllib.parse.quote(text)
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        return f"🚀 **Action Executed**: Triggered smart search for '{text}'!\n\n🔗 **Direct Link**: [View Search Results]({url})"


# ============================================================
# APP LAYOUT
# ============================================================

col_title, col_id = st.columns([2.5, 1.5])

with col_title:
    st.title("🎙️ Multi-Modal Speech-to-Text & AI Assistant")

with col_id:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 15px;">
            <p style="margin: 0; font-weight: bold; font-size: 16px; color: #ffffff;">N.Navya Asritha Reddy</p>
            <p style="margin: 0; font-size: 14px; color: #a0aec0;">323103282064</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("Convert spoken audio into text, translate into supported languages using Phi-3, and execute smart tasks.")
st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ System Information")
    st.write("**Speech Model:** SeamlessM4T Small S2T")
    st.write("**Information Model:** Phi-3 via Ollama")
    st.divider()
    st.header("📚 Transcription History")
    history_files = get_transcription_history()

    if history_files:
        st.write(f"Total saved: **{len(history_files)}**")
        for filename in history_files:
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                saved_text = file.read()
            audio_filepath = get_audio_for_transcription(filename)
            short_name = filename.replace("transcription_", "").replace(".txt", "")

            with st.expander(f"📄 {short_name}"):
                st.write(saved_text)
                if audio_filepath:
                    st.audio(audio_filepath, format="audio/wav")
                    with open(audio_filepath, "rb") as audio_file:
                        st.download_button(
                            label="🎵 Download",
                            data=audio_file,
                            file_name=os.path.basename(audio_filepath),
                            mime="audio/wav",
                            key=f"sidebar_audio_{filename}"
                        )
    else:
        st.write("No history available yet.")


# ============================================================
# CONFIGURATION CONTROLS
# ============================================================

st.subheader("⚙️ Assistant Configuration")
col_lang, col_mode = st.columns(2)

with col_lang:
    language_options = {
        "English": "eng",
        "Hindi": "hin",
        "Spanish": "spa",
        "French": "fra",
        "Portuguese": "por",
        "German": "deu",
        "Italian": "ita",
        "Japanese": "jpn"
    }
    selected_lang_name = st.selectbox("🌐 Choose Target Language:", list(language_options.keys()))
    target_lang_code = language_options[selected_lang_name]

with col_mode:
    task_mode_options = [
        "Speech-to-Text & Information Answering",
        "Conversational Language Practice",
        "Transcription Analysis & Insights",
        "Speech-to-Action (Alexa-style Commands)"
    ]
    selected_task_mode = st.selectbox("🎯 Choose Assistant Task Mode:", task_mode_options)

st.divider()


# ============================================================
# INPUT METHOD
# ============================================================

st.subheader("🎤 Select Audio Input")
input_method = st.radio("Choose how you want to provide audio:", ["📁 Upload Audio", "🎙 Record Audio"], horizontal=True)


# ============================================================
# SESSION STATE
# ============================================================

if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "saved_file" not in st.session_state:
    st.session_state.saved_file = ""
if "saved_audio" not in st.session_state:
    st.session_state.saved_audio = ""
if "information_answer" not in st.session_state:
    st.session_state.information_answer = ""
if "is_question" not in st.session_state:
    st.session_state.is_question = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# PROCESS TRANSCRIPTION & ROUTING
# ============================================================

def process_transcription(transcription, mode):
    if transcription is None:
        transcription = ""

    transcription = str(transcription).strip()
    transcription = re.sub(r"<lang:[a-z]+>", "", transcription).strip()

    # Handle phonetic typo corrections
    corrections = {
        "by ton": "Python", "by-ton": "Python", "paten": "Python", "pi-ton": "Python",
        "mission learning": "machine learning", "marching learning": "machine learning", "mashin learning": "machine learning"
    }
    for wrong_word, correct_word in corrections.items():
        if wrong_word in transcription.lower():
            transcription = re.sub(re.escape(wrong_word), correct_word, transcription, flags=re.IGNORECASE)

    st.session_state.information_answer = ""
    st.session_state.is_question = False

    if not transcription:
        return transcription

    if mode == "Speech-to-Action (Alexa-style Commands)":
        action_result = handle_speech_action(transcription)
        if action_result:
            st.session_state.is_question = True
            st.session_state.information_answer = action_result
            return transcription

    is_question = detect_question(transcription)
    st.session_state.is_question = is_question

    if is_question:
        with st.spinner(f"🤖 Processing with Phi-3 AI in {selected_lang_name}..."):
            if mode == "Conversational Language Practice":
                prompt = f"Provide a detailed and friendly conversational response in native {selected_lang_name} script:\nUser: {transcription}"
                answer = generate_information(prompt)
                st.session_state.chat_history.append({"role": "user", "content": transcription})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            elif mode == "Transcription Analysis & Insights":
                prompt = f"Provide a comprehensive summary and detailed bullet points in native {selected_lang_name} script:\n{transcription}"
                answer = generate_information(prompt)
            else:
                prompt = f"Provide a complete, detailed, and structured explanation of the following topic in native {selected_lang_name} script:\n{transcription}"
                answer = generate_information(prompt)
                
        st.session_state.information_answer = answer

    return transcription


# ============================================================
# UPLOAD OR RECORD EXECUTION
# ============================================================

if input_method == "📁 Upload Audio":
    st.subheader("📁 Upload WAV Audio")
    uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("🎯 Transcribe & Process Audio", use_container_width=True):
            with st.spinner("🎙️ Processing audio..."):
                try:
                    transcription = transcribe_audio(uploaded_file, tgt_lang=target_lang_code)
                    transcription = process_transcription(transcription, selected_task_mode)
                    
                    if not transcription:
                        st.warning("⚠️ No speech was detected.")
                    else:
                        st.session_state.transcription = transcription
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.session_state.saved_audio = save_audio(uploaded_file.getvalue(), timestamp)
                        st.session_state.saved_file = save_transcription(transcription, timestamp)
                        st.success("✅ Processing completed successfully!")
                except Exception as e:
                    st.error(f"❌ Error while processing: {e}")

else:
    st.subheader("🎙️ Record Your Voice")
    audio_bytes = audio_recorder(text="Click to record", recording_color="#e74c3c", neutral_color="#3498db", icon_name="microphone", icon_size="2x")

    if audio_bytes:
        st.success("✅ Recording completed successfully!")
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("🎯 Transcribe & Process Recording", use_container_width=True):
            with st.spinner("🎙️ Processing recording..."):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_audio = save_audio(audio_bytes, timestamp)
                    
                    temporary_file = "audio/recorded_audio.wav"
                    os.makedirs("audio", exist_ok=True)
                    with open(temporary_file, "wb") as file:
                        file.write(audio_bytes)

                    transcription = transcribe_audio(temporary_file, tgt_lang=target_lang_code)
                    transcription = process_transcription(transcription, selected_task_mode)

                    if not transcription:
                        st.warning("⚠️ No speech was detected.")
                    else:
                        st.session_state.transcription = transcription
                        st.session_state.saved_file = save_transcription(transcription, timestamp)
                        st.session_state.saved_audio = saved_audio
                        st.success("✅ Recording processed successfully!")
                except Exception as e:
                    st.error(f"❌ Error while processing: {e}")


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.transcription:
    st.divider()
    st.subheader("📝 Transcription / Recognized Speech")
    st.text_area("Recognized Speech", st.session_state.transcription, height=140)

    if selected_task_mode == "Conversational Language Practice" and st.session_state.chat_history:
        st.subheader("💬 Continuous Conversation Log")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        if st.session_state.is_question:
            st.subheader("💡 Assistant Output")
            if st.session_state.information_answer:
                st.markdown(st.session_state.information_answer)
            else:
                st.warning("⚠️ Output could not be generated.")
        else:
            st.info("ℹ️ Normal speech recorded. No assistant action triggered.")

    col_dl1, col_dl2, col_clr = st.columns(3)
    with col_dl1:
        st.download_button(label="💾 Download Text", data=st.session_state.transcription, file_name="transcription.txt", mime="text/plain", use_container_width=True)
    with col_dl2:
        if st.session_state.saved_audio:
            with open(st.session_state.saved_audio, "rb") as audio_file:
                st.download_button(label="🎵 Download Audio", data=audio_file, file_name=os.path.basename(st.session_state.saved_audio), mime="audio/wav", use_container_width=True)
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.transcription = ""
            st.session_state.saved_file = ""
            st.session_state.saved_audio = ""
            st.session_state.information_answer = ""
            st.session_state.is_question = False
            st.session_state.chat_history = []
            st.rerun()

st.divider()
st.caption("Speech-to-Text + Multi-Task AI Assistant | SeamlessM4T Small S2T + Phi-3")