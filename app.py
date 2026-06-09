import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

load_dotenv()

from pipeline.transcriber import load_transcriber, transcribe_audio
from pipeline.llm_client   import (
    summarise_transcript, generate_mcq_quiz,
    generate_flashcards, chat_with_context
)
from pipeline.rag          import build_rag_index, retrieve_chunks
from utils.audio_utils     import convert_to_wav, save_uploaded_file, cleanup_temp_file

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Lecture Companion",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── UI Theme & CSS ────────────────────────────────────────────────────────────
current_is_dark = st.session_state.get("theme_toggle", True)
toggle_label = "☀️ Light Mode" if current_is_dark else "🌙 Dark Mode"
is_dark = st.sidebar.toggle(toggle_label, value=current_is_dark, key="theme_toggle")

if is_dark:
    bg_gradient = "linear-gradient(135deg, #0B101E 0%, #1B263B 100%)"
    sidebar_bg  = "rgba(11, 16, 30, 0.9)"
    card_bg     = "rgba(255, 255, 255, 0.03)"
    border_color= "rgba(255, 255, 255, 0.1)"
    text_color  = "#F8FAFC"
    accent_color= "#38BDF8"
    user_chat_bg= "rgba(56, 189, 248, 0.15)"
    bot_chat_bg = "rgba(255, 255, 255, 0.05)"
    box_shadow  = "0 8px 30px rgba(0, 0, 0, 0.4)"
    button_bg   = "linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)"
    button_text = "#FFFFFF"
    header_gradient = "-webkit-linear-gradient(45deg, #4facfe, #00f2fe)"
else:
    bg_gradient = "linear-gradient(135deg, #FDFBF7 0%, #F4F1EA 100%)"
    sidebar_bg  = "rgba(253, 251, 247, 0.9)"
    card_bg     = "#FFFFFF"
    border_color= "#E8E2D9"
    text_color  = "#2C2825"
    accent_color= "#D97757"
    user_chat_bg= "rgba(217, 119, 87, 0.1)"
    bot_chat_bg = "#FFFFFF"
    box_shadow  = "0 8px 30px rgba(139, 115, 85, 0.08)"
    button_bg   = "#D97757"
    button_text = "#FFFFFF"
    header_gradient = "-webkit-linear-gradient(45deg, #E76F51, #F4A261)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Hide Native Header */
    header {{ visibility: hidden !important; }}
    .stAppHeader {{ display: none !important; }}
    
    /* Main Background */
    .stApp {{
        background: {bg_gradient} !important;
    }}
    
    /* Typography override */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, label {{
        color: {text_color} !important;
    }}
    
    /* Streamlit Native Widget Overrides */
    div[data-baseweb="select"] > div,
    input[type="text"],
    input[type="number"],
    div[data-baseweb="popover"] > div {{
        background-color: {card_bg} !important;
        border-color: {border_color} !important;
        color: {text_color} !important;
    }}
    
    div[data-baseweb="select"] * {{
        color: {text_color} !important;
    }}
    
    /* File Uploader Fixes */
    section[data-testid="stFileUploader"] {{
        background-color: transparent !important;
    }}
    section[data-testid="stFileUploader"] div.stFileDropzone {{
        background-color: {card_bg} !important;
        border: 2px dashed {border_color} !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }}
    section[data-testid="stFileUploader"] div.stFileDropzone:hover {{
        background-color: {user_chat_bg} !important;
        border-color: {accent_color} !important;
    }}
    section[data-testid="stFileUploader"] div.stFileDropzone * {{
        color: {text_color} !important;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid {border_color} !important;
    }}
    
    /* Glassmorphism Cards & Expanders */
    .result-card, .stExpander, div[data-testid="stChatInput"] {{
        background: {card_bg} !important;
        backdrop-filter: blur(10px);
        border: 1px solid {border_color} !important;
        border-radius: 16px !important;
        box-shadow: {box_shadow};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    /* Premium Buttons */
    .stButton>button {{
        background: {button_bg} !important;
        border: none !important;
        color: {button_text} !important;
        border-radius: 30px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 500 !important;
        box-shadow: {box_shadow};
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px) scale(1.02);
        opacity: 0.9;
    }}
    
    /* Stage Headers */
    .stage-header {{ 
        font-size: 22px; 
        font-weight: 600; 
        margin: 2.5rem 0 1.5rem 0; 
        color: {accent_color} !important;
        border-bottom: 2px solid {accent_color};
        display: inline-block;
        padding-bottom: 8px;
        letter-spacing: 0.5px;
    }}
    
    /* Chat Bubbles */
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 20px;
    }}
    .chat-user {{ 
        background: {user_chat_bg}; 
        border-radius: 18px 18px 0 18px; 
        padding: 14px 20px; 
        max-width: 80%; 
        align-self: flex-end; 
        color: {text_color}; 
        border: 1px solid {border_color}; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }}
    .chat-bot {{ 
        background: {bot_chat_bg}; 
        border-radius: 18px 18px 18px 0; 
        padding: 14px 20px; 
        max-width: 80%; 
        align-self: flex-start; 
        color: {text_color}; 
        border: 1px solid {border_color}; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_transcriber(model_size: str):
    return load_transcriber(model_size)

@st.cache_resource(show_spinner=False)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ── Callbacks ─────────────────────────────────────────────────────────────────
def update_quiz():
    if "transcript" in st.session_state:
        st.toast("Regenerating Quiz...")
        st.session_state["quiz"] = generate_mcq_quiz(st.session_state["transcript"], st.session_state.num_questions_slider)
        st.toast("Quiz updated! 🎉")

def update_flashcards():
    if "transcript" in st.session_state:
        st.toast("Regenerating Flashcards...")
        st.session_state["flashcards"] = generate_flashcards(st.session_state["transcript"], st.session_state.num_flashcards_slider)
        st.toast("Flashcards updated! 🎉")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    whisper_size   = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=1,
        help="'base' is recommended for CPU. 'large-v3' is highest quality."
    )
    ollama_model   = st.text_input(
        "Ollama model",
        value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        help="Must be pulled first: ollama pull llama3.1:8b"
    )
    num_questions  = st.slider("Quiz questions",  3, 10, 5, key="num_questions_slider", on_change=update_quiz)
    num_flashcards = st.slider("Flashcards",      4, 15, 8, key="num_flashcards_slider", on_change=update_flashcards)
    show_transcript = st.toggle("Show full transcript", value=True)
    show_timestamps = st.toggle("Show timed segments",  value=False)

    st.divider()
    st.markdown("**Stack**")
    st.caption("🗣️ faster-whisper (CTranslate2)")
    st.caption("🧠 Ollama local LLM")
    st.caption("🔍 FAISS + all-MiniLM-L6-v2 (RAG)")
    st.caption("All inference local. No API key needed.")

    st.divider()
    with st.expander("Ollama setup commands"):
        st.code(
            "# Install Ollama\ncurl -fsSL https://ollama.com/install.sh | sh\n\n"
            "# Pull the model (~4.7 GB, once only)\nollama pull llama3.1:8b\n\n"
            "# Start the server\nollama serve",
            language="bash"
        )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align: center; margin-top: 1rem; margin-bottom: 2.5rem;'>
    <h1 style='font-size: 3.8rem; font-weight: 800; background: {header_gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>
        🎓 Smart Lecture Companion
    </h1>
    <p style='font-size: 1.2rem; opacity: 0.8; margin-top: 0; color: {text_color} !important;'>
        Upload a lecture → get a transcript, summary, MCQ quiz, flashcards, and a chatbot grounded in the lecture.
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()


# ── Stage 1: Input ────────────────────────────────────────────────────────────
st.markdown('<p class="stage-header">Stage 1 — Input</p>', unsafe_allow_html=True)

upload_tab, record_tab = st.tabs(["📁 Upload File", "🎙️ Record Audio"])
audio_path_raw = None

with upload_tab:
    uploaded = st.file_uploader(
        "Upload lecture audio or video",
        type=["mp3", "wav", "m4a", "ogg", "mp4", "mov", "webm"]
    )
    if uploaded:
        st.audio(uploaded)
        audio_path_raw = save_uploaded_file(uploaded)
        st.success(f"✅ Ready: {uploaded.name}")

with record_tab:
    try:
        from audiorecorder import audiorecorder
        audio_data = audiorecorder("🔴 Start", "⏹️ Stop")
        if len(audio_data) > 0:
            st.audio(audio_data.export().read())
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_data.export(tmp.name, format="wav")
            audio_path_raw = tmp.name
            st.success("✅ Recording saved.")
    except ImportError:
        st.info("Install audiorecorder to enable mic input: pip install audiorecorder")


# ── Stage 2: Process ──────────────────────────────────────────────────────────
st.markdown('<p class="stage-header">Stage 2 — Process</p>', unsafe_allow_html=True)

if audio_path_raw is None:
    st.info("👆 Upload or record a lecture above to continue.")
    st.stop()

os.environ["OLLAMA_MODEL"] = ollama_model

if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):

    for key in ["transcript", "segments", "language", "summary", "quiz", "flashcards", "rag_state", "chat_history"]:
        st.session_state.pop(key, None)

    progress = st.progress(0)
    status   = st.empty()
    wav_path = None

    try:
        # 1/4: Transcribe
        status.markdown("**1/4 — Transcribing audio with faster-whisper...**")
        progress.progress(5)
        with st.spinner("Loading Whisper model..."):
            transcriber = get_transcriber(whisper_size)
        wav_path = convert_to_wav(audio_path_raw)
        progress.progress(15)

        result = transcribe_audio(wav_path, transcriber)
        transcript = result["text"].strip()
        if not transcript:
            st.error("Transcription returned empty text. Check the audio file.")
            st.stop()

        st.session_state["transcript"] = transcript
        st.session_state["segments"]   = result["segments"]
        st.session_state["language"]   = result["language"]
        progress.progress(35)

        # 2/4: Summarise
        status.markdown("**2/4 — Generating structured summary via Ollama...**")
        summary = summarise_transcript(transcript)
        st.session_state["summary"] = summary
        progress.progress(55)

        # 3/4: Quiz + Flashcards
        status.markdown("**3/4 — Generating MCQ quiz and flashcards...**")
        quiz       = generate_mcq_quiz(summary, num_questions=num_questions)
        flashcards = generate_flashcards(summary, num_cards=num_flashcards)
        st.session_state["quiz"]       = quiz
        st.session_state["flashcards"] = flashcards
        progress.progress(75)

        # 4/4: Build RAG index
        status.markdown("**4/4 — Indexing transcript for chatbot (FAISS)...**")
        rag_state = build_rag_index(transcript)
        st.session_state["rag_state"]    = rag_state
        st.session_state["chat_history"] = []
        progress.progress(100)

        status.markdown("✅ **Done.** Scroll down to explore your results.")

    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        raise
    finally:
        if wav_path:
            cleanup_temp_file(wav_path)


# ── Stage 3: Results ──────────────────────────────────────────────────────────
if "transcript" not in st.session_state:
    st.stop()

st.markdown('<p class="stage-header">Stage 3 — Results</p>', unsafe_allow_html=True)

transcript = st.session_state["transcript"]
segments   = st.session_state.get("segments", [])
language   = st.session_state.get("language", "unknown")
summary    = st.session_state.get("summary", "")
quiz       = st.session_state.get("quiz", [])
flashcards = st.session_state.get("flashcards", [])
rag_state  = st.session_state.get("rag_state")

st.caption(f"Language: **{language.upper()}** | Words: **{len(transcript.split())}** | Segments: **{len(segments)}**")

tab_summary, tab_quiz, tab_flash, tab_chat, tab_transcript = st.tabs([
    "📝 Summary", "❓ Quiz", "🃏 Flashcards", "💬 Ask the Lecture", "📜 Transcript"
])

# Summary tab
with tab_summary:
    st.markdown(summary)

# Quiz tab
with tab_quiz:
    if not quiz:
        st.warning("No quiz generated. Try running the pipeline again.")
    else:
        score_key = "quiz_score"
        if score_key not in st.session_state:
            st.session_state[score_key] = {}

        for i, q in enumerate(quiz):
            with st.container():
                st.markdown(f"**Q{i+1}. {q['question']}**")
                user_answer = st.radio(
                    label=f"q{i+1}",
                    options=q["options"],
                    key=f"quiz_q{i}",
                    label_visibility="collapsed"
                )
                correct_letter = q["answer"]
                correct_index  = ord(correct_letter) - ord("A")
                correct_text   = q["options"][correct_index] if correct_index < len(q["options"]) else ""

                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("Check", key=f"check_q{i}"):
                        st.session_state[score_key][i] = (user_answer == correct_text)

                if i in st.session_state[score_key]:
                    if st.session_state[score_key][i]:
                        st.success(f"✅ Correct! {q['explanation']}")
                    else:
                        st.error(f"❌ Correct answer: **{correct_letter}. {correct_text}**\n\n{q['explanation']}")
                st.divider()

        answered = len(st.session_state.get(score_key, {}))
        correct  = sum(1 for v in st.session_state.get(score_key, {}).values() if v)
        if answered > 0:
            st.metric("Score so far", f"{correct}/{answered}")

# Flashcards tab
with tab_flash:
    if not flashcards:
        st.warning("No flashcards generated.")
    else:
        cols = st.columns(2)
        for i, card in enumerate(flashcards):
            with cols[i % 2]:
                with st.expander(f"🃏 {card['term']}", expanded=False):
                    st.markdown(card["definition"])

# RAG Chat tab
with tab_chat:
    st.caption("Ask questions about this lecture. Answers are grounded only in the transcript.")
    if rag_state is None:
        st.warning("Run the pipeline first to enable the chatbot.")
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for turn in st.session_state.get("chat_history", []):
            if turn["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {turn["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {turn["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        user_q = st.chat_input("Ask a question about the lecture...")
        if user_q:
            with st.spinner("Searching transcript and generating answer..."):
                relevant_chunks = retrieve_chunks(user_q, rag_state, top_k=6)
                answer = chat_with_context(
                    user_q,
                    relevant_chunks,
                    st.session_state.get("chat_history", [])
                )
            st.session_state["chat_history"].append({"role": "user",      "content": user_q})
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

# Transcript tab
with tab_transcript:
    if show_transcript:
        st.text_area("Full transcript", transcript, height=350, label_visibility="collapsed")
    if show_timestamps and segments:
        st.subheader("Timed segments")
        for seg in segments[:100]:
            mins  = int(seg["start"] // 60)
            secs  = int(seg["start"] % 60)
            st.markdown(f"`[{mins:02d}:{secs:02d}]` {seg['text']}")


# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
quiz_text = "\n\n".join(
    f"Q{i+1}. {q['question']}\n"
    + "\n".join([f"  {chr(65+j)}. {opt}" for j, opt in enumerate(q['options'])])
    + f"\nAnswer: {q['answer']}. {q['explanation']}"
    for i, q in enumerate(quiz)
)
flash_text = "\n".join(f"- {c['term']}: {c['definition']}" for c in flashcards)

download_txt = (
    f"SMART LECTURE COMPANION — RESULTS\n"
    f"Language: {language.upper()} | Words: {len(transcript.split())}\n\n"
    f"{'='*60}\nSUMMARY\n{'='*60}\n{summary}\n\n"
    f"{'='*60}\nQUIZ\n{'='*60}\n{quiz_text}\n\n"
    f"{'='*60}\nFLASHCARDS\n{'='*60}\n{flash_text}\n\n"
    f"{'='*60}\nFULL TRANSCRIPT\n{'='*60}\n{transcript}\n"
)

st.download_button(
    "⬇️ Download All Results (.txt)",
    data=download_txt,
    file_name="lecture_results.txt",
    mime="text/plain",
    use_container_width=True
)
