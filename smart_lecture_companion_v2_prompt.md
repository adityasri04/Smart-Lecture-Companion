# Smart Lecture Companion v2 — Antigravity IDE Build Prompt (2026 Stack)

> Paste everything below this line directly into Antigravity IDE as your prompt.

---

## Project Brief

Build a complete, production-ready Python web application called **Smart Lecture Companion v2**. It is a modern, fully local AI pipeline that takes an uploaded or recorded lecture and produces:

1. **Transcript** — via `faster-whisper` (4× faster than original Whisper, same model weights, CTranslate2 backend)
2. **Structured Summary** — via a local LLM served through **Ollama** (no cloud API, no key required)
3. **MCQ Quiz with distractors** — same Ollama LLM, prompted to generate 4-choice questions with the correct answer flagged
4. **Flashcards** — same LLM, prompted to produce term → definition pairs
5. **RAG Chatbot** — ask questions about *this specific lecture* using FAISS vector store + Ollama, grounded only in the transcript (no hallucination from general knowledge)

The UI is **Streamlit**. All inference is local — Ollama must be running on the user's machine. The project is intentionally modelled after the open-source NotebookLM pattern: source-grounded QA plus multi-format study material generation.

---

## Exact Tech Stack

| Layer | Library / Tool | Notes |
|---|---|---|
| UI | `streamlit>=1.40.0` | Single-page app |
| Transcription | `faster-whisper>=1.0.0` | CTranslate2 backend, 4× faster than `openai-whisper` |
| LLM (all generation) | `ollama` Python client + local Ollama server | Default model: `llama3.1:8b`. Handles summary, quiz, flashcards, and chat |
| RAG embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Embed transcript chunks |
| RAG vector store | `faiss-cpu` | In-memory similarity search |
| Text splitting | `langchain-text-splitters` | Chunk transcript for RAG |
| Audio recording | `audiorecorder>=0.0.5` | In-browser mic recording |
| Audio conversion | `pydub>=0.25.1` + system `ffmpeg` | Format normalisation |
| Env management | `python-dotenv` | Optional config (Ollama host override) |

---

## File Structure to Create

```
smart_lecture_companion/
│
├── app.py                    # Main Streamlit app — all UI and orchestration
├── pipeline/
│   ├── __init__.py
│   ├── transcriber.py        # faster-whisper transcription
│   ├── llm_client.py         # Ollama wrapper (summary, quiz, flashcards, chat)
│   └── rag.py                # FAISS RAG pipeline for chatbot
├── utils/
│   ├── __init__.py
│   └── audio_utils.py        # Audio conversion and temp file helpers
├── requirements.txt
├── .env.example              # OLLAMA_HOST=http://localhost:11434
└── README.md
```

---

## Detailed Instructions for Each File

---

### `requirements.txt`

```
streamlit>=1.40.0
faster-whisper>=1.0.0
ollama>=0.2.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
langchain-text-splitters>=0.2.0
pydub>=0.25.1
audiorecorder>=0.0.5
python-dotenv>=1.0.0
numpy>=1.26.0
```

---

### `.env.example`

```
# Override if Ollama is running on a different host or port
OLLAMA_HOST=http://localhost:11434

# Whisper model size: tiny, base, small, medium, large-v3
# 'base' is good for CPU. 'large-v3' is best quality (needs 10GB+ RAM).
WHISPER_MODEL=base

# Ollama model to use. Must be pulled first: ollama pull llama3.1:8b
OLLAMA_MODEL=llama3.1:8b
```

---

### `pipeline/transcriber.py`

```python
"""
Audio transcription using faster-whisper.
faster-whisper uses CTranslate2 — it is 4x faster than openai-whisper
on CPU and uses ~50% less memory, with identical output quality.
Word-level timestamps are included for free (used by the RAG chunker).
"""
from faster_whisper import WhisperModel


def load_transcriber(model_size: str = "base", device: str = "cpu") -> WhisperModel:
    """
    Load a faster-whisper model.

    Args:
        model_size: "tiny", "base", "small", "medium", "large-v3"
                    Weights download once to ~/.cache/huggingface/hub/
        device:     "cpu" or "cuda" (if NVIDIA GPU available)

    Returns:
        A WhisperModel instance ready for transcription.
    """
    compute_type = "int8" if device == "cpu" else "float16"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(audio_path: str, model: WhisperModel) -> dict:
    """
    Transcribe an audio file with word-level timestamps.

    Returns a dict with:
      - 'text':     full transcript as a single string
      - 'segments': list of dicts {start, end, text} — one per sentence segment
      - 'language': auto-detected language code (e.g. 'en')
    """
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,           # built-in Voice Activity Detection: skip silence
        vad_parameters=dict(
            min_silence_duration_ms=500
        )
    )

    segments = []
    full_text_parts = []
    for seg in segments_gen:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip()
        })
        full_text_parts.append(seg.text.strip())

    return {
        "text":     " ".join(full_text_parts),
        "segments": segments,
        "language": info.language
    }
```

---

### `pipeline/llm_client.py`

```python
"""
All LLM generation via Ollama (local, no API key needed).
Handles: summary, MCQ quiz with distractors, flashcards, and single-turn chat.

Requires Ollama to be running: https://ollama.com
Pull the model before first use: ollama pull llama3.1:8b
"""
import ollama
import json
import re
import os


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _chat(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """
    Single-turn Ollama chat. Returns the assistant's response string.
    Raises a clear RuntimeError if Ollama is not running.
    """
    model = model or get_ollama_model()
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        if "Connection refused" in str(e) or "ConnectError" in str(e):
            raise RuntimeError(
                "Ollama is not running. Start it with: `ollama serve`\n"
                f"Then pull the model: `ollama pull {model}`"
            ) from e
        raise


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise_transcript(transcript: str) -> str:
    """
    Generate a structured summary of a lecture transcript.
    Output is markdown-formatted with three sections:
    Overview, Key Concepts, and Important Details.
    """
    system = (
        "You are an expert academic note-taker. "
        "You receive a raw lecture transcript and produce a clean, structured summary. "
        "Format your response in Markdown with three sections:\n"
        "## Overview\n(2-3 sentence high-level summary)\n\n"
        "## Key Concepts\n(bullet list of the main ideas, terms, and arguments)\n\n"
        "## Important Details\n(bullet list of specific facts, examples, or data mentioned)\n\n"
        "Be concise. Do not invent information not present in the transcript."
    )
    user = f"Lecture transcript:\n\n{transcript[:6000]}"
    return _chat(system, user)


# ── MCQ Quiz ──────────────────────────────────────────────────────────────────

def generate_mcq_quiz(summary: str, num_questions: int = 5) -> list[dict]:
    """
    Generate multiple-choice questions with 4 options and a correct answer flag.

    Returns a list of dicts, each with:
      - 'question':    str
      - 'options':     list of 4 strings (A, B, C, D)
      - 'answer':      str — correct option letter ('A', 'B', 'C', or 'D')
      - 'explanation': str — one-sentence explanation of the correct answer
    """
    system = (
        "You are a quiz writer for university-level students. "
        "Given a lecture summary, generate multiple-choice questions. "
        "Each question must have exactly 4 options labelled A, B, C, D. "
        "One option is correct; the other three are plausible but wrong distractors. "
        "Respond ONLY with a valid JSON array — no preamble, no markdown fences. "
        "Each element must have these exact keys: "
        '"question", "options" (array of 4 strings), "answer" (A/B/C/D), "explanation".'
    )
    user = (
        f"Generate exactly {num_questions} MCQ questions from this lecture summary.\n\n"
        f"{summary}"
    )

    raw = _chat(system, user)
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        questions = json.loads(raw.strip())
        validated = []
        for q in questions:
            if all(k in q for k in ("question", "options", "answer", "explanation")):
                if isinstance(q["options"], list) and len(q["options"]) == 4:
                    validated.append(q)
        return validated[:num_questions]
    except json.JSONDecodeError:
        return [{"question": "Could not parse quiz. Raw output:", "options": [raw[:200], "", "", ""], "answer": "A", "explanation": ""}]


# ── Flashcards ────────────────────────────────────────────────────────────────

def generate_flashcards(summary: str, num_cards: int = 8) -> list[dict]:
    """
    Generate term → definition flashcard pairs from a summary.
    Returns a list of dicts: [{"term": str, "definition": str}, ...]
    """
    system = (
        "You are a study aid generator. "
        "Given a lecture summary, identify the most important terms, concepts, or people. "
        "For each, write a clear one-sentence definition based only on the provided material. "
        "Respond ONLY with a valid JSON array. "
        'Each element must have exactly two keys: "term" and "definition". '
        "No markdown fences, no preamble."
    )
    user = f"Generate exactly {num_cards} flashcard pairs from this summary.\n\n{summary}"

    raw = _chat(system, user)
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        cards = json.loads(raw.strip())
        return [c for c in cards if "term" in c and "definition" in c][:num_cards]
    except json.JSONDecodeError:
        return [{"term": "Parse error", "definition": raw[:300]}]


# ── RAG Chat ──────────────────────────────────────────────────────────────────

def chat_with_context(question: str, context_chunks: list[str], chat_history: list[dict]) -> str:
    """
    Answer a question using ONLY the provided transcript context chunks.
    Called by the RAG pipeline after retrieval.

    Args:
        question:       The user's question string.
        context_chunks: Top-k retrieved text chunks from the transcript.
        chat_history:   Prior turns as [{"role": "user"|"assistant", "content": str}]

    Returns:
        The assistant's answer as a string.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are a helpful study assistant. "
        "You have access ONLY to the following excerpts from a lecture transcript. "
        "Answer the user's question using ONLY this material. "
        "If the answer is not in the excerpts, say: "
        "'I could not find that in the lecture. Could you rephrase or ask something else?'\n\n"
        f"LECTURE EXCERPTS:\n{context_text}"
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(chat_history[-6:])  # Keep last 3 turns for conversational context
    messages.append({"role": "user", "content": question})

    try:
        response = ollama.chat(model=get_ollama_model(), messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        if "Connection refused" in str(e):
            raise RuntimeError("Ollama is not running. Start it with: `ollama serve`") from e
        raise
```

---

### `pipeline/rag.py`

```python
"""
RAG (Retrieval-Augmented Generation) pipeline over the lecture transcript.
Uses sentence-transformers for embedding and FAISS for in-memory vector search.
This is the open-source equivalent of how NotebookLM grounds answers in source material.
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_rag_index(transcript: str) -> dict:
    """
    Build a FAISS vector index from a lecture transcript.
    Call this once after transcription; store the result in st.session_state.

    Steps:
      1. Split transcript into overlapping chunks (400 chars, 80 char overlap)
      2. Embed each chunk with all-MiniLM-L6-v2 (~90MB, downloads once)
      3. Store vectors in a FAISS flat L2 index (in memory)

    Returns a dict with:
      - 'index':   FAISS IndexFlatL2 object
      - 'chunks':  list of text strings (parallel to index vectors)
      - 'model':   SentenceTransformer instance (reused for query encoding)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(transcript)

    embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    embeddings  = embed_model.encode(chunks, show_progress_bar=False)
    embeddings  = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return {"index": index, "chunks": chunks, "model": embed_model}


def retrieve_chunks(question: str, rag_state: dict, top_k: int = 4) -> list[str]:
    """
    Retrieve the top_k most relevant transcript chunks for a question.

    Args:
        question:  The user's question string.
        rag_state: The dict returned by build_rag_index().
        top_k:     Number of chunks to return.

    Returns:
        List of text strings — the most relevant excerpts from the transcript.
    """
    query_vec = rag_state["model"].encode([question])
    query_vec = np.array(query_vec, dtype="float32")

    distances, indices = rag_state["index"].search(query_vec, top_k)
    return [rag_state["chunks"][i] for i in indices[0] if i < len(rag_state["chunks"])]
```

---

### `utils/audio_utils.py`

```python
"""
Audio file handling: format conversion and temp file management.
"""
import os
import tempfile
from pydub import AudioSegment


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to 16kHz mono WAV.
    faster-whisper works best on WAV; this handles MP3, M4A, MP4, OGG, etc.
    Returns path to a new temp WAV file.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return tmp.name


def save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to disk.
    Returns the temp file path.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".audio"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def cleanup_temp_file(path: str):
    """Silently delete a temp file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass
```

---

### `app.py` — Main Streamlit Application

Write the complete app with this exact code:

```python
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 900px; }
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #4A90D9;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 1rem 0;
    }
    .stage-header { font-size: 18px; font-weight: 600; margin: 1.5rem 0 0.5rem 0; }
    .chat-user { background: #e8f4fd; border-radius: 8px; padding: 8px 12px; margin: 4px 0; }
    .chat-bot  { background: #f0f0f0; border-radius: 8px; padding: 8px 12px; margin: 4px 0; }
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
    num_questions  = st.slider("Quiz questions",  3, 10, 5)
    num_flashcards = st.slider("Flashcards",      4, 15, 8)
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
st.title("🎓 Smart Lecture Companion")
st.caption("Upload a lecture → get a transcript, summary, MCQ quiz, flashcards, and a chatbot grounded in the lecture.")
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
        for turn in st.session_state.get("chat_history", []):
            if turn["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {turn["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {turn["content"]}</div>', unsafe_allow_html=True)

        user_q = st.chat_input("Ask a question about the lecture...")
        if user_q:
            with st.spinner("Searching transcript and generating answer..."):
                relevant_chunks = retrieve_chunks(user_q, rag_state, top_k=4)
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
```

---

### `README.md`

Write a README with the following sections:

**1. What it does** — one paragraph explaining this is a fully local, open-source study companion modelled after NotebookLM, producing a structured markdown summary, MCQ quiz with distractors, flashcards, and a RAG chatbot grounded in the transcript.

**2. Architecture**

```
Audio/Video File
      │
      ▼
faster-whisper (CTranslate2)
      │
      ▼
  Transcript
      │
  ┌───┴────────────────────────┐
  ▼                            ▼
Ollama LLM                  FAISS Index
(Summary, MCQ, Flashcards)  (all-MiniLM-L6-v2 embeddings)
  │                            │
  └─────────────┬──────────────┘
                ▼
          Streamlit UI
    [Summary | Quiz | Flashcards | Chat | Transcript]
```

**3. Prerequisites**

- Python 3.10+
- `ffmpeg` system binary (not just the Python package)
- Ollama (https://ollama.com)

ffmpeg install commands for macOS (`brew install ffmpeg`), Ubuntu (`sudo apt install ffmpeg`), and Windows (link to ffmpeg.org).

Ollama setup:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve
```

**4. Installation**
```bash
git clone <repo>
cd smart_lecture_companion
pip install -r requirements.txt
cp .env.example .env
```

**5. Running**
```bash
# Terminal 1 — keep running
ollama serve

# Terminal 2
streamlit run app.py
```

**6. First-run download sizes** table:

| Component | Download size | Cached at |
|---|---|---|
| faster-whisper base | ~145 MB | `~/.cache/huggingface/hub/` |
| llama3.1:8b (Ollama) | ~4.7 GB | `~/.ollama/models/` |
| all-MiniLM-L6-v2 | ~90 MB | `~/.cache/huggingface/hub/` |

**7. Whisper model options** table:

| Model | Size | CPU speed | Best for |
|---|---|---|---|
| tiny | 75 MB | Fastest | Quick tests |
| base | 145 MB | Fast | Default — good for CPU |
| small | 480 MB | Medium | Better accuracy |
| medium | 1.5 GB | Slow | High quality on CPU |
| large-v3 | 3 GB | Slowest | Best quality, prefer GPU |

**8. Alternative Ollama models** (update `OLLAMA_MODEL` in `.env`):
- `phi4` — Microsoft Phi-4, strong quality, smaller (~9 GB)
- `qwen2.5:7b` — fast, good multilingual support
- `mistral:7b` — fast summarisation
- `llama3.2:3b` — lightweight, runs on 8 GB RAM

**9. Troubleshooting**:
- `Connection refused` → Ollama is not running. Run `ollama serve`.
- `model not found` → Run `ollama pull llama3.1:8b`.
- `ffmpeg not found` → Install the system binary, not just the Python package.
- Out of memory on Whisper → Switch to `tiny` or `base`.
- Slow on CPU → Normal for 8B models. Expect 1-3 min per 1-hour lecture.
- RAG answers seem wrong → The chatbot only knows the transcript. It will say so if the answer is absent.

---

## Behaviour Requirements

1. **Single Ollama model for all generation** — do not load any HuggingFace models for summarisation or question generation. All LLM calls go through `ollama.chat()`.

2. **JSON output discipline** — `generate_mcq_quiz()` and `generate_flashcards()` must strip markdown fences from LLM output before parsing, validate the schema after parsing, and never raise an unhandled exception on malformed output.

3. **RAG is strictly grounded** — the system prompt for `chat_with_context()` must instruct the model to answer ONLY from the provided excerpts. If the answer is absent, the model must say so explicitly — not hallucinate.

4. **Session state persistence** — `transcript`, `summary`, `quiz`, `flashcards`, `rag_state`, and `chat_history` must all live in `st.session_state`. Nothing is recomputed on tab switches or slider changes.

5. **`@st.cache_resource` on model loaders** — `get_transcriber()` and `get_embed_model()` must use this decorator. The Ollama client is stateless (HTTP), so it does not need caching.

6. **Graceful Ollama error handling** — catch connection errors and show a clear `st.error()` with the `ollama serve` command. Never show a raw Python traceback to the user.

7. **Temp file cleanup** — always in a `finally` block. Convert to WAV in a temp file, process, then delete.

8. **Timed segments** — `faster-whisper` returns `start`/`end` per segment. Store them and render them in the Transcript tab when the toggle is on.

---

## What NOT to Do

- Do NOT use `openai-whisper` — use `faster-whisper` only.
- Do NOT use `facebook/bart-large-cnn` or `valhalla/t5-small-e2e-qg` — Ollama handles all generation.
- Do NOT call any external API (OpenAI, Anthropic, HuggingFace Inference, etc.) — everything is local.
- Do NOT use Gradio — Streamlit only.
- Do NOT hardcode file paths — use `tempfile` for all intermediate files.
- Do NOT call `build_rag_index()` on every rerun — build it once, store in `st.session_state`.
- Do NOT skip JSON schema validation in `generate_mcq_quiz()` and `generate_flashcards()`.

---

## Final Checklist Before Finishing

- [ ] `requirements.txt` includes every package imported across all `.py` files
- [ ] `pipeline/` and `utils/` both have `__init__.py`
- [ ] `load_dotenv()` is called at the top of `app.py` before any `os.getenv()` call
- [ ] `@st.cache_resource` is present on `get_transcriber()` and `get_embed_model()`
- [ ] The RAG system prompt says "answer ONLY from the provided excerpts"
- [ ] MCQ tab renders 4 radio options per question and reveals the correct answer and explanation on "Check"
- [ ] Flashcards render in a 2-column grid using `st.expander`
- [ ] Chat history renders above the `st.chat_input` box; `st.rerun()` is called after each answer
- [ ] Download button assembles summary + quiz + flashcards + transcript into one `.txt` file
- [ ] README includes Ollama setup, download size table, alternative models, and troubleshooting
