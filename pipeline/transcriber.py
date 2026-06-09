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
