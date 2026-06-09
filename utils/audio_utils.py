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
