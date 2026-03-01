"""
transcriber.py
--------------
Whisper ASR wrapper. Always uses a FIXED model version ("base") so that
all benchmark runs are directly comparable.
"""

import whisper
import os

WHISPER_MODEL_VERSION = "base"

_model = None  # load once, reuse across all calls


def _load_model():
    global _model
    if _model is None:
        print(f"[Whisper] Loading model: {WHISPER_MODEL_VERSION}")
        _model = whisper.load_model(WHISPER_MODEL_VERSION)
        print(f"[Whisper] Model loaded successfully.")
    return _model


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribe an audio file using Whisper.

    Parameters
    ----------
    audio_path : path to audio file (.mp3, .wav, .ogg, etc.)
    language   : ISO 639-1 language code hint e.g. 'hi', 'ta', 'en'
                 Providing this improves accuracy for non-English audio.

    Returns
    -------
    dict:
        text        : transcribed string
        language    : detected / provided language
        model_used  : always WHISPER_MODEL_VERSION
        audio_path  : echoed back for traceability
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model()

    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    return {
        "text"      : result["text"].strip(),
        "language"  : result.get("language", language or "unknown"),
        "model_used": WHISPER_MODEL_VERSION,
        "audio_path": audio_path,
    }
