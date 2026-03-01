"""
metrics.py
----------
Three evaluation metrics for TTS benchmarking:

  1. WER  - Word Error Rate        (lexical, word level)
  2. CER  - Character Error Rate   (lexical, char level — better for Indic scripts)

Why metrics?
  WER/CER are purely lexical: "Please switch off" vs "Switch off" → WER penalises.
"""

import os
import re
import json
from jiwer import wer, cer


# --------------------------------------------------------------------------- #
# WER / CER                                                                   #
# --------------------------------------------------------------------------- #

def calculate_wer_cer(original: str, transcribed: str) -> dict:
    """
    Calculate WER and CER between original script and Whisper transcription.

    Returns dict:
        wer           : float 0-1  (lower is better)
        cer           : float 0-1  (lower is better)
        word_accuracy : float 0-100 (higher is better)
        char_accuracy : float 0-100 (higher is better)
    """
    orig_norm  = _normalise(original)
    trans_norm = _normalise(transcribed)

    word_error = min(wer(orig_norm, trans_norm), 1.0)
    char_error = min(cer(orig_norm, trans_norm), 1.0)

    return {
        "wer"          : round(word_error, 4),
        "cer"          : round(char_error, 4),
        "word_accuracy": round((1 - word_error) * 100, 2),
        "char_accuracy": round((1 - char_error) * 100, 2),
    }


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for fair comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# --------------------------------------------------------------------------- #
# Combined scorer — this is what evaluator.py calls                           #
# --------------------------------------------------------------------------- #

def score_all(original: str, transcribed: str, api_key: str = None) -> dict:
    """Run WER and CER and return a single merged dict."""
    return calculate_wer_cer(original, transcribed)
