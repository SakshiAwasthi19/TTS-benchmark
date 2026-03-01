"""
metrics.py
----------
Three evaluation metrics for TTS benchmarking:

  1. WER  - Word Error Rate        (lexical, word level)
  2. CER  - Character Error Rate   (lexical, char level — better for Indic scripts)
  3. Semantic Score                (Groq LLM, 0-100 meaning similarity)

Why three metrics?
  WER/CER are purely lexical: "Please switch off" vs "Switch off" → WER penalises.
  Semantic score catches meaning-preserving differences that WER would mark wrong.
"""

import os
import re
import json
from jiwer import wer, cer
from groq import Groq


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
# Semantic Similarity via Groq                                                #
# --------------------------------------------------------------------------- #

SEMANTIC_PROMPT_TEMPLATE = """You are an expert linguist evaluating speech recognition quality.

Compare the ORIGINAL text and the TRANSCRIBED text below.

ORIGINAL:
{original}

TRANSCRIBED:
{transcribed}

Your task:
Rate the SEMANTIC SIMILARITY on a scale of 0 to 100, where:
  100 = Identical meaning, all key information preserved
   80 = Mostly correct, minor differences that don't affect meaning
   60 = Core meaning preserved but some important details missing or changed
   40 = Partial meaning preserved, significant information lost
   20 = Mostly incorrect, meaning significantly altered
    0 = Completely different meaning or unintelligible

Rules:
- Focus on MEANING, not exact word matching
- "Please switch off the fan" vs "Switch off the fan" should score ~95 (same intent)
- Ignore minor filler words unless they change meaning
- For non-English text, evaluate meaning in context of that language

Respond with ONLY a JSON object in this exact format, no extra text:
{{"score": <integer 0-100>, "reason": "<one sentence explanation>"}}"""


def calculate_semantic_score(original: str, transcribed: str, api_key: str = None) -> dict:
    """
    Use Groq LLM to rate semantic similarity between original and transcribed text.

    Parameters
    ----------
    original    : ground-truth script text
    transcribed : Whisper output
    api_key     : Groq API key (falls back to GROQ_API_KEY env var)

    Returns dict:
        semantic_score  : int 0-100  (or None if skipped)
        semantic_reason : str explanation
        semantic_error  : str or None
    """
    key = api_key or os.getenv("GROQ_API_KEY")

    if not key:
        return {
            "semantic_score" : None,
            "semantic_reason": "Skipped",
            "semantic_error" : "GROQ_API_KEY not set",
        }

    client = Groq(api_key=key)
    prompt = SEMANTIC_PROMPT_TEMPLATE.format(
        original   =original.strip(),
        transcribed=transcribed.strip(),
    )

    try:
        response = client.chat.completions.create(
            model      ="llama3-8b-8192",   # free Groq model, fast and accurate
            messages   =[{"role": "user", "content": prompt}],
            temperature=0,                  # deterministic scoring
            max_tokens =100,
        )
        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        return {
            "semantic_score" : int(parsed["score"]),
            "semantic_reason": parsed.get("reason", ""),
            "semantic_error" : None,
        }

    except Exception as e:
        return {
            "semantic_score" : None,
            "semantic_reason": "Error during Groq call",
            "semantic_error" : str(e),
        }


# --------------------------------------------------------------------------- #
# Combined scorer — this is what evaluator.py calls                           #
# --------------------------------------------------------------------------- #

def score_all(original: str, transcribed: str, api_key: str = None) -> dict:
    """Run all three metrics and return a single merged dict."""
    wer_cer  = calculate_wer_cer(original, transcribed)
    semantic = calculate_semantic_score(original, transcribed, api_key)
    return {**wer_cer, **semantic}
