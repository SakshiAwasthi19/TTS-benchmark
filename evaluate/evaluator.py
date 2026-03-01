"""
evaluator.py
------------
Orchestrates the full pipeline for a single audio file:

    audio_path + original_text
          ↓
    Whisper transcription
          ↓
    WER + CER
          ↓
    Save to results/results.csv
          ↓
    Return result dict
"""

import os
from evaluate.transcriber import transcribe_audio
from evaluate.metrics     import score_all
from evaluate.storage     import save_result


def evaluate_single(
    audio_path      : str,
    original_text   : str,
    tool            : str,
    language        : str,
    whisper_language: str  = None,
    groq_api_key    : str  = None,
    persist         : bool = True,
) -> dict:
    """
    Run the full eval pipeline for one audio file.

    Parameters
    ----------
    audio_path       : path to TTS-generated audio file
    original_text    : ground-truth text used to generate the audio
    tool             : "sarvam" or "elevenlabs"
    language         : e.g. "hindi"
    whisper_language : ISO 639-1 hint for Whisper e.g. "hi"
    groq_api_key     : optional, falls back to GROQ_API_KEY env var
    persist          : if True, saves to CSV (set False for quick tests)

    Returns
    -------
    Full result dict with all metrics + metadata
    """
    print(f"\n[Evaluator] tool={tool} | lang={language} | file={os.path.basename(audio_path)}")

    # Step 1 — Transcribe
    print("  [1/3] Transcribing with Whisper...")
    asr_result       = transcribe_audio(audio_path, language=whisper_language)
    transcribed_text = asr_result["text"]
    print(f"        Original   : {original_text[:80]}")
    print(f"        Transcribed: {transcribed_text[:80]}")

    # Step 2 — Score
    print("  [2/3] Calculating metrics...")
    scores = score_all(original_text, transcribed_text, api_key=groq_api_key)
    print(f"        WER={scores['wer']}  CER={scores['cer']}  "
          f"WordAcc={scores['word_accuracy']}%  CharAcc={scores['char_accuracy']}%")

    # Step 3 — Assemble result
    result = {
        "tool"            : tool,
        "language"        : language,
        "audio_file"      : os.path.basename(audio_path),
        "original_text"   : original_text,
        "transcribed_text": transcribed_text,
        "whisper_model"   : asr_result["model_used"],
        **scores,
    }

    # Step 4 — Persist
    if persist:
        print("  [3/3] Saving to CSV...")
        save_result(result)
    else:
        print("  [3/3] Skipped (persist=False)")

    return result
