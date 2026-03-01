"""
storage.py
----------
Logs evaluation results to a single CSV file:
    results/results.csv

Columns:
    tool | language | audio_file | original_text | transcribed_text |
    wer  | cer      | word_accuracy | char_accuracy |
    whisper_model  | timestamp
"""

import csv
import os
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
CSV_PATH    = os.path.join(RESULTS_DIR, "results.csv")

CSV_COLUMNS = [
    "tool", "language", "audio_file",
    "original_text", "transcribed_text",
    "wer", "cer", "word_accuracy", "char_accuracy",
    "whisper_model", "timestamp",
]


def _ensure_csv():
    """Create results/ dir and write CSV header if file doesn't exist yet."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def save_result(result: dict):
    """
    Append one evaluation result row to results/results.csv.

    Expected keys (produced by evaluator.py):
        tool, language, audio_file, original_text, transcribed_text,
        wer, cer, word_accuracy, char_accuracy,
        semantic_score, semantic_reason, semantic_error, whisper_model
    """
    _ensure_csv()

    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerow(result)

    print(f"  [Storage] Saved → tool={result['tool']} | "
          f"lang={result['language']} | file={result.get('audio_file', '')}")


def fetch_all_results() -> list[dict]:
    """
    Read all rows from results.csv as list of dicts.
    Used by report.py. Returns empty list if file doesn't exist yet.
    """
    if not os.path.exists(CSV_PATH):
        return []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))