"""
run_evaluation.py
-----------------
MAIN ENTRY POINT — run this script to evaluate all TTS outputs.

Usage:
    python run_evaluation.py

What it does:
  - Reads each language .txt from dataset/
  - Finds matching audio in outputs/sarvam/<lang>/ and outputs/elevenlabs/<lang>/
  - Runs Whisper + WER + CER on each audio file
  - Saves every result row to results/results.csv
"""

import os
import sys
import glob

# Add repo root to path so `evaluate` package is importable
sys.path.insert(0, os.path.dirname(__file__))

from evaluate.evaluator import evaluate_single


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TOOLS = ["sarvam", "elevenlabs"]

LANGUAGE_WHISPER_MAP = {
    "english"  : "en",
    "hindi"    : "hi",
    "bengali"  : "bn",
    "gujarati" : "gu",
    "kannada"  : "kn",
    "malayalam": "ml",
    "tamil"    : "ta",
    "telugu"   : "te",
}

AUDIO_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def read_sentences(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [l for l in lines if l]


def find_audio_files(tool: str, language: str) -> list[str]:
    lang_dir = os.path.join(OUTPUTS_DIR, tool, language)
    if not os.path.isdir(lang_dir):
        return []

    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(lang_dir, f"*{ext}")))

    return sorted(files)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("  TTS Benchmark Evaluation Runner  (WER + CER)")
    print("=" * 60)

    txt_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    if not txt_files:
        print(f"\n  ERROR: No .txt files found in {DATASET_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(txt_files)} language dataset(s): "
          f"{[os.path.basename(f) for f in txt_files]}\n")

    all_results = []
    skipped     = []

    for txt_path in txt_files:
        lang_name    = os.path.splitext(os.path.basename(txt_path))[0].lower()
        whisper_lang = LANGUAGE_WHISPER_MAP.get(lang_name, None)
        sentences    = read_sentences(txt_path)

        print(f"{'─'*60}")
        print(f"  Language: {lang_name} ({len(sentences)} sentences)")

        for tool in TOOLS:
            audio_files = find_audio_files(tool, lang_name)

            if not audio_files:
                print(f"\n  [SKIP] No audio found for {tool}/{lang_name}")
                print(f"         Expected folder: outputs/{tool}/{lang_name}/")
                skipped.append(f"{tool}/{lang_name}")
                continue

            pairs = list(zip(sentences, audio_files))

            if len(sentences) != len(audio_files):
                print(f"\n  [WARN] {tool}/{lang_name}: "
                      f"{len(sentences)} sentences vs {len(audio_files)} audio files. "
                      f"Evaluating first {len(pairs)} pairs.")

            for idx, (sentence, audio_path) in enumerate(pairs, start=1):
                print(f"\n  [{tool.upper()} | {lang_name} | {idx}/{len(pairs)}]")
                try:
                    result = evaluate_single(
                        audio_path      =audio_path,
                        original_text   =sentence,
                        tool            =tool,
                        language        =lang_name,
                        whisper_language=whisper_lang,
                        persist         =True,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  [ERROR] {audio_path}: {e}")
                    skipped.append(audio_path)

    print(f"\n{'='*60}")
    print(f"  Evaluation complete.")
    print(f"  Total evaluated : {len(all_results)}")
    print(f"  Skipped / errors: {len(skipped)}")
    print(f"  Results saved to: results/results.csv")
    print(f"\n  Run  python report.py  to see the comparison report.")
    print("=" * 60)


if __name__ == "__main__":
    main()