"""
report.py
---------
Reads results/results.csv and prints a comparison report:

  1. Per-language breakdown — Sarvam vs ElevenLabs side by side
  2. Overall winner table
  3. Saves summary to results/summary_report.csv

Usage:
    python report.py
"""

import os
import sys
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from evaluate.storage import fetch_all_results

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary_report.csv")


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def aggregate(results: list[dict]) -> dict:
    """
    Group results by (tool, language) and average all numeric metrics.
    Returns: { (tool, language): { metric: avg_value, count: n } }
    """
    buckets = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r["tool"], r["language"])
        for metric in ["wer", "cer", "word_accuracy", "char_accuracy"]:
            val = r.get(metric)
            if val not in (None, "", "None"):
                try:
                    buckets[key][metric].append(float(val))
                except (TypeError, ValueError):
                    pass

    aggregated = {}
    for key, metrics in buckets.items():
        aggregated[key] = {
            m: round(sum(v) / len(v), 2) for m, v in metrics.items() if v
        }
        aggregated[key]["count"] = len(metrics.get("wer", metrics.get("cer", [])))

    return aggregated


# --------------------------------------------------------------------------- #
# Display                                                                      #
# --------------------------------------------------------------------------- #

def print_report(agg: dict):
    languages = sorted(set(lang for (_, lang) in agg.keys()))
    tools     = sorted(set(tool for (tool, _) in agg.keys()))

    print("\n" + "=" * 80)
    print("  TTS BENCHMARK — SARVAM vs ELEVENLABS")
    print("=" * 80)

    for lang in languages:
        print(f"\n  Language: {lang.upper()}")
        print(f"  {'Tool':<14} {'WER':>6} {'CER':>6} {'WordAcc%':>10} {'CharAcc%':>10} {'N':>4}")
        print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*4}")

        for tool in tools:
            key = (tool, lang)
            if key not in agg:
                print(f"  {tool:<14} {'—':>6} {'—':>6} {'—':>10} {'—':>10} {'—':>4}")
                continue
            m   = agg[key]
            print(
                f"  {tool:<14} "
                f"{m.get('wer', 'N/A'):>6} "
                f"{m.get('cer', 'N/A'):>6} "
                f"{m.get('word_accuracy', 'N/A'):>10} "
                f"{m.get('char_accuracy', 'N/A'):>10} "
                f"{m.get('count', 0):>4}"
            )

    # Overall averages
    print("\n" + "=" * 80)
    print("  OVERALL WINNER (averaged across all languages)")
    print("=" * 80)

    overall = defaultdict(lambda: defaultdict(list))
    for (tool, lang), m in agg.items():
        for metric in ["word_accuracy", "char_accuracy"]:
            if metric in m:
                overall[tool][metric].append(m[metric])

    print(f"\n  {'Tool':<14} {'WordAcc%':>10} {'CharAcc%':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")

    summary_rows = []
    for tool in sorted(overall.keys()):
        m   = overall[tool]
        wa  = round(sum(m["word_accuracy"])  / len(m["word_accuracy"]),  2) if m["word_accuracy"]  else "N/A"
        ca  = round(sum(m["char_accuracy"])  / len(m["char_accuracy"]),  2) if m["char_accuracy"]  else "N/A"
        print(f"  {tool:<14} {wa:>10} {ca:>10}")
        summary_rows.append({
            "tool"              : tool,
            "avg_word_accuracy" : wa,
            "avg_char_accuracy" : ca,
        })

    return summary_rows


def save_summary_csv(agg: dict, summary_rows: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["=== PER-LANGUAGE BREAKDOWN ==="])
        writer.writerow(["tool", "language", "wer", "cer",
                         "word_accuracy", "char_accuracy", "count"])

        for (tool, lang), m in sorted(agg.items()):
            writer.writerow([
                tool, lang,
                m.get("wer", ""),           m.get("cer", ""),
                m.get("word_accuracy", ""), m.get("char_accuracy", ""),
                m.get("count", ""),
            ])

        writer.writerow([])
        writer.writerow(["=== OVERALL AVERAGES ==="])
        writer.writerow(["tool", "avg_word_accuracy", "avg_char_accuracy"])
        for row in summary_rows:
            writer.writerow([row["tool"], row["avg_word_accuracy"],
                             row["avg_char_accuracy"]])

    print(f"\n  Summary saved → {SUMMARY_PATH}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    results = fetch_all_results()

    if not results:
        print("\n  No results found. Run  python run_evaluation.py  first.")
        sys.exit(0)

    print(f"\n  Loaded {len(results)} evaluation records from results.csv")

    agg          = aggregate(results)
    summary_rows = print_report(agg)
    save_summary_csv(agg, summary_rows)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()