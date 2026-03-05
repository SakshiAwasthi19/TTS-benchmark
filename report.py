"""
report.py
---------
Reads results/results.csv and prints a comparison report:

  1. Per-language breakdown — Sarvam vs ElevenLabs side by side
  2. Per-language WINNER — which tool is better based on WER + CER
  3. Overall winner
  4. Saves summary to results/summary_report.csv

Winner logic:
  Primary   → lower WER  (word level accuracy)
  Tiebreak  → lower CER  (character level accuracy)

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
    Group by (tool, language) and average all numeric metrics.
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
            m: round(sum(v) / len(v), 4) for m, v in metrics.items() if v
        }
        aggregated[key]["count"] = len(metrics.get("wer", []))

    return aggregated


# --------------------------------------------------------------------------- #
# Winner logic                                                                 #
# --------------------------------------------------------------------------- #

def determine_winners(agg: dict) -> list[dict]:
    """
    For each language, pick winner based on:
      1. Lower WER  (primary)
      2. Lower CER  (tiebreaker)
    """
    languages = sorted(set(lang for (_, lang) in agg.keys()))
    winners   = []

    for lang in languages:
        s  = agg.get(("sarvam",     lang), {})
        el = agg.get(("elevenlabs", lang), {})

        if not s or not el:
            continue

        s_wer  = s.get("wer",  1.0)
        el_wer = el.get("wer", 1.0)
        s_cer  = s.get("cer",  1.0)
        el_cer = el.get("cer", 1.0)

        s_wa  = s.get("word_accuracy",  0)
        el_wa = el.get("word_accuracy", 0)
        s_ca  = s.get("char_accuracy",  0)
        el_ca = el.get("char_accuracy", 0)

        # Primary: lower WER wins
        if s_wer < el_wer:
            winner = "sarvam"
            reason = f"Lower WER ({s_wer} vs {el_wer})"
        elif el_wer < s_wer:
            winner = "elevenlabs"
            reason = f"Lower WER ({el_wer} vs {s_wer})"
        # Tiebreak: lower CER wins
        elif s_cer < el_cer:
            winner = "sarvam"
            reason = f"Equal WER — lower CER ({s_cer} vs {el_cer})"
        elif el_cer < s_cer:
            winner = "elevenlabs"
            reason = f"Equal WER — lower CER ({el_cer} vs {s_cer})"
        else:
            winner = "tie"
            reason = "WER and CER are equal"

        winners.append({
            "language"        : lang,
            "winner"          : winner,
            "reason"          : reason,
            "sarvam_wer"      : s_wer,
            "elevenlabs_wer"  : el_wer,
            "sarvam_cer"      : s_cer,
            "elevenlabs_cer"  : el_cer,
            "sarvam_wordacc"  : s_wa,
            "elevenlabs_wordacc": el_wa,
            "sarvam_characc"  : s_ca,
            "elevenlabs_characc": el_ca,
        })

    return winners


# --------------------------------------------------------------------------- #
# Display                                                                      #
# --------------------------------------------------------------------------- #

def print_report(agg: dict) -> tuple:
    languages = sorted(set(lang for (_, lang) in agg.keys()))
    tools     = sorted(set(tool for (tool, _) in agg.keys()))

    print("\n" + "=" * 80)
    print("  TTS BENCHMARK — SARVAM vs ELEVENLABS")
    print("=" * 80)

    # Per-language metrics table
    for lang in languages:
        print(f"\n  Language: {lang.upper()}")
        print(f"  {'Tool':<14} {'WER':>8} {'CER':>8} {'WordAcc%':>10} {'CharAcc%':>10} {'N':>5}")
        print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*5}")

        for tool in tools:
            key = (tool, lang)
            if key not in agg:
                print(f"  {tool:<14} {'—':>8} {'—':>8} {'—':>10} {'—':>10} {'—':>5}")
                continue
            m = agg[key]
            print(
                f"  {tool:<14} "
                f"{m.get('wer', 'N/A'):>8} "
                f"{m.get('cer', 'N/A'):>8} "
                f"{m.get('word_accuracy', 'N/A'):>10} "
                f"{m.get('char_accuracy', 'N/A'):>10} "
                f"{m.get('count', 0):>5}"
            )

    # Per-language winner table
    winners = determine_winners(agg)

    print("\n" + "=" * 80)
    print("  PER-LANGUAGE WINNER  (based on WER → CER)")
    print("=" * 80)
    print(f"\n  {'Language':<12} {'Winner':<14} {'Reason'}")
    print(f"  {'-'*12} {'-'*14} {'-'*46}")

    sarvam_wins     = 0
    elevenlabs_wins = 0

    for w in winners:
        if w["winner"] == "sarvam":
            trophy = "🏆"
            sarvam_wins += 1
        elif w["winner"] == "elevenlabs":
            trophy = "🏆"
            elevenlabs_wins += 1
        else:
            trophy = "🤝"
        print(f"  {w['language']:<12} {w['winner']:<14} {trophy}  {w['reason']}")

    # Overall winner
    print("\n" + "=" * 80)
    print("  OVERALL WINNER")
    print("=" * 80)

    # Average WER across all languages per tool
    overall = defaultdict(lambda: defaultdict(list))
    for (tool, lang), m in agg.items():
        for metric in ["wer", "cer", "word_accuracy", "char_accuracy"]:
            if metric in m:
                overall[tool][metric].append(m[metric])

    print(f"\n  {'Tool':<14} {'Avg WER':>10} {'Avg CER':>10} {'Avg WordAcc%':>14} {'Avg CharAcc%':>14} {'Languages Won':>14}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*14} {'-'*14} {'-'*14}")

    summary_rows  = []
    best_tool     = None
    best_avg_wer  = 999

    for tool in sorted(overall.keys()):
        m    = overall[tool]
        awer = round(sum(m["wer"]) / len(m["wer"]), 4)                         if m["wer"]           else "N/A"
        acer = round(sum(m["cer"]) / len(m["cer"]), 4)                         if m["cer"]           else "N/A"
        awa  = round(sum(m["word_accuracy"]) / len(m["word_accuracy"]), 2)     if m["word_accuracy"] else "N/A"
        aca  = round(sum(m["char_accuracy"]) / len(m["char_accuracy"]), 2)     if m["char_accuracy"] else "N/A"
        wins = sarvam_wins if tool == "sarvam" else elevenlabs_wins

        print(f"  {tool:<14} {awer:>10} {acer:>10} {awa:>14} {aca:>14} {wins:>14}")

        summary_rows.append({
            "tool"            : tool,
            "avg_wer"         : awer,
            "avg_cer"         : acer,
            "avg_word_accuracy": awa,
            "avg_char_accuracy": aca,
            "languages_won"   : wins,
        })

        if isinstance(awer, float) and awer < best_avg_wer:
            best_avg_wer = awer
            best_tool    = tool

    if best_tool:
        print(f"\n  🏆 OVERALL BEST TOOL: {best_tool.upper()}"
              f"  (lower average WER across all languages)")

    return summary_rows, winners


# --------------------------------------------------------------------------- #
# Save CSV                                                                     #
# --------------------------------------------------------------------------- #

def save_summary_csv(agg: dict, summary_rows: list[dict], winners: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Per-language metrics
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

        # Per-language winners
        writer.writerow([])
        writer.writerow(["=== PER-LANGUAGE WINNER ==="])
        writer.writerow(["language", "winner", "reason",
                         "sarvam_wer", "elevenlabs_wer",
                         "sarvam_cer", "elevenlabs_cer",
                         "sarvam_wordacc%", "elevenlabs_wordacc%",
                         "sarvam_characc%", "elevenlabs_characc%"])
        for w in winners:
            writer.writerow([
                w["language"], w["winner"], w["reason"],
                w["sarvam_wer"],       w["elevenlabs_wer"],
                w["sarvam_cer"],       w["elevenlabs_cer"],
                w["sarvam_wordacc"],   w["elevenlabs_wordacc"],
                w["sarvam_characc"],   w["elevenlabs_characc"],
            ])

        # Overall averages
        writer.writerow([])
        writer.writerow(["=== OVERALL AVERAGES ==="])
        writer.writerow(["tool", "avg_wer", "avg_cer",
                         "avg_word_accuracy", "avg_char_accuracy", "languages_won"])
        for row in summary_rows:
            writer.writerow([
                row["tool"],       row["avg_wer"],
                row["avg_cer"],    row["avg_word_accuracy"],
                row["avg_char_accuracy"], row["languages_won"],
            ])

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

    agg                   = aggregate(results)
    summary_rows, winners = print_report(agg)
    save_summary_csv(agg, summary_rows, winners)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()