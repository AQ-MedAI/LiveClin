"""Analyze evaluation results and produce CLI summary + enriched JSON."""

from collections import defaultdict
from typing import Any, Dict, List


def analyze(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics from evaluated results and write them
    into results["summary"]. Returns the mutated results dict.
    """
    cases = results.get("cases", [])
    if not cases:
        results["summary"] = {}
        return results

    total_cases = len(cases)
    total_mcqs = 0
    correct_mcqs = 0
    correct_cases = 0

    rarity_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "correct_cases": 0, "mcqs": 0, "correct_mcqs": 0}
    )
    chapter_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "correct_cases": 0, "mcqs": 0, "correct_mcqs": 0}
    )

    for case in cases:
        mcqs = case.get("mcqs", [])
        n = len(mcqs)
        c = case.get("correct_mcqs", sum(1 for m in mcqs if m.get("is_correct")))
        total_mcqs += n
        correct_mcqs += c

        case_correct = (c == n and n > 0)
        if case_correct:
            correct_cases += 1

        rarity = str(case.get("rarity", "unknown")).lower()
        r = rarity_stats[rarity]
        r["cases"] += 1
        r["correct_cases"] += int(case_correct)
        r["mcqs"] += n
        r["correct_mcqs"] += c

        chapter = case.get("level1", "Unknown")
        ch = chapter_stats[chapter]
        ch["cases"] += 1
        ch["correct_cases"] += int(case_correct)
        ch["mcqs"] += n
        ch["correct_mcqs"] += c

    def _acc(correct: int, total: int) -> float:
        return round(correct / total, 4) if total > 0 else 0.0

    def _block(s: Dict[str, int]) -> Dict[str, Any]:
        return {
            "cases": s["cases"],
            "correct_cases": s["correct_cases"],
            "case_accuracy": _acc(s["correct_cases"], s["cases"]),
            "mcqs": s["mcqs"],
            "correct_mcqs": s["correct_mcqs"],
            "question_accuracy": _acc(s["correct_mcqs"], s["mcqs"]),
        }

    results["summary"] = {
        "total_cases": total_cases,
        "total_mcqs": total_mcqs,
        "correct_mcqs": correct_mcqs,
        "correct_cases": correct_cases,
        "question_accuracy": _acc(correct_mcqs, total_mcqs),
        "case_accuracy": _acc(correct_cases, total_cases),
        "by_rarity": {k: _block(v) for k, v in sorted(rarity_stats.items())},
        "by_chapter": {
            k: _block(v)
            for k, v in sorted(chapter_stats.items(), key=lambda x: -x[1]["mcqs"])
        },
    }

    return results


# ── CLI summary printer ─────────────────────────────────────────────────

def print_summary(results: Dict[str, Any]) -> None:
    """Print a concise evaluation summary to the terminal."""
    meta = results.get("meta", {})
    s = results.get("summary", {})

    if not s:
        print("No results to summarize.")
        return

    model = meta.get("model", "?")
    dataset = meta.get("dataset", "?")

    w = 60
    print()
    print("=" * w)
    print(f"  LiveClin Results: {model} ({dataset})")
    print("=" * w)

    total_q = s.get("total_mcqs", 0)
    correct_q = s.get("correct_mcqs", 0)
    total_c = s.get("total_cases", 0)
    correct_c = s.get("correct_cases", 0)
    q_acc = s.get("question_accuracy", 0)
    c_acc = s.get("case_accuracy", 0)

    print(f"  Question Accuracy:  {correct_q}/{total_q} ({q_acc:.1%})")
    print(f"  Case Accuracy:      {correct_c}/{total_c} ({c_acc:.1%})")
    print("-" * w)

    by_rarity = s.get("by_rarity", {})
    if by_rarity:
        print("  By Rarity:")
        for rarity, rs in by_rarity.items():
            label = rarity.capitalize()
            n = rs["cases"]
            qa = rs["question_accuracy"]
            ca = rs["case_accuracy"]
            print(f"    {label:<12} ({n:>4} cases)  Q-Acc {qa:.1%}  C-Acc {ca:.1%}")
        print("-" * w)

    by_chapter = s.get("by_chapter", {})
    if by_chapter:
        print("  By Chapter (top 5):")
        for i, (ch, cs) in enumerate(by_chapter.items()):
            if i >= 5:
                break
            short = ch[:40] + ("..." if len(ch) > 40 else "")
            n = cs["cases"]
            qa = cs["question_accuracy"]
            print(f"    {short:<44} ({n:>3}) Q-Acc {qa:.1%}")
        if len(by_chapter) > 5:
            print(f"    ... and {len(by_chapter) - 5} more chapters")
        print("-" * w)

    print(f"  Results: {meta.get('image_mode', '?')} mode | "
          f"v{meta.get('liveclin_version', '?')}")
    print("=" * w)
    print()
