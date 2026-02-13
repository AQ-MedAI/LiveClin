"""
stats_analyzer.py (Open-source friendly version)

Changes vs. original:
- Translated Chinese comments/prompts to English.
- Removed emoji from user-facing output (optional; keep if you want).
- Added argparse support for non-interactive usage (open-source friendly).
- No hard-coded absolute paths or secrets.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
from datetime import datetime
import argparse

TOPNUM = 5  # Number of hardest/easiest cases to display/copy


class EvaluationStatistics:
    def __init__(self, folder_path: str, ignored_models: Optional[List[str]] = None):
        """
        Args:
            folder_path: Folder containing evaluated JSON files.
            ignored_models: Optional list of model_ids to exclude from statistics.
        """
        self.folder_path = Path(folder_path)
        self.ignored_models_set: Set[str] = set(ignored_models) if ignored_models else set()

        if self.ignored_models_set:
            print(f"Info: Ignoring models in statistics: {', '.join(sorted(self.ignored_models_set))}")

        self.json_files = sorted([f for f in self.folder_path.glob("*.json") if f.is_file()])

        self.statistics: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "folder_path": str(self.folder_path),
            "ignored_models": sorted(list(self.ignored_models_set)),
            "total_files": 0,
            "total_questions": 0,
            "models": {},
            "case_accuracy_distribution": {},
            "case_details": [],
            "hardest_cases": [],
            "easiest_cases": [],
            "discrimination_analysis": {},
            "overall_case_average_accuracy": 0.0,
            "average_model_case_accuracy": 0.0,
            "rarity_clusters": {},
        }

        self.case_scores: List[Tuple[float, Dict[str, Any]]] = []
        self.report_lines: List[str] = []

    def calculate_discrimination_metrics(self, file_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Discrimination metrics are computed from per-model accuracies for this file.
        Note: file_result['model_results'] already excludes ignored models.
        """
        model_accuracies = [r["accuracy"] for _, r in file_result["model_results"].items()]

        if len(model_accuracies) < 2:
            return {
                "std_dev": 0.0,
                "range": 0.0,
                "iqr": 0.0,
                "cv": 0.0,
                "discrimination_score": 0.0,
                "model_count": len(model_accuracies),
            }

        std_dev = float(np.std(model_accuracies))
        range_val = float(max(model_accuracies) - min(model_accuracies))
        q75, q25 = np.percentile(model_accuracies, [75, 25])
        iqr = float(q75 - q25)
        mean_acc = float(np.mean(model_accuracies))
        cv = float(std_dev / mean_acc) if mean_acc > 0 else 0.0

        discrimination_score = (
            0.4 * min(std_dev / 0.5, 1.0) +
            0.3 * min(range_val, 1.0) +
            0.3 * min(iqr / 0.5, 1.0)
        )

        return {
            "std_dev": std_dev,
            "range": range_val,
            "iqr": iqr,
            "cv": cv,
            "discrimination_score": float(discrimination_score),
            "model_count": len(model_accuracies),
        }

    def analyze_single_file(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Parse and compute per-file statistics."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path.name}: {e}")
            return None

        final_policy = data.get("exam_creation", {}).get("final_policy", {})
        mcqs = final_policy.get("mcqs", [])
        if not mcqs:
            return None

        file_result: Dict[str, Any] = {
            "filename": json_path.name,
            "path": str(json_path),
            "total_questions": len(mcqs),
            "rarity": data.get("Rarity", "Unknown"),
            "model_results": {},
            "model_case_correct": {},
            "average_accuracy": 0.0,
            "scenario": (final_policy.get("scenario", "")[:100] + "...") if final_policy.get("scenario") else "",
        }

        model_correct_counts = defaultdict(int)
        model_total_counts = defaultdict(int)

        for question_idx, mcq in enumerate(mcqs):
            evaluations = mcq.get("evaluate", [])
            for eval_item in evaluations:
                model_id = eval_item.get("model_id")
                if not model_id:
                    continue

                # Ignore selected models
                if model_id in self.ignored_models_set:
                    continue

                # If raw response is missing, treat it as a critical data integrity error
                model_raw_response = eval_item.get("model_raw_response")
                if model_raw_response is None:
                    error_msg = (
                        "ERROR: model_raw_response is null!\n"
                        f"  File: {json_path.name}\n"
                        f"  Question #{question_idx + 1}\n"
                        f"  Model: {model_id}"
                    )
                    raise ValueError(error_msg)

                model_total_counts[model_id] += 1
                if eval_item.get("is_correct", False):
                    model_correct_counts[model_id] += 1

        total_accuracy_sum = 0.0
        model_count = 0

        for model_id, total in model_total_counts.items():
            if total <= 0:
                continue
            correct = model_correct_counts[model_id]
            accuracy = correct / total
            file_result["model_results"][model_id] = {
                "correct": int(correct),
                "total": int(total),
                "accuracy": float(accuracy),
            }
            file_result["model_case_correct"][model_id] = (correct == total)
            total_accuracy_sum += accuracy
            model_count += 1

        if model_count > 0:
            file_result["average_accuracy"] = float(total_accuracy_sum / model_count)

        file_result["discrimination"] = self.calculate_discrimination_metrics(file_result)
        return file_result

    def calculate_overall_discrimination(self, all_file_results: List[Dict[str, Any]]):
        """Compute overall discrimination stats across files."""
        discrimination_stats: Dict[str, Any] = {
            "average_metrics": {"avg_std_dev": 0.0, "avg_iqr": 0.0, "avg_cv": 0.0},
            "high_discrimination_cases": 0,
            "medium_discrimination_cases": 0,
            "low_discrimination_cases": 0,
            "discrimination_distribution": {},
            "top_discriminating_cases": [],
        }

        scores: List[float] = []
        std_devs: List[float] = []
        iqrs: List[float] = []
        cvs: List[float] = []
        case_discriminations: List[Tuple[float, Dict[str, Any]]] = []

        for result in all_file_results:
            disc = result.get("discrimination", {})
            score = float(disc.get("discrimination_score", 0.0))
            scores.append(score)
            std_devs.append(float(disc.get("std_dev", 0.0)))
            iqrs.append(float(disc.get("iqr", 0.0)))
            cvs.append(float(disc.get("cv", 0.0)))
            case_discriminations.append((score, result))

            if score > 0.7:
                discrimination_stats["high_discrimination_cases"] += 1
            elif score > 0.3:
                discrimination_stats["medium_discrimination_cases"] += 1
            else:
                discrimination_stats["low_discrimination_cases"] += 1

        if scores:
            discrimination_stats["average_metrics"]["avg_std_dev"] = float(np.mean(std_devs))
            discrimination_stats["average_metrics"]["avg_iqr"] = float(np.mean(iqrs))
            discrimination_stats["average_metrics"]["avg_cv"] = float(np.mean(cvs))

            discrimination_stats["discrimination_distribution"] = {
                "high": {
                    "count": discrimination_stats["high_discrimination_cases"],
                    "percentage": discrimination_stats["high_discrimination_cases"] / len(scores) * 100,
                },
                "medium": {
                    "count": discrimination_stats["medium_discrimination_cases"],
                    "percentage": discrimination_stats["medium_discrimination_cases"] / len(scores) * 100,
                },
                "low": {
                    "count": discrimination_stats["low_discrimination_cases"],
                    "percentage": discrimination_stats["low_discrimination_cases"] / len(scores) * 100,
                },
            }

            case_discriminations.sort(key=lambda x: x[0], reverse=True)
            discrimination_stats["top_discriminating_cases"] = [
                {
                    "filename": case[1]["filename"],
                    "std_dev": case[1]["discrimination"]["std_dev"],
                    "iqr": case[1]["discrimination"]["iqr"],
                    "cv": case[1]["discrimination"]["cv"],
                }
                for case in case_discriminations[:5]
            ]

        self.statistics["discrimination_analysis"] = discrimination_stats

    def calculate_accuracy_distribution(self):
        """Build a histogram-like distribution over case average accuracies."""
        if not self.case_scores:
            return

        accuracies = [score for score, _ in self.case_scores]
        bins = [
            (0.8, 1.0, "80%-100%"),
            (0.6, 0.8, "60%-80%"),
            (0.4, 0.6, "40%-60%"),
            (0.2, 0.4, "20%-40%"),
            (0.0, 0.2, "0%-20%"),
        ]

        distribution: Dict[str, Any] = {}
        cumulative_count = 0
        total_count = len(accuracies)
        if total_count == 0:
            return

        for min_acc, max_acc, label in bins:
            count = sum(1 for acc in accuracies if min_acc <= acc < max_acc)
            if max_acc == 1.0:
                count += sum(1 for acc in accuracies if acc == 1.0)

            cumulative_count += count
            distribution[label] = {
                "count": int(count),
                "percentage": (count / total_count * 100),
                "cumulative_count": int(cumulative_count),
                "cumulative_percentage": (cumulative_count / total_count * 100),
            }

        self.statistics["case_accuracy_distribution"] = distribution

    def identify_extreme_cases(self):
        """Identify hardest/easiest cases by average accuracy."""
        hardest = self.case_scores[:TOPNUM] if len(self.case_scores) >= TOPNUM else self.case_scores
        self.statistics["hardest_cases"] = [
            {"filename": r["filename"], "average_accuracy": score, "scenario_preview": r.get("scenario", "")}
            for score, r in hardest
        ]

        easiest = self.case_scores[-TOPNUM:] if len(self.case_scores) >= TOPNUM else self.case_scores
        self.statistics["easiest_cases"] = [
            {"filename": r["filename"], "average_accuracy": score, "scenario_preview": r.get("scenario", "")}
            for score, r in reversed(easiest)
        ]

    def calculate_statistics(self):
        """Main statistics computation pipeline."""
        print("Analyzing evaluation results...")
        all_file_results: List[Dict[str, Any]] = []
        error_count = 0

        for json_file in self.json_files:
            try:
                result = self.analyze_single_file(json_file)
                if result:
                    all_file_results.append(result)
                    self.case_scores.append((result["average_accuracy"], result))
            except ValueError as e:
                error_count += 1
                print(f"\nSkipping file due to data error: {json_file.name}")
                print(str(e))

        if error_count > 0:
            print(f"\nFiles with data errors: {error_count}")

        self.statistics["total_files"] = len(all_file_results)
        if not all_file_results:
            print("No valid evaluation results found.")
            return

        # Aggregate per-model stats across files
        model_stats = defaultdict(lambda: {
            "total_questions": 0,
            "correct_questions": 0,
            "total_cases": 0,
            "correct_cases": 0,
            "question_accuracy": 0.0,
            "case_accuracy": 0.0,
        })

        for file_result in all_file_results:
            for model_id, model_result in file_result["model_results"].items():
                stats = model_stats[model_id]
                stats["total_questions"] += int(model_result["total"])
                stats["correct_questions"] += int(model_result["correct"])
                stats["total_cases"] += 1
                if file_result["model_case_correct"].get(model_id, False):
                    stats["correct_cases"] += 1

        for model_id, stats in model_stats.items():
            if stats["total_questions"] > 0:
                stats["question_accuracy"] = stats["correct_questions"] / stats["total_questions"]
            if stats["total_cases"] > 0:
                stats["case_accuracy"] = stats["correct_cases"] / stats["total_cases"]

        sorted_models = dict(sorted(model_stats.items(), key=lambda x: x[1]["question_accuracy"], reverse=True))
        self.statistics["models"] = sorted_models

        # Average number of questions per file (computed from model totals; assumes all models answered same questions)
        self.statistics["total_questions"] = (
            sum(stats["total_questions"] for stats in model_stats.values()) // len(model_stats)
            if model_stats else 0
        )

        # Discrimination and case-level details
        self.calculate_overall_discrimination(all_file_results)

        self.case_scores.sort(key=lambda x: x[0])
        self.statistics["case_details"] = [
            {
                "filename": r["filename"],
                "average_accuracy": score,
                "model_results": dict(r["model_results"]),
                "discrimination": r.get("discrimination", {}),
            }
            for score, r in self.case_scores
        ]

        self.statistics["overall_case_average_accuracy"] = float(
            np.mean([score for score, _ in self.case_scores]) if self.case_scores else 0.0
        )

        all_models_case_accuracies = [
            stats["case_accuracy"] for _, stats in model_stats.items() if stats["total_cases"] > 0
        ]
        self.statistics["average_model_case_accuracy"] = float(
            np.mean(all_models_case_accuracies) if all_models_case_accuracies else 0.0
        )

        self.calculate_accuracy_distribution()
        self.identify_extreme_cases()

        # Rarity cluster stats (merging rarity 0 and 1 into "0&1")
        rarity_stats = defaultdict(lambda: {
            "case_count": 0,
            "models": defaultdict(lambda: {
                "total_questions": 0,
                "correct_questions": 0,
                "total_cases": 0,
                "correct_cases": 0,
                "question_accuracy": 0.0,
                "case_accuracy": 0.0,
            }),
        })

        for file_result in all_file_results:
            rarity = str(file_result.get("rarity", "Unknown"))
            if rarity in ("0", "1"):
                rarity = "0&1"

            rarity_stats[rarity]["case_count"] += 1
            for model_id, model_result in file_result["model_results"].items():
                stats = rarity_stats[rarity]["models"][model_id]
                stats["total_questions"] += int(model_result["total"])
                stats["correct_questions"] += int(model_result["correct"])
                stats["total_cases"] += 1
                if file_result["model_case_correct"].get(model_id, False):
                    stats["correct_cases"] += 1

        for rarity, data in rarity_stats.items():
            for model_id, stats in data["models"].items():
                if stats["total_questions"] > 0:
                    stats["question_accuracy"] = stats["correct_questions"] / stats["total_questions"]
                if stats["total_cases"] > 0:
                    stats["case_accuracy"] = stats["correct_cases"] / stats["total_cases"]

        # Convert defaultdicts to normal dicts for JSON serialization friendliness
        self.statistics["rarity_clusters"] = {
            rarity: {
                "case_count": int(data["case_count"]),
                "models": dict(data["models"]),
            }
            for rarity, data in rarity_stats.items()
        }

    # =============================================================================
    # Output helpers
    # =============================================================================

    def copy_extreme_cases(self):
        """Copy hardest/easiest JSON files into ./hard and ./easy subfolders."""
        hard_dir = self.folder_path / "hard"
        easy_dir = self.folder_path / "easy"
        hard_dir.mkdir(exist_ok=True)
        easy_dir.mkdir(exist_ok=True)

        print(f"\nCopying {len(self.statistics['hardest_cases'])} hardest cases to 'hard'...")
        for case in self.statistics["hardest_cases"]:
            src = self.folder_path / case["filename"]
            dst = hard_dir / case["filename"]
            try:
                shutil.copy2(src, dst)
                print(f"  Copied: {case['filename']} (accuracy: {case['average_accuracy']:.2%})")
            except Exception as e:
                print(f"  Failed to copy {case['filename']}: {e}")

        print(f"\nCopying {len(self.statistics['easiest_cases'])} easiest cases to 'easy'...")
        for case in self.statistics["easiest_cases"]:
            src = self.folder_path / case["filename"]
            dst = easy_dir / case["filename"]
            try:
                shutil.copy2(src, dst)
                print(f"  Copied: {case['filename']} (accuracy: {case['average_accuracy']:.2%})")
            except Exception as e:
                print(f"  Failed to copy {case['filename']}: {e}")

    def save_statistics(self):
        """Write JSON stats report to 0.statistics.json."""
        output_file = self.folder_path / "0.statistics.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.statistics, f, indent=2, ensure_ascii=False)
            print(f"\nStatistics saved to: {output_file}")
        except Exception as e:
            print(f"\nFailed to save statistics: {e}")

    def print_and_log(self, text: str):
        print(text)
        self.report_lines.append(text)

    def save_report(self):
        """Write human-readable report to 0.evaluation_report.txt."""
        report_file = self.folder_path / "0.evaluation_report.txt"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.report_lines))
            print(f"Evaluation report saved to: {report_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")

    def print_summary(self):
        """Print and store a human-readable summary."""
        self.report_lines = []

        self.print_and_log("\n" + "=" * 60)
        self.print_and_log("EVALUATION STATISTICS SUMMARY")
        self.print_and_log("=" * 60)
        self.print_and_log(f"\nTotal files analyzed: {self.statistics['total_files']}")
        self.print_and_log(f"Total questions: {self.statistics['total_questions']}")

        if self.ignored_models_set:
            self.print_and_log(f"Ignored models: {', '.join(sorted(self.ignored_models_set))}")

        self.print_and_log(f"\nOverall Case Average Accuracy: {self.statistics.get('overall_case_average_accuracy', 0):.2%}")
        self.print_and_log(f"Average Model Case Accuracy: {self.statistics.get('average_model_case_accuracy', 0):.2%}")

        self.print_and_log("\nModel Performance (sorted by Question Accuracy):")
        self.print_and_log("-" * 60)
        self.print_and_log(f"{'Model':<30} {'Case Acc':<15} {'Question Acc':<15}")
        self.print_and_log("-" * 60)

        sorted_models = sorted(
            self.statistics["models"].items(),
            key=lambda x: x[1]["question_accuracy"],
            reverse=True,
        )
        for model_id, stats in sorted_models:
            self.print_and_log(f"{model_id:<30} {stats['case_accuracy']:<15.2%} {stats['question_accuracy']:<15.2%}")

        self.print_and_log("\nDiscrimination Analysis (Question Differentiation Metrics):")
        self.print_and_log("-" * 60)

        avg_metrics = self.statistics.get("discrimination_analysis", {}).get("average_metrics", {})
        self.print_and_log(f"Average Standard Deviation: {avg_metrics.get('avg_std_dev', 0):.4f}")
        self.print_and_log(f"Average IQR (Interquartile Range): {avg_metrics.get('avg_iqr', 0):.4f}")
        self.print_and_log(f"Average CV (Coefficient of Variation): {avg_metrics.get('avg_cv', 0):.4f}")

        disc_stats = self.statistics.get("discrimination_analysis", {})
        self.print_and_log("\nDiscrimination Distribution:")
        dist = disc_stats.get("discrimination_distribution", {})
        self.print_and_log(
            f"  High (>0.7):      {disc_stats.get('high_discrimination_cases', 0)} cases "
            f"({dist.get('high', {}).get('percentage', 0):.1f}%)"
        )
        self.print_and_log(
            f"  Medium (0.3-0.7): {disc_stats.get('medium_discrimination_cases', 0)} cases "
            f"({dist.get('medium', {}).get('percentage', 0):.1f}%)"
        )
        self.print_and_log(
            f"  Low (<0.3):       {disc_stats.get('low_discrimination_cases', 0)} cases "
            f"({dist.get('low', {}).get('percentage', 0):.1f}%)"
        )

        top_disc_cases = disc_stats.get("top_discriminating_cases", [])
        if top_disc_cases:
            self.print_and_log(f"\nTop {len(top_disc_cases)} Most Discriminating Cases:")
            self.print_and_log(f"{'Rank':<6} {'Filename':<40} {'Std Dev':<12} {'IQR':<12} {'CV':<12}")
            self.print_and_log("-" * 82)
            for i, case in enumerate(top_disc_cases, 1):
                self.print_and_log(
                    f"{i:<6} {case['filename']:<40} {case['std_dev']:<12.4f} {case['iqr']:<12.4f} {case['cv']:<12.4f}"
                )

        self.print_and_log("\nCase Accuracy Distribution:")
        self.print_and_log("-" * 60)
        self.print_and_log(f"{'Range':<15} {'Count':<10} {'Percentage':<15} {'Cumulative':<15}")
        self.print_and_log("-" * 60)

        dist_data = self.statistics.get("case_accuracy_distribution", {})
        for range_label, dist_item in dist_data.items():
            self.print_and_log(
                f"{range_label:<15} {dist_item['count']:<10} {dist_item['percentage']:<15.1f}% {dist_item['cumulative_percentage']:<15.1f}%"
            )

        self.print_and_log(f"\nTop {min(5, len(self.statistics['hardest_cases']))} Hardest Cases:")
        self.print_and_log("-" * 60)
        for i, case in enumerate(self.statistics["hardest_cases"][:5], 1):
            self.print_and_log(f"{i}. {case['filename']} - Accuracy: {case['average_accuracy']:.2%}")

        self.print_and_log(f"\nTop {min(5, len(self.statistics['easiest_cases']))} Easiest Cases:")
        self.print_and_log("-" * 60)
        for i, case in enumerate(self.statistics["easiest_cases"][:5], 1):
            self.print_and_log(f"{i}. {case['filename']} - Accuracy: {case['average_accuracy']:.2%}")

        rarity_clusters = self.statistics.get("rarity_clusters", {})
        if rarity_clusters:
            self.print_and_log("\nRarity Cluster Analysis:")
            self.print_and_log("-" * 60)
            for rarity, data in rarity_clusters.items():
                self.print_and_log(f"\nCluster: {rarity}  (Cases: {data['case_count']})")
                self.print_and_log(f"{'Model':<30} {'Case Acc':<15} {'Question Acc':<15}")
                self.print_and_log("-" * 60)
                for model_id, stats in sorted(
                    data["models"].items(),
                    key=lambda x: x[1]["question_accuracy"],
                    reverse=True,
                ):
                    self.print_and_log(f"{model_id:<30} {stats['case_accuracy']:<15.2%} {stats['question_accuracy']:<15.2%}")

    def run(self, copy_extremes: bool = True):
        """Run the full pipeline."""
        if not self.json_files:
            print("No JSON files found in the specified folder.")
            return
        self.calculate_statistics()
        self.print_summary()
        if copy_extremes:
            self.copy_extreme_cases()
        self.save_statistics()
        self.save_report()


def main():
    parser = argparse.ArgumentParser(description="MCQ Evaluation Statistics Analyzer")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing evaluated JSON files.")
    parser.add_argument(
        "--ignore-models",
        type=str,
        default="",
        help="Comma-separated model_ids to ignore in statistics (e.g., gpt-4o,model-x).",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy hardest/easiest cases into hard/ easy subfolders.",
    )
    args = parser.parse_args()

    resolved_path = Path(args.folder).resolve()
    if not resolved_path.is_dir():
        print(f"Error: '{resolved_path}' is not a valid directory.")
        return

    ignored_models: List[str] = []
    if args.ignore_models.strip():
        ignored_models = [m.strip() for m in args.ignore_models.split(",") if m.strip()]

    print("MCQ Evaluation Statistics Analyzer")
    print("=" * 50)
    print(f"Target folder: {resolved_path}")

    analyzer = EvaluationStatistics(str(resolved_path), ignored_models=ignored_models)
    analyzer.run(copy_extremes=not args.no_copy)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
