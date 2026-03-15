#!/usr/bin/env python3
"""Evaluation script for CausalSpatial-Bench.

Computes accuracy metrics for model predictions against ground-truth answers,
broken down by level and question type.

Usage:
    python evaluation/evaluate.py \
        --benchmark output/benchmark.json \
        --predictions predictions/gpt4o.json \
        --output_report evaluation/report_gpt4o.json

Predictions JSON format:
    [
        {"question_id": 0, "prediction": "A"},
        {"question_id": 1, "prediction": "C"},
        ...
    ]
    OR:
    [
        {"scene_id": "xxx", "image_name": "yyy", "question": "...", "prediction": "B"},
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data


def load_predictions(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_predictions(
    questions: list[dict],
    predictions: list[dict],
) -> list[tuple[dict, str]]:
    """Match predictions to ground-truth questions.

    Supports matching by index (question_id) or by (scene_id, question) text.
    Returns list of (question, predicted_answer) pairs.
    """
    matched: list[tuple[dict, str]] = []

    # Try index-based matching first
    if predictions and "question_id" in predictions[0]:
        pred_map = {p["question_id"]: p["prediction"] for p in predictions}
        for i, q in enumerate(questions):
            if i in pred_map:
                matched.append((q, pred_map[i]))
    else:
        # Text-based matching
        pred_map = {}
        for p in predictions:
            key = (p.get("scene_id", ""), p.get("question", ""))
            pred_map[key] = p["prediction"]
        for q in questions:
            key = (q.get("scene_id", ""), q.get("question", ""))
            if key in pred_map:
                matched.append((q, pred_map[key]))

    logger.info("Matched %d / %d predictions", len(matched), len(predictions))
    return matched


def compute_accuracy(
    matched: list[tuple[dict, str]],
) -> dict[str, float | dict]:
    """Compute accuracy metrics broken down by level and type."""
    results: dict[str, Any] = {}

    # Overall
    correct = sum(1 for q, pred in matched if pred == q["answer"])
    total = len(matched)
    results["overall_accuracy"] = correct / total if total > 0 else 0.0
    results["overall_correct"] = correct
    results["overall_total"] = total

    # By level
    level_stats: dict[str, dict] = {}
    for level in ("L1", "L2", "L3"):
        level_matched = [(q, p) for q, p in matched if q.get("level") == level]
        n = len(level_matched)
        c = sum(1 for q, p in level_matched if p == q["answer"])
        level_stats[level] = {
            "accuracy": c / n if n > 0 else 0.0,
            "correct": c,
            "total": n,
        }
    results["by_level"] = level_stats

    # By type
    type_stats: dict[str, dict] = {}
    all_types = set(q.get("type", "?") for q, _ in matched)
    for qtype in sorted(all_types):
        type_matched = [(q, p) for q, p in matched if q.get("type") == qtype]
        n = len(type_matched)
        c = sum(1 for q, p in type_matched if p == q["answer"])
        type_stats[qtype] = {
            "accuracy": c / n if n > 0 else 0.0,
            "correct": c,
            "total": n,
        }
    results["by_type"] = type_stats

    # L1 vs L2 gap (core hypothesis)
    l1_acc = level_stats.get("L1", {}).get("accuracy", 0)
    l2_acc = level_stats.get("L2", {}).get("accuracy", 0)
    results["l1_l2_gap"] = l1_acc - l2_acc

    return results


def print_report(results: dict):
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 60)
    print("CAUSAL-SPATIAL BENCH EVALUATION REPORT")
    print("=" * 60)
    print(f"\nOverall accuracy: {results['overall_accuracy']:.1%} "
          f"({results['overall_correct']}/{results['overall_total']})")

    print("\n--- By Level ---")
    for level, stats in results["by_level"].items():
        print(f"  {level}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    print("\n--- By Type ---")
    for qtype, stats in results["by_type"].items():
        print(f"  {qtype}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    gap = results.get("l1_l2_gap", 0)
    print(f"\n--- Core Hypothesis ---")
    print(f"  L1 - L2 gap: {gap:.1%}")
    if gap > 0.20:
        print("  [PASS] Gap > 20% — intervention reasoning is significantly harder")
    elif gap > 0.10:
        print("  [WARN] Gap 10-20% — marginal difference")
    else:
        print("  [FAIL] Gap < 10% — benchmark may not discriminate well")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CausalSpatial-Bench evaluation")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark.json")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to model predictions JSON")
    parser.add_argument("--output_report", type=str, default=None,
                        help="Path to save JSON report")
    args = parser.parse_args()

    questions = load_benchmark(args.benchmark)
    predictions = load_predictions(args.predictions)
    matched = match_predictions(questions, predictions)

    if not matched:
        logger.error("No predictions matched — check file format")
        sys.exit(1)

    results = compute_accuracy(matched)
    print_report(results)

    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
