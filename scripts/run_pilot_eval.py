#!/usr/bin/env python3
"""Run Qwen2.5-VL-72B on pilot study questions and evaluate results.

All-in-one script: inference → evaluate → per-question detail.

Usage:
    python scripts/run_pilot_eval.py \
        --benchmark output/pilot/benchmark.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans

Requires: pip install openai tqdm
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Qwen3-VL-32B endpoint (deployed locally on zju-47) ───────────────────

QWEN_BASE_URL = "http://localhost:60029/v1"
QWEN_MODEL = "/home/shenyl/hf/model/Qwen/Qwen3-VL-32B-Instruct"
QWEN_API_KEY = "empty"  # vLLM doesn't need a real key

_SYSTEM = (
    "You are a visual spatial-reasoning assistant. "
    "Answer multiple-choice questions about spatial relationships in images."
)
_MCQ_SUFFIX = "\n\nAnswer with a single letter only (A, B, C, or D). Do not explain."


# ── Helpers ───────────────────────────────────────────────────────────────

def build_prompt(question: dict) -> str:
    parts = [question["question"], ""]
    for i, opt in enumerate(question["options"]):
        parts.append(f"{chr(65 + i)}) {opt}")
    parts.append(_MCQ_SUFFIX)
    return "\n".join(parts)


def to_base64(path: Path) -> tuple[str, str]:
    ext = path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def resolve_image(question: dict, image_root: Path) -> Path:
    return image_root / question["scene_id"] / "color" / question["image_name"]


def parse_answer(raw: str) -> str | None:
    if not raw:
        return None
    first = raw.strip()[0].upper()
    if first in "ABCD":
        return first
    m = re.search(r"\b([ABCD])\b", raw.upper())
    return m.group(1) if m else None


# ── Inference ─────────────────────────────────────────────────────────────

def run_inference(questions: list[dict], image_root: Path, delay: float = 0.3) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    try:
        from tqdm import tqdm
        loop = tqdm(enumerate(questions), total=len(questions), desc="Qwen2.5-VL")
    except ImportError:
        loop = enumerate(questions)

    results = []
    for idx, q in loop:
        image_path = resolve_image(q, image_root)

        if not image_path.exists():
            logger.warning("[%d] Image not found: %s", idx, image_path)
            results.append({
                "question_id": idx,
                "prediction": None,
                "raw_response": None,
                "error": "image_not_found",
            })
            continue

        prompt = build_prompt(q)
        b64, mime = to_base64(image_path)

        pred, raw = None, None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=QWEN_MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    max_tokens=16,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                pred = parse_answer(raw)
                break
            except Exception as exc:
                wait = 2.0 * (2 ** attempt)
                if attempt < 2:
                    logger.warning("[%d] Attempt %d failed: %s. Retry in %.0fs", idx, attempt+1, exc, wait)
                    time.sleep(wait)
                else:
                    logger.error("[%d] All attempts failed: %s", idx, exc)

        results.append({
            "question_id": idx,
            "prediction": pred,
            "raw_response": raw,
            "gt_answer": q["answer"],
            "level": q.get("level"),
            "type": q.get("type"),
            "scene_id": q.get("scene_id"),
            "image_name": q.get("image_name"),
            "question": q.get("question"),
            "correct_value": q.get("correct_value"),
        })

        if delay > 0:
            time.sleep(delay)

    return results


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(questions: list[dict], predictions: list[dict]):
    """Print per-type accuracy and per-question results."""

    # Match predictions to questions by index
    pred_map = {p["question_id"]: p for p in predictions}

    # Collect per-question results
    details = []
    for i, q in enumerate(questions):
        p = pred_map.get(i)
        if p is None or p["prediction"] is None:
            correct = False
            pred_letter = "?"
        else:
            pred_letter = p["prediction"]
            correct = (pred_letter == q["answer"])

        details.append({
            "idx": i,
            "level": q.get("level", "?"),
            "type": q.get("type", "?"),
            "scene_id": q.get("scene_id", ""),
            "image_name": q.get("image_name", ""),
            "question": q.get("question", "")[:80],
            "gt": q["answer"],
            "pred": pred_letter,
            "correct": correct,
            "gt_value": q.get("correct_value", ""),
        })

    # ── Overall ───────────────────────────────────────────────────────────
    total = len(details)
    n_correct = sum(1 for d in details if d["correct"])
    print("\n" + "=" * 70)
    print("PILOT EVALUATION: Qwen2.5-VL-72B-Instruct")
    print("=" * 70)
    print(f"Overall: {n_correct}/{total} = {n_correct/total:.1%}")

    # ── By Level ──────────────────────────────────────────────────────────
    print("\n--- By Level ---")
    for level in ("L1", "L2", "L3"):
        lvl = [d for d in details if d["level"] == level]
        if not lvl:
            continue
        c = sum(1 for d in lvl if d["correct"])
        print(f"  {level}: {c}/{len(lvl)} = {c/len(lvl):.1%}")

    # ── By Type ───────────────────────────────────────────────────────────
    print("\n--- By Type ---")
    type_groups = defaultdict(list)
    for d in details:
        type_groups[d["type"]].append(d)

    type_order = ["direction", "distance", "occlusion",
                  "object_move", "viewpoint_move", "object_remove",
                  "support_chain", "coordinate_rotation"]
    for t in type_order:
        if t not in type_groups:
            continue
        grp = type_groups[t]
        c = sum(1 for d in grp if d["correct"])
        print(f"  {t:25s}: {c}/{len(grp)} = {c/len(grp):.1%}")

    # Print any remaining types not in the fixed order
    for t in sorted(type_groups.keys()):
        if t not in type_order:
            grp = type_groups[t]
            c = sum(1 for d in grp if d["correct"])
            print(f"  {t:25s}: {c}/{len(grp)} = {c/len(grp):.1%}")

    # ── L1-L2 Gap ─────────────────────────────────────────────────────────
    l1 = [d for d in details if d["level"] == "L1"]
    l2 = [d for d in details if d["level"] == "L2"]
    l1_acc = sum(1 for d in l1 if d["correct"]) / len(l1) if l1 else 0
    l2_acc = sum(1 for d in l2 if d["correct"]) / len(l2) if l2 else 0
    gap = l1_acc - l2_acc
    print(f"\n--- Core Hypothesis ---")
    print(f"  L1={l1_acc:.1%}  L2={l2_acc:.1%}  gap={gap:.1%}")
    if gap > 0.20:
        print("  [PASS] Gap > 20%")
    elif gap > 0.10:
        print("  [WARN] Gap 10-20%")
    else:
        print("  [FAIL] Gap < 10%")

    # ── Per-question detail ───────────────────────────────────────────────
    print("\n--- Per-Question Detail ---")
    print(f"{'#':>4s}  {'Lv':>2s}  {'Type':>20s}  {'GT':>2s}  {'Pred':>4s}  {'':>1s}  Question")
    print("-" * 100)
    for d in details:
        mark = "✓" if d["correct"] else "✗"
        print(f"{d['idx']:4d}  {d['level']:>2s}  {d['type']:>20s}  "
              f"{d['gt']:>2s}  {d['pred']:>4s}  {mark}  {d['question']}")

    print("=" * 70)

    return details


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pilot eval with Qwen2.5-VL-72B")
    parser.add_argument(
        "--benchmark", type=str, default="output/pilot/benchmark.json",
        help="Path to pilot benchmark.json",
    )
    parser.add_argument(
        "--image_root", type=str,
        default="/home/lihongxing/datasets/ScanNet/data/scans",
        help="Root of ScanNet scans directory",
    )
    parser.add_argument(
        "--output", type=str, default="output/pilot/eval_qwen.json",
        help="Path to save detailed results",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between API calls (default 0.3)",
    )
    parser.add_argument(
        "--max_questions", type=int, default=None,
        help="Cap number of questions (for quick test)",
    )
    args = parser.parse_args()

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) and "questions" in data else data
    logger.info("Loaded %d questions from %s", len(questions), benchmark_path)

    if args.max_questions:
        questions = questions[:args.max_questions]
        logger.info("Capped at %d questions", len(questions))

    # Run inference
    print(f"\nCalling Qwen2.5-VL-72B @ {QWEN_BASE_URL} ...")
    predictions = run_inference(questions, Path(args.image_root), delay=args.delay)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Predictions saved to %s", output_path)

    # Evaluate
    details = evaluate(questions, predictions)

    # Also save full detail
    detail_path = output_path.with_name("eval_qwen_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info("Detail saved to %s", detail_path)


if __name__ == "__main__":
    main()
