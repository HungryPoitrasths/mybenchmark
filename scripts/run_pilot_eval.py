#!/usr/bin/env python3
"""Run VLM on pilot study questions and evaluate results.

All-in-one script: load model → inference → evaluate → per-question detail.
Loads model directly via transformers (no API server needed).

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/run_pilot_eval.py \
        --benchmark output/pilot/benchmark.json \
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \
        --model /home/shenyl/hf/model/Qwen/Qwen3-VL-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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


# ── Inference (transformers, direct model loading) ────────────────────────

def load_model(model_path: str):
    """Load Qwen2/3-VL model and processor."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    logger.info("Loading model from %s ...", model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded on %d GPUs", torch.cuda.device_count())
    return model, processor


def run_inference(
    questions: list[dict],
    image_root: Path,
    model,
    processor,
) -> list[dict]:
    from qwen_vl_utils import process_vision_info

    try:
        from tqdm import tqdm
        loop = tqdm(enumerate(questions), total=len(questions), desc="VLM Eval")
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

        prompt_text = build_prompt(q)

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=16)

        # Only decode newly generated tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

        pred = parse_answer(raw)

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

    return results


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(questions: list[dict], predictions: list[dict], model_name: str):
    """Print per-type accuracy and per-question results."""

    pred_map = {p["question_id"]: p for p in predictions}

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
    print(f"PILOT EVALUATION: {model_name}")
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
                  "attachment_chain", "coordinate_rotation"]
    for t in type_order:
        if t not in type_groups:
            continue
        grp = type_groups[t]
        c = sum(1 for d in grp if d["correct"])
        print(f"  {t:25s}: {c}/{len(grp)} = {c/len(grp):.1%}")

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
        mark = "v" if d["correct"] else "x"
        print(f"{d['idx']:4d}  {d['level']:>2s}  {d['type']:>20s}  "
              f"{d['gt']:>2s}  {d['pred']:>4s}  {mark}  {d['question']}")

    print("=" * 70)

    return details


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pilot eval: load model directly via transformers")
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
        "--model", type=str,
        default="/home/shenyl/hf/model/Qwen/Qwen3-VL-32B-Instruct",
        help="Path to local model (HuggingFace format)",
    )
    parser.add_argument(
        "--output", type=str, default="output/pilot/eval_qwen.json",
        help="Path to save detailed results",
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

    # Load model
    model, processor = load_model(args.model)
    model_name = Path(args.model).name

    # Run inference
    print(f"\nRunning {model_name} on {len(questions)} questions ...")
    predictions = run_inference(questions, Path(args.image_root), model, processor)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Predictions saved to %s", output_path)

    # Evaluate
    details = evaluate(questions, predictions, model_name)

    # Also save full detail
    detail_path = output_path.with_name("eval_qwen_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info("Detail saved to %s", detail_path)


if __name__ == "__main__":
    main()
