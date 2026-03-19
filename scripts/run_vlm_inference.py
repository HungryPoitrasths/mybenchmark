#!/usr/bin/env python3
"""VLM inference script for CausalSpatial-Bench.

Sends each benchmark question (image + MCQ text) to a VLM and saves
predictions in the format expected by evaluation/evaluate.py.

Supported models
----------------
  gpt-4o          OpenAI GPT-4o          (env: OPENAI_API_KEY)
  gemini-2.5-pro  Google Gemini 2.5 Pro  (env: GOOGLE_API_KEY)
  qwen2.5-vl      Qwen2.5-VL via DashScope (env: DASHSCOPE_API_KEY)

Usage
-----
  python scripts/run_vlm_inference.py \\
      --benchmark  output/pilot/benchmark.json \\
      --image_root data/scannetpp \\
      --model      gpt-4o \\
      --output     predictions/pilot_gpt4o.json

Optional dependencies (install only what you need)
----------------------------------------------------
  pip install openai                  # GPT-4o and Qwen (compatible endpoint)
  pip install google-generativeai Pillow  # Gemini
  pip install tqdm                    # progress bar
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a visual spatial-reasoning assistant. "
    "Answer multiple-choice questions about spatial relationships in images."
)

_MCQ_SUFFIX = "\n\nAnswer with a single letter only (A, B, C, or D). Do not explain."


def build_prompt(question: dict) -> str:
    parts = [question["question"], ""]
    for i, opt in enumerate(question["options"]):
        parts.append(f"{chr(65 + i)}) {opt}")
    parts.append(_MCQ_SUFFIX)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_base64(path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type)."""
    ext = path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def resolve_image(question: dict, image_root: Path) -> Path:
    """Map scene_id + image_name → full filesystem path.

    ScanNet stores colour frames in ``<scene_id>/color/<frame_id>.jpg``.
    """
    return (
        image_root
        / question.get("scene_id", "")
        / "color"
        / question.get("image_name", "")
    )


# ---------------------------------------------------------------------------
# Model callers  (each returns a raw string from the model)
# ---------------------------------------------------------------------------

def make_gpt4o_caller(api_key: str) -> Callable[[Path, str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def call(image_path: Path, prompt: str) -> str:
        b64, mime = _to_base64(image_path)
        resp = client.chat.completions.create(
            model="gpt-4o",
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
        return resp.choices[0].message.content.strip()

    return call


def make_gemini_caller(
    api_key: str,
    model_name: str = "gemini-2.5-pro-preview-05-06",
) -> Callable[[Path, str], str]:
    import google.generativeai as genai
    from PIL import Image as PILImage

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        system_instruction=_SYSTEM,
    )

    def call(image_path: Path, prompt: str) -> str:
        img = PILImage.open(image_path)
        resp = model.generate_content(
            [img, prompt],
            generation_config={"max_output_tokens": 16, "temperature": 0},
        )
        return resp.text.strip()

    return call


def make_qwen_caller(api_key: str) -> Callable[[Path, str], str]:
    """Qwen2.5-VL via DashScope OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def call(image_path: Path, prompt: str) -> str:
        b64, mime = _to_base64(image_path)
        resp = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
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
        return resp.choices[0].message.content.strip()

    return call


def make_openai_local_caller(
    base_url: str,
    model_name: str,
    api_key: str = "EMPTY",
) -> Callable[[Path, str], str]:
    """Any OpenAI-compatible local VLM endpoint (e.g. vLLM / SGLang serving).

    Usage example for the lab's Qwen2.5-VL-72B server::

        python scripts/run_vlm_inference.py \\
            --model openai_local \\
            --base_url http://183.129.178.195:60029/v1 \\
            --model_name Qwen2.5-VL-72B-Instruct \\
            ...
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(image_path: Path, prompt: str) -> str:
        b64, mime = _to_base64(image_path)
        resp = client.chat.completions.create(
            model=model_name,
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
        return resp.choices[0].message.content.strip()

    return call


_CALLER_FACTORIES: dict[str, tuple[str, Callable]] = {
    "gpt-4o":         ("OPENAI_API_KEY",    make_gpt4o_caller),
    "gemini-2.5-pro": ("GOOGLE_API_KEY",    make_gemini_caller),
    "qwen2.5-vl":     ("DASHSCOPE_API_KEY", make_qwen_caller),
    # openai_local: handled separately in main() — uses --base_url / --model_name
    "openai_local":   ("EMPTY",             None),
}


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------

def parse_answer(raw: str) -> str | None:
    """Extract A/B/C/D from raw model output."""
    if not raw:
        return None
    first = raw.strip()[0].upper()
    if first in "ABCD":
        return first
    # Fallback: find any standalone letter
    m = re.search(r"\b([ABCD])\b", raw.upper())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _call_with_retry(
    caller: Callable,
    image_path: Path,
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> tuple[str | None, str | None]:
    """Return (parsed_letter, raw_response). Retries on transient errors."""
    for attempt in range(max_retries):
        try:
            raw = caller(image_path, prompt)
            return parse_answer(raw), raw
        except Exception as exc:
            wait = base_delay * (2 ** attempt)
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.0fs…",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed: %s", max_retries, exc)
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_benchmark(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Support both top-level list and {"questions": [...]} wrapper
    return data["questions"] if isinstance(data, dict) else data


def _save(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a VLM on CausalSpatial-Bench and save predictions."
    )
    parser.add_argument("--benchmark",   required=True,
                        help="Path to benchmark.json")
    parser.add_argument("--image_root",  required=True,
                        help="Root directory of ScanNet++ data")
    parser.add_argument("--model",       required=True,
                        choices=list(_CALLER_FACTORIES),
                        help="VLM to evaluate")
    parser.add_argument("--output",      required=True,
                        help="Path to save predictions.json")
    parser.add_argument("--api_key",     default=None,
                        help="API key (falls back to env variable)")
    # Local OpenAI-compatible endpoint options
    parser.add_argument("--base_url",    default=None,
                        help="Base URL for openai_local model (e.g. http://host:port/v1)")
    parser.add_argument("--model_name",  default=None,
                        help="Model name to pass to openai_local endpoint")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Cap number of questions (for quick smoke-tests)")
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="Seconds to sleep between API calls (default 0.5)")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save intermediate results every N questions (default 50)")
    # Optional: filter to specific level/type for targeted runs
    parser.add_argument("--level",  default=None,
                        choices=["L1", "L2", "L3"],
                        help="Restrict to one level (optional)")
    parser.add_argument("--qtype",  default=None,
                        help="Restrict to one question type, e.g. support_chain (optional)")
    args = parser.parse_args()

    # ── Load benchmark ────────────────────────────────────────────────────────
    questions = _load_benchmark(Path(args.benchmark))
    logger.info("Loaded %d questions from %s", len(questions), args.benchmark)

    # Optional filters
    if args.level:
        questions = [q for q in questions if q.get("level") == args.level]
        logger.info("Filtered to level=%s: %d questions", args.level, len(questions))
    if args.qtype:
        questions = [q for q in questions if q.get("type") == args.qtype]
        logger.info("Filtered to type=%s: %d questions", args.qtype, len(questions))
    if args.max_questions:
        questions = questions[: args.max_questions]
        logger.info("Capped at %d questions", len(questions))

    # ── Build caller ──────────────────────────────────────────────────────────
    logger.info("Initialising model: %s", args.model)

    if args.model == "openai_local":
        if not args.base_url:
            raise SystemExit("--base_url is required for openai_local (e.g. http://host:port/v1)")
        if not args.model_name:
            raise SystemExit("--model_name is required for openai_local")
        api_key = args.api_key or "EMPTY"
        caller = make_openai_local_caller(args.base_url, args.model_name, api_key)
    else:
        env_var, factory = _CALLER_FACTORIES[args.model]
        api_key = args.api_key or os.environ.get(env_var)
        if not api_key:
            raise SystemExit(
                f"API key required. Pass --api_key or set the {env_var} environment variable."
            )
        caller = factory(api_key)

    # ── Inference loop ────────────────────────────────────────────────────────
    image_root   = Path(args.image_root)
    output_path  = Path(args.output)
    predictions: list[dict] = []

    n_missing  = 0
    n_errors   = 0
    n_unparsed = 0

    try:
        from tqdm import tqdm
        loop = tqdm(enumerate(questions), total=len(questions), desc=args.model)
    except ImportError:
        loop = enumerate(questions)

    for idx, q in loop:
        image_path = resolve_image(q, image_root)

        if not image_path.exists():
            logger.warning("[%d] Image not found: %s", idx, image_path)
            predictions.append({
                "question_id": idx,
                "prediction":  None,
                "error":       "image_not_found",
            })
            n_missing += 1
            continue

        prompt       = build_prompt(q)
        pred, raw    = _call_with_retry(caller, image_path, prompt)

        if raw is None:
            n_errors += 1
        elif pred is None:
            n_unparsed += 1
            logger.warning("[%d] Unparseable reply: %r", idx, raw[:80])

        predictions.append({
            "question_id":  idx,
            "prediction":   pred,
            "raw_response": raw,
            # Extra fields for offline debugging / text-based matching
            "scene_id":     q.get("scene_id"),
            "image_name":   q.get("image_name"),
            "level":        q.get("level"),
            "type":         q.get("type"),
            "gt_answer":    q.get("answer"),
        })

        if args.delay > 0:
            time.sleep(args.delay)

        if (idx + 1) % args.checkpoint_every == 0:
            _save(predictions, output_path)
            logger.info("Checkpoint saved: %d / %d", idx + 1, len(questions))

    _save(predictions, output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    answered = sum(1 for p in predictions if p["prediction"] is not None)
    print()
    print("─" * 50)
    print(f"  Model           : {args.model}")
    print(f"  Total questions : {len(predictions)}")
    print(f"  Answered        : {answered}")
    print(f"  Missing images  : {n_missing}")
    print(f"  API errors      : {n_errors}")
    print(f"  Unparsed reply  : {n_unparsed}")
    print(f"  Output          : {output_path}")
    print("─" * 50)


if __name__ == "__main__":
    main()
