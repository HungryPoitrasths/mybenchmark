#!/usr/bin/env python3
"""VLM-based question quality filter for CausalSpatial-Bench.

For each question, asks Qwen2.5-VL-72B whether every object mentioned in the
question is clearly visible in the image.  Questions where any object is not
clearly visible are removed.

Usage:
    python scripts/run_vlm_filter.py \\
        --questions output/pilot/benchmark.json \\
        --image_root /home/lihongxing/datasets/ScanNet/data/scans \\
        --output    output/pilot/benchmark_filtered.json \\
        --workers   8

The script is resumable: if --output already exists, questions that were
already processed are loaded from it and only the remaining ones are sent to
the VLM.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_filter")

VLM_BASE_URL = "http://183.129.178.195:60029/v1"
VLM_MODEL    = "Qwen2.5-VL-72B-Instruct"  # default; override with --vlm_model

# Objects mentioned in a question are extracted from these fields
OBJECT_FIELDS = [
    "question",   # parsed via regex
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_object_names(question_text: str) -> list[str]:
    """Heuristically extract object names from a question string.

    Works for all our template patterns:
      "... mirror ... shower ..."  → ["mirror", "shower"]
    Strategy: grab words/phrases that look like nouns in the question.
    We rely on the fact that our templates use object labels directly.
    """
    # Remove filler phrases, keep candidate nouns
    text = question_text.lower()
    # Strip common question scaffolding
    for phrase in [
        "from the image", "looking at the scene", "in this view",
        "approximately", "how far apart are", "what is the approximate distance between",
        "what is the spatial relationship of", "is in which direction relative to",
        "where is", "positioned relative to", "can you see", "completely",
        "or is it blocked by", "from the current viewpoint",
        "if the observer moves", "from the current position",
        "would", "become visible or occluded",
        "if", "is moved", "by", "what would be the new spatial relationship between",
        "and", "after this change", "what is the relative position of", "to",
        "imagine moving", "in which direction is", "relative to",
        "suppose this room had originally been designed with its orientation rotated",
        "degrees with all objects keeping their relative positions",
        "observed from the original camera position and viewing direction unchanged",
        "suppose", "were moved to a different location",
        "which of the following objects would also be displaced from their current positions",
        "if were relocated elsewhere in the room",
        "imagine is moved to a new spot",
        "which of the following objects would also be displaced as a result",
    ]:
        text = text.replace(phrase, " ")

    # Match multi-word labels like "coffee table", "kitchen counter"
    # Remove punctuation, split by common delimiters
    text = re.sub(r"[?.!,]", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?m?\b", " ", text)   # strip numbers/distances
    text = re.sub(r"\s+", " ", text).strip()

    # Split on "between", "of", "relative", etc. and collect 1-3 word chunks
    chunks = re.split(r"\b(?:between|relative|of|from|and|to|is|in|the|this|a|an)\b", text)
    objects = []
    for chunk in chunks:
        chunk = chunk.strip()
        # Keep 1-3 word phrases that look like object names (not stop words)
        words = chunk.split()
        if 1 <= len(words) <= 3 and all(len(w) > 2 for w in words):
            objects.append(chunk)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for obj in objects:
        if obj not in seen:
            seen.add(obj)
            unique.append(obj)
    return unique


def _load_image_b64(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _is_object_visible(client, img_b64: str, obj_name: str, model: str) -> bool:
    """Ask the VLM whether *obj_name* is clearly visible in the image."""
    prompt = (
        f'Is there a "{obj_name}" clearly visible in this image? '
        f"Answer with only Yes or No."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=4,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        logger.warning("VLM call failed for object '%s': %s", obj_name, e)
        return True


def _filter_question(client, q: dict, image_root: Path, model: str) -> dict:
    """Return the question dict with a 'vlm_visible' flag added."""
    scene_id   = q.get("scene_id", "")
    image_name = q.get("image_name", "")
    image_path = image_root / scene_id / "color" / image_name

    img_b64 = _load_image_b64(image_path)
    if img_b64 is None:
        logger.warning("Image not found: %s — keeping question", image_path)
        q["vlm_visible"] = True
        return q

    objects = _extract_object_names(q["question"])
    if not objects:
        q["vlm_visible"] = True
        return q

    for obj in objects:
        if not _is_object_visible(client, img_b64, obj, model):
            q["vlm_visible"] = False
            q["vlm_invisible_object"] = obj
            return q

    q["vlm_visible"] = True
    return q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VLM-based QA visibility filter")
    parser.add_argument("--questions",   required=True,  help="Input benchmark.json")
    parser.add_argument("--image_root",  required=True,  help="ScanNet scans root dir")
    parser.add_argument("--output",      required=True,  help="Output filtered JSON")
    parser.add_argument("--workers",     type=int, default=8,
                        help="ThreadPoolExecutor max_workers")
    parser.add_argument("--vlm_url",     default=VLM_BASE_URL,
                        help="VLM API base URL")
    parser.add_argument("--vlm_model",   default=None,
                        help="Model name to use; if not set, auto-detected from /v1/models")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=args.vlm_url)

    # Test connection and resolve model name
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        logger.info("VLM available models: %s", available)
    except Exception as e:
        logger.error("Cannot reach VLM at %s: %s", args.vlm_url, e)
        sys.exit(1)

    model_name = args.vlm_model if args.vlm_model else available[0]
    logger.info("Using model: %s", model_name)

    # Load questions
    with open(args.questions, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
        meta = {k: v for k, v in data.items() if k != "questions"}
    else:
        questions = data
        meta = {}

    logger.info("Loaded %d questions", len(questions))

    # Resumability: load already-processed questions
    output_path = Path(args.output)
    processed: dict[str, dict] = {}  # question index → result
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            prev = json.load(f)
        prev_qs = prev.get("questions", prev) if isinstance(prev, dict) else prev
        # Use (scene_id, image_name, question) as key
        for q in prev_qs:
            key = (q.get("scene_id"), q.get("image_name"), q.get("question"))
            processed[str(key)] = q
        logger.info("Resuming: %d questions already processed", len(processed))

    image_root = Path(args.image_root)
    to_process = []
    already_done = []
    for q in questions:
        key = str((q.get("scene_id"), q.get("image_name"), q.get("question")))
        if key in processed:
            already_done.append(processed[key])
        else:
            to_process.append(q)

    logger.info(
        "%d to process, %d already done", len(to_process), len(already_done)
    )

    results = list(already_done)

    if to_process:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_filter_question, client, q, image_root, model_name): i
                for i, q in enumerate(to_process)
            }
            done_count = 0
            for future in as_completed(futures):
                results.append(future.result())
                done_count += 1
                if done_count % 100 == 0:
                    logger.info(
                        "Progress: %d / %d processed",
                        done_count + len(already_done), len(questions)
                    )

    # Split into kept / discarded
    kept      = [q for q in results if q.get("vlm_visible", True)]
    discarded = [q for q in results if not q.get("vlm_visible", True)]

    logger.info(
        "Filter complete: %d kept, %d discarded (%.1f%% removed)",
        len(kept), len(discarded),
        100 * len(discarded) / max(len(results), 1),
    )

    # Log discard breakdown by level
    from collections import Counter
    disc_by_level = Counter(q.get("level") for q in discarded)
    logger.info("Discarded by level: %s", dict(disc_by_level))

    # Save output — same structure as input
    output = {**meta, "questions": kept}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Also save discard log for inspection
    discard_path = output_path.with_suffix(".discarded.json")
    with open(discard_path, "w", encoding="utf-8") as f:
        json.dump(discarded, f, indent=2, ensure_ascii=False)

    logger.info("Saved filtered questions to %s", output_path)
    logger.info("Saved discard log to %s", discard_path)


if __name__ == "__main__":
    main()
