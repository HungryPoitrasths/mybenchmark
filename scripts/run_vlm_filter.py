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

def _get_object_labels(q: dict) -> list[str]:
    """Extract object labels directly from stored question fields.

    All question types now store label fields explicitly in the JSON so we
    don't need fragile regex parsing of the question text.

    Returns a deduplicated list of object label strings to check visibility.
    """
    labels = set()

    for mention in q.get("mentioned_objects", []):
        label = mention.get("label")
        if label:
            labels.add(label)

    # L1/L3 direction, distance, occlusion, coordinate_rotation (ego-centric & allocentric)
    for key in ("obj_a_label", "obj_b_label"):
        v = q.get(key)
        if v:
            labels.add(v)

    # L2 object_move (ego-centric)
    for key in ("moved_obj_label", "obj_c_label"):
        v = q.get(key)
        if v:
            labels.add(v)

    # Object-centric types (L1/L2/L3)
    for key in ("obj_ref_label", "obj_face_label", "obj_target_label"):
        v = q.get(key)
        if v:
            labels.add(v)

    # L3 support_chain (labels stored explicitly since #chain_ids)
    for key in ("grandparent_label", "parent_label", "grandchild_label", "neighbor_label"):
        v = q.get(key)
        if v:
            labels.add(v)

    _EXCLUDED = {
        "object", "unknown", "", "floor", "wall", "ceiling",
        "otherfurniture", "otherprop", "otherstructure",
        "room", "ground", "door", "window", "stairs",
        "mirror", "glass", "monitor", "tv",
        "doorframe", "windowsill", "hand rail", "shower",
        "shower curtain rod", "bathroom stall", "bathroom stall door",
        "ledge", "structure", "closet", "breakfast bar", "shower curtain",
        "case", "tube", "board", "sign", "frame", "paper", "lotion",
        "counter", "couch", "clothing", "blanket", "rug",
        "power outlet", "light switch", "fire alarm", "controller",
        "power strip", "soda can", "starbucks cup", "battery disposal jar",
        "can", "water bottle", "paper cutter",
        "pillar", "column",
    }
    labels -= _EXCLUDED
    return list(labels)


def _load_image_b64(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _ask_vlm_yes_no(client, img_b64: str, prompt: str, model: str) -> bool:
    """Send a Yes/No question to the VLM about an image. Returns True for Yes."""
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
        logger.warning("VLM call failed for prompt '%s': %s", prompt[:60], e)
        return True  # fail-open: keep the question


def _is_object_visible(client, img_b64: str, obj_name: str, model: str) -> bool:
    """Ask the VLM whether *obj_name* is present in the image."""
    prompt = (
        f'Is there a "{obj_name}" present in this image? '
        f"Answer with only Yes or No."
    )
    return _ask_vlm_yes_no(client, img_b64, prompt, model)


def _is_object_ambiguous(client, img_b64: str, obj_name: str, model: str) -> bool:
    """Ask the VLM whether there are multiple instances of *obj_name*.

    Returns True if ambiguous (more than one instance visible).
    """
    prompt = (
        f'Is there more than one "{obj_name}" in this image? '
        f"Answer with only Yes or No."
    )
    return _ask_vlm_yes_no(client, img_b64, prompt, model)


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

    objects = _get_object_labels(q)
    if not objects:
        q["vlm_visible"] = True
        return q

    for obj in objects:
        if not _is_object_visible(client, img_b64, obj, model):
            q["vlm_visible"] = False
            q["vlm_invisible_object"] = obj
            return q

    # Check for ambiguity: if multiple instances of the same object are
    # visible, the question is unanswerable (e.g., "which direction is
    # the chair relative to the table?" when there are 2 chairs).
    for obj in objects:
        if _is_object_ambiguous(client, img_b64, obj, model):
            q["vlm_visible"] = False
            q["vlm_ambiguous_object"] = obj
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
