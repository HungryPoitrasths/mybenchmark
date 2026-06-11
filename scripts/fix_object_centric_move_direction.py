#!/usr/bin/env python3
"""Fix the movement direction word in already-generated L2_object_move_object_centric questions.

Background
----------
The question text puts the reader in the MOVED object's frame
("imagine you are {move_source} and initially facing the camera. If you were
shifted {direction} by {distance}..."), but the stored text was rendered with
``_delta_to_description(delta, camera_pose)`` which expresses the delta in the
CAMERA optical frame.  Because the object faces the camera, the object frame is
rotated ~180deg from the camera frame, so the stored word is mirror-reversed
(forward<->backward AND left<->right), and for off-axis objects / diagonal
deltas it is not even a clean flip (the projection axis differs by the object's
angle in the FOV, which can flip forward/back<->left/right dominance).

This script recomputes the word in the object-facing frame using the SAME
geometry the answer was computed in (``_delta_to_object_facing_description``),
and rewrites ONLY the ``shifted <word> by`` token in the question text.

It does NOT touch answer / options / correct_value / new_correct_value /
old_correct_value -- those were always computed in the query object's frame and
are already correct.

Run this where the ScanNet / ScanNet++ scene data lives (the pipeline box); the
question JSON does not store object centers or camera poses, so the scenes must
be reloaded to recompute geometry.

Example:
    python scripts/fix_object_centric_move_direction.py \
        --input output/benchmark_subset.json \
        --output output/benchmark_subset.object_centric_dir_fixed.json \
        --report output/benchmark_subset.object_centric_dir_report.json \
        --scannet-root /home/lihongxing/datasets/ScanNet/data/scans \
        --scannetpp-root /home/sujinyue/datasets/scannetpp
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import make_data_source
from src.qa_generator import _delta_to_object_facing_description
from src.support_graph import enrich_scene_with_attachment

TARGET_TYPE = "object_move_object_centric"

# "shifted <word> by" appears once in every template variant of this type.
_DIRECTION_TOKEN = re.compile(r"(shifted\s+)([a-zA-Z\-]+)(\s+by\b)")

# A naive forward<->backward (+ left<->right) flip, used ONLY to measure how
# many questions a blind string-swap would get wrong vs. the geometric recompute.
_NAIVE_FLIP = {
    "forward": "backward",
    "backward": "forward",
    "left": "right",
    "right": "left",
    "up": "up",
    "down": "down",
}

DEFAULT_SCANNET_ROOT = Path("/home/lihongxing/datasets/ScanNet/data/scans")
DEFAULT_SCANNETPP_ROOT = Path("/home/sujinyue/datasets/scannetpp")

# Skip reason codes.
SKIP_NOT_TARGET = "not_target_question"
SKIP_NO_DELTA = "missing_delta"
SKIP_NO_TOKEN = "direction_token_not_found"
SKIP_MISSING_ROOT = "missing_scene_root"
SKIP_SCENE_NOT_FOUND = "scene_dir_not_found"
SKIP_SCENE_LOAD_FAILED = "scene_load_failed"
SKIP_POSE_MISSING = "pose_missing_for_frame"
SKIP_SOURCE_MISSING = "move_source_object_missing"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _benchmark_questions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("questions", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Could not locate a question list in the benchmark payload")


def _infer_dataset(question: dict[str, Any]) -> str | None:
    explicit = str(question.get("_dataset", "")).strip().lower()
    if explicit in {"scannet", "scannetpp"}:
        return explicit
    source_text = str(question.get("_source_benchmark", "")).lower()
    scene_id = str(question.get("scene_id", "")).strip().lower()
    if "scannetpp" in source_text:
        return "scannetpp"
    if scene_id.startswith("scene"):
        return "scannet"
    if scene_id:
        return "scannetpp"
    return None


def _resolve_scene_dir(root: Path, scene_id: str) -> Path:
    candidates = [root / scene_id, root / "scans" / scene_id]
    if root.name == scene_id and root.is_dir():
        candidates.insert(0, root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(scene_id)


def _load_scene_context(
    *,
    dataset: str,
    scene_id: str,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str,
) -> dict[str, Any]:
    root = scannetpp_root if dataset == "scannetpp" else scannet_root
    if root is None:
        raise FileNotFoundError(SKIP_MISSING_ROOT)
    scene_dir = _resolve_scene_dir(root, scene_id)
    data_source = make_data_source(dataset, scene_dir, sensor=scannetpp_sensor)
    scene = data_source.load_scene()
    if scene is None:
        raise RuntimeError(f"Failed to parse scene {scene_id}")
    enrich_scene_with_attachment(scene)
    objects = [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]
    obj_map = {
        int(obj["id"]): obj
        for obj in objects
        if _coerce_int(obj.get("id")) is not None
    }
    return {"obj_map": obj_map, "poses": data_source.load_poses()}


def fix_questions(
    questions: list[dict[str, Any]],
    *,
    scannet_root: Path | None,
    scannetpp_root: Path | None,
    scannetpp_sensor: str,
    log=print,
) -> dict[str, Any]:
    scene_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    changed: list[dict[str, Any]] = []
    naive_flip_divergent: list[dict[str, Any]] = []
    unchanged = 0
    skipped: list[dict[str, Any]] = []

    targets = [q for q in questions if q.get("type") == TARGET_TYPE]
    log(f"[info] {len(targets)} {TARGET_TYPE} questions out of {len(questions)} total")

    for question in targets:
        delta = question.get("delta")
        if not delta:
            skipped.append({"reason": SKIP_NO_DELTA, "question": _qref(question)})
            continue
        match = _DIRECTION_TOKEN.search(question.get("question", ""))
        if match is None:
            skipped.append({"reason": SKIP_NO_TOKEN, "question": _qref(question)})
            continue
        old_word = match.group(2).lower()

        dataset = _infer_dataset(question)
        scene_id = str(question.get("scene_id", "")).strip()
        image_name = str(question.get("image_name", "")).strip()
        if dataset is None or not scene_id:
            skipped.append({"reason": SKIP_SCENE_NOT_FOUND, "question": _qref(question)})
            continue

        cache_key = (dataset, scene_id)
        if cache_key not in scene_cache:
            try:
                scene_cache[cache_key] = _load_scene_context(
                    dataset=dataset,
                    scene_id=scene_id,
                    scannet_root=scannet_root,
                    scannetpp_root=scannetpp_root,
                    scannetpp_sensor=scannetpp_sensor,
                )
                ctx = scene_cache[cache_key]
                log(
                    f"[scene] loaded {dataset}:{scene_id} "
                    f"objects={len(ctx['obj_map'])} poses={len(ctx['poses'])}"
                )
            except FileNotFoundError as exc:
                reason = SKIP_MISSING_ROOT if str(exc) == SKIP_MISSING_ROOT else SKIP_SCENE_NOT_FOUND
                scene_cache[cache_key] = None
                log(f"[scene] {reason} for {dataset}:{scene_id}")
            except Exception as exc:  # noqa: BLE001 - record and move on
                scene_cache[cache_key] = None
                log(f"[scene] load failed {dataset}:{scene_id}: {exc}")

        ctx = scene_cache[cache_key]
        if ctx is None:
            skipped.append({"reason": SKIP_SCENE_LOAD_FAILED, "question": _qref(question)})
            continue

        pose = ctx["poses"].get(image_name)
        if pose is None:
            skipped.append({"reason": SKIP_POSE_MISSING, "question": _qref(question)})
            continue
        source_id = _coerce_int(question.get("moved_obj_id"))
        source_obj = ctx["obj_map"].get(source_id) if source_id is not None else None
        if source_obj is None or "center" not in source_obj:
            skipped.append({"reason": SKIP_SOURCE_MISSING, "question": _qref(question)})
            continue

        source_center = np.asarray(source_obj["center"], dtype=float)
        camera_center = np.asarray(pose.position, dtype=float)
        new_word = _delta_to_object_facing_description(
            np.asarray(delta, dtype=float), source_center, camera_center
        )

        if new_word == old_word:
            unchanged += 1
            continue

        # Rewrite ONLY the direction token; answer/options untouched.
        new_text = (
            question["question"][: match.start()]
            + match.group(1)
            + new_word
            + match.group(3)
            + question["question"][match.end():]
        )
        question["question"] = new_text
        question["direction_word_fix"] = {
            "old": old_word,
            "new": new_word,
            "naive_flip_would_be": _NAIVE_FLIP.get(old_word, old_word),
        }

        record = {**_qref(question), "old": old_word, "new": new_word}
        changed.append(record)
        # Would a blind forward<->backward(+L/R) swap have produced a DIFFERENT
        # (i.e. wrong) word than the geometric recompute?
        if _NAIVE_FLIP.get(old_word, old_word) != new_word:
            naive_flip_divergent.append(record)

    report = {
        "target_type": TARGET_TYPE,
        "target_count": len(targets),
        "changed_count": len(changed),
        "unchanged_count": unchanged,
        "skipped_count": len(skipped),
        "naive_flip_divergent_count": len(naive_flip_divergent),
        "changed": changed,
        "naive_flip_divergent": naive_flip_divergent,
        "skipped": skipped,
    }
    return report


def _qref(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": question.get("scene_id"),
        "image_name": question.get("image_name"),
        "moved_obj_id": question.get("moved_obj_id"),
        "obj_ref_id": question.get("obj_ref_id"),
        "question_uid": question.get("question_uid"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite the movement direction word in L2_object_move_object_centric questions to the object-facing frame.",
    )
    parser.add_argument("--input", default="output/benchmark_subset.json")
    parser.add_argument("--output", default="output/benchmark_subset.object_centric_dir_fixed.json")
    parser.add_argument("--report", default="output/benchmark_subset.object_centric_dir_report.json")
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report changes but do not write the fixed benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _read_json(Path(args.input))
    questions = _benchmark_questions(payload)

    report = fix_questions(
        questions,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        scannetpp_sensor=args.scannetpp_sensor,
    )

    _write_json(Path(args.report), report)
    if not args.dry_run:
        _write_json(Path(args.output), payload)

    print(
        f"[done] target={report['target_count']} "
        f"changed={report['changed_count']} unchanged={report['unchanged_count']} "
        f"skipped={report['skipped_count']} "
        f"naive_flip_would_be_wrong_on={report['naive_flip_divergent_count']}"
    )
    if args.dry_run:
        print("[dry-run] no benchmark written; report only")
    else:
        print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
