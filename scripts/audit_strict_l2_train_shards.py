#!/usr/bin/env python3
"""Audit the six fresh strict-L2 train shards without merging them."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.repair_strict_l2_move_chain import _visible_identity
from src.cot.facts import build_fact_record
from src.cot.images import collect_image_names
from src.cot.validators import validate_answer_mapping, validate_fact_consistency


SHARDS = ("40-49", "50-59", "60-69", "70-79", "80-89", "90-99")
EXPECTED_TYPES = (
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "object_remove",
)
MOVE_TYPES = {
    "object_move_agent": ("agent", "frame_1"),
    "object_move_distance": ("agent", "frame_1"),
    "object_move_allocentric": ("allocentric", None),
}


def _load_document(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise ValueError(f"{path}: expected an object with a questions list")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_marker(question: dict[str, Any], index: int) -> None:
    question_type = str(question.get("type") or "")
    prefix = f"question {index} ({question_type})"
    if question_type in MOVE_TYPES:
        expected_frame, expected_binding = MOVE_TYPES[question_type]
        if question.get("movement_semantics_version") != 2:
            raise AssertionError(f"{prefix}: movement_semantics_version is not 2")
        if question.get("movement_reference_frame") != expected_frame:
            raise AssertionError(f"{prefix}: wrong movement_reference_frame")
        if question.get("movement_camera_binding") != expected_binding:
            raise AssertionError(f"{prefix}: wrong movement_camera_binding")
        delta = question.get("delta")
        if not isinstance(delta, list) or len(delta) != 3:
            raise AssertionError(f"{prefix}: invalid delta")
        distance = float(question["movement_distance_m"])
        if not math.isclose(math.sqrt(sum(float(value) ** 2 for value in delta)), distance, abs_tol=2e-6):
            raise AssertionError(f"{prefix}: delta norm differs from movement_distance_m")
        if question_type == "object_move_allocentric":
            unit = {
                "north": (0.0, 1.0),
                "south": (0.0, -1.0),
                "east": (1.0, 0.0),
                "west": (-1.0, 0.0),
                "northeast": (math.sqrt(0.5), math.sqrt(0.5)),
                "northwest": (-math.sqrt(0.5), math.sqrt(0.5)),
                "southeast": (math.sqrt(0.5), -math.sqrt(0.5)),
                "southwest": (-math.sqrt(0.5), -math.sqrt(0.5)),
            }.get(str(question.get("movement_direction") or ""))
            if unit is None or not (
                math.isclose(float(delta[0]), unit[0] * distance, abs_tol=2e-6)
                and math.isclose(float(delta[1]), unit[1] * distance, abs_tol=2e-6)
                and math.isclose(float(delta[2]), 0.0, abs_tol=2e-6)
            ):
                raise AssertionError(f"{prefix}: allocentric delta/direction mismatch")
    elif question_type == "object_move_occlusion":
        bindings = question.get("camera_bindings") or {}
        if question.get("occlusion_semantics_version") != 2:
            raise AssertionError(f"{prefix}: occlusion_semantics_version is not 2")
        if bindings.get("movement") != "frame_1" or bindings.get("visibility") != "frame_2":
            raise AssertionError(f"{prefix}: wrong occlusion camera bindings")
        if len(collect_image_names(question)) < 2:
            raise AssertionError(f"{prefix}: occlusion question has no cross-frame route")
    elif question_type == "object_rotate_object_centric":
        if question.get("reference_frame") != "object_centric":
            raise AssertionError(f"{prefix}: rotation is not object-centric")
        if (question.get("camera_bindings") or {}).get("answer") != "object_defined":
            raise AssertionError(f"{prefix}: wrong rotation answer frame")
    elif question_type == "object_remove":
        required = ("removed_obj_id", "old_visibility", "new_visibility")
        if any(key not in question for key in required):
            raise AssertionError(f"{prefix}: incomplete removal semantics")


def audit(
    shard_root: Path,
    val_benchmark: Path,
) -> dict[str, Any]:
    expected_paths = {
        shard: shard_root / f"{shard}l2strict" / "benchmark.json" for shard in SHARDS
    }
    missing = [str(path) for path in expected_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed shard benchmarks: {missing}")

    val_document = _load_document(val_benchmark)
    val_scenes = {str(row.get("scene_id") or "") for row in val_document["questions"]}
    val_identities = {_visible_identity(row) for row in val_document["questions"]}
    identities: set[tuple[Any, ...]] = set()
    all_questions: list[dict[str, Any]] = []
    shard_reports: dict[str, Any] = {}
    for shard, path in expected_paths.items():
        document = _load_document(path)
        questions = document["questions"]
        counts = Counter(str(row.get("type") or "") for row in questions)
        unknown = set(counts) - set(EXPECTED_TYPES)
        if unknown:
            raise AssertionError(f"{shard}: unexpected question types {sorted(unknown)}")
        for index, question in enumerate(questions):
            identity = _visible_identity(question)
            scene_id = str(question.get("scene_id") or "")
            if identity in identities:
                raise AssertionError(f"{shard}:{index}: duplicate visible identity")
            if identity in val_identities or scene_id in val_scenes:
                raise AssertionError(f"{shard}:{index}: train/val isolation failure")
            identities.add(identity)
            _validate_marker(question, index)
            record = build_fact_record(question)
            validate_fact_consistency(record)
            validate_answer_mapping(question, record)
        all_questions.extend(questions)
        shard_reports[shard] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "question_count": len(questions),
            "by_type": dict(sorted(counts.items())),
        }

    total_counts = Counter(str(row.get("type") or "") for row in all_questions)
    scene_counts = Counter(str(row.get("scene_id") or "") for row in all_questions)
    return {
        "schema_version": "strict-l2-train-shards-audit-v1",
        "status": "pass",
        "question_identity": "exact question text plus complete ordered image route",
        "shard_count": len(expected_paths),
        "question_count": len(all_questions),
        "unique_visible_identity_count": len(identities),
        "by_type": dict(sorted(total_counts.items())),
        "by_scene": dict(sorted(scene_counts.items())),
        "val_benchmark": str(val_benchmark.resolve()),
        "val_benchmark_sha256": _sha256(val_benchmark),
        "val_scene_overlap_count": 0,
        "val_visible_identity_overlap_count": 0,
        "shards": shard_reports,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-root", type=Path, default=Path("output_train/scannetpp_polit")
    )
    parser.add_argument(
        "--val-benchmark", type=Path, default=Path("output_val/benchmark.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_train/scannetpp_polit/strict_l2_train_shards_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args.shard_root, args.val_benchmark)
    _atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
