#!/usr/bin/env python3
"""Append audited strict camera-facing object-centric train questions."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recompute_l2_move_benchmark import (
    SceneResources,
    _object_facing_ground_axes,
    _pose_for_question,
)
from scripts.repair_strict_l2_move_chain import (
    QUESTION_TYPE,
    _file_sha256,
    _load_document,
    _load_resources,
    _stable_json,
    _validate_strict_questions,
    _visible_identity,
)
from src.cot.images import collect_image_names, resolve_image_paths
from src.quality_control import compute_statistics


@dataclass(frozen=True)
class SourceQuestion:
    path: Path
    index: int
    question: dict[str, Any]


def _identity_digest(question: dict[str, Any]) -> str:
    payload = json.dumps(
        [str(question.get("question") or ""), collect_image_names(question)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _document_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_sources(pattern: str) -> tuple[list[SourceQuestion], list[Path]]:
    paths = sorted(Path(value).resolve() for value in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no shard benchmark matches {pattern!r}")
    records: list[SourceQuestion] = []
    for path in paths:
        document = _load_document(path)
        for index, question in enumerate(document["questions"]):
            records.append(SourceQuestion(path, index, copy.deepcopy(question)))
    return records, paths


def _validate_real_geometry(
    question: dict[str, Any], resource: SceneResources, *, atol: float = 2e-6
) -> None:
    pose = _pose_for_question(question, resource)
    objects = resource.objects_by_id
    moved_id = int(question["moved_obj_id"])
    query_id = int(question["query_obj_id"])
    if moved_id not in objects or query_id not in objects:
        raise KeyError(
            f"{resource.scene_id}: missing moved/query object {moved_id}/{query_id}"
        )
    camera_position = np.asarray(pose.position, dtype=np.float64)
    expected_movement = _object_facing_ground_axes(
        np.asarray(objects[moved_id]["center"], dtype=np.float64), camera_position
    )
    expected_answer = _object_facing_ground_axes(
        np.asarray(objects[query_id]["center"], dtype=np.float64), camera_position
    )
    if expected_movement is None or expected_answer is None:
        raise ValueError(f"{resource.scene_id}: degenerate camera-facing object axis")
    for prefix, expected in (
        ("movement_frame", expected_movement),
        ("answer_frame", expected_answer),
    ):
        for suffix, vector in zip(("forward_world", "right_world"), expected):
            stored = np.asarray(question[f"{prefix}_{suffix}"], dtype=np.float64)
            if stored.shape != (3,) or not np.allclose(stored, vector, atol=atol):
                raise AssertionError(
                    f"{resource.scene_id}/{question.get('image_name')}: "
                    f"{prefix}_{suffix} disagrees with the first camera pose"
                )


def _scene_ids(document: dict[str, Any]) -> set[str]:
    return {str(question.get("scene_id") or "") for question in document["questions"]}


def prepare_merge(
    canonical: dict[str, Any],
    validation: dict[str, Any],
    sources: Sequence[SourceQuestion],
    *,
    resources: dict[str, SceneResources],
    image_validator: Callable[[SourceQuestion], None],
    expected_source_count: int | None = 462,
    expected_existing_count: int | None = 51,
    expected_append_count: int | None = 411,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if expected_source_count is not None and len(sources) != expected_source_count:
        raise AssertionError(
            f"source row count is {len(sources)}, expected {expected_source_count}"
        )
    source_questions = [record.question for record in sources]
    non_target = [index for index, row in enumerate(source_questions) if row.get("type") != QUESTION_TYPE]
    if non_target:
        raise AssertionError(f"source shards contain non-target rows: {non_target[:10]}")
    _validate_strict_questions(source_questions)

    canonical_identities = {_visible_identity(row) for row in canonical["questions"]}
    val_identities = {_visible_identity(row) for row in validation["questions"]}
    val_scenes = _scene_ids(validation)
    canonical_val_scenes = _scene_ids(canonical) & val_scenes
    canonical_val_identities = canonical_identities & val_identities
    if canonical_val_scenes or canonical_val_identities:
        raise AssertionError(
            "existing train/val isolation failed: "
            f"scene_overlap={sorted(canonical_val_scenes)[:10]}, "
            f"visible_identity_overlap_count={len(canonical_val_identities)}"
        )
    additions: list[dict[str, Any]] = []
    already_present: list[dict[str, Any]] = []
    for record in sources:
        question = record.question
        identity = _visible_identity(question)
        scene_id = str(question.get("scene_id") or "")
        if identity in val_identities or scene_id in val_scenes:
            raise AssertionError(
                f"train/val isolation failed for {scene_id}/{record.path.name}:{record.index}"
            )
        resource = resources.get(scene_id)
        if resource is None:
            raise KeyError(f"missing scene resources for {scene_id}")
        _validate_real_geometry(question, resource)
        image_validator(record)
        if identity in canonical_identities:
            already_present.append(
                {
                    "source": str(record.path),
                    "source_index": record.index,
                    "scene_id": scene_id,
                    "identity_sha256": _identity_digest(question),
                    "reason": "same_question_text_and_complete_ordered_image_route",
                }
            )
            continue
        canonical_identities.add(identity)
        additions.append(copy.deepcopy(question))

    if expected_existing_count is not None and len(already_present) != expected_existing_count:
        raise AssertionError(
            f"already-present count is {len(already_present)}, expected {expected_existing_count}"
        )
    if expected_append_count is not None and len(additions) != expected_append_count:
        raise AssertionError(
            f"append count is {len(additions)}, expected {expected_append_count}"
        )

    before = copy.deepcopy(canonical["questions"])
    merged = copy.deepcopy(canonical)
    merged["questions"].extend(additions)
    if merged["questions"][: len(before)] != before:
        raise AssertionError("existing canonical rows or order changed during merge")
    merged["statistics"] = compute_statistics(merged["questions"])

    scene_distribution = Counter(str(row.get("scene_id") or "") for row in additions)
    triple_distribution = Counter(
        (
            str(row.get("scene_id") or ""),
            int(row["moved_obj_id"]),
            int(row["query_obj_id"]),
            int(row["obj_ref_id"]),
        )
        for row in additions
    )
    audit = {
        "schema_version": "strict-object-centric-train-addon-merge-v1",
        "question_identity": "exact question text plus complete ordered image route",
        "source_count": len(sources),
        "strict_pass_count": len(source_questions),
        "internal_duplicate_count": 0,
        "already_present_count": len(already_present),
        "appended_count": len(additions),
        "canonical_before_count": len(before),
        "canonical_after_count": len(merged["questions"]),
        "already_present": already_present,
        "added_scene_distribution": dict(sorted(scene_distribution.items())),
        "added_unique_scene_object_triples": len(triple_distribution),
        "added_scene_object_triple_distribution": [
            {
                "scene_id": key[0],
                "moved_obj_id": key[1],
                "query_obj_id": key[2],
                "obj_ref_id": key[3],
                "count": value,
            }
            for key, value in sorted(triple_distribution.items())
        ],
    }
    return merged, audit


def _stage_json(path: Path, value: dict[str, Any]) -> tuple[Path, bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _document_bytes(value)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    with handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary, data


def _install(
    benchmark_path: Path,
    audit_path: Path,
    merged: dict[str, Any],
    audit: dict[str, Any],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = benchmark_path.with_name(f"{benchmark_path.stem}.before_objcentric_addon_{timestamp}{benchmark_path.suffix}")
    if backup.exists():
        raise FileExistsError(backup)
    benchmark_temp: Path | None = None
    audit_temp: Path | None = None
    try:
        final_bytes = _document_bytes(merged)
        audit = dict(audit)
        audit["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        audit["canonical_before_sha256"] = _file_sha256(benchmark_path)
        audit["canonical_after_sha256"] = _sha256(final_bytes)
        benchmark_temp, staged_final = _stage_json(benchmark_path, merged)
        if staged_final != final_bytes:
            raise AssertionError("staged benchmark bytes changed unexpectedly")
        audit_temp, _ = _stage_json(audit_path, audit)
        shutil.copy2(benchmark_path, backup)
        os.replace(benchmark_temp, benchmark_path)
        benchmark_temp = None
        os.replace(audit_temp, audit_path)
        audit_temp = None
    finally:
        for temporary in (benchmark_temp, audit_temp):
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return backup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("output_train/benchmark.json"))
    parser.add_argument("--val-benchmark", type=Path, default=Path("output_val/benchmark.json"))
    parser.add_argument("--shard-glob", default="output_train/objcentric/*0-*9obj/benchmark.json")
    parser.add_argument("--metadata-root", action="append", type=Path)
    parser.add_argument("--scannet-root", type=Path, required=True)
    parser.add_argument("--scannetpp-root", type=Path, required=True)
    parser.add_argument("--scannet-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-image-root", action="append", type=Path, default=[])
    parser.add_argument("--scannetpp-sensor", choices=("iphone", "dslr"), default="iphone")
    parser.add_argument("--audit", type=Path, default=Path("output_train/object_centric_addon_merge_audit.json"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sources, shard_paths = _load_sources(args.shard_glob)
    metadata_roots = args.metadata_root or [path.parent for path in shard_paths]
    resources, metadata_paths = _load_resources(
        [record.question for record in sources],
        metadata_roots=[path.resolve() for path in metadata_roots],
        scannet_root=args.scannet_root.resolve(),
        scannetpp_root=args.scannetpp_root.resolve(),
    )

    def validate_images(record: SourceQuestion) -> None:
        resolve_image_paths(
            record.question,
            benchmark_path=record.path,
            scannet_roots=[path.resolve() for path in args.scannet_image_root],
            scannetpp_roots=[path.resolve() for path in args.scannetpp_image_root],
            scannetpp_sensor=args.scannetpp_sensor,
            require_exists=True,
        )

    canonical = _load_document(args.benchmark)
    validation = _load_document(args.val_benchmark)
    merged, audit = prepare_merge(
        canonical,
        validation,
        sources,
        resources=resources,
        image_validator=validate_images,
    )
    audit.update(
        benchmark=str(args.benchmark.resolve()),
        val_benchmark=str(args.val_benchmark.resolve()),
        shard_files=[
            {"path": str(path), "sha256": _file_sha256(path)} for path in shard_paths
        ],
        scene_metadata=metadata_paths,
        val_benchmark_sha256=_file_sha256(args.val_benchmark),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        audit["canonical_before_sha256"] = _file_sha256(args.benchmark)
        audit["canonical_after_sha256"] = _sha256(_document_bytes(merged))
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    backup = _install(args.benchmark, args.audit, merged, audit)
    print(
        json.dumps(
            {
                "benchmark": str(args.benchmark.resolve()),
                "backup": str(backup.resolve()),
                "audit": str(args.audit.resolve()),
                "appended_count": audit["appended_count"],
                "final_count": len(merged["questions"]),
                "final_sha256": _file_sha256(args.benchmark),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
