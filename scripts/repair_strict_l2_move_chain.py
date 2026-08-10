#!/usr/bin/env python3
"""Stage and validate strict camera-facing L2 object-move repairs by split.

The stage command never mutates canonical inputs. It repairs every
``object_move_object_centric`` question using raw scene geometry and the first
frame camera pose, writes an isolated output tree, and records source hashes in
a manifest. The apply command verifies those hashes, creates timestamped
backups, and atomically installs the staged files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recompute_l2_move_benchmark import (
    L2_OBJECT_MOVE_SEMANTICS_VERSION,
    SceneResources,
    _load_scene_resources,
    _load_templates,
    _metadata_index,
    _pose_for_question,
    repair_benchmark,
)
from src.cot.images import collect_image_names
from src.quality_control import compute_statistics


QUESTION_TYPE = "object_move_object_centric"
STRICT_MOVEMENT_FRAME = "moved_object_facing_first_camera"
STRICT_ANSWER_FRAME = "query_object_facing_first_camera"
SAMPLE_SEED = "strict-camera-facing-val-sample-v1"
ATTACHMENT_RESULT_SCHEMA = "l2-attachment-audit-result-v1"


def _load_document(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("questions"), list):
        raise ValueError(f"{path}: expected an object with a questions list")
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
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


def _load_trusted_attachment_evidence(
    root: Path, *, expected_benchmark_sha256: str
) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    evidence: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    result_paths = sorted(root.rglob("*.json"))
    if not result_paths:
        raise ValueError(f"no attachment audit results found under {root}")
    for path in result_paths:
        payload = _load_document_like(path)
        if payload.get("schema_version") != ATTACHMENT_RESULT_SCHEMA:
            continue
        if payload.get("benchmark_sha256") != expected_benchmark_sha256:
            raise ValueError(f"{path}: attachment audit benchmark hash mismatch")
        if payload.get("verdict") != "pass":
            continue
        scene_id = str(payload.get("scene_id") or "")
        parent_id = int(payload["parent"]["obj_id"])
        child_id = int(payload["child"]["obj_id"])
        key = (parent_id, child_id)
        record = {
            "check_id": str(payload["check_id"]),
            "relation": str(payload.get("relation") or ""),
            "confidence": payload.get("confidence"),
            "frame_key": str(payload.get("frame_key") or ""),
            "result_path": str(path.resolve()),
        }
        existing = evidence[scene_id].get(key)
        if existing is None or float(record["confidence"] or 0.0) > float(
            existing["confidence"] or 0.0
        ):
            evidence[scene_id][key] = record
    if not evidence:
        raise ValueError(f"no trusted pass attachment results found under {root}")
    return dict(evidence)


def _apply_trusted_attachment_evidence(
    resources: dict[str, SceneResources],
    evidence: dict[str, dict[tuple[int, int], dict[str, Any]]],
) -> dict[str, int]:
    applied = already_present = 0
    for scene_id, edges in evidence.items():
        resource = resources.get(scene_id)
        if resource is None:
            continue
        trusted = getattr(resource, "trusted_attachment_evidence", {})
        for (parent_id, child_id), record in edges.items():
            children = resource.attachment_graph.setdefault(parent_id, [])
            if child_id in children:
                already_present += 1
            else:
                children.append(child_id)
                children.sort()
                applied += 1
            trusted[(parent_id, child_id)] = record
        resource.trusted_attachment_evidence = trusted
        resource.motion_cache.clear()
    return {"applied_missing_edges": applied, "already_present_edges": already_present}


def _scene_ids(document: dict[str, Any]) -> set[str]:
    return {str(question.get("scene_id") or "") for question in document["questions"]}


def _visible_identity(question: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(question.get("type") or ""),
        str(question.get("question") or ""),
        tuple(collect_image_names(question)),
    )


def _source_link_key(question: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(question.get("scene_id") or ""),
        tuple(collect_image_names(question)),
        str(question.get("question") or ""),
        int(question.get("moved_obj_id", -1)),
        int(question.get("query_obj_id", -1)),
        int(question.get("obj_ref_id", -1)),
    )


def _assert_split_isolation(
    split: str,
    document: dict[str, Any],
    other_path: Path | None,
) -> dict[str, Any]:
    if other_path is None:
        raise ValueError("--other-split-benchmark is required for the hard isolation check")
    other = _load_document(other_path)
    scene_overlap = sorted(_scene_ids(document) & _scene_ids(other))
    identity_overlap = {
        _visible_identity(question) for question in document["questions"]
    } & {_visible_identity(question) for question in other["questions"]}
    if scene_overlap or identity_overlap:
        raise AssertionError(
            f"{split}: val/train isolation failed: scene_overlap={scene_overlap[:10]}, "
            f"visible_identity_overlap_count={len(identity_overlap)}"
        )
    return {
        "other_split_benchmark": str(other_path.resolve()),
        "scene_overlap_count": 0,
        "visible_identity_overlap_count": 0,
    }


def _target_questions(documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        question
        for document in documents
        for question in document["questions"]
        if question.get("type") == QUESTION_TYPE
    ]


def _load_resources(
    questions: Sequence[dict[str, Any]],
    *,
    metadata_roots: Sequence[Path],
    scannet_root: Path,
    scannetpp_root: Path,
) -> tuple[dict[str, SceneResources], dict[str, str]]:
    index = _metadata_index(metadata_roots)
    resources: dict[str, SceneResources] = {}
    errors: dict[str, str] = {}
    scene_ids = sorted({str(question.get("scene_id") or "") for question in questions})
    for position, scene_id in enumerate(scene_ids, start=1):
        print(f"loading scene {position}/{len(scene_ids)}: {scene_id}", flush=True)
        try:
            resources[scene_id] = _load_scene_resources(
                scene_id,
                metadata_index=index,
                scannet_root=scannet_root,
                scannetpp_root=scannetpp_root,
                distance_geometry="aabb",
                needs_distance=False,
                needs_camera_pose=True,
            )
        except Exception as exc:
            errors[scene_id] = f"{type(exc).__name__}: {exc}"
    if errors:
        raise RuntimeError(
            "systemic scene-resource failure; no staged outputs were written: "
            + _stable_json(errors)
        )

    pose_errors: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        scene_id = str(question.get("scene_id") or "")
        try:
            _pose_for_question(question, resources[scene_id])
        except Exception as exc:
            pose_errors[scene_id].append(
                f"{question.get('image_name')}: {type(exc).__name__}: {exc}"
            )
    if pose_errors:
        raise RuntimeError(
            "systemic camera-pose failure; no staged outputs were written: "
            + _stable_json(dict(sorted(pose_errors.items())))
        )
    return resources, {
        scene_id: str(resource.metadata_path) if resource.metadata_path else ""
        for scene_id, resource in sorted(resources.items())
    }


def _repair_document(
    source: dict[str, Any],
    *,
    resources: dict[str, SceneResources],
    templates: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[Any, ...], dict[str, Any]]]:
    repaired, audit = repair_benchmark(
        source,
        resources_by_scene=resources,
        scene_errors={},
        templates=templates,
        target_types={QUESTION_TYPE},
        legacy_only=False,
        rebalance=False,
        deduplicate_against_preserved=False,
        recover_object_centric_from_text=True,
        object_centric_template="freeze",
    )
    repaired["version"] = source.get("version", "1.0")
    repaired["statistics"] = compute_statistics(repaired["questions"])

    replacements: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in audit["questions"]:
        if row.get("status") != "kept":
            continue
        source_index = int(row["source_index"])
        output_index = int(row["output_index"])
        key = _source_link_key(source["questions"][source_index])
        candidate = repaired["questions"][output_index]
        existing = replacements.get(key)
        if existing is not None and _stable_json(existing) != _stable_json(candidate):
            raise AssertionError("one source identity resolved to two different repairs")
        replacements[key] = candidate
    _validate_strict_questions(repaired["questions"])
    return repaired, audit, replacements


def _validate_strict_questions(questions: Sequence[dict[str, Any]]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for index, question in enumerate(questions):
        if question.get("type") != QUESTION_TYPE:
            continue
        if question.get("movement_semantics_version") != L2_OBJECT_MOVE_SEMANTICS_VERSION:
            raise AssertionError(f"question {index}: non-v2 object-centric record")
        required = {
            "movement_reference_frame": STRICT_MOVEMENT_FRAME,
            "movement_camera_binding": "frame_1",
            "movement_frame_frozen": True,
            "answer_reference_frame": STRICT_ANSWER_FRAME,
            "answer_camera_binding": "frame_1",
            "answer_frame_frozen": True,
        }
        for key, expected in required.items():
            if question.get(key) != expected:
                raise AssertionError(
                    f"question {index}: {key}={question.get(key)!r}, expected {expected!r}"
                )
        if int(question["movement_frame_anchor_obj_id"]) != int(question["moved_obj_id"]):
            raise AssertionError(f"question {index}: movement anchor is not the moved object")
        if int(question["answer_frame_anchor_obj_id"]) != int(question["query_obj_id"]):
            raise AssertionError(f"question {index}: answer anchor is not the query object")
        for prefix in ("movement_frame", "answer_frame"):
            forward = np.asarray(question[f"{prefix}_forward_world"], dtype=np.float64)
            right = np.asarray(question[f"{prefix}_right_world"], dtype=np.float64)
            if not (
                forward.shape == right.shape == (3,)
                and np.allclose(np.linalg.norm(forward), 1.0, atol=1e-6)
                and np.allclose(np.linalg.norm(right), 1.0, atol=1e-6)
                and np.allclose(right, [forward[1], -forward[0], 0.0], atol=1e-6)
            ):
                raise AssertionError(f"question {index}: invalid {prefix} axes")
        movement_forward = np.asarray(
            question["movement_frame_forward_world"], dtype=np.float64
        )
        movement_right = np.asarray(
            question["movement_frame_right_world"], dtype=np.float64
        )
        direction_vectors = {
            "forward": movement_forward,
            "forward-right": movement_forward + movement_right,
            "right": movement_right,
            "backward-right": -movement_forward + movement_right,
            "backward": -movement_forward,
            "backward-left": -movement_forward - movement_right,
            "left": -movement_right,
            "forward-left": movement_forward - movement_right,
        }
        movement_direction = str(question.get("movement_direction") or "")
        if movement_direction not in direction_vectors:
            raise AssertionError(f"question {index}: invalid movement direction")
        expected_unit = direction_vectors[movement_direction]
        expected_unit /= np.linalg.norm(expected_unit)
        distance = float(question["movement_distance_m"])
        delta = np.asarray(question["delta"], dtype=np.float64)
        if not np.allclose(delta, expected_unit * distance, atol=2e-6):
            raise AssertionError(
                f"question {index}: delta does not match the strict movement frame"
            )
        options = question.get("options")
        answer = str(question.get("answer") or "")
        if not isinstance(options, list) or len(answer) != 1:
            raise AssertionError(f"question {index}: invalid options/answer")
        answer_index = ord(answer) - ord("A")
        if not 0 <= answer_index < len(options) or options[answer_index] != question.get(
            "correct_value"
        ):
            raise AssertionError(f"question {index}: answer mapping is inconsistent")
        identity = _visible_identity(question)
        if identity in seen:
            raise AssertionError(f"question {index}: duplicate text/image identity")
        seen.add(identity)


def _propagate_subset(
    source: dict[str, Any],
    *,
    canonical: dict[str, Any],
    replacements: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    canonical_by_visible: dict[tuple[Any, ...], dict[str, Any]] = {}
    for question in canonical["questions"]:
        canonical_by_visible.setdefault(_visible_identity(question), question)

    output_questions: list[dict[str, Any]] = []
    replaced = dropped = canonicalized = 0
    for question in source["questions"]:
        if question.get("type") == QUESTION_TYPE:
            candidate = replacements.get(_source_link_key(question))
            if candidate is None:
                dropped += 1
                continue
            output_questions.append(copy.deepcopy(candidate))
            replaced += 1
            continue
        canonical_question = canonical_by_visible.get(_visible_identity(question))
        if canonical_question is None:
            raise AssertionError(
                "validation subset contains a non-object-centric question absent from "
                "the canonical output_val/benchmark.json"
            )
        output_questions.append(copy.deepcopy(canonical_question))
        canonicalized += 1
    output = copy.deepcopy(source)
    output["questions"] = output_questions
    output["statistics"] = compute_statistics(output_questions)
    return output, {
        "object_centric_replaced": replaced,
        "object_centric_dropped": dropped,
        "non_object_centric_canonicalized": canonicalized,
    }


def _sample_exact_per_type(
    source: dict[str, Any],
    *,
    canonical: dict[str, Any],
    per_type: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    type_order = list(dict.fromkeys(str(q.get("type") or "") for q in source["questions"]))
    canonical_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in canonical["questions"]:
        canonical_by_type[str(question.get("type") or "")].append(question)

    output_questions: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for question_type in type_order:
        pool = canonical_by_type[question_type]
        if len(pool) < per_type:
            raise RuntimeError(
                f"{question_type}: need {per_type} repaired val questions, found {len(pool)}"
            )
        pool_by_visible = {_visible_identity(question): question for question in pool}
        selected: list[dict[str, Any]] = []
        selected_ids: set[tuple[Any, ...]] = set()
        preserved = 0
        for old in source["questions"]:
            if old.get("type") != question_type:
                continue
            candidate = pool_by_visible.get(_visible_identity(old))
            if candidate is None:
                continue
            identity = _visible_identity(candidate)
            if identity in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(identity)
            preserved += 1
        remaining = [q for q in pool if _visible_identity(q) not in selected_ids]
        remaining.sort(
            key=lambda q: _sha256_bytes(
                f"{SAMPLE_SEED}|{question_type}|{_stable_json(_visible_identity(q))}".encode(
                    "utf-8"
                )
            )
        )
        selected.extend(remaining[: max(0, per_type - len(selected))])
        selected = selected[:per_type]
        if len(selected) != per_type:
            raise AssertionError(f"{question_type}: failed to fill sample quota")
        output_questions.extend(copy.deepcopy(selected))
        report[question_type] = {
            "quota": per_type,
            "preserved_exact": preserved,
            "backfilled": per_type - preserved,
            "available": len(pool),
        }
    output = copy.deepcopy(source)
    output["questions"] = output_questions
    output["statistics"] = compute_statistics(output_questions)
    return output, report


def _assert_subset_of_canonical(
    name: str, subset: dict[str, Any], canonical: dict[str, Any]
) -> None:
    canonical_ids = {_visible_identity(question) for question in canonical["questions"]}
    missing = [
        _visible_identity(question)
        for question in subset["questions"]
        if _visible_identity(question) not in canonical_ids
    ]
    if missing:
        raise AssertionError(f"{name}: {len(missing)} questions are absent from canonical val")


def _stage_output(
    output_root: Path,
    relative: Path,
    source_path: Path,
    document: dict[str, Any],
) -> dict[str, Any]:
    staged_path = output_root / relative
    _atomic_write_json(staged_path, document)
    return {
        "relative_path": relative.as_posix(),
        "source_path": str(source_path.resolve()),
        "source_sha256": _file_sha256(source_path),
        "staged_sha256": _file_sha256(staged_path),
        "question_count": len(document["questions"]),
        "by_type": dict(sorted(Counter(str(q.get("type") or "") for q in document["questions"]).items())),
    }


def stage(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_path = args.benchmark.resolve()
    benchmark = _load_document(benchmark_path)
    isolation = _assert_split_isolation(
        args.split, benchmark, args.other_split_benchmark.resolve()
    )
    optional_sources: list[tuple[str, Path, dict[str, Any]]] = []
    for name in ("benchmark_all", "audit_benchmark", "sample", "sample_no_occlusion"):
        path = getattr(args, name)
        if path is not None:
            resolved = path.resolve()
            optional_sources.append((name, resolved, _load_document(resolved)))
    if args.split == "train" and optional_sources:
        raise ValueError("train staging accepts only --benchmark")
    if args.split == "val":
        required = {"benchmark_all", "audit_benchmark", "sample", "sample_no_occlusion"}
        present = {name for name, _path, _doc in optional_sources}
        if present != required:
            raise ValueError(f"val staging requires {sorted(required - present)}")

    repair_documents = [benchmark]
    repair_documents.extend(
        doc for name, _path, doc in optional_sources if name == "benchmark_all"
    )
    targets = _target_questions(repair_documents)
    resources, metadata_paths = _load_resources(
        targets,
        metadata_roots=[path.resolve() for path in args.metadata_root],
        scannet_root=args.scannet_root.resolve(),
        scannetpp_root=args.scannetpp_root.resolve(),
    )
    attachment_override_report = None
    if args.trusted_attachment_results_root is not None:
        evidence = _load_trusted_attachment_evidence(
            args.trusted_attachment_results_root.resolve(),
            expected_benchmark_sha256=_file_sha256(benchmark_path),
        )
        attachment_override_report = _apply_trusted_attachment_evidence(
            resources, evidence
        )
    templates = _load_templates()
    repaired_main, main_audit, replacements = _repair_document(
        benchmark, resources=resources, templates=templates
    )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_root / "benchmark.repair_audit.json", main_audit)
    output_records = [
        _stage_output(output_root, Path("benchmark.json"), benchmark_path, repaired_main)
    ]
    reports: dict[str, Any] = {"benchmark": main_audit}

    if args.split == "val":
        source_by_name = {name: (path, doc) for name, path, doc in optional_sources}
        all_path, all_source = source_by_name["benchmark_all"]
        repaired_all, all_audit, _all_replacements = _repair_document(
            all_source, resources=resources, templates=templates
        )
        _atomic_write_json(output_root / "benchmark_all.repair_audit.json", all_audit)
        output_records.append(
            _stage_output(output_root, Path("benchmark_all.json"), all_path, repaired_all)
        )
        reports["benchmark_all"] = all_audit

        audit_path, audit_source = source_by_name["audit_benchmark"]
        repaired_audit, propagation_report = _propagate_subset(
            audit_source, canonical=repaired_main, replacements=replacements
        )
        _assert_subset_of_canonical("audit_benchmark", repaired_audit, repaired_main)
        output_records.append(
            _stage_output(
                output_root,
                Path("l2_attachment_audit_full/benchmark.json"),
                audit_path,
                repaired_audit,
            )
        )
        reports["audit_benchmark"] = propagation_report

        for source_name, relative in (
            ("sample", Path("l2_attachment_audit_full/benchmark_per_type_50.json")),
            (
                "sample_no_occlusion",
                Path("l2_attachment_audit_full/benchmark_per_type_50_v2_no_occlusion.json"),
            ),
        ):
            source_path, source_doc = source_by_name[source_name]
            repaired_sample, sample_report = _sample_exact_per_type(
                source_doc, canonical=repaired_main, per_type=50
            )
            _assert_subset_of_canonical(source_name, repaired_sample, repaired_main)
            output_records.append(
                _stage_output(output_root, relative, source_path, repaired_sample)
            )
            reports[source_name] = sample_report

    manifest = {
        "schema_version": "strict-camera-facing-l2-repair-stage-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "semantics": {
            "movement_semantics_version": L2_OBJECT_MOVE_SEMANTICS_VERSION,
            "movement_reference_frame": STRICT_MOVEMENT_FRAME,
            "answer_reference_frame": STRICT_ANSWER_FRAME,
        },
        "isolation": isolation,
        "metadata_paths": metadata_paths,
        "trusted_attachment_overrides": attachment_override_report,
        "outputs": output_records,
        "reports": reports,
    }
    _atomic_write_json(output_root / "repair_manifest.json", manifest)
    return manifest


def apply(args: argparse.Namespace) -> dict[str, Any]:
    stage_root = args.stage_dir.resolve()
    manifest = _load_document_like(stage_root / "repair_manifest.json")
    if manifest.get("schema_version") != "strict-camera-facing-l2-repair-stage-v1":
        raise ValueError("unsupported repair manifest")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checked: list[tuple[Path, Path, Path]] = []
    for record in manifest["outputs"]:
        source = Path(record["source_path"])
        staged = stage_root / record["relative_path"]
        if _file_sha256(source) != record["source_sha256"]:
            raise RuntimeError(f"source changed after staging: {source}")
        if _file_sha256(staged) != record["staged_sha256"]:
            raise RuntimeError(f"staged file hash mismatch: {staged}")
        backup = source.with_name(f"{source.stem}.before_strict_camera_{timestamp}{source.suffix}")
        if backup.exists():
            raise FileExistsError(f"backup already exists: {backup}")
        checked.append((source, staged, backup))
    for source, _staged, backup in checked:
        shutil.copy2(source, backup)
    for source, staged, _backup in checked:
        temporary = source.with_name(f".{source.name}.strict-camera.tmp")
        shutil.copy2(staged, temporary)
        os.replace(temporary, source)
    return {
        "applied": len(checked),
        "backups": {str(source): str(backup) for source, _staged, backup in checked},
    }


def _load_document_like(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--split", choices=("val", "train"), required=True)
    stage_parser.add_argument("--benchmark", type=Path, required=True)
    stage_parser.add_argument("--other-split-benchmark", type=Path, required=True)
    stage_parser.add_argument("--benchmark-all", type=Path)
    stage_parser.add_argument("--audit-benchmark", type=Path)
    stage_parser.add_argument("--sample", type=Path)
    stage_parser.add_argument("--sample-no-occlusion", type=Path)
    stage_parser.add_argument("--metadata-root", action="append", type=Path, required=True)
    stage_parser.add_argument("--scannet-root", type=Path, required=True)
    stage_parser.add_argument("--scannetpp-root", type=Path, required=True)
    stage_parser.add_argument("--output-dir", type=Path, required=True)
    stage_parser.add_argument("--trusted-attachment-results-root", type=Path)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--stage-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = stage(args) if args.command == "stage" else apply(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
