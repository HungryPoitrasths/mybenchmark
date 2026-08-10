#!/usr/bin/env python3
"""Build the capped, diversity-selected strict object-centric val benchmark."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.repair_strict_l2_move_chain import (
    QUESTION_TYPE,
    _assert_subset_of_canonical,
    _stable_json,
    _validate_strict_questions,
    _visible_identity,
)
from src.cot.images import collect_image_names
from src.quality_control import compute_statistics


TARGET_COUNT = 293
DOMINANT_SCENE = "0d2ee665be"
DOMINANT_PAIR = "27->88"
DOMINANT_COUNT = 67
SELECTION_SEED = "strict-camera-diverse-val-293-v1"
RESTORED_SCENE = "c49a8c6cff"
RESTORED_PAIR = "45->44"
RESTORED_COUNT = 14


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("questions"), list):
        raise ValueError(f"{path}: expected an object with a questions list")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}|{_stable_json(value)}".encode("utf-8")
    ).hexdigest()


def _route(question: dict[str, Any]) -> tuple[str, ...]:
    return tuple(collect_image_names(question))


def _pair_key(question: dict[str, Any]) -> tuple[str, str]:
    return str(question.get("scene_id") or ""), str(
        question.get("attachment_pair_id") or ""
    )


def _distribution(questions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(questions)
    return {
        "count": len(values),
        "scene_counts": dict(sorted(Counter(str(q.get("scene_id")) for q in values).items())),
        "pair_counts": dict(
            sorted(
                Counter(f"{scene}|{pair}" for scene, pair in map(_pair_key, values)).items()
            )
        ),
        "unique_scene_count": len({_pair_key(q)[0] for q in values}),
        "unique_pair_count": len({_pair_key(q) for q in values}),
    }


def _union_questions(
    sources: Sequence[tuple[str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], list[str]]]:
    output: list[dict[str, Any]] = []
    provenance: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for source_name, document in sources:
        for question in document["questions"]:
            if question.get("type") != QUESTION_TYPE:
                continue
            identity = _visible_identity(question)
            provenance[identity].append(source_name)
            if identity not in seen:
                output.append(copy.deepcopy(question))
                seen.add(identity)
    return output, dict(provenance)


def _select_dominant(
    candidates: Sequence[dict[str, Any]], mandatory_ids: set[tuple[Any, ...]]
) -> set[tuple[Any, ...]]:
    dominant = [
        q
        for q in candidates
        if _pair_key(q) == (DOMINANT_SCENE, DOMINANT_PAIR)
    ]
    if len(dominant) != 86:
        raise AssertionError(f"expected 86 dominant-pair candidates, found {len(dominant)}")
    selected = [q for q in dominant if _visible_identity(q) in mandatory_ids]
    if len(selected) > DOMINANT_COUNT:
        raise AssertionError("mandatory sample exceeds dominant-pair quota")
    remaining = [q for q in dominant if _visible_identity(q) not in mandatory_ids]
    while len(selected) < DOMINANT_COUNT:
        ref_counts = Counter(int(q.get("obj_ref_id", -1)) for q in selected)
        frame_counts = Counter(
            (str(q.get("image_name")), str(q.get("reasoning_frame_2"))) for q in selected
        )
        movement_counts = Counter(
            (str(q.get("movement_direction")), float(q.get("movement_distance_m", 0.0)))
            for q in selected
        )
        answer_counts = Counter(str(q.get("correct_value")) for q in selected)

        def score(q: dict[str, Any]) -> tuple[Any, ...]:
            identity = _visible_identity(q)
            return (
                ref_counts[int(q.get("obj_ref_id", -1))],
                frame_counts[(str(q.get("image_name")), str(q.get("reasoning_frame_2")))],
                movement_counts[(str(q.get("movement_direction")), float(q.get("movement_distance_m", 0.0)))],
                answer_counts[str(q.get("correct_value"))],
                _sha(identity),
            )

        chosen = min(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return {_visible_identity(q) for q in selected}


def _canonicalize_subset(
    source: dict[str, Any], canonical: dict[str, Any]
) -> dict[str, Any]:
    canonical_by_id = {
        _visible_identity(q): q for q in canonical["questions"]
    }
    output = copy.deepcopy(source)
    output["questions"] = [
        copy.deepcopy(canonical_by_id[_visible_identity(q)])
        for q in source["questions"]
        if _visible_identity(q) in canonical_by_id
    ]
    output["statistics"] = compute_statistics(output["questions"])
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    main = _load(args.repaired_main.resolve())
    all_doc = _load(args.repaired_all.resolve())
    audit = _load(args.repaired_audit.resolve())
    sample = _load(args.sample.resolve())
    sample_no_occ = _load(args.sample_no_occlusion.resolve())
    train = _load(args.train_benchmark.resolve())

    candidates, provenance = _union_questions(
        (("repaired_main", main), ("repaired_all", all_doc))
    )
    if len(candidates) != 312:
        raise AssertionError(f"expected 312 union candidates, found {len(candidates)}")
    _validate_strict_questions(candidates)

    mandatory_ids = {
        _visible_identity(q)
        for document in (sample, sample_no_occ)
        for q in document["questions"]
        if q.get("type") == QUESTION_TYPE
    }
    restored = [
        q
        for q in candidates
        if _pair_key(q) == (RESTORED_SCENE, RESTORED_PAIR)
    ]
    if len(restored) != RESTORED_COUNT:
        raise AssertionError(
            f"expected {RESTORED_COUNT} restored human-pass questions, found {len(restored)}"
        )
    mandatory_ids.update(_visible_identity(q) for q in restored)
    dominant_selected = _select_dominant(candidates, mandatory_ids)

    selected: list[dict[str, Any]] = []
    for question in candidates:
        if _pair_key(question) == (DOMINANT_SCENE, DOMINANT_PAIR):
            if _visible_identity(question) not in dominant_selected:
                continue
        selected.append(question)
    if len(selected) != TARGET_COUNT:
        raise AssertionError(f"expected {TARGET_COUNT} selected questions, found {len(selected)}")
    selected_ids = {_visible_identity(q) for q in selected}
    if not mandatory_ids <= selected_ids:
        raise AssertionError("a mandatory restored/sample question was not selected")

    output_questions: list[dict[str, Any]] = []
    inserted = False
    for question in main["questions"]:
        if question.get("type") == QUESTION_TYPE:
            if not inserted:
                output_questions.extend(copy.deepcopy(selected))
                inserted = True
            continue
        output_questions.append(copy.deepcopy(question))
    if not inserted:
        output_questions.extend(copy.deepcopy(selected))
    final = copy.deepcopy(main)
    final["questions"] = output_questions
    final["statistics"] = compute_statistics(output_questions)
    _validate_strict_questions(output_questions)

    selected_dist = _distribution(selected)
    if selected_dist["unique_scene_count"] != 12 or selected_dist["unique_pair_count"] != 27:
        raise AssertionError("scene/pair diversity coverage regressed")
    if selected_dist["pair_counts"].get(f"{DOMINANT_SCENE}|{DOMINANT_PAIR}") != DOMINANT_COUNT:
        raise AssertionError("dominant-pair cap was not enforced")

    final_ids = {_visible_identity(q) for q in output_questions}
    train_ids = {_visible_identity(q) for q in train["questions"]}
    final_scenes = {str(q.get("scene_id")) for q in output_questions}
    train_scenes = {str(q.get("scene_id")) for q in train["questions"]}
    if final_ids & train_ids or final_scenes & train_scenes:
        raise AssertionError("val/train isolation failed")

    final_audit = _canonicalize_subset(audit, final)
    final_sample = _canonicalize_subset(sample, final)
    final_sample_no_occ = _canonicalize_subset(sample_no_occ, final)
    if len(final_sample["questions"]) != len(sample["questions"]):
        raise AssertionError("the 6x50 sample changed membership")
    if len(final_sample_no_occ["questions"]) != len(sample_no_occ["questions"]):
        raise AssertionError("the 5x50 sample changed membership")
    _assert_subset_of_canonical("audit", final_audit, final)
    _assert_subset_of_canonical("sample", final_sample, final)
    _assert_subset_of_canonical("sample_no_occlusion", final_sample_no_occ, final)

    report_rows = []
    for question in candidates:
        identity = _visible_identity(question)
        is_selected = identity in selected_ids
        if identity in mandatory_ids:
            reason = "mandatory_sample_or_human_restore"
        elif _pair_key(question) != (DOMINANT_SCENE, DOMINANT_PAIR):
            reason = "preserve_scene_pair_coverage"
        elif is_selected:
            reason = "dominant_pair_diversity_selected"
        else:
            reason = "dominant_pair_cap_excluded"
        report_rows.append({
            "selected": is_selected,
            "reason": reason,
            "sources": provenance[identity],
            "scene_id": question.get("scene_id"),
            "attachment_pair_id": question.get("attachment_pair_id"),
            "obj_ref_id": question.get("obj_ref_id"),
            "obj_ref_label": question.get("obj_ref_label"),
            "movement_direction": question.get("movement_direction"),
            "movement_distance_m": question.get("movement_distance_m"),
            "correct_value": question.get("correct_value"),
            "question": question.get("question"),
            "image_route": list(_route(question)),
        })

    output_dir = args.output_dir.resolve()
    outputs = {
        "benchmark.json": final,
        "l2_attachment_audit_full/benchmark.json": final_audit,
        "l2_attachment_audit_full/benchmark_per_type_50.json": final_sample,
        "l2_attachment_audit_full/benchmark_per_type_50_v2_no_occlusion.json": final_sample_no_occ,
    }
    for relative, document in outputs.items():
        _write(output_dir / relative, document)
    report = {
        "schema_version": "strict-camera-diverse-val-selection-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_distribution": _distribution(candidates),
        "selected_distribution": selected_dist,
        "mandatory_identity_count": len(mandatory_ids),
        "rows": report_rows,
    }
    _write(output_dir / "object_move_object_centric_selection_report.json", report)
    manifest = {
        "schema_version": "strict-camera-diverse-val-finalize-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_question_counts": {
            relative: len(document["questions"]) for relative, document in outputs.items()
        },
        "object_centric_count": len(selected),
        "scene_count": selected_dist["unique_scene_count"],
        "pair_count": selected_dist["unique_pair_count"],
        "val_train_scene_overlap_count": 0,
        "val_train_visible_identity_overlap_count": 0,
    }
    _write(output_dir / "finalize_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repaired-main", type=Path, required=True)
    parser.add_argument("--repaired-all", type=Path, required=True)
    parser.add_argument("--repaired-audit", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sample-no-occlusion", type=Path, required=True)
    parser.add_argument("--train-benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(build_parser().parse_args(argv))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
