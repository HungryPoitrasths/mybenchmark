#!/usr/bin/env python3
"""Validate future-rollout manifests before generation or VLM evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sampled_type_vlm_eval import (  # noqa: E402
    ROLLOUT_MODES,
    RolloutManifest,
    _sha256_file,
    load_rollout_manifest,
    resolve_rollout_images,
)


L2_ROLLOUT_TYPES = (
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_move_allocentric",
    "object_rotate_object_centric",
)

PROVENANCE_FIELDS = (
    "model",
    "checkpoint",
    "seed",
    "prompt_version",
    "request_sha256",
    "response_id",
    "elapsed_seconds",
    "retries",
    "status",
)


def _context_signature(manifest: RolloutManifest, entry: dict[str, Any], mode: str) -> list[tuple[str, str]]:
    signature: list[tuple[str, str]] = []
    for item in entry[mode].get("media", []):
        if item.get("kind") != "context":
            continue
        raw_path = Path(str(item["path"]))
        path = raw_path if raw_path.is_absolute() else manifest.path.parent / raw_path
        digest = str(item.get("sha256") or "").lower()
        if not digest and path.is_file():
            digest = _sha256_file(path)
        signature.append((str(item.get("role")), digest or str(path.resolve())))
    return signature


def validate_manifest(
    manifest: RolloutManifest,
    *,
    mode: str,
    expected_per_type: int,
    strict_provenance: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    eligible_uids: list[str] = []
    counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()

    for uid in manifest.entry_order:
        entry = manifest.entries[uid]
        branch = entry.get(mode)
        if branch is None:
            rejection_counts["missing_modality_branch"] += 1
            continue
        if not branch["eligible"]:
            reasons = branch.get("rejection_reasons") or []
            if not reasons:
                errors.append(f"{uid}: ineligible {mode} entry has no rejection reason")
            for reason in reasons:
                rejection_counts[str(reason)] += 1
            continue

        qtype = str(entry.get("question_type") or "")
        scene_id = str(entry.get("scene_id") or "")
        if qtype not in L2_ROLLOUT_TYPES:
            errors.append(f"{uid}: missing or unsupported question_type {qtype!r}")
        if not scene_id:
            errors.append(f"{uid}: scene_id is required")
        eligible_uids.append(uid)
        counts[qtype] += 1

        _, media_error = resolve_rollout_images(
            {"question_uid": uid},
            manifest,
            mode=mode,
            context_only=False,
        )
        if media_error:
            errors.append(f"{uid}: {media_error}")

        if strict_provenance:
            generation = branch.get("generation")
            if not isinstance(generation, dict):
                errors.append(f"{uid}: {mode}.generation provenance object is required")
            else:
                missing = [field for field in PROVENANCE_FIELDS if field not in generation]
                if missing:
                    errors.append(f"{uid}: missing generation fields: {', '.join(missing)}")
            for media_index, item in enumerate(branch.get("media", [])):
                digest = str(item.get("sha256") or "")
                if len(digest) != 64:
                    errors.append(f"{uid}: media[{media_index}] requires a SHA-256 in strict mode")

    if mode == "picture":
        for qtype in L2_ROLLOUT_TYPES:
            if counts[qtype] != expected_per_type:
                errors.append(
                    f"{qtype}: expected exactly {expected_per_type} eligible picture entries, "
                    f"found {counts[qtype]}"
                )
    else:
        for qtype in L2_ROLLOUT_TYPES:
            if counts[qtype] > expected_per_type:
                errors.append(
                    f"{qtype}: video eligible count {counts[qtype]} exceeds cap {expected_per_type}"
                )

    return {
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "mode": mode,
        "entry_count": len(manifest.entry_order),
        "eligible_count": len(eligible_uids),
        "eligible_question_uids": eligible_uids,
        "eligible_by_type": dict(sorted(counts.items())),
        "rejections": dict(sorted(rejection_counts.items())),
        "errors": errors,
        "valid": not errors,
    }


def compare_paired_picture_manifests(manifests: list[RolloutManifest]) -> list[str]:
    if len(manifests) < 2:
        return []
    errors: list[str] = []
    reference = manifests[0]
    reference_uids = [
        uid
        for uid in reference.entry_order
        if isinstance(reference.entries[uid].get("picture"), dict)
        and reference.entries[uid]["picture"].get("eligible")
    ]
    for candidate in manifests[1:]:
        candidate_uids = [
            uid
            for uid in candidate.entry_order
            if isinstance(candidate.entries[uid].get("picture"), dict)
            and candidate.entries[uid]["picture"].get("eligible")
        ]
        if candidate_uids != reference_uids:
            errors.append(
                f"Picture question_uid order differs: {reference.path} vs {candidate.path}"
            )
            continue
        for uid in reference_uids:
            reference_entry = reference.entries[uid]
            candidate_entry = candidate.entries[uid]
            if reference_entry.get("question_type") != candidate_entry.get("question_type"):
                errors.append(f"{uid}: paired picture question_type differs")
            if reference_entry.get("scene_id") != candidate_entry.get("scene_id"):
                errors.append(f"{uid}: paired picture scene_id differs")
            if _context_signature(reference, reference_entry, "picture") != _context_signature(
                candidate, candidate_entry, "picture"
            ):
                errors.append(f"{uid}: paired picture context media or order differs")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Manifest path; repeat to check pairing")
    parser.add_argument("--mode", choices=ROLLOUT_MODES, required=True)
    parser.add_argument("--expected_per_type", type=int, default=50)
    parser.add_argument("--strict_provenance", action="store_true")
    parser.add_argument("--report", default=None, help="Optional JSON validation report")
    args = parser.parse_args(argv)
    if args.expected_per_type <= 0:
        parser.error("--expected_per_type must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifests = [load_rollout_manifest(Path(path)) for path in args.manifest]
    reports = [
        validate_manifest(
            manifest,
            mode=args.mode,
            expected_per_type=args.expected_per_type,
            strict_provenance=args.strict_provenance,
        )
        for manifest in manifests
    ]
    pairing_errors = (
        compare_paired_picture_manifests(manifests) if args.mode == "picture" else []
    )
    payload = {
        "mode": args.mode,
        "manifests": reports,
        "pairing_errors": pairing_errors,
        "valid": all(report["valid"] for report in reports) and not pairing_errors,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
