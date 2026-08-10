from __future__ import annotations

import hashlib
import heapq
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


TYPES_BY_LEVEL: dict[str, tuple[str, ...]] = {
    "L1": (
        "direction_agent",
        "occlusion",
        "distance",
        "direction_object_centric",
        "direction_allocentric",
    ),
    "L2": (
        "object_move_agent",
        "object_move_distance",
        "object_move_occlusion",
        "object_rotate_object_centric",
        "object_move_object_centric",
        "object_move_allocentric",
        "object_remove",
    ),
    "L3": (
        "attachment_chain",
        "coordinate_rotation_agent",
        "coordinate_rotation_object_centric",
        "coordinate_rotation_allocentric",
    ),
}
SUPPORTED_TYPE_ORDER = tuple(
    question_type
    for level in ("L1", "L2", "L3")
    for question_type in TYPES_BY_LEVEL[level]
)

PILOT_TRAIN_8K_LEVEL_QUOTAS = {"L1": 3669, "L2": 661, "L3": 3670}
PILOT_TRAIN_LEVEL_QUOTAS = {"L1": 4669, "L2": 661, "L3": 4670}
PILOT_TRAIN_2K_LEVEL_QUOTAS = {"L1": 350, "L2": 1300, "L3": 350}
MONITOR_VALIDATION_PER_TYPE = 20


class SamplingError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionResult:
    indices: list[int]
    report: dict[str, Any]


def _row_level(row: dict[str, Any]) -> str:
    facts = row.get("facts")
    if isinstance(facts, dict):
        level = str(facts.get("level") or "").strip().upper()
        if level:
            return level
    question_type = str(row.get("question_type") or "")
    for level, question_types in TYPES_BY_LEVEL.items():
        if question_type in question_types:
            return level
    return ""


def _row_scene(row: dict[str, Any]) -> str:
    facts = row.get("facts")
    if isinstance(facts, dict):
        return str(facts.get("scene_id") or "")
    return ""


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _equal_allocation(total: int, capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise SamplingError("selection total must be non-negative")
    if total > sum(capacities.values()):
        raise SamplingError(
            f"requested {total} records but only {sum(capacities.values())} are available"
        )
    allocation = {key: 0 for key in capacities}
    remaining = total
    ordered = list(capacities)
    while remaining:
        progressed = False
        for key in ordered:
            if allocation[key] >= capacities[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise SamplingError("quota allocation stalled before reaching the requested total")
    return allocation


def _sqrt_allocation(total: int, capacities: dict[str, int]) -> dict[str, int]:
    if total > sum(capacities.values()):
        raise SamplingError(
            f"requested {total} signature records but only {sum(capacities.values())} are available"
        )
    allocation = {key: 0 for key in capacities}
    heap: list[tuple[float, str]] = []
    weights = {key: math.sqrt(value) for key, value in capacities.items() if value > 0}
    for key, weight in weights.items():
        heapq.heappush(heap, (-weight, key))
    for _ in range(total):
        while heap:
            _priority, key = heapq.heappop(heap)
            if allocation[key] < capacities[key]:
                break
        else:
            raise SamplingError("signature allocation exhausted before reaching quota")
        allocation[key] += 1
        if allocation[key] < capacities[key]:
            next_priority = -(weights[key] / (allocation[key] + 1))
            heapq.heappush(heap, (next_priority, key))
    return allocation


def _diverse_order(
    rows: list[dict[str, Any]], indices: Iterable[int], *, seed: int, key: str
) -> list[int]:
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in indices:
        row = rows[index]
        answers = " ".join(str(value) for value in row.get("answer_letters") or [])
        buckets[(answers, _row_scene(row))].append(index)

    rng = random.Random(_stable_seed(seed, key))
    bucket_keys = sorted(buckets)
    rng.shuffle(bucket_keys)
    for bucket_key in bucket_keys:
        rng.shuffle(buckets[bucket_key])

    ordered: list[int] = []
    while bucket_keys:
        next_keys: list[tuple[str, str]] = []
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            if bucket:
                ordered.append(bucket.pop())
            if bucket:
                next_keys.append(bucket_key)
        bucket_keys = next_keys
    return ordered


def _select_type_indices(
    rows: list[dict[str, Any]],
    candidate_indices: list[int],
    quota: int,
    *,
    seed: int,
    question_type: str,
) -> list[int]:
    by_signature: dict[str, list[int]] = defaultdict(list)
    for index in candidate_indices:
        signature = str(rows[index].get("signature_id") or "missing")
        by_signature[signature].append(index)
    signature_quotas = _sqrt_allocation(
        quota, {signature: len(indices) for signature, indices in sorted(by_signature.items())}
    )
    selected: list[int] = []
    for signature, indices in sorted(by_signature.items()):
        ordered = _diverse_order(
            rows,
            indices,
            seed=seed,
            key=f"{question_type}|{signature}",
        )
        selected.extend(ordered[: signature_quotas[signature]])
    rng = random.Random(_stable_seed(seed, question_type, "selected"))
    rng.shuffle(selected)
    return selected


def _deduplicated_indices(rows: list[dict[str, Any]]) -> tuple[list[int], int]:
    seen: set[str] = set()
    unique: list[int] = []
    duplicate_count = 0
    for index, row in enumerate(rows):
        uid = str(row.get("question_uid") or "").strip()
        if not uid:
            raise SamplingError(f"sidecar row {index} has no question_uid")
        if uid in seen:
            duplicate_count += 1
            continue
        seen.add(uid)
        unique.append(index)
    return unique, duplicate_count


def select_stratified(
    rows: list[dict[str, Any]],
    *,
    level_quotas: dict[str, int] | None = None,
    per_type_count: int | None = None,
    type_quotas: dict[str, int] | None = None,
    seed: int = 42,
    strict_type_capacity: bool = False,
) -> SelectionResult:
    provided_modes = sum(
        value is not None for value in (level_quotas, per_type_count, type_quotas)
    )
    if provided_modes != 1:
        raise SamplingError(
            "provide exactly one of level_quotas, per_type_count, or type_quotas"
        )

    unique_indices, duplicate_count = _deduplicated_indices(rows)
    by_type: dict[str, list[int]] = {question_type: [] for question_type in SUPPORTED_TYPE_ORDER}
    ignored_type_counts: Counter[str] = Counter()
    for index in unique_indices:
        row = rows[index]
        question_type = str(row.get("question_type") or "")
        if question_type not in by_type:
            ignored_type_counts[question_type or "missing"] += 1
            continue
        expected_level = next(
            level for level, values in TYPES_BY_LEVEL.items() if question_type in values
        )
        if _row_level(row) != expected_level:
            raise SamplingError(
                f"row {index} has level {_row_level(row)!r}, expected {expected_level!r} for {question_type}"
            )
        by_type[question_type].append(index)

    requested_type_quotas = type_quotas
    resolved_type_quotas: dict[str, int] = {}
    if requested_type_quotas is not None:
        unknown_types = set(requested_type_quotas) - set(SUPPORTED_TYPE_ORDER)
        if unknown_types:
            raise SamplingError(f"unknown types in quota: {sorted(unknown_types)}")
        for question_type in SUPPORTED_TYPE_ORDER:
            quota = int(requested_type_quotas.get(question_type, 0))
            available = len(by_type[question_type])
            if quota < 0:
                raise SamplingError(f"{question_type} quota must be non-negative")
            if quota > available:
                raise SamplingError(
                    f"{question_type} has {available} records, fewer than required {quota}"
                )
            resolved_type_quotas[question_type] = quota
    elif per_type_count is not None:
        if per_type_count <= 0:
            raise SamplingError("per_type_count must be positive")
        for question_type in SUPPORTED_TYPE_ORDER:
            available = len(by_type[question_type])
            if strict_type_capacity and available < per_type_count:
                raise SamplingError(
                    f"{question_type} has {available} records, fewer than required {per_type_count}"
                )
            resolved_type_quotas[question_type] = min(per_type_count, available)
    else:
        assert level_quotas is not None
        unknown_levels = set(level_quotas) - set(TYPES_BY_LEVEL)
        if unknown_levels:
            raise SamplingError(f"unknown levels in quota: {sorted(unknown_levels)}")
        for level in ("L1", "L2", "L3"):
            quota = int(level_quotas.get(level, 0))
            capacities = {
                question_type: len(by_type[question_type])
                for question_type in TYPES_BY_LEVEL[level]
            }
            resolved_type_quotas.update(_equal_allocation(quota, capacities))

    selected: list[int] = []
    for question_type in SUPPORTED_TYPE_ORDER:
        quota = resolved_type_quotas[question_type]
        if quota:
            selected.extend(
                _select_type_indices(
                    rows,
                    by_type[question_type],
                    quota,
                    seed=seed,
                    question_type=question_type,
                )
            )
    rng = random.Random(_stable_seed(seed, "global-selection"))
    rng.shuffle(selected)

    selected_by_type = Counter(str(rows[index]["question_type"]) for index in selected)
    selected_by_level = Counter(_row_level(rows[index]) for index in selected)
    selected_by_signature = Counter(str(rows[index]["signature_id"]) for index in selected)
    available_by_type = {question_type: len(by_type[question_type]) for question_type in SUPPORTED_TYPE_ORDER}
    report = {
        "seed": seed,
        "selected_count": len(selected),
        "unique_supported_count": sum(available_by_type.values()),
        "duplicate_uid_count": duplicate_count,
        "ignored_type_counts": dict(sorted(ignored_type_counts.items())),
        "available_by_type": available_by_type,
        "target_by_type": resolved_type_quotas,
        "selected_by_level": dict(sorted(selected_by_level.items())),
        "selected_by_type": dict(sorted(selected_by_type.items())),
        "selected_by_signature": dict(sorted(selected_by_signature.items())),
    }
    return SelectionResult(indices=selected, report=report)


def select_pilot_train(rows: list[dict[str, Any]], *, seed: int = 42) -> SelectionResult:
    return select_stratified(rows, level_quotas=PILOT_TRAIN_LEVEL_QUOTAS, seed=seed)


def select_pilot_train_2k(rows: list[dict[str, Any]], *, seed: int = 42) -> SelectionResult:
    return select_stratified(rows, level_quotas=PILOT_TRAIN_2K_LEVEL_QUOTAS, seed=seed)


def select_pilot_train_8k(rows: list[dict[str, Any]], *, seed: int = 42) -> SelectionResult:
    return select_stratified(rows, level_quotas=PILOT_TRAIN_8K_LEVEL_QUOTAS, seed=seed)


def select_monitor_validation(
    rows: list[dict[str, Any]], *, seed: int = 42
) -> SelectionResult:
    return select_stratified(
        rows,
        per_type_count=MONITOR_VALIDATION_PER_TYPE,
        seed=seed,
        strict_type_capacity=True,
    )
