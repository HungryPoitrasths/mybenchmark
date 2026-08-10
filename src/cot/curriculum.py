from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any


LEGACY_PROFILE_ID = "legacy-10k-v1"
FRESH_12K_PROFILE_ID = "fresh-12k-v2"
LEGACY_SCHEMA_VERSION = "predictive-spatial-cot-curriculum-v1"
PROFILE_SCHEMA_VERSION = "predictive-spatial-cot-curriculum-v2"


@dataclass(frozen=True)
class CurriculumProfile:
    profile_id: str
    schema_version: str
    global_batch: int
    stage1_exposures: int
    stage2_pattern: tuple[tuple[str, dict[str, int]], ...]
    stage2_pattern_repetitions: int
    target_exposures_by_level: dict[str, int]
    candidate_quotas_by_level: dict[str, int] | None = None
    candidate_quotas_by_type: dict[str, int] | None = None
    l2_exposures_by_type: dict[str, int] | None = None
    l2_max_uid_exposures: int | None = None
    l2_max_uid_exposures_by_type: dict[str, int] | None = None

    @property
    def stage2_exposures(self) -> int:
        return (
            self.global_batch
            * len(self.stage2_pattern)
            * self.stage2_pattern_repetitions
        )

    @property
    def total_exposures(self) -> int:
        return self.stage1_exposures + self.stage2_exposures

    @property
    def stage2_exposures_by_level(self) -> dict[str, int]:
        counts = {"L1": 0, "L2": 0, "L3": 0}
        for _name, composition in self.stage2_pattern:
            for level in counts:
                counts[level] += composition[level] * self.stage2_pattern_repetitions
        return counts


LEGACY_PROFILE = CurriculumProfile(
    profile_id=LEGACY_PROFILE_ID,
    schema_version=LEGACY_SCHEMA_VERSION,
    global_batch=32,
    stage1_exposures=6_144,
    stage2_pattern=(
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
        ("A", {"L1": 4, "L2": 14, "L3": 14}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
    ),
    stage2_pattern_repetitions=64,
    target_exposures_by_level={"L1": 8_192, "L2": 6_144, "L3": 6_144},
)

FRESH_12K_CANDIDATE_BY_TYPE = {
    "direction_agent": 907,
    "occlusion": 907,
    "distance": 907,
    "direction_object_centric": 906,
    "direction_allocentric": 906,
    "object_move_agent": 516,
    "object_move_distance": 845,
    "object_move_occlusion": 211,
    "object_rotate_object_centric": 381,
    "object_move_object_centric": 360,
    "object_move_allocentric": 294,
    "object_remove": 327,
    "attachment_chain": 50,
    "coordinate_rotation_agent": 1_495,
    "coordinate_rotation_object_centric": 1_494,
    "coordinate_rotation_allocentric": 1_494,
}

FRESH_12K_L2_EXPOSURES_BY_TYPE = {
    "object_move_agent": 1_246,
    "object_move_distance": 1_594,
    "object_move_occlusion": 797,
    "object_rotate_object_centric": 1_070,
    "object_move_object_centric": 1_041,
    "object_move_allocentric": 940,
    "object_remove": 992,
}

FRESH_12K_PROFILE = CurriculumProfile(
    profile_id=FRESH_12K_PROFILE_ID,
    schema_version=PROFILE_SCHEMA_VERSION,
    global_batch=32,
    stage1_exposures=6_144,
    stage2_pattern=(
        ("A", {"L1": 6, "L2": 13, "L3": 13}),
        ("B", {"L1": 5, "L2": 14, "L3": 13}),
        ("C", {"L1": 5, "L2": 13, "L3": 14}),
    ),
    stage2_pattern_repetitions=192,
    target_exposures_by_level={"L1": 9_216, "L2": 7_680, "L3": 7_680},
    candidate_quotas_by_level={"L1": 4_533, "L2": 2_934, "L3": 4_533},
    candidate_quotas_by_type=FRESH_12K_CANDIDATE_BY_TYPE,
    l2_exposures_by_type=FRESH_12K_L2_EXPOSURES_BY_TYPE,
    l2_max_uid_exposures=4,
    l2_max_uid_exposures_by_type={"object_move_object_centric": 2},
)

PROFILES = {
    LEGACY_PROFILE.profile_id: LEGACY_PROFILE,
    FRESH_12K_PROFILE.profile_id: FRESH_12K_PROFILE,
}


def get_curriculum_profile(profile_id: str) -> CurriculumProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown curriculum profile: {profile_id!r}") from exc


def profile_for_manifest(payload: dict[str, Any]) -> CurriculumProfile:
    schema_version = str(payload.get("schema_version") or "")
    profile_id = str(payload.get("profile_id") or "")
    if schema_version == LEGACY_SCHEMA_VERSION and not profile_id:
        return LEGACY_PROFILE
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported curriculum schema_version: {schema_version!r}")
    profile = get_curriculum_profile(profile_id)
    if profile.schema_version != schema_version:
        raise ValueError(
            f"profile {profile_id!r} requires schema_version {profile.schema_version!r}"
        )
    return profile


def sqrt_largest_remainder_allocation(
    total: int, capacities: dict[str, int]
) -> dict[str, int]:
    if total < 0 or any(value < 0 for value in capacities.values()):
        raise ValueError("allocation totals and capacities must be non-negative")
    weights = {key: math.sqrt(value) for key, value in capacities.items() if value}
    if total and not weights:
        raise ValueError("cannot allocate a positive total over empty capacities")
    weight_sum = sum(weights.values())
    raw = {key: total * weights.get(key, 0.0) / weight_sum for key in capacities}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(capacities, key=lambda key: (-(raw[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def stable_hash(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def l2_rarity_group(question: dict[str, Any]) -> tuple[str, ...]:
    question_type = str(question.get("type") or "")
    scene = str(question.get("scene_id") or "")
    keys_by_type = {
        "object_move_agent": ("moved_obj_id", "query_obj_id", "obj_c_id"),
        "object_move_distance": ("moved_obj_id", "query_obj_id", "obj_c_id"),
        "object_move_occlusion": ("moved_obj_id", "query_obj_id", "obj_ref_id"),
        "object_rotate_object_centric": (
            "moved_obj_id",
            "query_obj_id",
            "obj_face_id",
            "obj_ref_id",
        ),
        "object_move_object_centric": (
            "moved_obj_id",
            "query_obj_id",
            "obj_ref_id",
        ),
        "object_move_allocentric": (
            "moved_obj_id",
            "query_obj_id",
            "obj_ref_id",
        ),
        "object_remove": ("removed_obj_id", "obj_b_id"),
    }
    keys = keys_by_type.get(question_type, ())
    roles: list[str] = []
    for key in keys:
        value = question.get(key)
        if value is None and key == "query_obj_id":
            value = question.get("target_obj_id", question.get("obj_b_id"))
        if value is None and key == "obj_ref_id":
            value = question.get("obj_b_id", question.get("obj_c_id"))
        roles.append(f"{key}={value if value is not None else 'missing'}")
    return (question_type, scene, *roles)
