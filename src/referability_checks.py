from __future__ import annotations

from collections import defaultdict
from typing import Any

QUESTION_MENTION_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("obj_a_id", "obj_a_label", "obj_a"),
    ("obj_b_id", "obj_b_label", "obj_b"),
    ("obj_ref_id", "obj_ref_label", "obj_ref"),
    ("obj_face_id", "obj_face_label", "obj_face"),
    ("obj_target_id", "obj_target_label", "obj_target"),
    ("moved_obj_id", "moved_obj_label", "moved_obj"),
    ("query_obj_id", "query_obj_label", "query_obj"),
    ("obj_c_id", "obj_c_label", "obj_c"),
    ("target_obj_id", "target_obj_label", "target_object"),
    ("removed_obj_id", "removed_obj_label", "removed_obj"),
    ("grandparent_id", "grandparent_label", "grandparent"),
    ("parent_id", "parent_label", "parent"),
    ("grandchild_id", "grandchild_label", "grandchild"),
    ("neighbor_id", "neighbor_label", "neighbor"),
)

_VALID_LABEL_STATUSES = {"absent", "unique", "multiple", "unsure"}


def coerce_object_id(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_object_ids(value: object) -> list[int]:
    object_ids: list[int] = []
    if not isinstance(value, list):
        return object_ids
    for item in value:
        obj_id = coerce_object_id(item)
        if obj_id is None:
            continue
        object_ids.append(obj_id)
    return sorted(set(object_ids))


def normalize_label_statuses(value: object) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not isinstance(value, dict):
        return normalized
    for label, status in value.items():
        if not isinstance(label, str):
            continue
        label_text = label.strip().lower()
        status_text = str(status or "").strip().lower()
        if not label_text or status_text not in _VALID_LABEL_STATUSES:
            continue
        normalized[label_text] = status_text
    return dict(sorted(normalized.items()))


def normalize_label_to_object_ids(value: object) -> dict[str, list[int]]:
    label_to_object_ids: dict[str, list[int]] = {}
    if not isinstance(value, dict):
        return label_to_object_ids
    for key, obj_ids in value.items():
        if not isinstance(key, str):
            continue
        label = key.strip().lower()
        if not label:
            continue
        label_to_object_ids[label] = normalize_object_ids(obj_ids)
    return dict(sorted(label_to_object_ids.items()))


def collect_question_mentions(
    question: dict[str, Any],
    objects_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    mention_index_by_key: dict[tuple[object, str, str], int] = {}

    def _merge_source(existing_idx: int, source: str, *, label: str) -> None:
        sources = mentions[existing_idx].setdefault("sources", [])
        if source not in sources:
            sources.append(source)
        if not str(mentions[existing_idx].get("label", "")).strip() and label:
            mentions[existing_idx]["label"] = label

    def _merge_role(
        existing_idx: int,
        *,
        role_text: str,
        source: str,
        is_explicit: bool,
    ) -> None:
        observed_roles = mentions[existing_idx].setdefault("observed_roles", [])
        if role_text not in observed_roles:
            observed_roles.append(role_text)
        role_bucket = "explicit_roles" if is_explicit else "fallback_roles"
        role_values = mentions[existing_idx].setdefault(role_bucket, [])
        if role_text not in role_values:
            role_values.append(role_text)

    def _append_mention(
        *,
        role: object,
        label_value: object,
        obj_id_value: object,
        source: str,
        prefer_existing_object: bool,
        is_explicit: bool,
    ) -> None:
        obj_id = coerce_object_id(obj_id_value)
        label = str(label_value or "").strip()
        if not label and obj_id is not None:
            label = str(objects_by_id.get(obj_id, {}).get("label", "")).strip()
        if not label and obj_id is None:
            return
        role_text = str(role or "mentioned").strip() or "mentioned"
        key = (obj_id if obj_id is not None else "", label.lower(), role_text)
        existing_idx = mention_index_by_key.get(key)
        if existing_idx is not None:
            _merge_source(existing_idx, source, label=label)
            _merge_role(
                existing_idx,
                role_text=role_text,
                source=source,
                is_explicit=is_explicit,
            )
            return
        if prefer_existing_object:
            if obj_id is not None:
                for idx, mention in enumerate(mentions):
                    if mention.get("obj_id") == obj_id:
                        _merge_source(idx, source, label=label)
                        _merge_role(
                            idx,
                            role_text=role_text,
                            source=source,
                            is_explicit=is_explicit,
                        )
                        return
            elif label:
                label_key = label.lower()
                for idx, mention in enumerate(mentions):
                    if str(mention.get("label", "")).strip().lower() == label_key:
                        _merge_source(idx, source, label=label)
                        _merge_role(
                            idx,
                            role_text=role_text,
                            source=source,
                            is_explicit=is_explicit,
                        )
                        return
        mention_index_by_key[key] = len(mentions)
        mentions.append(
            {
                "role": role_text,
                "label": label,
                "obj_id": obj_id,
                "obj_id_parse_failed": (
                    obj_id is None and obj_id_value not in (None, "")
                ),
                "sources": [source],
                "observed_roles": [role_text],
                "explicit_roles": [role_text] if is_explicit else [],
                "fallback_roles": [] if is_explicit else [role_text],
            }
        )

    raw_mentions = question.get("mentioned_objects")
    if isinstance(raw_mentions, list):
        for idx, mention in enumerate(raw_mentions):
            if not isinstance(mention, dict):
                continue
            if any(
                key in mention
                for key in ("observed_roles", "explicit_roles", "fallback_roles")
            ):
                obj_id = coerce_object_id(mention.get("obj_id"))
                label = str(mention.get("label", "")).strip()
                if not label and obj_id is not None:
                    label = str(objects_by_id.get(obj_id, {}).get("label", "")).strip()
                if not label and obj_id is None:
                    continue
                role_text = str(mention.get("role") or "mentioned").strip() or "mentioned"
                key = (obj_id if obj_id is not None else "", label.lower(), role_text)
                existing_idx = mention_index_by_key.get(key)
                if existing_idx is None:
                    mention_index_by_key[key] = len(mentions)
                    mentions.append(
                        {
                            "role": role_text,
                            "label": label,
                            "obj_id": obj_id,
                            "obj_id_parse_failed": bool(mention.get("obj_id_parse_failed", False)),
                            "sources": [
                                str(source).strip()
                                for source in mention.get("sources", [])
                                if str(source).strip()
                            ] or [f"mentioned_objects[{idx}]"],
                            "observed_roles": [],
                            "explicit_roles": [],
                            "fallback_roles": [],
                        }
                    )
                    existing_idx = mention_index_by_key[key]
                for source in mention.get("sources", []):
                    _merge_source(existing_idx, str(source).strip(), label=label)
                if not mention.get("sources"):
                    _merge_source(existing_idx, f"mentioned_objects[{idx}]", label=label)
                for role_value in mention.get("explicit_roles", []):
                    role_text = str(role_value).strip()
                    if role_text:
                        _merge_role(
                            existing_idx,
                            role_text=role_text,
                            source=f"mentioned_objects[{idx}]",
                            is_explicit=True,
                        )
                for role_value in mention.get("fallback_roles", []):
                    role_text = str(role_value).strip()
                    if role_text:
                        _merge_role(
                            existing_idx,
                            role_text=role_text,
                            source=f"mentioned_objects[{idx}]",
                            is_explicit=False,
                        )
                if not mention.get("explicit_roles") and not mention.get("fallback_roles"):
                    _merge_role(
                        existing_idx,
                        role_text=role_text,
                        source=f"mentioned_objects[{idx}]",
                        is_explicit=True,
                    )
                continue
            _append_mention(
                role=mention.get("role"),
                label_value=mention.get("label"),
                obj_id_value=mention.get("obj_id"),
                source=f"mentioned_objects[{idx}]",
                prefer_existing_object=False,
                is_explicit=True,
            )

    for id_key, label_key, role in QUESTION_MENTION_FIELDS:
        _append_mention(
            role=role,
            label_value=question.get(label_key),
            obj_id_value=question.get(id_key),
            source=id_key,
            prefer_existing_object=True,
            is_explicit=False,
        )

    return mentions


def is_static_occlusion_absent_target(
    question: dict[str, Any],
    mention: dict[str, Any],
    *,
    label_status: str | None,
    referable_label_ids: list[int],
) -> bool:
    return (
        str(question.get("type", "")).strip().lower() == "occlusion"
        and str(question.get("correct_value", "")).strip().lower() == "not visible"
        and str(mention.get("role", "")).strip().lower() == "target"
        and mention.get("obj_id") is None
        and not bool(mention.get("obj_id_parse_failed", False))
        and (
            (
                str(label_status or "").strip().lower() == "absent"
                and not referable_label_ids
            )
            or str(question.get("occlusion_decision_source", "")).strip().lower()
            == "vlm_out_of_frame_label_review"
        )
    )


def build_question_referability_audit(
    question: dict[str, Any],
    *,
    objects_by_id: dict[int, dict[str, Any]],
    label_statuses: dict[str, Any] | None,
    label_to_object_ids: dict[str, Any] | None,
    frame_referable_ids: list[int] | None,
) -> dict[str, Any]:
    referable_set = set(normalize_object_ids(frame_referable_ids))
    normalized_statuses = normalize_label_statuses(label_statuses)
    normalized_label_to_ids = normalize_label_to_object_ids(label_to_object_ids)
    mentions = collect_question_mentions(question, objects_by_id)

    role_signatures: dict[str, set[tuple[object, str]]] = defaultdict(set)
    object_explicit_roles: dict[int, set[str]] = defaultdict(set)
    object_fallback_roles: dict[int, set[str]] = defaultdict(set)
    for mention in mentions:
        role_signatures[str(mention.get("role", ""))].add(
            (
                mention.get("obj_id") if mention.get("obj_id") is not None else "",
                str(mention.get("label", "")).strip().lower(),
            )
        )
        mention_obj_id = coerce_object_id(mention.get("obj_id"))
        if mention_obj_id is None:
            continue
        object_explicit_roles[mention_obj_id].update(
            str(role).strip()
            for role in mention.get("explicit_roles", [])
            if str(role).strip()
        )
        object_fallback_roles[mention_obj_id].update(
            str(role).strip()
            for role in mention.get("fallback_roles", [])
            if str(role).strip()
        )
    conflicting_roles = {
        role for role, signatures in role_signatures.items()
        if len(signatures) > 1
    }
    multi_role_object_ids = {
        obj_id
        for obj_id, explicit_roles in object_explicit_roles.items()
        if len(explicit_roles) > 1
        or (
            not explicit_roles
            and len(object_fallback_roles.get(obj_id, set())) > 1
        )
    }

    audited_mentions: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    seen_reasons: set[str] = set()

    def _add_reason(code: str, mention_reasons: list[str]) -> None:
        if code not in mention_reasons:
            mention_reasons.append(code)
        if code not in seen_reasons:
            seen_reasons.add(code)
            reason_codes.append(code)

    for mention in mentions:
        label = str(mention.get("label", "")).strip()
        label_key = label.lower()
        mention_obj_id = coerce_object_id(mention.get("obj_id"))
        label_status = normalized_statuses.get(label_key)
        candidate_ids = normalized_label_to_ids.get(label_key, [])
        referable_label_ids = [
            obj_id for obj_id in candidate_ids
            if int(obj_id) in referable_set
        ]
        mention_reasons: list[str] = []
        exempt = is_static_occlusion_absent_target(
            question,
            mention,
            label_status=label_status,
            referable_label_ids=referable_label_ids,
        )

        if str(mention.get("role", "")) in conflicting_roles:
            _add_reason("mentioned_role_conflict", mention_reasons)
        same_object_roles = sorted(
            object_explicit_roles.get(mention_obj_id, set())
            or object_fallback_roles.get(mention_obj_id, set())
        ) if mention_obj_id is not None else []
        if mention_obj_id in multi_role_object_ids:
            _add_reason("mentioned_object_multi_role", mention_reasons)

        if not exempt:
            if bool(mention.get("obj_id_parse_failed", False)):
                _add_reason("mentioned_label_not_resolved", mention_reasons)
            if mention_obj_id is not None and mention_obj_id not in referable_set:
                _add_reason("mentioned_nonreferable_object", mention_reasons)

            if label_key:
                if label_status == "multiple":
                    _add_reason("mentioned_label_not_unique", mention_reasons)
                elif label_status != "unique":
                    _add_reason("mentioned_label_not_resolved", mention_reasons)

                if len(referable_label_ids) != 1:
                    _add_reason("mentioned_label_not_resolved", mention_reasons)
                elif mention_obj_id is not None and mention_obj_id not in set(referable_label_ids):
                    _add_reason("mentioned_nonreferable_object", mention_reasons)
            elif mention_obj_id is None:
                _add_reason("mentioned_label_not_resolved", mention_reasons)

        audited_mentions.append(
            {
                "role": str(mention.get("role", "mentioned")),
                "label": label,
                "obj_id": mention_obj_id,
                "obj_id_parse_failed": bool(mention.get("obj_id_parse_failed", False)),
                "label_status": label_status,
                "candidate_object_ids": candidate_ids,
                "referable_object_ids": referable_label_ids,
                "passes_referability_check": not mention_reasons,
                "reason_codes": mention_reasons,
                "exempt": exempt,
                "sources": list(mention.get("sources", [])),
                "observed_roles": list(mention.get("observed_roles", [])),
                "explicit_roles": list(mention.get("explicit_roles", [])),
                "fallback_roles": list(mention.get("fallback_roles", [])),
                "same_object_roles": same_object_roles,
            }
        )

    return {
        "decision": "drop" if reason_codes else "pass",
        "reason_codes": reason_codes,
        "mentioned_objects": audited_mentions,
        "frame_referable_object_ids": sorted(referable_set),
    }
