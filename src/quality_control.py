"""Stage 7: Quality control for generated questions.

Includes automatic filtering, answer distribution balancing, and utilities
for human validation.
"""

from __future__ import annotations

import itertools
import logging
import random
import re
from collections import Counter
from typing import Any, Callable

from .qa_generator import _cap_l3_unchanged_ratio

logger = logging.getLogger(__name__)

MAX_ANSWER_RATIO = 0.35  # no single option should exceed 35% of correct answers
MAX_L1_OCCLUSION_NOT_VISIBLE_RATIO = 1.0 / 3.0
L1_OCCLUSION_NOT_VISIBLE_DOWNSAMPLE_SEED = 42
ATTACHMENT_NEAR_DUP_TYPES = {
    "object_move",
    "object_move_agent",
    "object_move_distance",
    "object_move_occlusion",
    "object_move_object_centric",
    "object_rotate_object_centric",
    "object_move_allocentric",
    "object_remove",
    "attachment_chain",
    "attachment_type",
    "support_move_consequence",
}
QUESTION_TYPE_ALIASES: dict[str, str] = {}
ATTACHMENT_ID_FIELDS = (
    "obj_a_id",
    "moved_obj_id",
    "obj_target_id",
    "removed_obj_id",
    "query_obj_id",
    "obj_b_id",
    "obj_c_id",
    "obj_ref_id",
    "obj_face_id",
    "grandparent_id",
    "parent_id",
    "grandchild_id",
    "neighbor_id",
)
def _label_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _id_key(value: Any) -> int | str:
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _object_id_signature(q: dict[str, Any]) -> tuple:
    return tuple(
        (field, value)
        for field in ATTACHMENT_ID_FIELDS
        if (value := _id_key(q.get(field))) != ""
    )


def _stable_id_sort_key(value: int | str) -> tuple[int, int | str]:
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def _attachment_pair_object_ids(pair_id: Any) -> tuple[int | str, ...]:
    pair_text = str(pair_id or "").strip()
    if not pair_text:
        return ()

    parts = re.findall(r"\d+", pair_text)
    if len(parts) < 2:
        return ()
    return tuple(_id_key(part) for part in parts[:2])


def _other_attachment_object_ids(
    q: dict[str, Any],
    pair_object_ids: tuple[int | str, ...],
) -> tuple[int | str, ...]:
    excluded_ids = set(pair_object_ids)
    mentioned_ids: list[int | str] = []

    mentions = q.get("mentioned_objects") or []
    if mentions:
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            obj_id = _id_key(mention.get("obj_id"))
            if obj_id == "" or obj_id in excluded_ids:
                continue
            mentioned_ids.append(obj_id)

    if not mentions:
        for field in ATTACHMENT_ID_FIELDS:
            obj_id = _id_key(q.get(field))
            if obj_id == "" or obj_id in excluded_ids:
                continue
            mentioned_ids.append(obj_id)

    return tuple(sorted(set(mentioned_ids), key=_stable_id_sort_key))


def _near_duplicate_key(q: dict[str, Any]) -> tuple:
    qtype = QUESTION_TYPE_ALIASES.get(q.get("type"), q.get("type"))
    base = (
        q.get("scene_id"),
        q.get("image_name"),
        qtype,
    )
    object_signature = _object_id_signature(q)
    if object_signature:
        return base + ("ids", object_signature)

    primary_label = _label_key(
        q.get("obj_a_label")
        or q.get("moved_obj_label")
        or q.get("obj_target_label")
    )
    secondary_labels = (
        _label_key(q.get("query_obj_label")),
        _label_key(q.get("obj_b_label")),
        _label_key(q.get("obj_c_label")),
        _label_key(q.get("obj_ref_label")),
    )
    return base + (primary_label, *secondary_labels)


def _question_preview(question_text: Any, limit: int = 160) -> str:
    text = " ".join(str(question_text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _near_duplicate_signature(q: dict[str, Any]) -> dict[str, Any]:
    qtype = QUESTION_TYPE_ALIASES.get(q.get("type"), q.get("type"))
    signature: dict[str, Any] = {
        "scene_id": q.get("scene_id"),
        "image_name": q.get("image_name"),
        "type": qtype,
    }
    object_signature = _object_id_signature(q)
    if object_signature:
        signature["mode"] = "object_ids"
        signature["object_ids"] = {
            field: value for field, value in object_signature
        }
        return signature

    signature["mode"] = "labels"
    signature["labels"] = {
        "obj_a_label": _label_key(
            q.get("obj_a_label")
            or q.get("moved_obj_label")
            or q.get("obj_target_label")
        ),
        "query_obj_label": _label_key(q.get("query_obj_label")),
        "obj_b_label": _label_key(q.get("obj_b_label")),
        "obj_c_label": _label_key(q.get("obj_c_label")),
        "obj_ref_label": _label_key(q.get("obj_ref_label")),
    }
    return signature


def _near_duplicate_detail(
    question: dict[str, Any],
    kept_question: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    signature = _near_duplicate_signature(question)
    duplicate_of_id = kept_question.get("trace_question_id")
    duplicate_of_question = _question_preview(kept_question.get("question"))
    if signature.get("mode") == "object_ids":
        object_ids = signature.get("object_ids", {})
        signature_text = ", ".join(f"{field}={value}" for field, value in object_ids.items())
        if not signature_text:
            signature_text = "no distinguishing object ids"
        detail = (
            f"same scene/frame/type and object-id signature as "
            f"{duplicate_of_id or 'earlier kept question'} ({signature_text})"
        )
    else:
        labels = signature.get("labels", {})
        signature_text = ", ".join(
            f"{field}={value}"
            for field, value in labels.items()
            if str(value).strip()
        )
        if not signature_text:
            signature_text = "all distinguishing labels empty"
        detail = (
            f"same scene/frame/type and label signature as "
            f"{duplicate_of_id or 'earlier kept question'} ({signature_text})"
        )
    if duplicate_of_question:
        detail += f'; kept question: "{duplicate_of_question}"'
    return detail, signature


def _emit_trace_event(
    trace_recorder: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if trace_recorder is not None:
        trace_recorder(payload)


def quality_filter(
    questions: list[dict],
    trace_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict]:
    """Apply automatic quality filters to remove problematic questions.

    Filters:
        1. Direction ambiguity > 0.7 (too close to boundary)
        2. Near-duplicate questions (same frame + type + object ids or label tuple, keep one)
        3. Cross-frame duplicates (same scene + text + object ids, keep one)

    Duplicate filtering applies to all levels and question types, including L3.
    """
    filtered: list[dict] = []
    removed_counts: Counter = Counter()

    for q in questions:
        # Filter 1: directional ambiguity
        if q.get("type") == "direction" and q.get("ambiguity_score", 0) > 0.7:
            removed_counts["ambiguous_direction"] += 1
            _emit_trace_event(
                trace_recorder,
                {
                    "event": "question_removed",
                    "stage": "quality_filter",
                    "filter": "ambiguous_direction",
                    "reason": "ambiguous_direction",
                    "detail": "direction ambiguity score exceeded threshold 0.7",
                    "details": {
                        "ambiguity_score": q.get("ambiguity_score"),
                        "threshold": 0.7,
                    },
                    "trace_question_id": q.get("trace_question_id"),
                    "question": q,
                },
            )
            continue

        filtered.append(q)

    # Filter 2: deduplicate near-identical questions across all question types.
    # Same (scene, frame, type, object signature/labels) -> keep only one.
    seen_keys: dict[tuple, dict] = {}
    deduped: list[dict] = []
    for q in filtered:
        key = _near_duplicate_key(q)
        kept_question = seen_keys.get(key)
        if kept_question is not None:
            removed_counts["near_duplicate"] += 1
            detail, signature = _near_duplicate_detail(q, kept_question)
            _emit_trace_event(
                trace_recorder,
                {
                    "event": "question_removed",
                    "stage": "quality_filter",
                    "filter": "near_duplicate",
                    "reason": "near_duplicate",
                    "detail": detail,
                    "details": {
                        "signature": signature,
                        "duplicate_of_trace_question_id": kept_question.get("trace_question_id"),
                        "duplicate_of_question": kept_question.get("question"),
                    },
                    "duplicate_of_trace_question_id": kept_question.get("trace_question_id"),
                    "duplicate_of_question": kept_question.get("question"),
                    "trace_question_id": q.get("trace_question_id"),
                    "question": q,
                },
            )
            continue
        seen_keys[key] = q
        deduped.append(q)

    # Filter 3: cross-frame dedup within same scene for all question types.
    # Same (scene_id, question_text, object signature) -> keep only first.
    seen_text: dict[tuple, dict] = {}
    final: list[dict] = []
    for q in deduped:
        text_key = (q.get("scene_id"), q.get("question"), _object_id_signature(q))
        kept_question = seen_text.get(text_key)
        if kept_question is not None:
            removed_counts["cross_frame_duplicate"] += 1
            _emit_trace_event(
                trace_recorder,
                {
                    "event": "question_removed",
                    "stage": "quality_filter",
                    "filter": "cross_frame_duplicate",
                    "reason": "cross_frame_duplicate",
                    "detail": (
                        f'same scene + identical question text + object-id signature as '
                        f'{kept_question.get("trace_question_id") or "earlier kept question"}'
                    ),
                    "details": {
                        "object_id_signature": _object_id_signature(q),
                        "duplicate_of_trace_question_id": kept_question.get("trace_question_id"),
                        "duplicate_of_question": kept_question.get("question"),
                        "duplicate_of_image_name": kept_question.get("image_name"),
                    },
                    "duplicate_of_trace_question_id": kept_question.get("trace_question_id"),
                    "duplicate_of_question": kept_question.get("question"),
                    "trace_question_id": q.get("trace_question_id"),
                    "question": q,
                },
            )
            continue
        seen_text[text_key] = q
        final.append(q)

    for reason, count in removed_counts.items():
        logger.info("Removed %d questions: %s", count, reason)
    logger.info(
        "Quality filter: %d → %d questions (removed %d)",
        len(questions), len(final), len(questions) - len(final),
    )
    _emit_trace_event(
        trace_recorder,
        {
            "event": "quality_filter_summary",
            "stage": "quality_filter",
            "input_count": len(questions),
            "output_count": len(final),
            "removed_count": len(questions) - len(final),
            "removed_counts": dict(removed_counts),
        },
    )
    return final


def cap_l1_occlusion_not_visible_ratio(
    questions: list[dict],
    max_ratio: float = MAX_L1_OCCLUSION_NOT_VISIBLE_RATIO,
    seed: int = L1_OCCLUSION_NOT_VISIBLE_DOWNSAMPLE_SEED,
) -> list[dict]:
    """Globally cap the share of L1 occlusion questions answered as not visible."""
    if max_ratio <= 0.0 or max_ratio >= 1.0:
        raise ValueError(f"max_ratio must be in (0, 1), got {max_ratio}")

    l1_occlusion_indices = [
        idx for idx, q in enumerate(questions)
        if q.get("level") == "L1" and q.get("type") == "occlusion"
    ]
    if not l1_occlusion_indices:
        return questions

    not_visible_indices = [
        idx for idx in l1_occlusion_indices
        if questions[idx].get("correct_value") == "not visible"
    ]
    not_visible_count = len(not_visible_indices)
    if not not_visible_count:
        return questions

    other_count = len(l1_occlusion_indices) - not_visible_count
    max_not_visible = not_visible_count
    while (
        max_not_visible > 0
        and max_not_visible / (other_count + max_not_visible) > max_ratio
    ):
        max_not_visible -= 1
    if not_visible_count <= max_not_visible:
        return questions

    rng = random.Random(seed)
    kept_not_visible_indices = (
        set(rng.sample(not_visible_indices, max_not_visible))
        if max_not_visible > 0 else set()
    )
    removed_count = not_visible_count - len(kept_not_visible_indices)
    capped_questions = [
        q
        for idx, q in enumerate(questions)
        if idx not in not_visible_indices or idx in kept_not_visible_indices
    ]

    final_l1_occlusion = len(l1_occlusion_indices) - removed_count
    final_not_visible = len(kept_not_visible_indices)
    logger.info(
        "Capped L1 occlusion not-visible questions: total=%d, original_not_visible=%d, kept=%d, removed=%d, final_ratio=%.3f",
        final_l1_occlusion + removed_count,
        not_visible_count,
        final_not_visible,
        removed_count,
        0.0 if final_l1_occlusion == 0 else final_not_visible / final_l1_occlusion,
    )
    return capped_questions


def balance_l2_attachment_per_scene(questions: list[dict]) -> list[dict]:
    """Balance L2 attachment variants per batch and canonical qtype.

    Applies only to L2 qtypes whose canonical type starts with ``object_move_``
    or equals ``object_rotate_object_centric``.

    For each canonical qtype across the full batch:

    1. **Changed attachment** (``attachment_remapped=True``,
       ``relation_unchanged=False``):
       Keep all questions after deduping repeated attachment changes within the
       same scene by ``(attachment_pair_id, other_mentioned_object_ids)``.
    2. **Unattached** (``attachment_remapped=False``/absent):
       First keep at most 3 per scene, then keep at most
       ``floor(len(changed_kept) / 4)`` across the whole batch.
    3. **Attached unchanged** (``attachment_remapped=True``,
       ``relation_unchanged=True``):
       First keep at most 3 per scene, then keep at most
       ``floor(len(changed_kept) / 4)`` across the whole batch.

    Earlier-generated questions win whenever trimming is required. Questions of
    other types pass through unchanged.
    """

    def _bucket(qtype: str) -> str:
        canonical = str(QUESTION_TYPE_ALIASES.get(qtype, qtype)).strip()
        if canonical.startswith("object_move_") or canonical == "object_rotate_object_centric":
            return canonical
        return ""

    def _cap_candidates_per_scene(indices: list[int]) -> list[int]:
        kept: list[int] = []
        scene_counts: Counter[str] = Counter()
        for idx in indices:
            scene_id = str(questions[idx].get("scene_id", ""))
            if scene_counts[scene_id] >= 3:
                continue
            scene_counts[scene_id] += 1
            kept.append(idx)
        return kept

    grouped: dict[str, list[int]] = {}
    pass_through: list[int] = []
    for idx, q in enumerate(questions):
        qtype = str(q.get("type", "")).strip()
        bucket = _bucket(qtype)
        if bucket and str(q.get("level", "")) == "L2":
            grouped.setdefault(bucket, []).append(idx)
        else:
            pass_through.append(idx)

    keep = [False] * len(questions)
    for pt_idx in pass_through:
        keep[pt_idx] = True

    for qtype, indices in grouped.items():
        changed: list[int] = []
        unattached: list[int] = []
        unchanged: list[int] = []

        for idx in indices:
            q = questions[idx]
            if q.get("attachment_remapped", False):
                if q.get("relation_unchanged", False):
                    unchanged.append(idx)
                else:
                    changed.append(idx)
            else:
                unattached.append(idx)

        seen_changed_keys: set[tuple[str, str, tuple[int | str, ...]]] = set()
        changed_kept: list[int] = []
        for idx in changed:
            scene_id = str(questions[idx].get("scene_id", ""))
            pair_id = str(questions[idx].get("attachment_pair_id", "") or "").strip()
            if not pair_id:
                changed_kept.append(idx)
                continue
            pair_object_ids = _attachment_pair_object_ids(pair_id)
            other_object_ids = _other_attachment_object_ids(questions[idx], pair_object_ids)
            dedup_key = (scene_id, pair_id, other_object_ids)
            if dedup_key in seen_changed_keys:
                continue
            seen_changed_keys.add(dedup_key)
            changed_kept.append(idx)

        batch_cap = max(0, len(changed_kept) // 4)
        unattached_scene_capped = _cap_candidates_per_scene(unattached)
        unchanged_scene_capped = _cap_candidates_per_scene(unchanged)

        unattached_kept = unattached_scene_capped[:batch_cap]
        unchanged_kept = unchanged_scene_capped[:batch_cap]

        kept_count = len(changed_kept) + len(unattached_kept) + len(unchanged_kept)
        removed_count = len(indices) - kept_count
        if removed_count:
            logger.info(
                "Attachment balance (%s): changed=%d/%d, unattached=%d/%d after scene_cap=%d/%d, unchanged=%d/%d after scene_cap=%d/%d, batch_cap=%d",
                qtype,
                len(changed_kept),
                len(changed),
                len(unattached_kept),
                len(unattached),
                len(unattached_scene_capped),
                len(unattached),
                len(unchanged_kept),
                len(unchanged),
                len(unchanged_scene_capped),
                len(unchanged),
                batch_cap,
            )

        for idx in changed_kept:
            keep[idx] = True
        for idx in unattached_kept:
            keep[idx] = True
        for idx in unchanged_kept:
            keep[idx] = True

    return [q for idx, q in enumerate(questions) if keep[idx]]


def balance_answer_values(
    questions: list[dict],
    target_types: tuple[str, ...] = (
        "distance", "direction",
        "direction_object_centric", "direction_allocentric",
    ),
) -> list[dict]:
    """Downsample questions so correct_value distribution is roughly uniform.

    For question types in *target_types*, groups questions by correct_value and
    downsamples each group to the size of the smallest group.  This prevents
    answer-value imbalance (e.g., 66% of distance answers being "very close").

    Questions of other types are passed through unchanged.
    """
    from collections import defaultdict

    other: list[dict] = []
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        if q.get("type") in target_types:
            by_type[q["type"]].append(q)
        else:
            other.append(q)

    balanced: list[dict] = list(other)
    for qtype, qs in by_type.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for q in qs:
            groups[q["correct_value"]].append(q)

        if not groups:
            continue

        min_count = min(len(g) for g in groups.values())
        if min_count == 0:
            # Some bin has zero questions — keep all to avoid losing data
            balanced.extend(qs)
            continue

        before = len(qs)
        for val, group in groups.items():
            if len(group) > min_count:
                balanced.extend(random.sample(group, min_count))
            else:
                balanced.extend(group)

        after = sum(min(len(g), min_count) for g in groups.values())
        logger.info(
            "Answer-value balance (%s): %d → %d (min_bin=%d, bins=%s)",
            qtype, before, after, min_count,
            {v: len(g) for v, g in groups.items()},
        )

    return balanced


def balance_answer_distribution(
    questions: list[dict],
    max_ratio: float = MAX_ANSWER_RATIO,
) -> list[dict]:
    """Re-shuffle options in questions where one answer letter is overrepresented.

    Groups questions by (level, type) and within each group ensures no single
    correct-answer letter exceeds *max_ratio*.
    """
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for q in questions:
        key = (q.get("level", ""), q.get("type", ""))
        groups[key].append(q)

    balanced: list[dict] = []
    for key, group in groups.items():
        group_copy: list[dict] = []
        for q in group:
            q_copy = dict(q)
            q_copy["options"] = list(q["options"])
            group_copy.append(q_copy)

        answer_counts = Counter(q["answer"] for q in group_copy)
        total = len(group_copy)
        needs_rebalance = any(c / total > max_ratio for c in answer_counts.values())

        if needs_rebalance:
            logger.info(
                "Rebalancing %s: %s (total=%d)", key, dict(answer_counts), total
            )
            for q in group_copy:
                original_answer = q["answer"]
                if answer_counts[original_answer] / total <= max_ratio:
                    continue

                correct_val = q["correct_value"]
                options = list(q["options"])
                best_options = options
                best_answer = original_answer
                best_counts = answer_counts
                best_overflow = max(
                    max(0.0, count / total - max_ratio)
                    for count in answer_counts.values()
                )

                for perm in {tuple(p) for p in itertools.permutations(options)}:
                    new_options = list(perm)
                    new_answer = chr(65 + new_options.index(correct_val))
                    trial_counts = answer_counts.copy()
                    trial_counts[original_answer] -= 1
                    if trial_counts[original_answer] <= 0:
                        del trial_counts[original_answer]
                    trial_counts[new_answer] += 1
                    trial_overflow = max(
                        (max(0.0, count / total - max_ratio) for count in trial_counts.values()),
                        default=0.0,
                    )
                    if trial_overflow < best_overflow:
                        best_options = new_options
                        best_answer = new_answer
                        best_counts = trial_counts
                        best_overflow = trial_overflow
                        if best_overflow == 0.0:
                            break

                q["options"] = best_options
                q["answer"] = best_answer
                answer_counts = best_counts

        balanced.extend(group_copy)

    return balanced


def compute_statistics(questions: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for a question set."""
    stats: dict[str, Any] = {}

    # Overall
    stats["total"] = len(questions)

    # By level
    level_counts = Counter(q.get("level", "?") for q in questions)
    stats["by_level"] = dict(level_counts)

    # By type
    type_counts = Counter(q.get("type", "?") for q in questions)
    stats["by_type"] = dict(type_counts)

    # Answer distribution per level
    for level in ("L1", "L2", "L3"):
        level_qs = [q for q in questions if q.get("level") == level]
        if level_qs:
            ans_dist = Counter(q["answer"] for q in level_qs)
            total = len(level_qs)
            stats[f"{level}_answer_dist"] = {
                k: round(v / total, 3) for k, v in sorted(ans_dist.items())
            }

    return stats


def sample_for_human_validation(
    questions: list[dict],
    n_per_level: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Random sample of questions for human annotation.

    Samples up to *n_per_level* questions per level.
    Returns a list of question dicts with an added ``_validation_id`` field.
    """
    rng = random.Random(seed)
    from collections import defaultdict

    by_level: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_level[q.get("level", "?")].append(q)

    sampled: list[dict] = []
    vid = 1
    for level in sorted(by_level):
        pool = by_level[level]
        n = min(n_per_level, len(pool))
        chosen = rng.sample(pool, n)
        for q in chosen:
            q_copy = dict(q)
            q_copy["_validation_id"] = vid
            sampled.append(q_copy)
            vid += 1

    logger.info("Sampled %d questions for human validation", len(sampled))
    return sampled


def compute_inter_annotator_agreement(
    annotations_a: list[str],
    annotations_b: list[str],
) -> float:
    """Compute Cohen's kappa between two annotators.

    Both inputs should be lists of the same length containing answer labels.
    """
    assert len(annotations_a) == len(annotations_b), "Annotation lists must be same length"
    n = len(annotations_a)
    if n == 0:
        return 0.0

    labels = sorted(set(annotations_a) | set(annotations_b))
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(annotations_a, annotations_b):
        matrix[label_idx[a]][label_idx[b]] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[j][i] for j in range(k)) for i in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def full_quality_pipeline(questions: list[dict]) -> list[dict]:
    """Run the benchmark quality-control pipeline.

    Steps:
        1. Automatic quality filter
        2. Batch-level-per-qtype L2 attachment balance
        3. Batch-level-per-qtype L3 unchanged ratio cap
        4. Cap global L1 occlusion not-visible ratio
        5. Answer-letter distribution balancing
        6. Log statistics
    """
    questions = quality_filter(questions)
    questions = balance_l2_attachment_per_scene(questions)
    questions = _cap_l3_unchanged_ratio(questions)
    questions = cap_l1_occlusion_not_visible_ratio(questions)
    questions = balance_answer_distribution(questions)
    stats = compute_statistics(questions)
    logger.info("Final statistics: %s", stats)
    return questions
