"""Stage 7: Quality control for generated questions.

Includes automatic filtering, answer distribution balancing, and utilities
for human validation.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections import Counter
from typing import Any

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
    "object_move_allocentric",
    "object_remove",
    "attachment_chain",
    "attachment_type",
    "support_move_consequence",
}
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


def _near_duplicate_key(q: dict[str, Any]) -> tuple:
    base = (
        q.get("scene_id"),
        q.get("image_name"),
        q.get("type"),
    )
    if q.get("type") in ATTACHMENT_NEAR_DUP_TYPES:
        return base + tuple(_id_key(q.get(field)) for field in ATTACHMENT_ID_FIELDS)

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


def quality_filter(questions: list[dict]) -> list[dict]:
    """Apply automatic quality filters to remove problematic questions.

    Filters:
        1. Direction ambiguity > 0.7 (too close to boundary)
        2. Distance questions near bin boundaries
        3. Near-duplicate questions (same frame + type + attachment ids or label tuple, keep one)
    """
    filtered: list[dict] = []
    removed_counts: Counter = Counter()

    for q in questions:
        # Filter 1: directional ambiguity
        if q.get("type") == "direction" and q.get("ambiguity_score", 0) > 0.7:
            removed_counts["ambiguous_direction"] += 1
            continue

        # Filter 2: distance near boundary
        if q.get("type") == "distance" and q.get("near_boundary", False):
            removed_counts["near_boundary"] += 1
            continue

        filtered.append(q)

    # Filter 3: deduplicate near-identical questions.
    # Same (scene, frame, type, primary_object) → keep only one.
    seen_keys: set[tuple] = set()
    deduped: list[dict] = []
    for q in filtered:
        key = _near_duplicate_key(q)
        if key in seen_keys:
            removed_counts["near_duplicate"] += 1
            continue
        seen_keys.add(key)
        deduped.append(q)

    # Filter 5: cross-frame dedup within same scene.
    # Same (scene_id, question_text) on different frames → keep only first.
    seen_text: set[tuple] = set()
    final: list[dict] = []
    for q in deduped:
        text_key = (q.get("scene_id"), q.get("question"))
        if text_key in seen_text:
            removed_counts["cross_frame_duplicate"] += 1
            continue
        seen_text.add(text_key)
        final.append(q)

    for reason, count in removed_counts.items():
        logger.info("Removed %d questions: %s", count, reason)
    logger.info(
        "Quality filter: %d → %d questions (removed %d)",
        len(questions), len(final), len(questions) - len(final),
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
    kept_not_visible_indices = set(rng.sample(not_visible_indices, max_not_visible)) if max_not_visible > 0 else set()
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


def enforce_l2_attachment_dominance(
    questions: list[dict],
) -> list[dict]:
    """Globally enforce without_attachment <= 2 * with_attachment for L2 object-move types."""
    target_types = sorted({
        str(q.get("type", "")).strip()
        for q in questions
        if q.get("level") == "L2" and str(q.get("type", "")).strip().startswith("object_move_")
    })
    if not target_types:
        return questions

    keep_mask = [True] * len(questions)
    removed_counts: Counter = Counter()

    for qtype in target_types:
        attached_indices = [
            idx for idx, q in enumerate(questions)
            if q.get("level") == "L2"
            and str(q.get("type", "")).strip() == qtype
            and bool(q.get("attachment_remapped", False))
        ]
        unattached_indices = [
            idx for idx, q in enumerate(questions)
            if q.get("level") == "L2"
            and str(q.get("type", "")).strip() == qtype
            and not bool(q.get("attachment_remapped", False))
        ]
        allowed_unattached = 2 * len(attached_indices)
        if len(unattached_indices) <= allowed_unattached:
            continue

        kept_unattached = set(
            random.sample(unattached_indices, allowed_unattached)
        ) if allowed_unattached > 0 else set()
        for idx in unattached_indices:
            if idx not in kept_unattached:
                keep_mask[idx] = False
                removed_counts[qtype] += 1

        logger.info(
            "Attachment dominance (%s): kept %d attached and %d/%d unattached",
            qtype,
            len(attached_indices),
            allowed_unattached,
            len(unattached_indices),
        )

    if not removed_counts:
        return questions

    final_questions = [
        q for idx, q in enumerate(questions)
        if keep_mask[idx]
    ]
    logger.info(
        "Attachment dominance removed %d questions: %s",
        sum(removed_counts.values()),
        dict(removed_counts),
    )
    return final_questions


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
    """Run the complete quality control pipeline.

    Steps:
        1. Automatic quality filter
        2. Cap global L1 occlusion not-visible ratio
        3. Enforce global L2 attachment dominance
        4. Answer distribution balancing
        5. Log statistics
    """
    questions = quality_filter(questions)
    questions = cap_l1_occlusion_not_visible_ratio(questions)
    questions = enforce_l2_attachment_dominance(questions)
    questions = balance_answer_values(questions)
    questions = balance_answer_distribution(questions)
    stats = compute_statistics(questions)
    logger.info("Final statistics: %s", stats)
    return questions
