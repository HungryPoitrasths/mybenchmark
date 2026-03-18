"""Stage 7: Quality control for generated questions.

Includes automatic filtering, answer distribution balancing, and utilities
for human validation.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

MAX_ANSWER_RATIO = 0.35  # no single option should exceed 35% of correct answers


def quality_filter(questions: list[dict]) -> list[dict]:
    """Apply automatic quality filters to remove problematic questions.

    Filters:
        1. Direction ambiguity > 0.7 (too close to boundary)
        2. L2/L3 questions where the relation didn't actually change
        3. Distance questions near bin boundaries
        4. Near-duplicate questions (same frame + type + obj_a, keep one)
    """
    filtered: list[dict] = []
    removed_counts: Counter = Counter()

    for q in questions:
        # Filter 1: directional ambiguity
        if q.get("type") == "direction" and q.get("ambiguity_score", 0) > 0.7:
            removed_counts["ambiguous_direction"] += 1
            continue

        # Filter 2: intervention questions must actually change something
        if q.get("level") in ("L2", "L3") and q.get("relation_unchanged", False):
            removed_counts["unchanged_relation"] += 1
            continue

        # Filter 3: distance near boundary
        if q.get("type") == "distance" and q.get("near_boundary", False):
            removed_counts["near_boundary"] += 1
            continue

        filtered.append(q)

    # Filter 4: deduplicate near-identical questions.
    # Same (scene, frame, type, primary_object) → keep only one.
    seen_keys: set[tuple] = set()
    deduped: list[dict] = []
    for q in filtered:
        # Pick the most specific object label for dedup
        primary_label = (
            q.get("obj_a_label")
            or q.get("moved_obj_label")
            or q.get("obj_target_label")
            or ""
        )
        key = (
            q.get("scene_id"),
            q.get("image_name"),
            q.get("type"),
            primary_label,
        )
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
        answer_counts = Counter(q["answer"] for q in group)
        total = len(group)
        needs_rebalance = any(c / total > max_ratio for c in answer_counts.values())

        if needs_rebalance:
            logger.info(
                "Rebalancing %s: %s (total=%d)", key, dict(answer_counts), total
            )
            for q in group:
                correct_val = q["correct_value"]
                options = q["options"]
                random.shuffle(options)
                new_idx = options.index(correct_val)
                q["answer"] = chr(65 + new_idx)

        balanced.extend(group)

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
        2. Answer distribution balancing
        3. Log statistics
    """
    questions = quality_filter(questions)
    questions = balance_answer_values(questions)
    questions = balance_answer_distribution(questions)
    stats = compute_statistics(questions)
    logger.info("Final statistics: %s", stats)
    return questions
