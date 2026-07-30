from __future__ import annotations

import re
from typing import Any

from .models import CotFactRecord, FactExtractionError
from .render import reasoning_clauses


def _normalize(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return re.sub(r"^(?:the|a|an)\s+", "", text)


def _normalize_occlusion_answer(value: Any) -> str:
    text = _normalize(value)
    return re.sub(r"\s*\([^)]*(?:query|reference)[^)]*\)", "", text)


def validate_answer_mapping(question: dict[str, Any], record: CotFactRecord) -> None:
    options = question.get("options")
    if not isinstance(options, list):
        raise FactExtractionError("missing_options", "options is not a list")
    selected = [options[ord(letter) - ord("A")] for letter in record.answer_letters]
    if question.get("multi_select"):
        expected = {_normalize(value) for value in record.semantic_answer}
        actual = {_normalize(value) for value in selected}
        if expected != actual:
            raise FactExtractionError(
                "answer_semantic_mismatch",
                f"letters select {sorted(actual)}, expected {sorted(expected)}",
            )
    elif len(selected) != 1 or _normalize(selected[0]) != _normalize(record.semantic_answer):
        raise FactExtractionError(
            "answer_semantic_mismatch",
            f"letter selects {selected!r}, expected {record.semantic_answer!r}",
        )


def validate_fact_consistency(record: CotFactRecord) -> None:
    facts = record.facts
    qtype = record.question_type
    answer_fact_keys = {
        "direction_agent": "result",
        "direction_object_centric": "result",
        "direction_allocentric": "result",
        "occlusion": "status",
        "distance": "distance_bin",
        "object_move_agent": "result",
        "object_move_distance": "distance_bin",
        "object_rotate_object_centric": "result",
        "object_move_object_centric": "result",
        "object_move_allocentric": "result",
        "object_remove": "new_status",
        "coordinate_rotation_agent": "result",
        "coordinate_rotation_object_centric": "result",
        "coordinate_rotation_allocentric": "result",
    }
    answer_key = answer_fact_keys.get(qtype)
    if answer_key and _normalize(record.semantic_answer) != _normalize(facts.get(answer_key)):
        raise FactExtractionError(
            "semantic_fact_mismatch",
            f"semantic answer {record.semantic_answer!r} does not match facts.{answer_key} "
            f"{facts.get(answer_key)!r}",
        )

    if qtype == "attachment_chain":
        expected = {_normalize(value) for value in facts.get("correct_values", [])}
        actual = {_normalize(value) for value in record.semantic_answer}
        if actual != expected:
            raise FactExtractionError(
                "semantic_fact_mismatch",
                f"attachment answers {sorted(actual)!r} do not match {sorted(expected)!r}",
            )

    if qtype == "object_move_occlusion":
        relation = facts.get("pairwise_relation")
        semantic = _normalize_occlusion_answer(record.semantic_answer)
        query = re.escape(_normalize(facts.get("query_object")))
        reference = re.escape(_normalize(facts.get("reference_object")))
        if relation == "query_occluded_by_ref":
            matches = re.search(
                rf"{query}\s+(?:is\s+)?occluded by\s+(?:the\s+)?{reference}", semantic
            )
        elif relation == "ref_occluded_by_query":
            matches = re.search(
                rf"{reference}\s+(?:is\s+)?occluded by\s+(?:the\s+)?{query}", semantic
            )
        else:
            matches = relation == "neither" and "neither" in semantic
        if not matches:
            raise FactExtractionError(
                "semantic_fact_mismatch",
                f"occlusion answer {record.semantic_answer!r} contradicts relation {relation!r}",
            )

    layout = facts.get("layout")
    if layout in {"single_axis", "diagonal"}:
        neutral_count = sum(
            value == "neutral"
            for value in (facts.get("horizontal_axis"), facts.get("depth_axis"))
        )
        expected_neutral_count = 1 if layout == "single_axis" else 0
        if neutral_count != expected_neutral_count:
            raise FactExtractionError(
                "axis_layout_mismatch",
                f"{layout} has {neutral_count} neutral axes, expected {expected_neutral_count}",
            )

    old_result = facts.get("old_result")
    result = facts.get("result")
    change_state = facts.get("change_state")
    if old_result not in {None, "unknown"} and result is not None:
        inconsistent = (
            change_state == "changed" and _normalize(old_result) == _normalize(result)
        ) or (
            change_state == "preserved" and _normalize(old_result) != _normalize(result)
        )
        if inconsistent:
            raise FactExtractionError(
                "change_state_mismatch",
                f"{change_state} conflicts with {old_result!r} -> {result!r}",
            )


def validate_response(response: str, record: CotFactRecord) -> None:
    expected = f"Answer: {' '.join(record.answer_letters)}"
    lines = response.rstrip().splitlines()
    if not lines or lines[-1] != expected:
        raise FactExtractionError("invalid_answer_line", f"expected final line {expected!r}")
    if response.count("Answer:") != 1:
        raise FactExtractionError("duplicate_answer_line", "response must contain one answer line")
    if "<think>" in response.lower() or "</think>" in response.lower():
        raise FactExtractionError("forbidden_think_tag", "responses must not use think tags")
    reasoning = "\n".join(lines[:-1]).strip()
    if len(reasoning.split()) < 18:
        raise FactExtractionError("reasoning_too_short", "reasoning has fewer than 18 words")


def validate_reasoning_consistency(response: str, record: CotFactRecord) -> None:
    """Verify that rendered reasoning retains the facts used to derive the answer."""
    reasoning = response.rsplit("\nAnswer:", 1)[0]
    normalized_reasoning = _normalize(reasoning)
    for clause_name, clause in zip(
        ("observation", "transformation", "conclusion"),
        reasoning_clauses(record),
        strict=True,
    ):
        if _normalize(clause) not in normalized_reasoning:
            raise FactExtractionError(
                "reasoning_fact_mismatch",
                f"rendered reasoning is missing the {clause_name} fact: {clause!r}",
            )

    if record.question_type != "object_rotate_object_centric":
        return

    direction = str(record.facts.get("rotation_direction") or "").strip().lower()
    if direction not in {"clockwise", "counterclockwise"}:
        raise FactExtractionError(
            "invalid_rotation_direction",
            f"unsupported rotation direction {direction!r}",
        )
    angle = int(record.facts["rotation_angle"])
    required_operation = rf"\b{angle}\s+degrees?\s+{re.escape(direction)}\b"
    if not re.search(required_operation, reasoning, flags=re.IGNORECASE):
        raise FactExtractionError(
            "reasoning_rotation_mismatch",
            f"reasoning does not state the {angle}-degree {direction} orbit",
        )

    opposite_pattern = (
        r"(?<!counter)clockwise"
        if direction == "counterclockwise"
        else r"counterclockwise"
    )
    if re.search(opposite_pattern, reasoning, flags=re.IGNORECASE):
        opposite = "clockwise" if direction == "counterclockwise" else "counterclockwise"
        raise FactExtractionError(
            "reasoning_rotation_mismatch",
            f"reasoning contradicts the saved direction with {opposite!r}",
        )


def validate_sft_item(item: dict[str, Any]) -> None:
    images = item.get("images")
    messages = item.get("messages")
    if not isinstance(images, list) or not images:
        raise FactExtractionError("missing_sft_images", "SFT item has no images")
    if not isinstance(messages, list) or len(messages) != 2:
        raise FactExtractionError("invalid_sft_messages", "expected user and assistant messages")
    user_content = str(messages[0].get("content") or "")
    if user_content.count("<image>") != len(images):
        raise FactExtractionError(
            "image_placeholder_mismatch",
            f"{user_content.count('<image>')} placeholders for {len(images)} images",
        )
