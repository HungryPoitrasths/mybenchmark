from __future__ import annotations

import re
from typing import Any

from .models import CotFactRecord, FactExtractionError


def _normalize(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return re.sub(r"^(?:the|a|an)\s+", "", text)


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
