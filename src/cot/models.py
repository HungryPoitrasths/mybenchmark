from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CotFactRecord:
    question_uid: str
    question_type: str
    signature_id: str
    facts: dict[str, Any]
    semantic_answer: Any
    answer_letters: list[str]
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FactExtractionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
