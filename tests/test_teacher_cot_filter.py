from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "filter_correct_teacher_cot.py"
    spec = importlib.util.spec_from_file_location("filter_correct_teacher_cot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question(image_path: Path) -> dict[str, Any]:
    return {
        "level": "L1",
        "type": "direction_agent",
        "question": "Where is the cup relative to the chair?",
        "options": ["left", "right", "front", "back"],
        "answer": "A",
        "correct_value": "left",
        "scene_id": "scene0000_00",
        "image_name": image_path.name,
        "image_path": str(image_path),
        "obj_a_label": "cup",
        "obj_b_label": "chair",
        "oracle_facts": {"result": "left"},
    }


def _reasoning(answer: str) -> str:
    return (
        "The visible cup sits clearly on the left side of the chair in the image. "
        "Their horizontal positions provide the relevant relation without needing any hidden scene information.\n"
        f"Answer: {answer}"
    )


class _FakeCompletions:
    def __init__(self, outcomes: list[Any], calls: list[dict[str, Any]]):
        self.outcomes = outcomes
        self.calls = calls

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


class _FakeClient:
    def __init__(self, outcomes: list[Any], calls: list[dict[str, Any]]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcomes, calls))


def _args(tmp_path: Path, benchmark: Path, output_name: str = "teacher.json") -> Namespace:
    return Namespace(
        benchmark=benchmark,
        output=tmp_path / output_name,
        cache_jsonl=tmp_path / "teacher.cache.jsonl",
        base_url="https://teacher.invalid/v1",
        model="teacher-vlm",
        api_key_env="TEST_API_KEY",
        max_attempts=2,
        transport_retries=2,
        retry_delay=0.0,
        timeout=10.0,
        workers=1,
        max_output_tokens=384,
        temperature=0.2,
        limit=None,
        limit_per_type=None,
        question_uid=[],
        seed=42,
        progress_every=1,
        scannet_image_root=[],
        scannetpp_image_root=[],
        scannetpp_sensor="iphone",
    )


def _mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_mapping_keys(child))
    return keys


def test_transport_retry_does_not_consume_two_semantic_attempts_and_cache_resumes(
    tmp_path: Path,
) -> None:
    script = _load_script()
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"synthetic-jpeg")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps({"questions": [_question(image)]}), encoding="utf-8"
    )
    args = _args(tmp_path, benchmark)
    calls: list[dict[str, Any]] = []
    outcomes = [RuntimeError("temporary network error"), _reasoning("B"), _reasoning("A")]
    client = _FakeClient(outcomes, calls)

    report = script.run_filter(args, client_factory=lambda: client)

    assert report["accepted_count"] == 1
    assert len(calls) == 3
    output = json.loads(args.output.read_text(encoding="utf-8"))
    assert output["questions"][0]["teacher_cot"].endswith("Answer: A")
    cache = [json.loads(line) for line in args.cache_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["kind"] for row in cache] == [
        "transport_error",
        "semantic_attempt",
        "semantic_attempt",
        "terminal",
    ]
    assert cache[1]["semantic_attempt"] == 1
    assert cache[2]["semantic_attempt"] == 2

    forbidden = {"answer", "correct_value", "correct_values", "facts", "oracle_facts"}
    assert not (forbidden & _mapping_keys(calls[0]["messages"]))
    user_content = calls[0]["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in user_content) == 1

    class _NoCallFactory:
        def __call__(self) -> Any:
            raise AssertionError("a terminal cache hit must not call the API")

    resumed_args = _args(tmp_path, benchmark, "teacher-resumed.json")
    resumed = script.run_filter(resumed_args, client_factory=_NoCallFactory())
    assert resumed["accepted_count"] == 1


def test_two_wrong_semantic_attempts_discard_question(tmp_path: Path) -> None:
    script = _load_script()
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"synthetic-jpeg")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps([_question(image)]), encoding="utf-8")
    args = _args(tmp_path, benchmark)
    calls: list[dict[str, Any]] = []
    client = _FakeClient([_reasoning("B"), _reasoning("C")], calls)

    report = script.run_filter(args, client_factory=lambda: client)

    assert report["accepted_count"] == 0
    assert report["rejected_count"] == 1
    assert len(calls) == 2
    output = json.loads(args.output.read_text(encoding="utf-8"))
    assert output["questions"] == []


def test_response_validation_normalizes_multiselect_and_rejects_repetition() -> None:
    script = _load_script()
    question = {
        "level": "L3",
        "type": "attachment_chain",
        "question": "Which objects move with the table?",
        "options": ["the tray", "the cup", "the chair"],
        "answer": "A B",
        "correct_value": "the tray; the cup",
        "correct_values": ["the tray", "the cup"],
        "multi_select": True,
        "scene_id": "scene0000_00",
        "image_name": "frame.jpg",
        "grandparent_label": "table",
        "parent_label": "tray",
        "grandchild_label": "cup",
        "neighbor_label": "chair",
    }
    response = (
        "The tray is directly attached to the table and the cup belongs to that attached chain. "
        "The separate chair has no attachment path connecting it to the moved table.\n"
        "Answer: B A"
    )

    normalized, reason, _ = script.validate_teacher_response(question, response)

    assert reason == "accepted"
    assert normalized is not None and normalized.endswith("Answer: A B")

    phrase = "one two three four five six seven eight"
    repeated = f"{phrase} {phrase} {phrase}\nAnswer: A"
    single = _question(Path("frame.jpg"))
    normalized, reason, _ = script.validate_teacher_response(single, repeated)
    assert normalized is None
    assert reason == "repeated_8gram"

    malformed = _reasoning("A").replace("Answer: A", "Final: A")
    normalized, reason, _ = script.validate_teacher_response(single, malformed)
    assert normalized is None
    assert reason == "invalid_answer_format"


def test_small_sample_selection_is_deterministic_and_per_type_limited() -> None:
    script = _load_script()
    questions = []
    for question_type in ("direction_agent", "distance"):
        for index in range(5):
            questions.append(
                {
                    "type": question_type,
                    "question_uid": f"{question_type}-{index}",
                }
            )

    selected_a = script.select_questions(
        questions,
        seed=9,
        limit=3,
        limit_per_type=2,
        requested_uids=[],
    )
    selected_b = script.select_questions(
        questions,
        seed=9,
        limit=3,
        limit_per_type=2,
        requested_uids=[],
    )

    assert selected_a == selected_b
    counts: dict[str, int] = {}
    for row in selected_a:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    assert len(selected_a) == 3
    assert max(counts.values()) <= 2
