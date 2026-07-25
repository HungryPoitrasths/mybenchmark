import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scripts.audit_benchmark_gpt import (
    ApiRequestLimiter,
    CHECK_NAMES,
    SceneMetadataResolver,
    applicable_checks,
    attachment_pairs,
    audit_question,
    cache_key,
    call_openai_responses,
    ordered_image_names,
    result_has_unresolved_api_failure,
    reusable_progress_result,
    resolve_api_key,
    response_json_schema,
    run_audit,
    run_with_api_failure_retries,
)


def model_payload(overrides: dict[str, str] | None = None) -> dict:
    overrides = overrides or {}
    checks = {}
    for name in CHECK_NAMES:
        verdict = overrides.get(name, "not_applicable")
        issue = {
            "code": "other",
            "message_zh": f"{name} needs review",
            "image_indices": [1],
            "object_labels": [],
        }
        base = {
            "verdict": verdict,
            "summary_zh": "ok" if verdict == "pass" else "review",
            "issues": [issue] if verdict in {"fail", "uncertain"} else [],
        }
        if name == "referability":
            base["object_checks"] = []
        elif name == "occlusion_visibility":
            base.update(observed_status=None, gt_correct=None)
        elif name == "attachment_pair":
            base["pair_checks"] = []
        elif name == "continuity":
            base["sequence_continuous"] = None
        elif name == "fairness":
            base.update(evidence_sufficient=None, answer_leakage=None)
        checks[name] = base
    return {"checks": checks}


class AuditBenchmarkGptTests(unittest.TestCase):
    def test_api_request_limiter_serializes_request_starts(self) -> None:
        now = 100.0
        sleeps: list[float] = []

        def clock() -> float:
            return now + sum(sleeps)

        limiter = ApiRequestLimiter(65, clock=clock, sleeper=sleeps.append)
        limiter.wait()
        limiter.wait()
        limiter.wait()

        self.assertEqual(sleeps, [65.0, 65.0])

    @staticmethod
    def api_error_result() -> dict:
        stage = {
            "model": "gpt-5.2",
            "status": "error",
            "error": "401 Unauthorized",
            "result": None,
        }
        return {
            "source_index": 4,
            "final_source": "review",
            "final_status": "flagged",
            "primary_result": {**stage, "model": "gpt-4.1-mini"},
            "review_result": stage,
            "final_result": stage,
            "completed_at": "2026-07-25T00:00:00+00:00",
        }

    def test_whole_question_api_failure_is_retried_until_success(self) -> None:
        successful = {
            "source_index": 4,
            "final_source": "primary",
            "final_status": "passed",
            "final_result": {"model": "gpt-4.1-mini", "status": "ok", "error": None},
        }
        results = iter([self.api_error_result(), successful])
        calls = 0

        def audit_once() -> dict:
            nonlocal calls
            calls += 1
            return next(results)

        result = run_with_api_failure_retries(
            audit_once,
            source_index=4,
            failed_api_retries=2,
            failed_api_retry_delay=0,
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result["final_status"], "passed")
        self.assertEqual(result["api_retry_count"], 1)
        self.assertEqual(len(result["api_failure_history"]), 1)
        self.assertFalse(result_has_unresolved_api_failure(result))

    def test_whole_question_api_failure_retry_is_bounded(self) -> None:
        calls = 0

        def audit_once() -> dict:
            nonlocal calls
            calls += 1
            return self.api_error_result()

        result = run_with_api_failure_retries(
            audit_once,
            source_index=4,
            failed_api_retries=2,
            failed_api_retry_delay=0,
        )
        self.assertEqual(calls, 3)
        self.assertEqual(result["api_retry_count"], 2)
        self.assertEqual(len(result["api_failure_history"]), 3)
        self.assertTrue(result_has_unresolved_api_failure(result))

    def test_resume_only_reuses_authoritative_or_input_validation_results(self) -> None:
        self.assertFalse(reusable_progress_result(self.api_error_result()))
        self.assertFalse(
            reusable_progress_result(
                {"final_source": "review", "final_result": {"status": "invalid"}}
            )
        )
        self.assertTrue(
            reusable_progress_result(
                {"final_source": "review", "final_result": {"status": "ok"}}
            )
        )
        self.assertTrue(
            reusable_progress_result(
                {"final_source": "input_validation", "final_result": {"status": "input_error"}}
            )
        )

    def test_resume_requeues_cached_api_failure(self) -> None:
        question = {
            "scene_id": "scene0000_00",
            "image_name": "missing.jpg",
            "question": "Q",
            "type": "direction_agent",
        }
        with tempfile.TemporaryDirectory(dir="tests") as tmp:
            root = Path(tmp)
            key = cache_key(
                question,
                0,
                primary_model="gpt-4.1-mini",
                review_model="gpt-5.2",
                max_image_edge=32,
                max_output_tokens=100,
                scannetpp_sensor="iphone",
                base_url="https://gateway.example/v1",
            )
            (root / "progress.jsonl").write_text(
                json.dumps({"cache_key": key, "result": self.api_error_result()}) + "\n",
                encoding="utf-8",
            )

            def caller(**kwargs):
                raise AssertionError("missing input must not call the API")

            results = run_audit(
                [question],
                benchmark_path=root / "benchmark.json",
                output_dir=root,
                metadata_root=None,
                scannet_roots=[root / "images"],
                scannetpp_roots=[],
                scannetpp_sensor="iphone",
                primary_model="gpt-4.1-mini",
                review_model="gpt-5.2",
                max_image_edge=32,
                max_output_tokens=100,
                timeout=1,
                max_workers=1,
                api_key="test",
                base_url="https://gateway.example/v1",
                resume=True,
                failed_api_retries=0,
                failed_api_retry_delay=0,
                caller=caller,
            )
            progress_lines = (root / "progress.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(results[0]["final_source"], "input_validation")
        self.assertEqual(len(progress_lines), 2)

    def test_api_key_falls_back_to_api_key_environment_variable(self) -> None:
        with mock.patch.dict("os.environ", {"API_KEY": "lab-secret"}, clear=True):
            self.assertEqual(resolve_api_key(), ("lab-secret", "API_KEY"))

    def test_explicit_api_key_environment_variable_is_strict(self) -> None:
        with mock.patch.dict("os.environ", {"API_KEY": "lab-secret"}, clear=True):
            self.assertIsNone(resolve_api_key("CUSTOM_API_KEY"))

    def test_responses_api_uses_strict_json_schema_without_verbosity(self) -> None:
        payload = model_payload({"referability": "pass"})

        class FakeResponse:
            output_text = json.dumps(payload)
            usage = None
            id = "response-1"

        class FakeResponses:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        fake_responses = FakeResponses()
        fake_client = type("FakeClient", (), {"responses": fake_responses})()
        with mock.patch("scripts.audit_benchmark_gpt._get_openai_client", return_value=fake_client):
            parsed, _, _, response_id = call_openai_responses(
                model="gpt-4.1-mini",
                content=[{"type": "input_text", "text": "audit"}],
                api_key="test",
                base_url=None,
                max_output_tokens=100,
                timeout=1,
            )
        self.assertEqual(parsed, payload)
        self.assertEqual(response_id, "response-1")
        text_config = fake_responses.kwargs["text"]
        self.assertNotIn("verbosity", text_config)
        self.assertTrue(text_config["format"]["strict"])
        self.assertEqual(text_config["format"]["type"], "json_schema")

    def test_responses_api_limits_every_retry_attempt(self) -> None:
        payload = model_payload({"referability": "pass"})

        class FakeResponse:
            output_text = json.dumps(payload)
            usage = None
            id = "response-1"

        class FakeResponses:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    error = RuntimeError("rate limit")
                    error.status_code = 429
                    raise error
                return FakeResponse()

        class FakeLimiter:
            def __init__(self) -> None:
                self.calls = 0

            def wait(self) -> None:
                self.calls += 1

        fake_responses = FakeResponses()
        fake_limiter = FakeLimiter()
        fake_client = type("FakeClient", (), {"responses": fake_responses})()
        with (
            mock.patch("scripts.audit_benchmark_gpt._get_openai_client", return_value=fake_client),
            mock.patch("scripts.audit_benchmark_gpt.time.sleep"),
        ):
            call_openai_responses(
                model="gpt-4.1-mini",
                content=[{"type": "input_text", "text": "audit"}],
                api_key="test",
                base_url=None,
                max_output_tokens=100,
                timeout=1,
                request_limiter=fake_limiter,
            )

        self.assertEqual(fake_responses.calls, 2)
        self.assertEqual(fake_limiter.calls, 2)

    def test_ordered_images_put_reasoning_frame_last_and_deduplicate(self) -> None:
        question = {
            "image_name": "first.jpg",
            "auxiliary_image_names": ["bridge.jpg", "last.jpg", "bridge.jpg"],
            "reasoning_frame_2": "last.jpg",
        }
        self.assertEqual(
            ordered_image_names(question),
            ["first.jpg", "bridge.jpg", "last.jpg"],
        )

    def test_applicable_checks_cover_attachment_occlusion_and_multiview(self) -> None:
        question = {
            "level": "L1",
            "type": "occlusion",
            "attachment_remapped": True,
        }
        self.assertEqual(applicable_checks(question, ["a.jpg", "b.jpg"]), list(CHECK_NAMES))

    def test_attachment_labels_are_resolved_from_scene_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir="tests") as tmp:
            root = Path(tmp)
            (root / "abc.json").write_text(
                json.dumps({"objects": [{"id": 27, "label": "counter"}, {"id": 88, "label": "water filter"}]}),
                encoding="utf-8",
            )
            pairs, errors = attachment_pairs(
                {"scene_id": "abc", "attachment_pair_id": "27->88"},
                SceneMetadataResolver(root),
            )
        self.assertEqual(errors, [])
        self.assertEqual(pairs[0]["parent_label"], "counter")
        self.assertEqual(pairs[0]["child_label"], "water filter")

    def test_strict_schema_requires_every_property(self) -> None:
        schema = response_json_schema()

        def visit(node: object) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))
                    self.assertFalse(node.get("additionalProperties", True))
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(schema)

    def test_review_pass_is_authoritative_after_primary_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir="tests") as tmp:
            image_root = Path(tmp)
            scene_dir = image_root / "abc12345"
            scene_dir.mkdir()
            Image.new("RGB", (8, 8), "white").save(scene_dir / "first.jpg")
            calls: list[str] = []

            def caller(**kwargs):
                calls.append(kwargs["model"])
                verdict = "fail" if len(calls) == 1 else "pass"
                payload = model_payload({"referability": verdict})
                return payload, json.dumps(payload), None, f"response-{len(calls)}"

            result = audit_question(
                {
                    "scene_id": "abc12345",
                    "image_name": "first.jpg",
                    "question": "Which object is visible?",
                    "type": "direction_agent",
                },
                0,
                metadata=SceneMetadataResolver(None),
                scannet_roots=[],
                scannetpp_roots=[image_root],
                scannetpp_sensor="iphone",
                primary_model="gpt-4.1-mini",
                review_model="gpt-5.2",
                max_image_edge=32,
                max_output_tokens=100,
                timeout=1,
                api_key="test",
                base_url=None,
                caller=caller,
            )
        self.assertEqual(calls, ["gpt-4.1-mini", "gpt-5.2"])
        self.assertEqual(result["final_source"], "review")
        self.assertEqual(result["final_status"], "passed")
        self.assertEqual(result["problem_checks"], [])

    def test_primary_pass_skips_review(self) -> None:
        with tempfile.TemporaryDirectory(dir="tests") as tmp:
            image_root = Path(tmp)
            scene_dir = image_root / "abc12345"
            scene_dir.mkdir()
            Image.new("RGB", (8, 8), "white").save(scene_dir / "first.jpg")
            calls: list[str] = []

            def caller(**kwargs):
                calls.append(kwargs["model"])
                payload = model_payload({"referability": "pass"})
                return payload, json.dumps(payload), None, "response-1"

            result = audit_question(
                {"scene_id": "abc12345", "image_name": "first.jpg", "question": "Q", "type": "direction_agent"},
                0,
                metadata=SceneMetadataResolver(None),
                scannet_roots=[],
                scannetpp_roots=[image_root],
                scannetpp_sensor="iphone",
                primary_model="gpt-4.1-mini",
                review_model="gpt-5.2",
                max_image_edge=32,
                max_output_tokens=100,
                timeout=1,
                api_key="test",
                base_url=None,
                caller=caller,
            )
        self.assertEqual(calls, ["gpt-4.1-mini"])
        self.assertIsNone(result["review_result"])
        self.assertEqual(result["final_status"], "passed")

    def test_missing_image_is_flagged_without_api_call(self) -> None:
        def caller(**kwargs):
            raise AssertionError("API should not be called for deterministic input errors")

        result = audit_question(
            {"scene_id": "abc12345", "image_name": "missing.jpg", "question": "Q", "type": "direction_agent"},
            0,
            metadata=SceneMetadataResolver(None),
            scannet_roots=[],
            scannetpp_roots=[Path("does-not-exist")],
            scannetpp_sensor="iphone",
            primary_model="gpt-4.1-mini",
            review_model="gpt-5.2",
            max_image_edge=32,
            max_output_tokens=100,
            timeout=1,
            api_key="test",
            base_url=None,
            caller=caller,
        )
        self.assertEqual(result["final_source"], "input_validation")
        self.assertEqual(result["final_status"], "flagged")
        self.assertTrue(result["input_errors"])


if __name__ == "__main__":
    unittest.main()
