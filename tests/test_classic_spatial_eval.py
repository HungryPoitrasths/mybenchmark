from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.classic_spatial_eval.common import (
    EXPECTED_TOTAL,
    MANIFEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    TARGET_COUNTS,
    parse_answer,
    sha256_file,
    write_json,
    write_jsonl,
)
from src.classic_spatial_eval.inference import ModelSpec, parse_model_specs, run_model
from src.classic_spatial_eval.inference import main as inference_main
from src.classic_spatial_eval import preparation
from src.classic_spatial_eval.preparation import (
    PrepareConfig,
    Sample,
    load_benchmark_samples,
    normalize_blink,
    normalize_clevrer,
    normalize_mindcube,
    normalize_mmsi,
    normalize_mvbench,
    normalize_spar,
    normalize_vsi,
    normalize_vsr,
    materialize_sample,
    select_samples,
)
from src.classic_spatial_eval.scoring import load_results, summarize


def manifest_row(sample_id: str, *, gold: str = "A", subset: str = "s") -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": "toy",
        "subset": subset,
        "split": "test",
        "sample_id": sample_id,
        "source_id": sample_id,
        "group_id": sample_id,
        "media": [
            {"kind": "image", "order": 0, "path": "image.jpg", "sha256": "unused"}
        ],
        "question": "Which answer is correct?",
        "options": ["one", "two"],
        "gold": [gold],
        "multi_select": False,
        "frame_indices": [],
    }


def result_row(
    sample_id: str,
    model: str,
    manifest_hash: str,
    *,
    correct: bool,
    gold: str = "A",
    prediction: str | None = None,
) -> dict:
    prediction = prediction or (gold if correct else ("B" if gold == "A" else "A"))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "model_label": model,
        "model_path": model,
        "backend": "fake",
        "benchmark": "toy",
        "subset": "s",
        "sample_id": sample_id,
        "source_id": sample_id,
        "group_id": sample_id,
        "raw_response": prediction,
        "prediction": [prediction],
        "prediction_text": prediction,
        "parse_status": "ok",
        "parse_source": "last_line",
        "gold": [gold],
        "correct": correct,
        "option_count": 2,
        "multi_select": False,
        "error": None,
        "elapsed_seconds": 0.1,
    }


def test_target_counts_are_decision_complete() -> None:
    assert EXPECTED_TOTAL == 3469
    assert sum(len(subsets) for subsets in TARGET_COUNTS.values()) == 22
    assert TARGET_COUNTS["mmsi"] == {
        "object_motion": 76,
        "multi_step_reasoning": 198,
    }
    assert sum(TARGET_COUNTS["blink"].values()) == 572


@pytest.mark.parametrize(
    ("raw", "multi", "expected", "status"),
    [
        ("<answer>B</answer>", False, ("B",), "ok"),
        ("Reasoning mentions A.\nC", False, ("C",), "ok"),
        ("A C", True, ("A", "C"), "ok"),
        ("<answer>A</answer><answer>B</answer>", False, (), "conflict"),
        ("I cannot tell", False, (), "invalid"),
    ],
)
def test_answer_parser(raw: str, multi: bool, expected: tuple[str, ...], status: str) -> None:
    parsed = parse_answer(raw, ["one", "two", "three"], multi_select=multi)
    assert parsed.letters == expected
    assert parsed.status == status


def test_model_specs_use_base_alias() -> None:
    specs = parse_model_specs(["base=base", "grpo=/ckpt/9216"], "/models/qwen")
    assert specs == [
        ModelSpec("base", "/models/qwen"),
        ModelSpec("grpo", "/ckpt/9216"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        parse_model_specs(["base=base", "base=/other"], "/models/qwen")


def test_normalizers_cover_all_eight_benchmarks() -> None:
    mmsi = normalize_mmsi(
        [
            {
                "id": 1,
                "question_type": "Motion (Obj.)",
                "question": "q",
                "options": ["x", "y"],
                "answer": "B",
                "images": [b"image"],
            },
            {
                "id": 2,
                "question_type": "MSR",
                "question": "q",
                "options": ["x", "y"],
                "answer": "A",
                "images": [b"image"],
            },
        ]
    )
    assert [row.subset for row in mmsi] == ["object_motion", "multi_step_reasoning"]

    spar = normalize_spar(
        [
            {
                "id": "s",
                "task": "view_change_infer",
                "question": "q",
                "options": ["x", "y"],
                "answer": "A",
                "image": [{"bytes": b"one"}, {"bytes": b"two"}],
            }
        ]
    )
    assert spar[0].subset == "ViewChg"
    assert spar[0].media_values == [{"bytes": b"one"}, {"bytes": b"two"}]

    mindcube = normalize_mindcube(
        [
            {
                "id": "dynamics_1",
                "category": "Dynamics",
                "question": "What if? A. left B. right",
                "gt_answer": "A",
                "images": ["a.jpg"],
            },
            {
                "id": "static_1",
                "category": "Cognitive Mapping",
                "question": "A. left B. right",
                "gt_answer": "A",
                "images": ["b.jpg"],
            },
        ]
    )
    assert len(mindcube) == 1 and mindcube[0].options == ["left", "right"]

    vsi = normalize_vsi(
        [
            {
                "id": 3,
                "question_type": "object_rel_direction_hard",
                "scene_name": "scene1",
                "dataset": "scannet",
                "question": "q",
                "options": ["x", "y"],
                "ground_truth": "A",
            }
        ]
    )
    assert vsi[0].subset == "relative_direction"
    assert vsi[0].media_kind == "video"

    mvbench = normalize_mvbench(
        {
            "moving_direction": [
                {
                    "video": "v.mp4",
                    "question": "q",
                    "candidates": ["x", "y"],
                    "answer": "y",
                }
            ]
        }
    )
    assert mvbench[0].answer == "y" and mvbench[0].media_kind == "video"

    clevrer = normalize_clevrer(
        [
            {
                "scene_index": 7,
                "questions": [
                    {
                        "question_type": "counterfactual",
                        "question": "q",
                        "choices": [
                            {"choice": "x", "answer": "wrong"},
                            {"choice": "y", "answer": "correct"},
                            {"choice": "z", "answer": "correct"},
                        ],
                    }
                ],
            }
        ]
    )
    assert clevrer[0].answer == ["B", "C"]
    assert clevrer[0].multi_select is True

    blink = normalize_blink(
        {
            "Relative_Depth": [
                {
                    "idx": "b",
                    "question": "q",
                    "choices": ["x", "y"],
                    "answer": "(B)",
                    "image_1": b"image",
                }
            ]
        }
    )
    assert blink[0].media_values == [b"image"]

    vsr = normalize_vsr(
        [
            {
                "caption": "The cup is left of the plate.",
                "label": 1,
                "image": "000000000001.jpg",
                "image_link": "http://example.test/000000000001.jpg",
            }
        ]
    )
    assert vsr[0].options == ["False", "True"]
    assert vsr[0].answer == "B"
    assert vsr[0].media_values == [
        {
            "path": "000000000001.jpg",
            "fallback_url": "http://example.test/000000000001.jpg",
        }
    ]


def test_selection_prefers_unique_groups_and_is_deterministic() -> None:
    samples = [
        Sample(
            "vsi",
            "relative_direction",
            "test",
            str(index),
            "q",
            ["x", "y"],
            "A",
            ["v.mp4"],
            group_id=f"scene-{index % 60}",
        )
        for index in range(100)
    ]
    for subset in ("relative_distance", "route_planning"):
        samples.extend(
            Sample(
                "vsi",
                subset,
                "test",
                f"{subset}-{index}",
                "q",
                ["x", "y"],
                "A",
                ["v.mp4"],
                group_id=f"{subset}-scene-{index}",
            )
            for index in range(50)
        )
    first = select_samples(samples, "vsi", seed=42)
    second = select_samples(list(reversed(samples)), "vsi", seed=42)
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    direction = [sample for sample in first if sample.subset == "relative_direction"]
    assert len({sample.group_id for sample in direction}) == 50


def test_materialized_media_is_relative_and_hashed(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (4, 4), color=(10, 20, 30)).save(buffer, format="PNG")
    sample = Sample(
        "blink",
        "Relative_Depth",
        "test",
        "sample",
        "Which option?",
        ["near", "far"],
        "B",
        [buffer.getvalue()],
    )
    row = materialize_sample(
        sample,
        source_root=None,
        output_dir=tmp_path,
        download_missing=False,
    )
    assert row["gold"] == ["B"]
    assert not Path(row["media"][0]["path"]).is_absolute()
    media_path = tmp_path / row["media"][0]["path"]
    assert media_path.is_file()
    assert row["media"][0]["sha256"] == sha256_file(media_path)

    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [row])
    assert (
        inference_main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "predictions"),
                "--model-spec",
                "base=base",
                "--base-model",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--dry-run",
            ]
        )
        == 0
    )


def test_vsr_missing_local_image_uses_fallback_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    source = tmp_path / "source.png"
    image_module.new("RGB", (4, 4), color=(30, 20, 10)).save(source)
    downloads: list[str] = []

    def fake_download(url: str, destination: str | Path) -> None:
        downloads.append(url)
        Path(destination).write_bytes(source.read_bytes())

    monkeypatch.setattr(preparation.urllib.request, "urlretrieve", fake_download)
    sample = Sample(
        "vsr",
        "zero_shot_test",
        "test",
        "fallback",
        "Is this statement true?",
        ["False", "True"],
        "B",
        [
            {
                "path": "missing.jpg",
                "fallback_url": "http://example.test/fallback.jpg",
            }
        ],
    )
    output_dir = tmp_path / "prepared"
    row = materialize_sample(
        sample,
        source_root=tmp_path / "vsr",
        output_dir=output_dir,
        download_missing=True,
    )
    assert downloads == ["http://example.test/fallback.jpg"]
    assert (output_dir / row["media"][0]["path"]).is_file()


def test_clevrer_loader_prefers_validation_annotations(tmp_path: Path) -> None:
    root = tmp_path / "clevrer"
    root.mkdir()

    def item(source_id: str) -> dict:
        return {
            "scene_index": 1,
            "questions": [
                {
                    "question_id": source_id,
                    "question_type": "predictive",
                    "question": "What happens next?",
                    "choices": [
                        {"choice": "x", "answer": "correct"},
                        {"choice": "y", "answer": "wrong"},
                    ],
                }
            ],
        }

    write_json(root / "CLEVR_train_questions.json", [item("train")])
    write_json(root / "CLEVR_val_questions.json", [item("validation")])
    config = PrepareConfig(
        output_dir=tmp_path / "prepared",
        cache_dir=tmp_path / "cache",
        roots={"clevrer": root},
        revisions={"clevrer_code": "pinned"},
        benchmarks=("clevrer",),
        download_missing=False,
        dry_run=True,
    )
    samples, media_root = load_benchmark_samples("clevrer", config)
    assert [sample.source_id for sample in samples] == ["validation"]
    assert media_root == root.resolve()


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, rows, manifest_path):
        self.calls += len(rows)
        return ["B" for _ in rows]

    def close(self) -> None:
        pass


def test_runner_resume_does_not_duplicate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [manifest_row("one", gold="B")]
    write_jsonl(manifest, rows)
    manifest_hash = sha256_file(manifest)
    output = tmp_path / "predictions.jsonl"
    backend = FakeBackend()
    kwargs = dict(
        model=ModelSpec("base", "base"),
        rows=rows,
        manifest_path=manifest,
        manifest_hash=manifest_hash,
        output_path=output,
        backend=backend,
        backend_name="fake",
        batch_size=1,
        resume=True,
    )
    assert run_model(**kwargs)["written"] == 1
    assert run_model(**kwargs)["written"] == 0
    assert backend.calls == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_scoring_reports_macro_delta_and_completeness(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest_rows = [manifest_row("one"), manifest_row("two")]
    write_jsonl(manifest, manifest_rows)
    manifest_hash = sha256_file(manifest)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            result_row("one", "base", manifest_hash, correct=True),
            result_row("two", "base", manifest_hash, correct=False),
            result_row("one", "grpo", manifest_hash, correct=True),
            result_row("two", "grpo", manifest_hash, correct=True),
        ],
    )
    loaded = load_results([predictions], expected_manifest_hash=manifest_hash)
    report = summarize(
        manifest_rows,
        loaded,
        base_label="base",
        iterations=200,
        seed=42,
        allow_incomplete=False,
    )
    overall = report["benchmarks"]["toy"]["models"]["grpo"]["overall"]
    assert overall["accuracy"] == 1.0
    assert overall["macro_accuracy"] == 1.0
    assert overall["delta_vs_base"] == 0.5
    assert report["completeness"]["grpo"]["missing"] == 0
