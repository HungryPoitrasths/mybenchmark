"""Prepare frozen manifests for the classic spatial benchmark suite."""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import (
    EXPECTED_TOTAL,
    LOCK_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    TARGET_COUNTS,
    answer_to_letters,
    sha256_file,
    split_inline_options,
    stable_rank,
    validate_manifest,
    write_json,
    write_jsonl,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_REVISIONS = {
    "mmsi": "ec7c92bfaf7728fcca1d61e3e224e190af309436",
    "spar": "ae9bbc5297fd277123c42b0628d1f40bf72f89f1",
    "mindcube": "9c941b46a6bd65b6914669ef7a579948fc9c8467",
    "mindcube_code": "6d8fd9acdf5b80769373a21cdbfcf068ab356e0f",
    "vsi": "bdcadb3fea447621a828a24911801faba3587c12",
    "mvbench": "230a2d4fac8900333c61754641c7a13e069ac9c6",
    "blink": "a3666eb249237ba3d5eca8db21176cc47967e040",
    "blink_code": "529b0ba055416dea5df79d594e7f62a910d8a308",
    "vsr": "b866daddec9717671066f55f0672eea05649d856",
    "vsr_source": "b27a0af0ee1462d2b6b92c8c83e869d9254a241a",
    "clevrer_code": "98b842082ba4f7c18b6b9e3f39145871782a65ef",
}

DATASET_IDS = {
    "mmsi": "RunsenXu/MMSI-Bench",
    "spar": "jasonzhango/SPAR-Bench-Tiny",
    "mindcube": "MLL-Lab/MindCube",
    "vsi": "nyu-visionx/VSI-Bench",
    "mvbench": "OpenGVLab/MVBench",
    "blink": "BLINK-Benchmark/BLINK",
    "vsr": "juletxara/visual-spatial-reasoning",
}

BENCHMARK_ORDER = (
    "mmsi",
    "spar",
    "mindcube",
    "vsi",
    "mvbench",
    "clevrer",
    "blink",
    "vsr",
)

VIDEO_FRAME_COUNTS = {"vsi": 32, "mvbench": 8, "clevrer": 32}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    benchmark: str
    subset: str
    split: str
    source_id: str
    question: str
    options: list[str]
    answer: Any
    media_values: list[Any]
    media_kind: str = "image"
    answer_type: str = "choice"
    group_id: str | None = None
    multi_select: bool = False
    start_seconds: float | None = None
    end_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return f"{self.benchmark}:{self.subset}:{self.source_id}"


@dataclass(frozen=True)
class PrepareConfig:
    output_dir: Path
    cache_dir: Path
    roots: Mapping[str, Path | None]
    revisions: Mapping[str, str]
    benchmarks: tuple[str, ...] = BENCHMARK_ORDER
    seed: int = 42
    download_missing: bool = True
    validate_media: bool = True
    smoke_per_subset: int = 0
    dry_run: bool = False


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _first(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _as_options(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        pairs = sorted(value.items(), key=lambda item: str(item[0]))
        return [str(item[1]).strip() for item in pairs]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return [str(item).strip() for item in converted]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded]
    return []


def _question_and_options(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    question = str(_first(row, "question", "prompt", "caption", default="")).strip()
    options = _as_options(_first(row, "options", "choices", "candidates"))
    if not options:
        stem, inline = split_inline_options(question)
        if inline:
            return stem, inline
    return question, options


def _media_from_row(row: Mapping[str, Any]) -> list[Any]:
    def expand(value: Any) -> list[Any]:
        if isinstance(value, (list, tuple)):
            return [item for item in value if item is not None]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if isinstance(converted, list):
                return [item for item in converted if item is not None]
        return [value]

    values = _first(row, "images", "image_paths", "frames", "media")
    if values is not None:
        return [value for value in expand(values) if value != ""]
    image_keys = sorted(
        (
            key
            for key in row
            if re.fullmatch(r"image(?:_\d+)?", str(key), flags=re.I)
        ),
        key=lambda key: (len(str(key)), str(key)),
    )
    media: list[Any] = []
    for key in image_keys:
        value = row[key]
        if value is None:
            continue
        media.extend(expand(value))
    return media


def _id(row: Mapping[str, Any], fallback: int) -> str:
    return str(
        _first(
            row,
            "id",
            "idx",
            "question_id",
            "sample_id",
            "uid",
            default=fallback,
        )
    )


def _category(row: Mapping[str, Any]) -> str:
    return _token(
        _first(
            row,
            "question_type",
            "task_type",
            "sub_task",
            "category",
            "type",
            "task",
            "setting",
            default="",
        )
    )


def normalize_mmsi(rows: Sequence[Mapping[str, Any]]) -> list[Sample]:
    aliases = {
        "object_motion": "object_motion",
        "object_movement": "object_motion",
        "motion_obj": "object_motion",
        "multi_step_reasoning": "multi_step_reasoning",
        "multistep_reasoning": "multi_step_reasoning",
        "multi_step": "multi_step_reasoning",
        "msr": "multi_step_reasoning",
    }
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        category = _category(row)
        subset = aliases.get(category)
        if subset is None and "object" in category and "motion" in category:
            subset = "object_motion"
        if subset is None and "multi" in category and "step" in category:
            subset = "multi_step_reasoning"
        if subset is None:
            continue
        question, options = _question_and_options(row)
        samples.append(
            Sample(
                "mmsi",
                subset,
                str(_first(row, "split", default="test")),
                _id(row, index),
                question,
                options,
                _first(row, "answer", "gt_answer", "ground_truth"),
                _media_from_row(row),
                group_id=str(_first(row, "scene_id", "scene", default=_id(row, index))),
            )
        )
    return samples


def normalize_spar(rows: Sequence[Mapping[str, Any]]) -> list[Sample]:
    aliases = {
        "viewchg": "ViewChg",
        "viewchgi": "ViewChg",
        "view_change_infer": "ViewChg",
        "spimag_oc": "SpImag_OC",
        "spatial_imagination_oc": "SpImag_OC",
        "spimag_oc_mv": "SpImag_OC_MV",
        "spatial_imagination_oc_mv": "SpImag_OC_MV",
        "spimag_oo": "SpImag_OO",
        "spatial_imagination_oo": "SpImag_OO",
        "spimag_oo_mv": "SpImag_OO_MV",
        "spatial_imagination_oo_mv": "SpImag_OO_MV",
    }
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        category = _category(row)
        subset = aliases.get(category)
        if subset is None:
            for alias in sorted(aliases, key=len, reverse=True):
                if category.startswith(alias + "_") or category.endswith("_" + alias):
                    subset = aliases[alias]
                    break
        if subset is None:
            continue
        question, options = _question_and_options(row)
        samples.append(
            Sample(
                "spar",
                subset,
                str(_first(row, "split", default="test")),
                _id(row, index),
                question,
                options,
                _first(row, "answer", "gt_answer", "ground_truth"),
                _media_from_row(row),
                answer_type="exact_text" if subset == "ViewChg" else "choice",
                group_id=str(_first(row, "scene_id", "image_group_id", default=_id(row, index))),
            )
        )
    return samples


def normalize_mindcube(rows: Sequence[Mapping[str, Any]]) -> list[Sample]:
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(rows):
        raw_category = row.get("category")
        if isinstance(raw_category, (list, tuple)):
            category_tokens = {_token(value) for value in raw_category}
        else:
            category_tokens = {_token(raw_category)} if raw_category is not None else set()

        # The official tinybench dynamics slice is all 153 sequence questions
        # plus rotation q2/q3 (79 + 60), for a fixed total of 292 examples.
        question_family = re.search(r"(?:^|_)q([0-9]+)(?:_|$)", _id(row, index))
        official_dynamic = "sequence" in category_tokens or (
            "rotation" in category_tokens
            and question_family is not None
            and question_family.group(1) in {"2", "3"}
        )
        if official_dynamic:
            selected.append((index, row))
            continue

        # Retain support for converted MindCube exports that expose a direct
        # dynamics/what-if task label instead of the official category array.
        category_values = " ".join(
            _token(row.get(key))
            for key in ("category", "task", "task_type", "question_type", "capability")
            if row.get(key) is not None
        )
        identifier = _token(_id(row, index))
        question = _token(row.get("question"))
        if "dynamic" in category_values or "what_if" in category_values:
            selected.append((index, row))
        elif not ({"sequence", "rotation"} & category_tokens) and any(
            term in f"{category_values} {identifier} {question}"
            for term in ("what_if", "dynamics")
        ):
            selected.append((index, row))
    samples: list[Sample] = []
    for index, row in selected:
        question, options = _question_and_options(row)
        samples.append(
            Sample(
                "mindcube",
                "dynamics",
                str(_first(row, "split", default="test")),
                _id(row, index),
                question,
                options,
                _first(row, "gt_answer", "answer", "ground_truth"),
                _media_from_row(row),
                group_id=str(_first(row, "image_group_id", "scene_id", default=_id(row, index))),
            )
        )
    return samples


def normalize_vsi(rows: Sequence[Mapping[str, Any]]) -> list[Sample]:
    aliases = {
        "object_rel_direction_easy": "relative_direction",
        "object_rel_direction_medium": "relative_direction",
        "object_rel_direction_hard": "relative_direction",
        "relative_direction": "relative_direction",
        "object_rel_distance": "relative_distance",
        "relative_distance": "relative_distance",
        "route_planning": "route_planning",
    }
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        subset = aliases.get(_category(row))
        if subset is None:
            continue
        question, options = _question_and_options(row)
        scene = str(_first(row, "scene_name", "scene_id", default=""))
        dataset = str(_first(row, "dataset", default=""))
        video = _first(row, "video", "video_path", "path")
        if video is None and scene:
            video = f"{dataset}/{scene}.mp4" if dataset else f"{scene}.mp4"
        samples.append(
            Sample(
                "vsi",
                subset,
                "test",
                _id(row, index),
                question,
                options,
                _first(row, "ground_truth", "answer", "gt_answer"),
                [video] if video is not None else [],
                media_kind="video",
                group_id=scene or _id(row, index),
                metadata={"source_dataset": dataset, "question_type": _category(row)},
            )
        )
    return samples


def normalize_mvbench(
    rows_by_subset: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[Sample]:
    samples: list[Sample] = []
    for subset, rows in rows_by_subset.items():
        for index, row in enumerate(rows):
            question, options = _question_and_options(row)
            video = _first(row, "video", "video_path", "path")
            samples.append(
                Sample(
                    "mvbench",
                    subset,
                    "test",
                    _id(row, index),
                    question,
                    options,
                    _first(row, "answer", "gt_answer", "ground_truth"),
                    [video] if video is not None else [],
                    media_kind="video",
                    group_id=str(video or _id(row, index)),
                    start_seconds=_float_or_none(_first(row, "start", "start_time")),
                    end_seconds=_float_or_none(_first(row, "end", "end_time")),
                )
            )
    return samples


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_clevrer(items: Sequence[Mapping[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    for item_index, item in enumerate(items):
        video = _first(
            item,
            "video",
            "video_path",
            "path",
        )
        scene_index = _first(
            item,
            "scene_index",
            "video_index",
            "id",
            default=video if video is not None else item_index,
        )
        if video is None:
            video = f"video_{int(scene_index):05d}.mp4"
        questions = item.get("questions")
        flat_item = not isinstance(questions, list)
        if flat_item:
            questions = [item]
        for question_index, row in enumerate(questions):
            if not isinstance(row, Mapping):
                continue
            subset = _token(_first(row, "question_type", "type", "category"))
            if subset not in TARGET_COUNTS["clevrer"]:
                continue
            question = str(_first(row, "question", "prompt", default="")).strip()
            raw_choices = _first(row, "choices", "options", default=[])
            if isinstance(raw_choices, Mapping):
                choice_values = _as_options(
                    _first(raw_choices, "choice", "text", "option", default=[])
                )
                answer_values = _first(
                    raw_choices, "answer", "label", "correct", default=[]
                )
                tolist = getattr(answer_values, "tolist", None)
                answer_list = tolist() if callable(tolist) else answer_values
                if not isinstance(answer_list, (list, tuple)):
                    answer_list = [answer_list] * len(choice_values)
                raw_choices = [
                    {"choice": choice, "answer": answer}
                    for choice, answer in zip(choice_values, answer_list)
                ]
            options: list[str] = []
            correct: list[str] = []
            if isinstance(raw_choices, list):
                for choice_index, choice in enumerate(raw_choices):
                    if isinstance(choice, Mapping):
                        option = _first(
                            choice, "choice", "text", "option", default=""
                        )
                        options.append(str(option).strip())
                        flag = _token(_first(choice, "answer", "label", "correct", default=""))
                        if flag in {"correct", "true", "1", "yes"}:
                            correct.append(chr(ord("A") + choice_index))
                    else:
                        options.append(str(choice).strip())
            if not correct:
                answer = _first(row, "answer", "gt_answer", "ground_truth")
            else:
                answer = correct
            raw_question_id = _first(row, "question_id", "id")
            source_id = (
                f"{video}:{raw_question_id}"
                if raw_question_id is not None and flat_item
                else str(raw_question_id or f"{scene_index}:{question_index}")
            )
            samples.append(
                Sample(
                    "clevrer",
                    subset,
                    str(
                        _first(
                            item,
                            "split",
                            default="validation"
                            if "validation" in str(video).lower()
                            else "test",
                        )
                    ),
                    source_id,
                    question,
                    options,
                    answer,
                    [video],
                    media_kind="video",
                    group_id=str(scene_index),
                    multi_select=True,
                    metadata={"scene_index": scene_index},
                )
            )
    return samples


def normalize_blink(
    rows_by_subset: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[Sample]:
    samples: list[Sample] = []
    for subset, rows in rows_by_subset.items():
        for index, row in enumerate(rows):
            question, options = _question_and_options(row)
            answer = _first(row, "answer", "gt_answer", "ground_truth")
            if _token(answer) == "hidden":
                raise ValueError(
                    f"BLINK {subset} has hidden test labels; use the labeled validation split"
                )
            source_id = _id(row, index)
            samples.append(
                Sample(
                    "blink",
                    subset,
                    "validation" if _token(source_id).startswith("val_") else "test",
                    source_id,
                    question,
                    options,
                    answer,
                    _media_from_row(row),
                    group_id=source_id,
                )
            )
    return samples


def normalize_vsr(rows: Sequence[Mapping[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        label = _first(row, "label", "answer", "ground_truth")
        if isinstance(label, str):
            answer = "B" if _token(label) in {"true", "1", "yes"} else "A"
        else:
            answer = "B" if bool(label) else "A"
        image_value = _first(row, "image", "image_path")
        if isinstance(image_value, str) and row.get("image_link"):
            image_value = {"path": image_value, "fallback_url": row["image_link"]}
        if image_value is None:
            image_value = row.get("image_link")
        samples.append(
            Sample(
                "vsr",
                "zero_shot_test",
                "test",
                _id(row, index),
                "Is this statement true for the image?\n"
                + str(_first(row, "caption", "question", default="")).strip(),
                ["False", "True"],
                answer,
                [image_value] if image_value is not None else [],
                group_id=str(_first(row, "image_link", "image_id", default=_id(row, index))),
                metadata={"relation": row.get("relation")},
            )
        )
    return samples


def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for key in ("questions", "data", "items", "samples", "annotations"):
                if isinstance(value.get(key), list):
                    return [row for row in value[key] if isinstance(row, dict)]
            return [value]
        raise ValueError(f"unsupported JSON structure: {path}")
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("reading local parquet requires pandas and pyarrow") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported annotation file: {path}")


def _find_annotation(root: Path, preferred: Sequence[str]) -> Path:
    if root.is_file():
        return root
    for name in preferred:
        matches = list(root.rglob(name))
        if matches:
            return sorted(matches)[0]
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".parquet"}
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"could not choose an annotation file under {root}; expected one of {list(preferred)}"
    )


def _find_clevrer_annotations(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".parquet"}
        and (
            path.suffix.lower() == ".parquet"
            or any(term in path.name.lower() for term in ("question", "annotation"))
        )
    )
    validation = [
        path
        for path in candidates
        if re.search(r"(?:^|[_-])(?:val|validation)(?:[_-]|$)", path.stem.lower())
    ]
    if validation:
        return validation
    if len(candidates) == 1:
        return candidates
    if not candidates:
        return [_find_annotation(root, ("*.json", "*.parquet"))]
    raise ValueError(
        "multiple CLEVRER annotation files found but no validation split could be "
        "identified; point --clevrer-root at the validation annotation tree"
    )


def _load_hf(
    dataset_id: str,
    *,
    revision: str,
    cache_dir: Path,
    config: str | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face loading requires the 'datasets' package") from exc
    kwargs: dict[str, Any] = {
        "revision": revision,
        "cache_dir": str(cache_dir),
    }
    try:
        loaded = load_dataset(dataset_id, config, split=split, **kwargs)
    except TypeError:
        loaded = load_dataset(dataset_id, config, split=split, trust_remote_code=True, **kwargs)
    if isinstance(loaded, DatasetDict) or isinstance(loaded, Mapping):
        for preferred in (split, "test", "validation", "train"):
            if preferred and preferred in loaded:
                loaded = loaded[preferred]
                break
        else:
            loaded = next(iter(loaded.values()))
    return [dict(row) for row in loaded]


def _snapshot(
    dataset_id: str,
    revision: str,
    cache_dir: Path,
    *,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("dataset snapshots require huggingface_hub") from exc
    return Path(
        snapshot_download(
            dataset_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    )


def _load_vsr_source(
    cache_dir: Path, revision: str
) -> tuple[list[dict[str, Any]], Path]:
    root = cache_dir / "vsr_source" / revision
    path = root / "test.jsonl"
    if not path.is_file():
        root.mkdir(parents=True, exist_ok=True)
        url = (
            "https://raw.githubusercontent.com/cambridgeltl/"
            "visual-spatial-reasoning/"
            f"{revision}/data/splits/zeroshot/test.jsonl"
        )
        urllib.request.urlretrieve(url, path)
    return _rows_from_file(path), root


def _extract_archives(root: Path, destination: Path, names: Sequence[str] | None = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archives = sorted(root.rglob("*.zip"))
    if names:
        wanted = {name.lower() for name in names}
        archives = [path for path in archives if path.name.lower() in wanted]
    for archive in archives:
        marker = destination / f".{archive.stem}.complete"
        if marker.is_file():
            continue
        LOGGER.info("Extracting %s", archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
        marker.write_text(sha256_file(archive) + "\n", encoding="ascii")
    return destination


def load_benchmark_samples(name: str, config: PrepareConfig) -> tuple[list[Sample], Path | None]:
    root = config.roots.get(name)
    revision = config.revisions.get(
        name, config.revisions.get(f"{name}_code", "")
    )
    if root is not None:
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"--{name}-root does not exist: {root}")
    elif not config.download_missing:
        raise ValueError(
            f"--no-download-missing requires --{name}-root for {name}"
        )

    if name == "mmsi":
        if root:
            rows = _rows_from_file(
                _find_annotation(root, ("MMSI_Bench.parquet", "*.jsonl"))
            )
        else:
            rows = _load_hf(
                DATASET_IDS[name],
                revision=revision,
                cache_dir=config.cache_dir,
            )
        return normalize_mmsi(rows), root.parent if root and root.is_file() else root
    if name == "spar":
        if root:
            rows = _rows_from_file(
                _find_annotation(
                    root, ("test-00000-of-00001.parquet", "*.jsonl")
                )
            )
        else:
            rows = _load_hf(
                DATASET_IDS[name],
                revision=revision,
                cache_dir=config.cache_dir,
                split="test",
            )
        return normalize_spar(rows), root.parent if root and root.is_file() else root
    if name == "mindcube":
        if root is None:
            snapshot = _snapshot(
                DATASET_IDS[name],
                revision,
                config.cache_dir,
                allow_patterns=("data.zip",),
            )
            root = _extract_archives(snapshot, config.cache_dir / "extracted" / "mindcube")
        elif root.is_file() and root.suffix.lower() == ".zip":
            root = _extract_archives(
                root.parent,
                config.cache_dir / "extracted" / "mindcube",
                (root.name,),
            )
        rows = _rows_from_file(
            _find_annotation(root, ("MindCube_tinybench.jsonl", "MindCube.jsonl"))
        )
        return normalize_mindcube(rows), root
    if name == "vsi":
        if root:
            rows = _rows_from_file(_find_annotation(root, ("test_debiased.parquet",)))
            media_root = root.parent if root.is_file() else root
            if (
                not config.dry_run
                and media_root.is_dir()
                and not any(
                    path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                    for path in media_root.rglob("*")
                )
                and any(media_root.rglob("*.zip"))
            ):
                media_root = _extract_archives(
                    media_root, config.cache_dir / "extracted" / "vsi"
                )
        else:
            rows = _load_hf(
                DATASET_IDS[name],
                config="debiased",
                split="test",
                revision=revision,
                cache_dir=config.cache_dir,
            )
            if config.dry_run:
                media_root = None
            else:
                snapshot = _snapshot(
                    DATASET_IDS[name],
                    revision,
                    config.cache_dir,
                    allow_patterns=("arkitscenes.zip", "scannet.zip", "scannetpp.zip"),
                )
                media_root = _extract_archives(
                    snapshot, config.cache_dir / "extracted" / "vsi"
                )
        return normalize_vsi(rows), media_root
    if name == "mvbench":
        if root is None:
            patterns = [
                "json/object_shuffle.json",
                "json/moving_direction.json",
                "json/egocentric_navigation.json",
            ]
            if not config.dry_run:
                patterns.extend(
                    ("video/perception.zip", "video/clevrer.zip", "video/vlnqa.zip")
                )
            root = _snapshot(
                DATASET_IDS[name],
                revision,
                config.cache_dir,
                allow_patterns=patterns,
            )
        rows_by_subset = {
            subset: _rows_from_file(_find_annotation(root, (f"{subset}.json",)))
            for subset in TARGET_COUNTS[name]
        }
        if config.dry_run:
            media_root = root
        elif any(
            path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            for path in root.rglob("*")
        ):
            media_root = root
        else:
            media_root = _extract_archives(
                root,
                config.cache_dir / "extracted" / "mvbench",
                ("perception.zip", "clevrer.zip", "vlnqa.zip"),
            )
        return normalize_mvbench(rows_by_subset), media_root
    if name == "clevrer":
        if root is None:
            raise ValueError(
                "CLEVRER requires --clevrer-root pointing to official videos "
                "and annotations"
            )
        annotation_files = _find_clevrer_annotations(root)
        items = [row for path in annotation_files for row in _rows_from_file(path)]
        media_root = root.parent if root.is_file() else root
        samples = normalize_clevrer(items)
        has_local_videos = media_root.is_dir() and any(
            path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            for path in media_root.rglob("*")
        )
        if has_local_videos:
            available: list[Sample] = []
            for sample in samples:
                try:
                    _resolve_string_media(
                        str(sample.media_values[0]), media_root, download_missing=False
                    )
                except FileNotFoundError:
                    continue
                available.append(sample)
            if len(available) != len(samples):
                LOGGER.warning(
                    "CLEVRER media root covers %d/%d labeled questions; "
                    "sampling only questions with verified local videos",
                    len(available),
                    len(samples),
                )
            samples = available
        return samples, media_root
    if name == "blink":
        rows_by_subset = {}
        for subset in TARGET_COUNTS[name]:
            if root:
                search_root = root / subset if (root / subset).exists() else root
                path = _find_annotation(
                    search_root,
                    (
                        "val-00000-of-00001.parquet",
                        "validation-00000-of-00001.parquet",
                        f"{subset}.jsonl",
                    ),
                )
                rows_by_subset[subset] = _rows_from_file(path)
            else:
                rows_by_subset[subset] = _load_hf(
                    DATASET_IDS[name],
                    config=subset,
                    split="validation",
                    revision=revision,
                    cache_dir=config.cache_dir,
                )
        return normalize_blink(rows_by_subset), root.parent if root and root.is_file() else root
    if name == "vsr":
        if root:
            rows = _rows_from_file(_find_annotation(root, ("test.jsonl",)))
            media_root = root.parent if root.is_file() else root
        else:
            rows, media_root = _load_vsr_source(
                config.cache_dir, config.revisions["vsr_source"]
            )
        return normalize_vsr(rows), media_root
    raise ValueError(f"unsupported benchmark: {name}")


def select_samples(
    samples: Sequence[Sample],
    benchmark: str,
    *,
    seed: int,
    smoke_per_subset: int = 0,
) -> list[Sample]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.subset].append(sample)
    selected: list[Sample] = []
    for subset, expected in TARGET_COUNTS[benchmark].items():
        candidates = grouped.get(subset, [])
        target = min(smoke_per_subset, expected) if smoke_per_subset else expected
        if len(candidates) < target:
            raise ValueError(
                f"{benchmark}/{subset}: found {len(candidates)} candidates, "
                f"expected at least {target}"
            )
        fixed_size_benchmarks = {
            "mmsi",
            "spar",
            "mindcube",
            "mvbench",
            "blink",
            "vsr",
        }
        if (
            not smoke_per_subset
            and benchmark in fixed_size_benchmarks
            and len(candidates) != expected
        ):
            raise ValueError(
                f"{benchmark}/{subset}: found {len(candidates)} rows, "
                f"expected exactly {expected}; "
                "check the pinned revision and category mapping"
            )
        ranked = sorted(
            candidates,
            key=lambda sample: stable_rank(
                benchmark, subset, sample.source_id, seed=seed
            ),
        )
        unique_groups: list[Sample] = []
        repeated_groups: list[Sample] = []
        seen_groups: set[str] = set()
        for sample in ranked:
            group = sample.group_id or sample.source_id
            if group in seen_groups:
                repeated_groups.append(sample)
            else:
                seen_groups.add(group)
                unique_groups.append(sample)
        chosen = (unique_groups + repeated_groups)[:target]
        if len(chosen) != target:
            raise AssertionError(f"selection bug for {benchmark}/{subset}")
        selected.extend(chosen)
    return selected


_PATH_INDEX: dict[Path, dict[str, list[Path]]] = {}


def _path_index(root: Path) -> dict[str, list[Path]]:
    root = root.resolve()
    if root not in _PATH_INDEX:
        index: dict[str, list[Path]] = defaultdict(list)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
                index[path.name].append(path)
                index[path.as_posix().lower()].append(path)
        _PATH_INDEX[root] = index
    return _PATH_INDEX[root]


def _resolve_string_media(value: str, root: Path | None, download_missing: bool) -> Path:
    if value.startswith(("http://", "https://")):
        if not download_missing:
            raise FileNotFoundError(f"remote media disabled in offline mode: {value}")
        destination_root = root or Path.cwd()
        destination = destination_root / ".classic_eval_downloads" / stable_rank(value)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            urllib.request.urlretrieve(value, destination)
        return destination
    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    if root is None:
        raise FileNotFoundError(value)
    direct = root / candidate
    if direct.is_file():
        return direct
    normalized = value.replace("\\", "/").lower()
    index = _path_index(root)
    matches = index.get(normalized, []) or index.get(candidate.name, [])
    unique = sorted({path.resolve() for path in matches})
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(f"media {value!r} not found under {root}")
    exact_suffix = [path for path in unique if path.as_posix().lower().endswith(normalized)]
    if len(exact_suffix) == 1:
        return exact_suffix[0]
    raise RuntimeError(f"media {value!r} is ambiguous under {root}: {unique[:5]}")


def _write_image_value(
    value: Any,
    destination: Path,
    root: Path | None,
    download_missing: bool,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("materializing benchmark images requires Pillow") from exc

    fallback_url = None
    if isinstance(value, Mapping):
        fallback_url = value.get("fallback_url")
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path") is not None:
            value = value["path"]
        elif fallback_url is not None:
            value = fallback_url
    if isinstance(value, (str, Path)):
        try:
            source = _resolve_string_media(str(value), root, download_missing)
        except FileNotFoundError:
            if fallback_url is None or str(value) == str(fallback_url):
                raise
            source = _resolve_string_media(
                str(fallback_url), root, download_missing
            )
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        return
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        return
    if isinstance(value, Image.Image):
        value.convert("RGB").save(destination, format="JPEG", quality=95)
        return
    try:
        image = Image.fromarray(value)
    except Exception as exc:
        raise TypeError(f"unsupported image value: {type(value).__name__}") from exc
    image.convert("RGB").save(destination, format="JPEG", quality=95)


def _uniform_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 0 or end < start:
        raise ValueError(f"invalid frame range/count: {start}, {end}, {count}")
    if count == 1:
        return [(start + end) // 2]
    return [round(start + index * (end - start) / (count - 1)) for index in range(count)]


def _extract_video_frames(
    source: Path,
    destination: Path,
    *,
    frame_count: int,
    start_seconds: float | None,
    end_seconds: float | None,
) -> tuple[list[Path], list[int]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video frame extraction requires opencv-python-headless") from exc
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
        if total <= 0:
            raise RuntimeError(f"video reports no frames: {source}")
        start = max(0, round((start_seconds or 0.0) * fps))
        end = total - 1
        if end_seconds is not None:
            end = min(end, max(start, round(end_seconds * fps) - 1))
        indices = _uniform_indices(start, end, frame_count)
        destination.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for order, frame_index in enumerate(indices):
            target = destination / f"frame_{order:03d}_{frame_index:06d}.jpg"
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed reading frame {frame_index} from {source}")
            if not cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"failed writing extracted frame: {target}")
            paths.append(target)
        return paths, indices
    finally:
        capture.release()


def materialize_sample(
    sample: Sample,
    *,
    source_root: Path | None,
    output_dir: Path,
    download_missing: bool,
    calculate_hashes: bool = True,
) -> dict[str, Any]:
    if not sample.question or (
        sample.answer_type == "choice" and len(sample.options) < 2
    ):
        raise ValueError(f"{sample.sample_id}: missing question or options")
    if sample.answer_type == "exact_text":
        gold: tuple[str, ...] = ()
        gold_text = str(sample.answer).strip()
        if not gold_text:
            raise ValueError(f"{sample.sample_id}: empty exact-text answer")
    else:
        gold = answer_to_letters(
            sample.answer,
            sample.options,
            multi_select=sample.multi_select,
        )
        gold_text = None
    sample_dir = output_dir / "media" / sample.benchmark / stable_rank(sample.sample_id)[:16]
    media_paths: list[Path] = []
    frame_indices: list[int] = []
    if sample.media_kind == "video":
        if len(sample.media_values) != 1:
            raise ValueError(f"{sample.sample_id}: video sample must have exactly one source")
        source = _resolve_string_media(
            str(sample.media_values[0]), source_root, download_missing
        )
        media_paths, frame_indices = _extract_video_frames(
            source,
            sample_dir,
            frame_count=VIDEO_FRAME_COUNTS[sample.benchmark],
            start_seconds=sample.start_seconds,
            end_seconds=sample.end_seconds,
        )
    else:
        if not sample.media_values:
            raise ValueError(f"{sample.sample_id}: no image media")
        sample_dir.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(sample.media_values):
            target = sample_dir / f"image_{index:03d}.jpg"
            _write_image_value(value, target, source_root, download_missing)
            media_paths.append(target)

    media = []
    for order, path in enumerate(media_paths):
        relative = path.relative_to(output_dir).as_posix()
        item = {"kind": "image", "order": order, "path": relative}
        if calculate_hashes:
            item["sha256"] = sha256_file(path)
        media.append(item)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": sample.benchmark,
        "subset": sample.subset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "source_id": sample.source_id,
        "group_id": sample.group_id or sample.source_id,
        "media": media,
        "question": sample.question,
        "options": sample.options,
        "gold": list(gold),
        "gold_text": gold_text,
        "answer_type": sample.answer_type,
        "multi_select": sample.multi_select,
        "frame_indices": frame_indices,
        "source_metadata": sample.metadata,
    }


def _counts(samples: Iterable[Sample]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        counters[sample.benchmark][sample.subset] += 1
    return {benchmark: dict(counter) for benchmark, counter in counters.items()}


def prepare(config: PrepareConfig) -> dict[str, Any]:
    all_selected: list[Sample] = []
    roots: dict[str, Path | None] = {}
    available_counts: dict[str, dict[str, int]] = {}
    for benchmark in config.benchmarks:
        LOGGER.info("Loading %s", benchmark)
        samples, media_root = load_benchmark_samples(benchmark, config)
        roots[benchmark] = media_root
        available_counts[benchmark] = _counts(samples).get(benchmark, {})
        all_selected.extend(
            select_samples(
                samples,
                benchmark,
                seed=config.seed,
                smoke_per_subset=config.smoke_per_subset,
            )
        )

    selected_counts = _counts(all_selected)
    preview = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "benchmarks": list(config.benchmarks),
        "seed": config.seed,
        "dry_run": config.dry_run,
        "smoke_per_subset": config.smoke_per_subset,
        "available_counts": available_counts,
        "selected_counts": selected_counts,
        "selected_total": len(all_selected),
    }
    if config.dry_run:
        return preview

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(all_selected, start=1):
        LOGGER.info("Materializing %d/%d %s", index, len(all_selected), sample.sample_id)
        rows.append(
            materialize_sample(
                sample,
                source_root=roots[sample.benchmark],
                output_dir=config.output_dir,
                download_missing=config.download_missing,
                calculate_hashes=config.validate_media,
            )
        )
    validate_manifest(rows)
    manifest_path = config.output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, rows)
    manifest_hash = sha256_file(manifest_path)

    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_hash,
        "seed": config.seed,
        "dataset_ids": {name: DATASET_IDS.get(name) for name in config.benchmarks},
        "revisions": dict(config.revisions),
        "source_roots": {
            name: str(root) if root is not None else None for name, root in roots.items()
        },
        "frame_counts": VIDEO_FRAME_COUNTS,
        "validate_media": config.validate_media,
        "selected_counts": selected_counts,
        "selected_total": len(rows),
    }
    write_json(config.output_dir / "benchmark.lock.json", lock)

    repeated_groups: dict[str, dict[str, int]] = defaultdict(dict)
    for benchmark in config.benchmarks:
        for subset in TARGET_COUNTS[benchmark]:
            group_counts = Counter(
                row["group_id"]
                for row in rows
                if row["benchmark"] == benchmark and row["subset"] == subset
            )
            repeated_groups[benchmark][subset] = sum(
                count - 1 for count in group_counts.values() if count > 1
            )
    summary = {
        **preview,
        "dry_run": False,
        "manifest_sha256": manifest_hash,
        "repeated_group_rows": dict(repeated_groups),
        "expected_full_total": EXPECTED_TOTAL,
        "media_file_count": sum(len(row["media"]) for row in rows),
    }
    write_json(config.output_dir / "manifest.summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARK_ORDER,
        default=list(BENCHMARK_ORDER),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-per-subset", type=int, default=0)
    parser.add_argument("--download-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-media", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    for benchmark in BENCHMARK_ORDER:
        parser.add_argument(f"--{benchmark}-root", type=Path)
        if benchmark != "clevrer":
            parser.add_argument(
                f"--{benchmark}-revision", default=DEFAULT_REVISIONS[benchmark]
            )
    parser.add_argument(
        "--clevrer-code-revision", default=DEFAULT_REVISIONS["clevrer_code"]
    )
    parser.add_argument(
        "--mindcube-code-revision", default=DEFAULT_REVISIONS["mindcube_code"]
    )
    parser.add_argument(
        "--blink-code-revision", default=DEFAULT_REVISIONS["blink_code"]
    )
    parser.add_argument(
        "--vsr-source-revision", default=DEFAULT_REVISIONS["vsr_source"]
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PrepareConfig:
    roots = {name: getattr(args, f"{name}_root") for name in BENCHMARK_ORDER}
    revisions = {
        name: getattr(args, f"{name}_revision")
        for name in BENCHMARK_ORDER
        if name != "clevrer"
    }
    revisions.update(
        {
            "clevrer_code": args.clevrer_code_revision,
            "mindcube_code": args.mindcube_code_revision,
            "blink_code": args.blink_code_revision,
            "vsr_source": args.vsr_source_revision,
        }
    )
    return PrepareConfig(
        output_dir=args.output_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        roots=roots,
        revisions=revisions,
        benchmarks=tuple(args.benchmarks),
        seed=args.seed,
        download_missing=args.download_missing,
        validate_media=args.validate_media,
        smoke_per_subset=args.smoke_per_subset,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.smoke_per_subset < 0:
        raise ValueError("--smoke-per-subset must be non-negative")
    summary = prepare(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
