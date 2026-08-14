"""Run frozen classic-spatial manifests with local Qwen3-VL checkpoints."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
import math
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .common import (
    RESULT_SCHEMA_VERSION,
    append_jsonl,
    canonical_exact_text,
    iter_jsonl,
    manifest_sha256,
    parse_answer,
    parse_exact_answer,
    resolve_media_path,
    sha256_file,
    validate_manifest,
)


LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = "You are a visual spatial-reasoning assistant."


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: str


class Backend(Protocol):
    def generate(self, rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> list[str]: ...

    def close(self) -> None: ...


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not label:
        raise ValueError(f"invalid empty model label from {value!r}")
    return label


def parse_model_specs(values: Sequence[str], base_model: str) -> list[ModelSpec]:
    if not values:
        return [ModelSpec("base", base_model)]
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-spec must be LABEL=PATH, got {value!r}")
        raw_label, raw_path = value.split("=", 1)
        label = safe_label(raw_label)
        path = base_model if raw_path.strip().lower() in {"base", "__base__"} else raw_path.strip()
        if not path:
            raise ValueError(f"model path is empty in {value!r}")
        if label in seen:
            raise ValueError(f"duplicate model label: {label}")
        seen.add(label)
        specs.append(ModelSpec(label, path))
    return specs


def build_prompt(row: Mapping[str, Any]) -> str:
    if row.get("answer_type") == "exact_text":
        return "\n".join(
            [
                str(row["question"]).strip(),
                "",
                "On the final line, output only the answer in the requested format.",
            ]
        )
    options = row["options"]
    option_lines = [f"{chr(ord('A') + index)}) {option}" for index, option in enumerate(options)]
    if row.get("multi_select"):
        suffix = (
            "Select every correct option. On the final line, output only the uppercase "
            "option letters in option order separated by single spaces."
        )
    else:
        suffix = "On the final line, output only the uppercase letter of the correct option."
    return "\n".join([str(row["question"]).strip(), "", *option_lines, "", suffix])


def _preflight_media(
    rows: Sequence[Mapping[str, Any]], manifest_path: Path, *, verify_hashes: bool
) -> None:
    errors: list[str] = []
    for row in rows:
        for media in row["media"]:
            path = resolve_media_path(str(media["path"]), manifest_path)
            if not path.is_file():
                errors.append(f"{row['sample_id']}: missing {path}")
                continue
            expected = media.get("sha256")
            if verify_hashes and expected and sha256_file(path) != expected:
                errors.append(f"{row['sample_id']}: checksum mismatch for {path}")
    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"media preflight failed ({len(errors)} errors):\n{preview}")


def _adapter_base(path: str) -> str | None:
    adapter_path = Path(path)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    value = config.get("base_model_name_or_path")
    return str(value) if value else None


class TransformersBackend:
    def __init__(
        self,
        *,
        model_path: str,
        base_model: str,
        dtype: str,
        attn_implementation: str,
        max_pixels: int,
        max_new_tokens: int,
        batch_size: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers backend requires torch, transformers, peft, and qwen-vl-utils"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        adapter_base = _adapter_base(model_path)
        model_source = base_model if adapter_base is not None else model_path
        if adapter_base and adapter_base != base_model:
            LOGGER.warning(
                "Adapter declares base %s but --base-model is %s; using --base-model",
                adapter_base,
                base_model,
            )
        processor_source = model_path
        if not (Path(model_path) / "processor_config.json").is_file():
            processor_source = model_source
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            max_pixels=max_pixels,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        model_class = self._model_class()
        self.model = model_class.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        if adapter_base is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("loading a LoRA checkpoint requires peft") from exc
            self.model = PeftModel.from_pretrained(self.model, model_path, is_trainable=False)
        self.model.eval()

    @staticmethod
    def _model_class() -> Any:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText

                return AutoModelForImageTextToText
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration

                return Qwen2_5_VLForConditionalGeneration

    def _messages(self, row: Mapping[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for media in sorted(row["media"], key=lambda value: int(value["order"])):
            path = resolve_media_path(str(media["path"]), manifest_path)
            content.append({"type": "image", "image": str(path)})
        content.append({"type": "text", "text": build_prompt(row)})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def generate(self, rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> list[str]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("transformers backend requires qwen-vl-utils") from exc

        messages_batch = [self._messages(row, manifest_path) for row in rows]
        texts: list[str] = []
        images: list[Any] = []
        for messages in messages_batch:
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs:
                raise RuntimeError("manifest must contain pre-extracted image frames, not videos")
            texts.append(text)
            images.extend(image_inputs or [])
        inputs = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return [
            value.strip()
            for value in self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class OpenAIBackend:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        max_new_tokens: int,
        timeout: float,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai backend requires the openai package") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    def generate(self, rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> list[str]:
        outputs: list[str] = []
        for row in rows:
            content: list[dict[str, Any]] = []
            for media in sorted(row["media"], key=lambda value: int(value["order"])):
                path = resolve_media_path(str(media["path"]), manifest_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._data_url(path)},
                    }
                )
            content.append({"type": "text", "text": build_prompt(row)})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                top_p=1,
                max_tokens=self.max_new_tokens,
            )
            outputs.append(response.choices[0].message.content or "")
        return outputs

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()


def _completed_keys(path: Path, expected_manifest_hash: str) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str]] = set()
    for row in iter_jsonl(path):
        if row.get("manifest_sha256") != expected_manifest_hash:
            raise ValueError(f"resume file was produced from a different manifest: {path}")
        completed.add((str(row.get("model_label")), str(row.get("sample_id"))))
    return completed


def _result_row(
    row: Mapping[str, Any],
    *,
    model: ModelSpec,
    raw: str | None,
    error: str | None,
    manifest_hash: str,
    backend_name: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    answer_type = str(row.get("answer_type", "choice"))
    if answer_type == "exact_text":
        parsed_exact = parse_exact_answer(raw)
        predicted = []
        prediction_text = parsed_exact.text
        parse_status = parsed_exact.status
        parse_source = parsed_exact.source
        gold = []
        gold_text = str(row["gold_text"])
        correct = bool(
            prediction_text
            and parse_status == "ok"
            and canonical_exact_text(prediction_text)
            == canonical_exact_text(gold_text)
            and error is None
        )
    else:
        parsed = parse_answer(
            raw,
            [str(value) for value in row["options"]],
            multi_select=bool(row.get("multi_select")),
        )
        predicted = list(parsed.letters)
        prediction_text = parsed.text
        parse_status = parsed.status
        parse_source = parsed.source
        gold = [str(value) for value in row["gold"]]
        gold_text = None
        correct = bool(predicted and predicted == gold and error is None)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "model_label": model.label,
        "model_path": model.path,
        "backend": backend_name,
        "benchmark": row["benchmark"],
        "subset": row["subset"],
        "sample_id": row["sample_id"],
        "source_id": row["source_id"],
        "group_id": row.get("group_id", row["source_id"]),
        "raw_response": raw,
        "prediction": predicted,
        "prediction_text": prediction_text,
        "parse_status": parse_status,
        "parse_source": parse_source,
        "gold": gold,
        "gold_text": gold_text,
        "answer_type": answer_type,
        "correct": correct,
        "option_count": len(row["options"]),
        "multi_select": bool(row.get("multi_select")),
        "error": error,
        "elapsed_seconds": elapsed_seconds,
    }


def _chunks(rows: Sequence[Mapping[str, Any]], size: int) -> list[Sequence[Mapping[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def run_model(
    *,
    model: ModelSpec,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest_hash: str,
    output_path: Path,
    backend: Backend,
    backend_name: str,
    batch_size: int,
    resume: bool,
) -> dict[str, int]:
    completed = _completed_keys(output_path, manifest_hash) if resume else set()
    pending = [
        row
        for row in rows
        if (model.label, str(row["sample_id"])) not in completed
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.is_file() else "w"
    counts = {
        "selected": len(rows),
        "resumed": len(rows) - len(pending),
        "written": 0,
        "errors": 0,
    }
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for batch_index, batch in enumerate(_chunks(pending, batch_size), start=1):
            LOGGER.info(
                "%s batch %d/%d (%d samples)",
                model.label,
                batch_index,
                math.ceil(len(pending) / batch_size) if pending else 0,
                len(batch),
            )
            started = time.monotonic()
            try:
                outputs = backend.generate(batch, manifest_path)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"backend returned {len(outputs)} outputs for {len(batch)} inputs"
                    )
                elapsed = (time.monotonic() - started) / max(1, len(batch))
                result_rows = [
                    _result_row(
                        row,
                        model=model,
                        raw=raw,
                        error=None,
                        manifest_hash=manifest_hash,
                        backend_name=backend_name,
                        elapsed_seconds=elapsed,
                    )
                    for row, raw in zip(batch, outputs)
                ]
            except Exception as batch_error:
                LOGGER.exception("Batch failed; retrying samples individually")
                result_rows = []
                for row in batch:
                    item_started = time.monotonic()
                    try:
                        raw = backend.generate([row], manifest_path)[0]
                        error = None
                    except Exception as item_error:
                        LOGGER.exception("Inference failed for %s", row["sample_id"])
                        raw = None
                        error = f"{type(item_error).__name__}: {item_error}"
                    result_rows.append(
                        _result_row(
                            row,
                            model=model,
                            raw=raw,
                            error=error,
                            manifest_hash=manifest_hash,
                            backend_name=backend_name,
                            elapsed_seconds=time.monotonic() - item_started,
                        )
                    )
            for result in result_rows:
                append_jsonl(handle, result)
                counts["written"] += 1
                if result["error"]:
                    counts["errors"] += 1
    return counts


def _make_backend(args: argparse.Namespace, spec: ModelSpec) -> Backend:
    if args.backend == "transformers":
        return TransformersBackend(
            model_path=spec.path,
            base_model=args.base_model,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    return OpenAIBackend(
        model_name=spec.path,
        base_url=args.openai_base_url,
        api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for base/SFT/GRPO variants; use LABEL=base for the base model.",
    )
    parser.add_argument("--backend", choices=("transformers", "openai"), default="transformers")
    parser.add_argument("--devices", help="CUDA device list, for example 0,1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=786432)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-media-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--openai-api-key")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    manifest_path = args.manifest.resolve()
    rows = list(iter_jsonl(manifest_path))
    validate_manifest(rows)
    _preflight_media(rows, manifest_path, verify_hashes=args.verify_media_hashes)
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    specs = parse_model_specs(args.model_spec, args.base_model)
    manifest_hash = manifest_sha256(manifest_path)
    preview = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "models": [spec.__dict__ for spec in specs],
        "backend": args.backend,
        "selected_samples": len(rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    summaries: dict[str, Any] = {}
    for spec in specs:
        backend = _make_backend(args, spec)
        try:
            output_path = (
                args.output_dir.resolve()
                / spec.label
                / f"predictions.shard-{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
            )
            summaries[spec.label] = run_model(
                model=spec,
                rows=rows,
                manifest_path=manifest_path,
                manifest_hash=manifest_hash,
                output_path=output_path,
                backend=backend,
                backend_name=args.backend,
                batch_size=args.batch_size,
                resume=args.resume,
            )
        finally:
            backend.close()
    print(json.dumps({**preview, "runs": summaries}, ensure_ascii=False, indent=2))
    return 0
