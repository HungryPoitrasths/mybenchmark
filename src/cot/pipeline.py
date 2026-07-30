from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .facts import build_fact_record, question_uid
from .images import resolve_image_paths
from .models import FactExtractionError
from .render import render_response
from .templates import load_template_library
from .validators import (
    validate_answer_mapping,
    validate_fact_consistency,
    validate_reasoning_consistency,
    validate_response,
    validate_sft_item,
)


def format_user_prompt(question: dict[str, Any], image_count: int) -> str:
    image_prefix = "\n".join("<image>" for _ in range(image_count))
    options = question.get("options") or []
    option_text = "\n".join(
        f"{chr(ord('A') + index)}. {value}" for index, value in enumerate(options)
    )
    return f"{image_prefix}\n{str(question.get('question') or '').strip()}\nOptions:\n{option_text}"


def build_dataset(
    questions: Iterable[dict[str, Any]],
    *,
    benchmark_path: Path | None = None,
    seed: int = 42,
    template_path: Path | None = None,
    scannet_roots: list[Path] | None = None,
    scannetpp_roots: list[Path] | None = None,
    scannetpp_sensor: str = "iphone",
    require_images: bool = True,
) -> dict[str, Any]:
    library = load_template_library(template_path)
    sidecar: list[dict[str, Any]] = []
    sft: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    signatures: Counter[str] = Counter()
    types: Counter[str] = Counter()
    rejection_codes: Counter[str] = Counter()

    for index, question in enumerate(questions):
        uid = question_uid(question)
        try:
            record = build_fact_record(question)
            validate_fact_consistency(record)
            validate_answer_mapping(question, record)
            response, template_id = render_response(
                record,
                seed=seed,
                template_library=library,
            )
            validate_response(response, record)
            validate_reasoning_consistency(response, record)
            images, image_diagnostics = resolve_image_paths(
                question,
                benchmark_path=benchmark_path,
                scannet_roots=scannet_roots,
                scannetpp_roots=scannetpp_roots,
                scannetpp_sensor=scannetpp_sensor,
                require_exists=require_images,
            )
            item = {
                "messages": [
                    {"role": "user", "content": format_user_prompt(question, len(images))},
                    {"role": "assistant", "content": response},
                ],
                "images": images,
                "question_uid": record.question_uid,
                "question_type": record.question_type,
                "signature_id": record.signature_id,
                "template_id": template_id,
            }
            validate_sft_item(item)
            sidecar_item = record.to_dict()
            sidecar_item.update(
                template_id=template_id,
                reasoning=response.rsplit("\nAnswer:", 1)[0],
                response=response,
                option_count=len(question.get("options") or []),
                multi_select=bool(question.get("multi_select")),
                images=images,
                image_resolution=image_diagnostics,
                validation={
                    "passed": True,
                    "fact_source": "benchmark_oracle",
                    "fact_consistency": "passed",
                    "answer_mapping": "passed",
                    "response_format": "passed",
                    "reasoning_consistency": "passed",
                    "image_count": len(images),
                },
            )
            sidecar.append(sidecar_item)
            sft.append(item)
            signatures[record.signature_id] += 1
            types[record.question_type] += 1
        except (FactExtractionError, FileNotFoundError, TypeError, ValueError) as exc:
            code = exc.code if isinstance(exc, FactExtractionError) else "missing_image" if isinstance(exc, FileNotFoundError) else "invalid_record"
            rejection_codes[code] += 1
            rejected.append(
                {
                    "index": index,
                    "question_uid": uid,
                    "question_type": str(question.get("type") or ""),
                    "code": code,
                    "message": str(exc),
                }
            )

    total = len(sidecar) + len(rejected)
    return {
        "sidecar": sidecar,
        "sft": sft,
        "rejected": rejected,
        "report": {
            "input_count": total,
            "accepted_count": len(sidecar),
            "rejected_count": len(rejected),
            "acceptance_rate": round(len(sidecar) / total, 6) if total else 0.0,
            "by_type": dict(sorted(types.items())),
            "by_signature": dict(sorted(signatures.items())),
            "rejection_codes": dict(sorted(rejection_codes.items())),
            "seed": seed,
            "template_count_per_signature": 12,
        },
    }
