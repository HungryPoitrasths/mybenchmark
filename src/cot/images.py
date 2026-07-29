from __future__ import annotations

from pathlib import Path
from typing import Any


def collect_image_names(question: dict[str, Any]) -> list[str]:
    primary = str(question.get("image_name") or "").strip()
    names = [primary] if primary else []
    raw_auxiliary = question.get("auxiliary_image_names")
    if isinstance(raw_auxiliary, list):
        names.extend(str(value).strip() for value in raw_auxiliary if str(value).strip())
    else:
        fallback = str(question.get("aux_image_name") or question.get("image_name_2") or "").strip()
        if fallback:
            names.append(fallback)
    destination = str(question.get("reasoning_frame_2") or "").strip()
    if destination and destination not in names:
        names.append(destination)
    deduplicated: list[str] = []
    for name in names:
        if name not in deduplicated:
            deduplicated.append(name)
    return deduplicated


def _is_scannetpp(question: dict[str, Any]) -> bool:
    explicit = str(question.get("dataset_source") or question.get("source_dataset") or "").lower()
    if "scannet++" in explicit or "scannetpp" in explicit:
        return True
    if explicit == "scannet":
        return False
    scene_id = str(question.get("scene_id") or "")
    return not scene_id.startswith("scene")


def _candidate_paths(
    question: dict[str, Any],
    image_name: str,
    *,
    benchmark_path: Path | None,
    scannet_roots: list[Path],
    scannetpp_roots: list[Path],
    scannetpp_sensor: str,
) -> list[Path]:
    scene_id = str(question.get("scene_id") or "")
    candidates: list[Path] = []
    if image_name == str(question.get("image_name") or ""):
        explicit = str(question.get("image_path") or "").strip()
        if explicit:
            candidates.append(Path(explicit))
    source = str(question.get("_source_benchmark") or "").strip()
    source_paths = [benchmark_path, Path(source) if source else None]
    for source_path in source_paths:
        if source_path is None:
            continue
        parent = source_path.resolve().parent
        candidates.extend(
            [
                parent / "images" / image_name,
                parent / "images" / scene_id / image_name,
                parent / scene_id / image_name,
            ]
        )

    if _is_scannetpp(question):
        if scannetpp_sensor not in {"iphone", "dslr"}:
            raise ValueError("scannetpp_sensor must be 'iphone' or 'dslr'")
        for root in scannetpp_roots:
            preferred = (
                root / scene_id / "iphone" / "rgb" / image_name
                if scannetpp_sensor == "iphone"
                else root / scene_id / "dslr" / "resized_images" / image_name
            )
            candidates.extend(
                [
                    preferred,
                    root / scene_id / image_name,
                    root / scene_id / "iphone" / "rgb" / image_name,
                    root / scene_id / "dslr" / "resized_images" / image_name,
                    root / "scans" / scene_id / image_name,
                    root / "scans" / scene_id / "iphone" / "rgb" / image_name,
                    root / "scans" / scene_id / "dslr" / "resized_images" / image_name,
                ]
            )
    else:
        for root in scannet_roots:
            candidates.extend(
                [
                    root / scene_id / "color" / image_name,
                    root / scene_id / image_name,
                    root / "scans" / scene_id / "color" / image_name,
                ]
            )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate.resolve())
    return unique


def resolve_image_paths(
    question: dict[str, Any],
    *,
    benchmark_path: Path | None = None,
    scannet_roots: list[Path] | None = None,
    scannetpp_roots: list[Path] | None = None,
    scannetpp_sensor: str = "iphone",
    require_exists: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    scannet_roots = scannet_roots or []
    scannetpp_roots = scannetpp_roots or []
    resolved: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for image_name in collect_image_names(question):
        candidates = _candidate_paths(
            question,
            image_name,
            benchmark_path=benchmark_path,
            scannet_roots=scannet_roots,
            scannetpp_roots=scannetpp_roots,
            scannetpp_sensor=scannetpp_sensor,
        )
        existing = next((candidate for candidate in candidates if candidate.is_file()), None)
        selected = existing or (candidates[0] if candidates else Path(image_name).resolve())
        diagnostics.append(
            {
                "image_name": image_name,
                "resolved": str(existing) if existing else None,
                "checked": [str(path) for path in candidates[:12]],
            }
        )
        if require_exists and existing is None:
            raise FileNotFoundError(f"image not found: {image_name}")
        resolved.append(str(selected))
    if not resolved:
        raise FileNotFoundError("question has no image_name")
    return resolved, diagnostics
