"""Summarize classic-spatial predictions with paired confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    RESULT_SCHEMA_VERSION,
    iter_jsonl,
    manifest_sha256,
    stable_rank,
    validate_manifest,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class ScoreRow:
    model: str
    benchmark: str
    subset: str
    sample_id: str
    group_id: str
    correct: bool
    prediction: tuple[str, ...]
    gold: tuple[str, ...]
    prediction_text: str | None
    gold_text: str | None
    answer_type: str
    option_count: int
    parse_status: str
    error: str | None


def _prediction_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("predictions*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    unique = sorted({path.resolve() for path in files})
    if not unique:
        raise ValueError("no prediction JSONL files found")
    return unique


def load_results(
    paths: Sequence[Path], *, expected_manifest_hash: str
) -> dict[str, dict[str, ScoreRow]]:
    by_model: dict[str, dict[str, ScoreRow]] = defaultdict(dict)
    for path in _prediction_files(paths):
        for raw in iter_jsonl(path):
            if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported result schema in {path}: "
                    f"{raw.get('schema_version')}"
                )
            if raw.get("manifest_sha256") != expected_manifest_hash:
                raise ValueError(f"manifest hash mismatch in {path}")
            model = str(raw["model_label"])
            sample_id = str(raw["sample_id"])
            row = ScoreRow(
                model=model,
                benchmark=str(raw["benchmark"]),
                subset=str(raw["subset"]),
                sample_id=sample_id,
                group_id=str(raw.get("group_id") or sample_id),
                correct=bool(raw.get("correct")),
                prediction=tuple(str(value) for value in raw.get("prediction") or []),
                gold=tuple(str(value) for value in raw.get("gold") or []),
                prediction_text=(
                    str(raw["prediction_text"])
                    if raw.get("prediction_text") is not None
                    else None
                ),
                gold_text=(
                    str(raw["gold_text"])
                    if raw.get("gold_text") is not None
                    else None
                ),
                answer_type=str(raw.get("answer_type") or "choice"),
                option_count=int(raw.get("option_count") or 0),
                parse_status=str(raw.get("parse_status") or "invalid"),
                error=str(raw["error"]) if raw.get("error") else None,
            )
            previous = by_model[model].get(sample_id)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate result for {model}/{sample_id}")
            by_model[model][sample_id] = row
    return dict(by_model)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_totals(rows: Sequence[ScoreRow]) -> list[tuple[int, int]]:
    grouped: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[f"{row.subset}|{row.group_id}"].append(row)
    return [
        (sum(1 for row in group if row.correct), len(group))
        for _, group in sorted(grouped.items())
    ]


def bootstrap_accuracy_ci(
    rows: Sequence[ScoreRow], *, iterations: int, seed: int
) -> tuple[float, float]:
    clusters = _cluster_totals(rows)
    if not clusters:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        correct = 0
        total = 0
        for _ in range(len(clusters)):
            cluster_correct, cluster_total = clusters[rng.randrange(len(clusters))]
            correct += cluster_correct
            total += cluster_total
        estimates.append(correct / total if total else 0.0)
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def paired_bootstrap_delta_ci(
    base_rows: Sequence[ScoreRow],
    candidate_rows: Sequence[ScoreRow],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    base = {row.sample_id: row for row in base_rows}
    candidate = {row.sample_id: row for row in candidate_rows}
    if set(base) != set(candidate):
        raise ValueError("paired bootstrap requires identical sample IDs")
    grouped: dict[str, list[tuple[ScoreRow, ScoreRow]]] = defaultdict(list)
    for sample_id in sorted(base):
        base_row = base[sample_id]
        candidate_row = candidate[sample_id]
        grouped[f"{base_row.subset}|{base_row.group_id}"].append((base_row, candidate_row))
    clusters = list(grouped.values())
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        base_correct = 0
        candidate_correct = 0
        total = 0
        for _ in range(len(clusters)):
            cluster = clusters[rng.randrange(len(clusters))]
            base_correct += sum(1 for base_row, _ in cluster if base_row.correct)
            candidate_correct += sum(1 for _, candidate_row in cluster if candidate_row.correct)
            total += len(cluster)
        estimates.append((candidate_correct - base_correct) / total if total else 0.0)
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _metric(rows: Sequence[ScoreRow], *, iterations: int, seed: int) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.correct)
    invalid = sum(1 for row in rows if row.parse_status != "ok")
    errors = sum(1 for row in rows if row.error)
    low, high = bootstrap_accuracy_ci(rows, iterations=iterations, seed=seed)
    metric = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "ci95": [low, high],
        "invalid": invalid,
        "errors": errors,
    }
    if rows and rows[0].benchmark == "clevrer":
        option_total = sum(row.option_count for row in rows)
        option_correct = 0
        for row in rows:
            predicted = set(row.prediction)
            gold = set(row.gold)
            allowed = [chr(ord("A") + index) for index in range(row.option_count)]
            option_correct += sum((letter in predicted) == (letter in gold) for letter in allowed)
        metric["per_option_total"] = option_total
        metric["per_option_correct"] = option_correct
        metric["per_option_accuracy"] = option_correct / option_total if option_total else None
    return metric


def _seed_for(seed: int, *parts: str) -> int:
    return int(stable_rank(*parts, seed=seed)[:16], 16)


def summarize(
    manifest_rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, ScoreRow]],
    *,
    base_label: str,
    iterations: int,
    seed: int,
    allow_incomplete: bool,
) -> dict[str, Any]:
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    if base_label not in results:
        raise ValueError(f"base model label {base_label!r} is absent from predictions")
    completeness: dict[str, Any] = {}
    for model, rows in results.items():
        result_ids = set(rows)
        missing = sorted(manifest_ids - result_ids)
        extra = sorted(result_ids - manifest_ids)
        completeness[model] = {
            "present": len(result_ids & manifest_ids),
            "missing": len(missing),
            "extra": len(extra),
            "missing_examples": missing[:20],
            "extra_examples": extra[:20],
        }
        if (missing or extra) and not allow_incomplete:
            raise ValueError(
                f"{model}: result coverage mismatch: {len(missing)} missing, {len(extra)} extra"
            )

    benchmark_names = []
    for row in manifest_rows:
        if row["benchmark"] not in benchmark_names:
            benchmark_names.append(str(row["benchmark"]))
    output: dict[str, Any] = {
        "base_label": base_label,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "completeness": completeness,
        "benchmarks": {},
    }
    for benchmark in benchmark_names:
        benchmark_output: dict[str, Any] = {"models": {}}
        subset_names = []
        for row in manifest_rows:
            if row["benchmark"] == benchmark and row["subset"] not in subset_names:
                subset_names.append(str(row["subset"]))
        for model, model_rows in results.items():
            rows = [row for row in model_rows.values() if row.benchmark == benchmark]
            if not rows:
                continue
            subset_metrics: dict[str, Any] = {}
            for subset in subset_names:
                subset_rows = [row for row in rows if row.subset == subset]
                subset_metrics[subset] = _metric(
                    subset_rows,
                    iterations=iterations,
                    seed=_seed_for(seed, benchmark, subset, model),
                )
            overall = _metric(
                rows,
                iterations=iterations,
                seed=_seed_for(seed, benchmark, model),
            )
            accuracies = [
                metric["accuracy"]
                for metric in subset_metrics.values()
                if metric["accuracy"] is not None
            ]
            overall["macro_accuracy"] = (
                sum(accuracies) / len(accuracies) if accuracies else None
            )
            benchmark_output["models"][model] = {
                "overall": overall,
                "subsets": subset_metrics,
            }

        base = benchmark_output["models"].get(base_label)
        if base:
            base_rows = [
                row
                for row in results[base_label].values()
                if row.benchmark == benchmark
            ]
            for model, metrics in benchmark_output["models"].items():
                if model == base_label:
                    metrics["overall"]["delta_vs_base"] = 0.0
                    metrics["overall"]["delta_ci95"] = [0.0, 0.0]
                    for subset in subset_names:
                        metrics["subsets"][subset]["delta_vs_base"] = 0.0
                        metrics["subsets"][subset]["delta_ci95"] = [0.0, 0.0]
                    continue
                candidate_rows = [
                    row
                    for row in results[model].values()
                    if row.benchmark == benchmark
                ]
                common = sorted(
                    set(row.sample_id for row in base_rows)
                    & set(row.sample_id for row in candidate_rows)
                )
                base_common = [results[base_label][sample_id] for sample_id in common]
                candidate_common = [results[model][sample_id] for sample_id in common]
                if not common:
                    continue
                delta = (
                    sum(row.correct for row in candidate_common)
                    - sum(row.correct for row in base_common)
                ) / len(common)
                delta_ci = paired_bootstrap_delta_ci(
                    base_common,
                    candidate_common,
                    iterations=iterations,
                    seed=_seed_for(seed, benchmark, base_label, model),
                )
                metrics["overall"]["delta_vs_base"] = delta
                metrics["overall"]["delta_ci95"] = list(delta_ci)
                for subset in subset_names:
                    base_subset = [row for row in base_common if row.subset == subset]
                    candidate_subset = [row for row in candidate_common if row.subset == subset]
                    if not base_subset:
                        continue
                    subset_delta = (
                        sum(row.correct for row in candidate_subset)
                        - sum(row.correct for row in base_subset)
                    ) / len(base_subset)
                    subset_ci = paired_bootstrap_delta_ci(
                        base_subset,
                        candidate_subset,
                        iterations=iterations,
                        seed=_seed_for(seed, benchmark, subset, base_label, model),
                    )
                    metrics["subsets"][subset]["delta_vs_base"] = subset_delta
                    metrics["subsets"][subset]["delta_ci95"] = list(subset_ci)
        output["benchmarks"][benchmark] = benchmark_output
    return output


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def write_csv_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "benchmark",
                "subset",
                "model",
                "correct",
                "total",
                "accuracy",
                "ci95_low",
                "ci95_high",
                "delta_vs_base",
                "delta_ci95_low",
                "delta_ci95_high",
                "invalid",
                "errors",
                "per_option_accuracy",
            ),
        )
        writer.writeheader()
        for benchmark, benchmark_data in report["benchmarks"].items():
            for model, model_data in benchmark_data["models"].items():
                metrics = {"__overall__": model_data["overall"], **model_data["subsets"]}
                for subset, metric in metrics.items():
                    delta_ci = metric.get("delta_ci95", [None, None])
                    writer.writerow(
                        {
                            "benchmark": benchmark,
                            "subset": subset,
                            "model": model,
                            "correct": metric["correct"],
                            "total": metric["total"],
                            "accuracy": metric["accuracy"],
                            "ci95_low": metric["ci95"][0],
                            "ci95_high": metric["ci95"][1],
                            "delta_vs_base": metric.get("delta_vs_base"),
                            "delta_ci95_low": delta_ci[0],
                            "delta_ci95_high": delta_ci[1],
                            "invalid": metric["invalid"],
                            "errors": metric["errors"],
                            "per_option_accuracy": metric.get("per_option_accuracy"),
                        }
                    )


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = ["# Classic Spatial Evaluation", ""]
    for benchmark, benchmark_data in report["benchmarks"].items():
        lines.extend(
            [
                f"## {benchmark}",
                "",
                "| Model | Accuracy | Macro | 95% CI | Delta vs Base | Invalid | Errors |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model, model_data in benchmark_data["models"].items():
            metric = model_data["overall"]
            ci = metric["ci95"]
            delta = metric.get("delta_vs_base")
            lines.append(
                (
                    "| {model} | {accuracy} | {macro} | [{low}, {high}] | "
                    "{delta} | {invalid} | {errors} |"
                ).format(
                    model=model,
                    accuracy=_format_percent(metric["accuracy"]),
                    macro=_format_percent(metric.get("macro_accuracy")),
                    low=_format_percent(ci[0]),
                    high=_format_percent(ci[1]),
                    delta=_format_percent(delta),
                    invalid=metric["invalid"],
                    errors=metric["errors"],
                )
            )
        lines.extend(["", "### Subsets", ""])
        for model, model_data in benchmark_data["models"].items():
            lines.append(f"**{model}**")
            lines.append("")
            lines.append("| Subset | Correct | Accuracy | Delta vs Base |")
            lines.append("|---|---:|---:|---:|")
            for subset, metric in model_data["subsets"].items():
                lines.append(
                    f"| {subset} | {metric['correct']}/{metric['total']} | "
                    f"{_format_percent(metric['accuracy'])} | "
                    f"{_format_percent(metric.get('delta_vs_base'))} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    manifest_path = args.manifest.resolve()
    manifest_rows = list(iter_jsonl(manifest_path))
    validate_manifest(manifest_rows)
    manifest_hash = manifest_sha256(manifest_path)
    results = load_results(args.predictions, expected_manifest_hash=manifest_hash)
    report = summarize(
        manifest_rows,
        results,
        base_label=args.base_label,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        allow_incomplete=args.allow_incomplete,
    )
    report["manifest"] = str(manifest_path)
    report["manifest_sha256"] = manifest_hash
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", report)
    write_csv_report(output_dir / "metrics.csv", report)
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8", newline="\n")
    error_rows = [
        {
            "model": row.model,
            "benchmark": row.benchmark,
            "subset": row.subset,
            "sample_id": row.sample_id,
            "prediction": list(row.prediction),
            "gold": list(row.gold),
            "prediction_text": row.prediction_text,
            "gold_text": row.gold_text,
            "answer_type": row.answer_type,
            "parse_status": row.parse_status,
            "error": row.error,
        }
        for model_rows in results.values()
        for row in model_rows.values()
        if not row.correct
    ]
    write_jsonl(output_dir / "errors.jsonl", error_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
