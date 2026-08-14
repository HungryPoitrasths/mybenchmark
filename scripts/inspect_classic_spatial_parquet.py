#!/usr/bin/env python3
"""Inspect parquet schemas without printing embedded image payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-values", type=int, default=40)
    return parser


def inspect(path: Path, *, max_values: int) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("parquet inspection requires pyarrow") from exc

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    scalar_columns = [
        field.name
        for field in schema
        if pa.types.is_boolean(field.type)
        or pa.types.is_integer(field.type)
        or pa.types.is_floating(field.type)
        or pa.types.is_string(field.type)
    ]
    table = pq.read_table(path, columns=scalar_columns)
    frame = table.to_pandas()
    distributions = {}
    for name in scalar_columns:
        counts = frame[name].astype(str).value_counts(dropna=False)
        if len(counts) <= max_values or any(
            token in name.lower()
            for token in ("type", "task", "category", "subset", "level", "split")
        ):
            distributions[name] = {
                str(key): int(value)
                for key, value in counts.head(max_values).items()
            }
    example_fields = [
        name
        for name in scalar_columns
        if any(
            token in name.lower()
            for token in ("type", "task", "format", "question", "answer")
        )
    ]
    category_examples = {}
    for category in scalar_columns:
        if not any(
            token in category.lower()
            for token in ("type", "task", "category", "subset")
        ):
            continue
        values = frame[category].astype(str)
        if values.nunique() > max_values:
            continue
        examples = {}
        for value in values.drop_duplicates():
            row = frame.loc[values == value, example_fields].iloc[0]
            examples[value] = {
                name: str(item)[:500] for name, item in row.items()
            }
        category_examples[category] = examples
    return {
        "path": str(path.resolve()),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": {field.name: str(field.type) for field in schema},
        "scalar_distributions": distributions,
        "category_examples": category_examples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_values <= 0:
        raise ValueError("--max-values must be positive")
    result = [inspect(path, max_values=args.max_values) for path in args.paths]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
