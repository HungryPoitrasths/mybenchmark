#!/usr/bin/env python3
"""Re-run ONLY the truncated questions from a type-sampled VLM eval.

Background
----------
In output/type_sample_vlm_eval/<TYPE>.json many rows were scored from a model
response that got cut off by the max_tokens limit BEFORE it emitted its
"Answer:" line. The original parser then fell back to grabbing the first bare
"\\b[A-D]\\b" token in the text, which is almost always the English article
"a" -> letter "A". That silently mis-scored most long-reasoning L2/L3 items.

This script re-asks the model for ONLY those truncated questions, using:
  * a STRICT prompt that forces "Answer: <letter>" on the FIRST line, so the
    choice survives even if the reasoning is later truncated;
  * a larger --max_tokens;
  * a STRICT parser that returns parse_fail instead of guessing "A".

Each per-type JSON row already embeds image_base64 + question + options +
gt_letter, so no benchmark.json or image-root lookup is needed.

By default it writes <TYPE>.json into a rerun/ subdir and prints an old-vs-new
accuracy comparison. It does NOT touch the originals unless --inplace is given
(which first writes a .bak backup).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Identical to run_sampled_type_vlm_eval.py so the model sees the same system role.
SYSTEM_PROMPT = (
    "You are a careful vision-language assistant solving multiple-choice "
    "spatial reasoning questions about an image."
)

# Stricter than the original PROMPT_SUFFIX: answer FIRST, on its own line.
STRICT_SUFFIX = (
    "First output your final choice on the very first line, in EXACTLY this "
    "format with nothing else on that line:\n"
    "Answer: <single option letter>\n"
    "Then, starting on the next line, give your full reasoning.\n"
    "Even if you run out of room, the first line must already contain the "
    "answer letter."
)

DEFAULT_EVAL_DIR = "output/type_sample_vlm_eval"
COMPLETED_RERUN_RESULTS = {"correct", "wrong"}


def build_prompt(question_text: str, options: list[dict[str, Any]]) -> str:
    parts = [str(question_text or "").strip(), ""]
    for opt in options:
        letter = opt.get("letter") or chr(65 + len(parts))
        parts.append(f"{letter}) {opt.get('text', '')}")
    parts.extend(["", STRICT_SUFFIX])
    return "\n".join(parts)


def allowed_letters(options: list[dict[str, Any]]) -> str:
    letters = [str(o.get("letter") or "").upper() for o in options if o.get("letter")]
    if letters:
        return "".join(letters)
    return "".join(chr(65 + i) for i in range(min(len(options), 26)))


def parse_answer_strict(raw: str | None, letters: str) -> str | None:
    """Return a letter only when the model clearly committed to one.

    Unlike the original parser this NEVER falls back to the first bare A-D
    token (which catches the article "a"). If no explicit answer is found we
    return None so the caller can mark it parse_fail instead of guessing "A".
    """
    if not raw:
        return None
    allowed = re.escape(letters.upper())
    upper = raw.strip().upper()

    # whole response is exactly one letter
    if re.fullmatch(rf"[{allowed}]", upper):
        return upper

    patterns = [
        rf"(?:FINAL\s+)?ANSWER\s*[:：]\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"(?:CHOICE|OPTION)\s*[:：]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^[\(\[]?\s*([{allowed}])\s*[\)\].:：\-]",
    ]
    for pattern in patterns:
        m = re.search(pattern, upper, re.MULTILINE)
        if m:
            return m.group(1)
    return None


def is_truncated_response(raw: str | None, letters: str) -> bool:
    """A response is 'truncated' if the strict parser cannot find an answer."""
    return parse_answer_strict(raw, letters) is None



def make_client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_model(
    client: Any,
    *,
    model: str,
    image_b64: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    # Per-type JSON stores bare base64 (no data: prefix); rows are JPEG.
    data_url = f"data:image/jpeg;base64,{image_b64}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def call_with_retries(
    client: Any,
    *,
    model: str,
    image_b64: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_delay: float,
) -> tuple[str | None, str | None]:
    """Return (raw_response, error). error is None on success."""
    attempt = 0
    delay = retry_delay
    while True:
        try:
            raw = call_model(
                client,
                model=model,
                image_b64=image_b64,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return raw, None
        except Exception as exc:  # noqa: BLE001 - surface any API error per row
            attempt += 1
            if attempt > retries:
                return None, f"{type(exc).__name__}: {exc}"
            time.sleep(delay)
            delay *= 2


def call_until_parsed(
    client: Any,
    *,
    model: str,
    image_b64: str,
    prompt: str,
    letters: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_delay: float,
) -> tuple[str | None, str | None, str | None, int]:
    """Return (raw_response, error, parsed_letter, parse_attempts)."""
    raw: str | None = None
    err: str | None = None
    pred: str | None = None
    parse_attempts = 0
    for parse_attempts in range(1, max(1, retries + 1) + 1):
        raw, err = call_with_retries(
            client,
            model=model,
            image_b64=image_b64,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            retry_delay=retry_delay,
        )
        if err is not None:
            return raw, err, None, parse_attempts
        pred = parse_answer_strict(raw, letters)
        if pred is not None:
            return raw, None, pred, parse_attempts
        if parse_attempts <= retries:
            time.sleep(retry_delay)
    return raw, None, None, parse_attempts



def load_truncated_map(eval_dir: Path, explicit: Path | None) -> dict[str, list[int]]:
    """Return {type_name: [question_id, ...]} of truncated questions."""
    path = explicit or (eval_dir / "_truncated_ids.json")
    if not path.exists():
        raise SystemExit(
            f"truncated-id file not found: {path}\n"
            "Generate it first (it maps each TYPE -> list of truncated ids)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: [int(i) for i in v] for k, v in data.items()}


def gold_letter_of(row: dict[str, Any]) -> str:
    g = str(row.get("gt_letter") or row.get("gold_answer") or "").strip().upper()
    if g:
        return g
    for opt in row.get("options") or []:
        if opt.get("is_gold"):
            return str(opt.get("letter") or "").upper()
    return ""


def is_completed_rerun(row: dict[str, Any]) -> bool:
    return row.get("rerun_result") in COMPLETED_RERUN_RESULTS


def output_path_for_type(
    *,
    type_name: str,
    eval_dir: Path,
    out_dir: Path,
    inplace: bool,
) -> Path:
    return (eval_dir if inplace else out_dir) / f"{type_name}.json"


def write_rows_checkpoint(
    *,
    rows: list[dict[str, Any]],
    dst: Path,
    inplace: bool,
    src: Path,
    original_source_text: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if inplace:
        bak = src.with_suffix(".json.bak")
        if not bak.exists():
            bak.write_text(original_source_text, encoding="utf-8")
    tmp = dst.with_name(f"{dst.name}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dst)


def merge_resume_rows(
    *,
    source_rows: list[dict[str, Any]],
    resume_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    if not resume_path.exists():
        return source_rows, 0

    resumed_rows = json.loads(resume_path.read_text(encoding="utf-8"))
    resumed_by_id = {
        int(row["id"]): row
        for row in resumed_rows
        if isinstance(row, dict) and "id" in row
    }
    merged = []
    restored = 0
    for row in source_rows:
        if "id" not in row:
            merged.append(row)
            continue
        resumed = resumed_by_id.get(int(row["id"]))
        if resumed is not None:
            merged.append(resumed)
            if is_completed_rerun(resumed) or resumed.get("rerun_result") == "error":
                restored += 1
        else:
            merged.append(row)
    return merged, restored


def rerun_type(
    *,
    type_name: str,
    ids: list[int],
    eval_dir: Path,
    out_dir: Path,
    client: Any,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_delay: float,
    delay: float,
    workers: int,
    checkpoint_every: int,
    resume: bool,
    inplace: bool,
) -> dict[str, Any]:
    src = eval_dir / f"{type_name}.json"
    if not src.exists():
        print(f"  !! {type_name}: source json missing ({src}), skipped")
        return {"type": type_name, "skipped": True}

    original_source_text = src.read_text(encoding="utf-8")
    source_rows = json.loads(original_source_text)
    original_by_id = {int(r["id"]): r for r in source_rows if "id" in r}
    dst = output_path_for_type(
        type_name=type_name,
        eval_dir=eval_dir,
        out_dir=out_dir,
        inplace=inplace,
    )
    rows = source_rows
    restored = 0
    if resume:
        rows, restored = merge_resume_rows(source_rows=source_rows, resume_path=dst)
        if restored:
            print(f"  .. {type_name}: resumed {restored} prior attempted row(s) from {dst}")

    by_id = {int(r["id"]): r for r in rows if "id" in r}
    target_ids = [i for i in ids if i in by_id]
    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"  !! {type_name}: {len(missing)} ids not found in json: {missing[:8]}...")

    old_correct_on_targets = sum(
        1
        for i in target_ids
        if original_by_id.get(i, {}).get(
            "orig_result", original_by_id.get(i, {}).get("result")
        )
        == "correct"
    )

    pending_ids = target_ids
    if resume:
        pending_ids = [qid for qid in target_ids if not is_completed_rerun(by_id[qid])]
    skipped_completed = len(target_ids) - len(pending_ids)
    if skipped_completed:
        print(f"  .. {type_name}: skipping {skipped_completed} completed row(s)")

    def _run_one(qid: int) -> dict[str, Any]:
        row = by_id[qid]
        options = row.get("options") or []
        letters = allowed_letters(options)
        prompt = build_prompt(row.get("question") or "", options)
        raw, err, pred, parse_attempts = call_until_parsed(
            client,
            model=model,
            image_b64=row["image_base64"],
            prompt=prompt,
            letters=letters,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            retry_delay=retry_delay,
        )
        if delay:
            time.sleep(delay)
        gold = gold_letter_of(row)
        return {
            "qid": qid,
            "raw": raw,
            "error": err,
            "pred": pred,
            "gold": gold,
            "parse_attempts": parse_attempts,
        }

    def _apply_result(result: dict[str, Any]) -> tuple[str, str]:
        row = by_id[int(result["qid"])]
        err = result["error"]
        pred = result["pred"]
        row["rerun_parse_attempts"] = result.get("parse_attempts")
        if err is not None:
            row["rerun_error"] = err
            row["rerun_model_letter"] = None
            row["rerun_raw_response"] = None
            row["rerun_result"] = "error"
            return "API_FAIL", "-"

        row["rerun_raw_response"] = result["raw"]
        row["rerun_error"] = None
        if pred is None:
            row["rerun_model_letter"] = None
            row["rerun_result"] = "parse_fail"
        else:
            row["rerun_model_letter"] = pred
            ok = bool(result["gold"]) and pred == result["gold"]
            row["rerun_result"] = "correct" if ok else "wrong"

        # Promote rerun result into the canonical fields so downstream
        # viewers/metrics pick up the corrected answer. Keep originals.
        original_row = original_by_id.get(int(result["qid"]), {})
        row.setdefault("orig_model_letter", original_row.get("model_letter"))
        row.setdefault("orig_result", original_row.get("result"))
        if pred is not None:
            row["model_letter"] = pred
            row["result"] = row["rerun_result"]
        else:
            row["model_letter"] = None
            row["result"] = "parse_fail"
        trunc_state = "TRUNCATED" if row["rerun_result"] == "parse_fail" else "complete"
        return "API_OK", trunc_state

    processed_now = 0
    worker_count = max(1, int(workers))
    if pending_ids:
        print(
            f"  .. {type_name}: running {len(pending_ids)} request(s) "
            f"with workers={worker_count}",
            flush=True,
        )

    if worker_count == 1:
        result_iter = (_run_one(qid) for qid in pending_ids)
        for result in result_iter:
            processed_now += 1
            api_state, trunc_state = _apply_result(result)
            if checkpoint_every and processed_now % checkpoint_every == 0:
                write_rows_checkpoint(
                    rows=rows,
                    dst=dst,
                    inplace=inplace,
                    src=src,
                    original_source_text=original_source_text,
                )
            qid = int(result["qid"])
            done = skipped_completed + processed_now
            retry_note = (
                f" attempts={result['parse_attempts']}"
                if int(result.get("parse_attempts") or 1) > 1
                else ""
            )
            print(
                f"    {type_name}: {done}/{len(target_ids)} id={qid} "
                f"{api_state} {trunc_state}{retry_note}",
                flush=True,
            )
    elif pending_ids:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(pending_ids))) as pool:
            futures = {pool.submit(_run_one, qid): qid for qid in pending_ids}
            for future in as_completed(futures):
                processed_now += 1
                result = future.result()
                api_state, trunc_state = _apply_result(result)
                if checkpoint_every and processed_now % checkpoint_every == 0:
                    write_rows_checkpoint(
                        rows=rows,
                        dst=dst,
                        inplace=inplace,
                        src=src,
                        original_source_text=original_source_text,
                    )
                qid = int(result["qid"])
                done = skipped_completed + processed_now
                retry_note = (
                    f" attempts={result['parse_attempts']}"
                    if int(result.get("parse_attempts") or 1) > 1
                    else ""
                )
                print(
                    f"    {type_name}: {done}/{len(target_ids)} id={qid} "
                    f"{api_state} {trunc_state}{retry_note}",
                    flush=True,
                )

    write_rows_checkpoint(
        rows=rows,
        dst=dst,
        inplace=inplace,
        src=src,
        original_source_text=original_source_text,
    )

    new_correct = sum(1 for i in target_ids if by_id[i].get("rerun_result") == "correct")
    parse_fail = sum(1 for i in target_ids if by_id[i].get("rerun_result") == "parse_fail")
    api_errors = sum(1 for i in target_ids if by_id[i].get("rerun_result") == "error")
    n_t = len(target_ids)
    summary = {
        "type": type_name,
        "truncated_targets": n_t,
        "old_correct_on_targets": old_correct_on_targets,
        "new_correct": new_correct,
        "parse_fail": parse_fail,
        "api_errors": api_errors,
        "skipped_completed": skipped_completed,
        "processed_now": processed_now,
        "output": str(dst),
    }
    print(
        f"  == {type_name}: targets={n_t} "
        f"old_correct={old_correct_on_targets} -> new_correct={new_correct} "
        f"(parse_fail={parse_fail}, err={api_errors}, resumed={skipped_completed}) "
        f"-> {dst.name}"
    )
    return summary



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-run only the truncated questions of a type-sampled VLM eval."
    )
    p.add_argument("--eval_dir", default=DEFAULT_EVAL_DIR,
                   help="Directory holding <TYPE>.json + _truncated_ids.json")
    p.add_argument("--truncated_ids", default=None,
                   help="Override path to the {type: [ids]} json")
    p.add_argument("--type", dest="types", action="append", default=None,
                   help="Limit to these type name(s); repeatable. Default: all in the id map")
    p.add_argument("--out_dir", default=None,
                   help="Where to write reran <TYPE>.json (default: <eval_dir>/rerun)")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the original <TYPE>.json (writes a .json.bak first)")
    p.add_argument("--resume", action="store_true",
                   help="Load existing output and skip rows with completed rerun_result")
    # model / API config — defaults match the original eval metadata
    p.add_argument("--model", default="qwen3.5-flash")
    p.add_argument("--base_url", default="https://www.packyapi.com/v1")
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_key_env", default=None)
    p.add_argument("--max_tokens", type=int, default=4096,
                   help="Larger than the original 1024/3072 so reasoning fits")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry_delay", type=float, default=2.0)
    p.add_argument("--delay", type=float, default=0.2)
    p.add_argument("--workers", "--vlm_workers", dest="workers", type=int, default=1,
                   help="Maximum number of concurrent VLM requests")
    p.add_argument("--checkpoint_every", type=int, default=1,
                   help="Write output every N completed requests (0 = final write only)")
    p.add_argument("--limit", type=int, default=0,
                   help="Debug: cap number of questions per type (0 = no cap)")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be re-run, make no API calls")
    args = p.parse_args(argv)
    if args.workers <= 0:
        p.error("--workers must be positive")
    if args.checkpoint_every < 0:
        p.error("--checkpoint_every must be non-negative")
    return args


def resolve_api_key(args: argparse.Namespace) -> str:
    key = (
        args.api_key
        or (os.getenv(args.api_key_env) if args.api_key_env else None)
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("PACKYAPI_API_KEY")
    )
    if not key:
        raise SystemExit(
            "No API key. Pass --api_key, or set --api_key_env, or export "
            "OPENAI_API_KEY / DASHSCOPE_API_KEY / PACKYAPI_API_KEY."
        )
    return key


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "rerun")

    trunc_map = load_truncated_map(eval_dir, Path(args.truncated_ids) if args.truncated_ids else None)
    if args.types:
        wanted = set(args.types)
        trunc_map = {k: v for k, v in trunc_map.items() if k in wanted}
        missing = wanted - set(trunc_map)
        if missing:
            print(f"warning: requested types not in id map: {sorted(missing)}")
    if args.limit:
        trunc_map = {k: v[: args.limit] for k, v in trunc_map.items()}

    total = sum(len(v) for v in trunc_map.values())
    print(f"types to re-run : {len(trunc_map)}")
    print(f"truncated qs    : {total}")
    for k in sorted(trunc_map, key=lambda t: -len(trunc_map[t])):
        print(f"  {k:42s} {len(trunc_map[k])}")
    print(f"model           : {args.model} @ {args.base_url}")
    print(f"max_tokens      : {args.max_tokens}  temperature: {args.temperature}")
    print(f"workers         : {args.workers}")
    print(f"resume          : {args.resume}")
    print(f"checkpoint_every: {args.checkpoint_every}")
    print(f"output          : {'IN-PLACE (.bak kept)' if args.inplace else out_dir}")

    if args.dry_run:
        print("\n[dry-run] no API calls made.")
        return

    api_key = resolve_api_key(args)
    client = make_client(args.base_url, api_key, args.timeout)

    summaries = []
    for type_name in sorted(trunc_map, key=lambda t: -len(trunc_map[t])):
        print(f"\n--- {type_name} ---")
        summaries.append(
            rerun_type(
                type_name=type_name,
                ids=trunc_map[type_name],
                eval_dir=eval_dir,
                out_dir=out_dir,
                client=client,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                retries=args.retries,
                retry_delay=args.retry_delay,
                delay=args.delay,
                workers=args.workers,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                inplace=args.inplace,
            )
        )

    # write a roll-up + print old-vs-new
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "inplace": args.inplace,
        "resume": args.resume,
        "workers": args.workers,
        "checkpoint_every": args.checkpoint_every,
        "per_type": summaries,
    }
    report_path = (out_dir if not args.inplace else eval_dir) / "_rerun_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ RERUN SUMMARY ================")
    print(
        f"{'type':42s} {'targets':>8s} {'old_ok':>6s} {'new_ok':>6s} "
        f"{'pfail':>6s} {'err':>4s} {'skip':>5s} {'done':>5s}"
    )
    tot = {"t": 0, "o": 0, "n": 0, "p": 0, "e": 0, "s": 0, "d": 0}
    for s in summaries:
        if s.get("skipped"):
            continue
        print(
            f"{s['type']:42s} {s['truncated_targets']:8d} "
            f"{s['old_correct_on_targets']:6d} {s['new_correct']:6d} "
            f"{s['parse_fail']:6d} {s['api_errors']:4d} "
            f"{s.get('skipped_completed', 0):5d} {s.get('processed_now', 0):5d}"
        )
        tot["t"] += s["truncated_targets"]
        tot["o"] += s["old_correct_on_targets"]
        tot["n"] += s["new_correct"]
        tot["p"] += s["parse_fail"]
        tot["e"] += s["api_errors"]
        tot["s"] += s.get("skipped_completed", 0)
        tot["d"] += s.get("processed_now", 0)
    print("-" * 74)
    print(
        f"{'TOTAL':42s} {tot['t']:8d} {tot['o']:6d} {tot['n']:6d} "
        f"{tot['p']:6d} {tot['e']:4d} {tot['s']:5d} {tot['d']:5d}"
    )
    print(f"\nreport written  : {report_path}")
    print(
        "note: old_ok = how many of these truncated rows were scored correct "
        "before (mostly lucky 'A' hits); new_ok = correct after re-run."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
