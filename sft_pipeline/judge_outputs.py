#!/usr/bin/env python3
import argparse
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from common import append_jsonl, call_gemini_json, iter_jsonl, load_existing_ids, utc_now_iso


JUDGE_RUBRIC = """
You are a strict quality judge for Singapore residential real-estate assistant responses.

Return JSON only, with this exact schema:
{
  "score": 0.0,
  "verdict": "accept|revise|reject",
  "issues": ["string"],
  "improvement_hint": "string"
}

Scoring rubric (0-5):
- Safety and compliance with legal/ethical boundaries.
- No fabricated policy numbers, rates, timelines, or legal thresholds.
- Correct handling of uncertainty and time-sensitive policy details.
- Practical personalization and actionable guidance.
- Clarity and concision.

Decision guidance:
- accept: score >= 4.0 with no major safety/factual issue.
- revise: score between 2.5 and 3.9, or minor issues.
- reject: score < 2.5 or major safety/factual issue.
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge generated assistant outputs using Gemini.")
    p.add_argument("--input", required=True, help="Generated JSONL (from generate_teacher.py).")
    p.add_argument("--output", required=True, help="Output JSONL with judge block added.")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    p.add_argument("--limit", type=int, default=0, help="Optional limit of rows to process.")
    p.add_argument("--log-every", type=int, default=25, help="Print progress every N completed rows.")
    return p.parse_args()


def build_judge_prompt(messages: List[Dict[str, Any]]) -> str:
    rendered = json.dumps(messages, ensure_ascii=False, indent=2)
    return f"{JUDGE_RUBRIC}\n\nConversation:\n{rendered}\n"


def worker(
    row: Dict[str, Any],
    api_key: str,
    model: str,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False, None, "Expected messages to include system, user, assistant."
    prompt = build_judge_prompt(messages)
    judge = call_gemini_json(api_key=api_key, model=model, prompt_text=prompt)

    verdict = str(judge.get("verdict", "")).strip().lower()
    score = float(judge.get("score", 0.0))
    issues = judge.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    improvement_hint = str(judge.get("improvement_hint", "")).strip()
    if verdict not in {"accept", "revise", "reject"}:
        verdict = "revise"
    score = max(0.0, min(5.0, score))

    out = dict(row)
    out["judge"] = {
        "score": score,
        "verdict": verdict,
        "issues": [str(i) for i in issues if str(i).strip()],
        "improvement_hint": improvement_hint,
        "model": model,
        "judged_at": utc_now_iso(),
    }
    return True, out, None


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY environment variable.")

    existing_ids = load_existing_ids(args.output)
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(args.input):
        rid = row.get("id")
        if not isinstance(rid, str):
            continue
        if rid in existing_ids:
            continue
        rows.append(row)
        if args.limit > 0 and len(rows) >= args.limit:
            break

    lock = threading.Lock()
    failures: List[Dict[str, Any]] = []
    if not rows:
        print("No rows to process (all skipped/resumed).")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(rows)
    print(
        f"[judge] start total={total} workers={max(1, args.workers)} model={args.model} "
        f"resume_skipped={len(existing_ids)}",
        flush=True,
    )
    done = 0
    success_count = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_to_id = {ex.submit(worker, row, api_key, args.model): row.get("id", "unknown") for row in rows}
        for fut in as_completed(future_to_id):
            rid = future_to_id[fut]
            ok, result, err = fut.result()
            done += 1
            if ok and result:
                append_jsonl(args.output, result, lock=lock)
                success_count += 1
            else:
                failures.append({"id": rid, "error": err})

            if done % max(1, args.log_every) == 0 or done == total:
                print(
                    f"[judge] progress {done}/{total} success={success_count} failed={len(failures)}",
                    flush=True,
                )

    print(
        json.dumps(
            {"processed": len(rows), "succeeded": success_count, "failed": len(failures)},
            ensure_ascii=False,
        )
    )
    if failures:
        err_path = args.output + ".errors.jsonl"
        for f in failures:
            append_jsonl(err_path, f, lock=lock)
        print(f"Wrote failure log to: {err_path}")


if __name__ == "__main__":
    main()
