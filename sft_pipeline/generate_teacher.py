#!/usr/bin/env python3
import argparse
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from common import OFFICIAL_LINKS, append_jsonl, call_gemini_json, iter_jsonl, load_existing_ids, stable_id, utc_now_iso


GENERATOR_RUBRIC = """
You are generating ground-truth assistant responses for supervised fine-tuning of a Singapore residential real-estate chatbot.

Return JSON only, with this exact schema:
{
  "assistant_answer": "string",
  "follow_up_questions": ["string"],
  "sources_used": ["HDB|CPF|IRAS|URA"]
}

Rules:
1) Be practical, concise, and step-by-step where useful.
2) Do NOT invent policy thresholds, timelines, rates, legal clauses, or exact figures.
3) If exact policy details are uncertain or time-sensitive, say so explicitly and advise verifying official sources.
4) Ask concise follow-up questions when missing key facts.
5) Refuse discriminatory housing guidance, fraud, tax evasion, document falsification, or policy circumvention.
6) If discussing policy eligibility/financing/taxes, include links to relevant official sources inside assistant_answer.

Official links:
HDB: {hdb}
CPF: {cpf}
IRAS: {iras}
URA: {ura}
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate assistant answers from prompt-only JSONL using Gemini.")
    p.add_argument("--input", required=True, help="Input prompt-only JSONL.")
    p.add_argument("--output", required=True, help="Output JSONL with generated assistant messages.")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    p.add_argument("--limit", type=int, default=0, help="Optional limit of rows to process.")
    p.add_argument("--log-every", type=int, default=25, help="Print progress every N completed rows.")
    return p.parse_args()


def get_system_and_user(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = ""
    user = ""
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "system":
            system = content
        elif role == "user":
            user = content
    if not user:
        raise ValueError("Missing user message.")
    return system, user


def build_generation_prompt(system_text: str, user_text: str) -> str:
    rubric = GENERATOR_RUBRIC
    rubric = rubric.replace("{hdb}", OFFICIAL_LINKS["HDB"])
    rubric = rubric.replace("{cpf}", OFFICIAL_LINKS["CPF"])
    rubric = rubric.replace("{iras}", OFFICIAL_LINKS["IRAS"])
    rubric = rubric.replace("{ura}", OFFICIAL_LINKS["URA"])
    return f"{rubric}\n\nConversation system prompt:\n{system_text}\n\nUser message:\n{user_text}\n"


def worker(
    row: Dict[str, Any],
    api_key: str,
    model: str,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, None, "Invalid or missing messages list."

    rid = row.get("id") or stable_id(messages)
    system_text, user_text = get_system_and_user(messages)
    prompt = build_generation_prompt(system_text, user_text)
    out = call_gemini_json(api_key=api_key, model=model, prompt_text=prompt)

    answer = str(out.get("assistant_answer", "")).strip()
    followups = out.get("follow_up_questions", [])
    if not isinstance(followups, list):
        followups = []
    sources = out.get("sources_used", [])
    if not isinstance(sources, list):
        sources = []
    if not answer:
        return False, None, "Model returned empty assistant_answer."

    assistant_message = {"role": "assistant", "content": answer}
    new_messages = messages + [assistant_message]
    result = {
        "id": rid,
        "messages": new_messages,
        "meta": {
            "stage": "generated",
            "model": model,
            "generated_at": utc_now_iso(),
            "follow_up_questions": [str(q) for q in followups if str(q).strip()],
            "sources_used": [str(s) for s in sources if str(s).strip()],
        },
    }
    return True, result, None


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY environment variable.")

    existing_ids = load_existing_ids(args.output)
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(args.input):
        messages = row.get("messages", [])
        rid = row.get("id") or stable_id(messages)
        if rid in existing_ids:
            continue
        row["id"] = rid
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
        f"[generate] start total={total} workers={max(1, args.workers)} model={args.model} "
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
                    f"[generate] progress {done}/{total} success={success_count} failed={len(failures)}",
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
