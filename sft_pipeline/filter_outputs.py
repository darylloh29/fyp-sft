#!/usr/bin/env python3
import argparse
import json
import re
from typing import Any, Dict, List, Tuple

from common import OFFICIAL_LINKS, append_jsonl, iter_jsonl


POLICY_KEYWORDS = [
    "eligible",
    "eligibility",
    "loan",
    "grant",
    "cpf",
    "hdb",
    "bto",
    "resale",
    "iras",
    "ura",
    "stamp duty",
    "absd",
    "mop",
    "hfe",
    "tax",
]

BANNED_INTENT_PATTERNS = [
    r"\b(fake|forge|falsify|fabricate)\b",
    r"\b(tax evasion|evade tax)\b",
    r"\b(bribe|kickback)\b",
    r"\b(discriminate|race-based|religion-based)\b",
]

# Heuristic: claims with specific percentages/amounts can be risky unless answer asks to verify.
HARD_NUMBER_PATTERNS = [
    r"\b\d+(\.\d+)?\s*%",
    r"\$\s*\d[\d,]*(\.\d+)?",
    r"\b\d+\s*(years?|months?)\b",
]

VERIFY_HINTS = [
    "verify",
    "check official",
    "subject to change",
    "latest rules",
    "confirm with",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter judged outputs into accepted subset.")
    p.add_argument("--input", required=True, help="Judged JSONL.")
    p.add_argument("--output", required=True, help="Accepted JSONL.")
    p.add_argument("--rejected-out", default="", help="Optional rejected JSONL path.")
    p.add_argument("--min-score", type=float, default=4.0, help="Minimum judge score.")
    p.add_argument("--allow-revise", action="store_true", help="Allow verdict=revise if score threshold passes.")
    return p.parse_args()


def contains_policy_topic(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in POLICY_KEYWORDS)


def has_official_link(text: str) -> bool:
    t = text.lower()
    return any(url.lower() in t for url in OFFICIAL_LINKS.values())


def has_banned_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in BANNED_INTENT_PATTERNS)


def has_hard_number_without_caution(text: str) -> bool:
    t = text.lower()
    has_num = any(re.search(pat, t) for pat in HARD_NUMBER_PATTERNS)
    if not has_num:
        return False
    has_verify_hint = any(h in t for h in VERIFY_HINTS)
    return not has_verify_hint


def check_row(row: Dict[str, Any], min_score: float, allow_revise: bool) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    judge = row.get("judge", {})
    verdict = str(judge.get("verdict", "")).lower()
    score = float(judge.get("score", 0.0))
    if score < min_score:
        reasons.append(f"low_score:{score}")
    if verdict != "accept" and not (allow_revise and verdict == "revise"):
        reasons.append(f"verdict:{verdict}")

    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 3:
        reasons.append("invalid_messages")
        return False, reasons

    user_text = ""
    assistant_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_text = str(m.get("content", ""))
        if m.get("role") == "assistant":
            assistant_text = str(m.get("content", ""))

    if has_banned_intent(assistant_text):
        reasons.append("banned_intent")
    if has_hard_number_without_caution(assistant_text):
        reasons.append("hard_number_without_caution")
    if contains_policy_topic(user_text) and not has_official_link(assistant_text):
        reasons.append("missing_official_link_for_policy_topic")

    return len(reasons) == 0, reasons


def main() -> None:
    args = parse_args()
    rejected_out = args.rejected_out.strip()
    accepted = 0
    rejected = 0

    for row in iter_jsonl(args.input):
        keep, reasons = check_row(row, min_score=args.min_score, allow_revise=args.allow_revise)
        if keep:
            append_jsonl(args.output, row)
            accepted += 1
            continue
        row = dict(row)
        row["filter_reasons"] = reasons
        rejected += 1
        if rejected_out:
            append_jsonl(rejected_out, row)

    print(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
