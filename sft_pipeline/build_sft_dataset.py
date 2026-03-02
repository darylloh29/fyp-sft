#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List

from common import append_jsonl, iter_jsonl, split_train_val_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train/val/test SFT JSONL files from accepted samples.")
    p.add_argument("--input", required=True, help="Accepted JSONL.")
    p.add_argument("--train-out", required=True, help="Train split output JSONL.")
    p.add_argument("--val-out", required=True, help="Validation split output JSONL.")
    p.add_argument("--test-out", required=True, help="Test split output JSONL.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--llama-style",
        choices=["messages", "prompt_completion"],
        default="messages",
        help="Output format. messages is recommended for chat SFT.",
    )
    return p.parse_args()


def to_prompt_completion(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    prompt_parts: List[str] = []
    completion = ""
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "assistant":
            completion = content
            break
        prompt_parts.append(f"{role.upper()}: {content}")
    return {"prompt": "\n\n".join(prompt_parts), "completion": completion}


def normalize_row(row: Dict[str, Any], style: str) -> Dict[str, Any]:
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Invalid messages array for SFT row.")
    if style == "messages":
        return {"messages": messages}
    return to_prompt_completion(messages)


def write_split(path: str, rows: List[Dict[str, Any]]) -> None:
    # Fresh rebuild: write from scratch.
    open(path, "w", encoding="utf-8").close()
    for r in rows:
        append_jsonl(path, r)


def main() -> None:
    args = parse_args()
    normalized: List[Dict[str, Any]] = []
    for row in iter_jsonl(args.input):
        normalized.append(normalize_row(row, style=args.llama_style))

    train, val, test = split_train_val_test(normalized, seed=args.seed)
    write_split(args.train_out, train)
    write_split(args.val_out, val)
    write_split(args.test_out, test)

    print(
        json.dumps(
            {
                "total": len(normalized),
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "format": args.llama_style,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
