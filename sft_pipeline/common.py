import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests


OFFICIAL_LINKS = {
    "HDB": "https://www.hdb.gov.sg/",
    "CPF": "https://www.cpf.gov.sg/",
    "IRAS": "https://www.iras.gov.sg/",
    "URA": "https://www.ura.gov.sg/",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def append_jsonl(path: str, row: Dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    ensure_parent_dir(path)
    payload = json.dumps(row, ensure_ascii=False)
    if lock is None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
        return
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")


def load_existing_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    ids: Set[str] = set()
    for row in iter_jsonl(path):
        rid = row.get("id")
        if isinstance(rid, str):
            ids.add(rid)
    return ids


def stable_id(messages: List[Dict[str, Any]], prefix: str = "sample") -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def extract_first_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def call_gemini_json(
    api_key: str,
    model: str,
    prompt_text: str,
    max_retries: int = 4,
    timeout_s: int = 90,
) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.95,
            "responseMimeType": "application/json",
        },
    }

    delay_s = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                params=params,
                json=body,
                timeout=timeout_s,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Retryable HTTP {resp.status_code}: {resp.text[:240]}")
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if not text:
                raise ValueError(f"Empty model text response. Raw: {json.dumps(data)[:400]}")
            return extract_first_json(text)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(delay_s + random.random())
            delay_s *= 2
    raise RuntimeError("Unexpected retry flow.")


def threaded_map(
    items: List[Any],
    worker_fn: Callable[[Any], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]],
    workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        future_map = {ex.submit(worker_fn, item): item for item in items}
        for fut in as_completed(future_map):
            ok, result, err = fut.result()
            if ok and result is not None:
                successes.append(result)
            else:
                failures.append({"item": future_map[fut], "error": err or "unknown_error"})
    return successes, failures


def split_train_val_test(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Any], List[Any], List[Any]]:
    rnd = random.Random(seed)
    shuffled = rows[:]
    rnd.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test
