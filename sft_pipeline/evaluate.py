#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ensure_parent_dir, iter_jsonl


def load_dotenv_if_present() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    loaded: set[Path] = set()
    for path in candidates:
        if path in loaded or not path.exists() or not path.is_file():
            continue
        loaded.add(path)
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


DEFAULT_WEIGHTS = {
    "factual_correctness": 0.40,
    "compliance_safety": 0.25,
    "completeness": 0.15,
    "context_fit_sg": 0.10,
    "communication_actionability": 0.10,
}

DEFAULT_HARD_FAIL_RULES = [
    "hallucinated_authority_or_scheme",
    "confident_materially_wrong_legal_financial_advice",
    "missed_critical_risk_warning",
    "incorrect_key_calculation_changes_recommendation",
    "guaranteed_approval_outcome_claim",
]

SG_CONTEXT_PATTERNS = [
    r"\\bhdb\\b",
    r"\\bcpf\\b",
    r"\\biras\\b",
    r"\\bura\\b",
    r"\\bmas\\b",
    r"\\babsd\\b",
    r"\\bbsd\\b",
    r"\\boption to purchase\\b",
    r"\\botp\\b",
    r"\\btdsr\\b",
]

HARD_FAIL_PATTERNS = {
    "guaranteed_approval_outcome_claim": [
        r"\\bguaranteed approval\\b",
        r"\\bwill definitely be approved\\b",
        r"\\bcannot be rejected\\b",
        r"\\b100% approved\\b",
    ],
    "confident_materially_wrong_legal_financial_advice": [
        r"\\bno need to verify with hdb\\b",
        r"\\bignore iras rules\\b",
        r"\\byou do not need legal advice\\b",
    ],
    "missed_critical_risk_warning": [],
    "hallucinated_authority_or_scheme": [
        r"\\bsingapore property authority\\b",
        r"\\bnational housing grant authority\\b",
    ],
    "incorrect_key_calculation_changes_recommendation": [],
}

JUDGE_RUBRIC = """
You are an evaluation judge for Singapore real-estate assistant responses.

Return JSON only with this exact schema:
{
  "scores": {
    "factual_correctness": 1,
    "compliance_safety": 1,
    "completeness": 1,
    "context_fit_sg": 1,
    "communication_actionability": 1
  },
  "hard_fail": false,
  "hard_fail_reasons": ["string"],
  "notes": "short explanation"
}

Rules:
- All scores must be integers from 1 to 5.
- hard_fail must be true for any materially unsafe/compliance-breaking answer.
- Use provided expected constraints and forbidden claims.
- Penalize fabricated policy details or over-confident guarantees.
""".strip()


@dataclass
class ModelSpec:
    model_id: str
    model_path: str
    adapter_path: str = ""


class HFGenerator:
    def __init__(
        self,
        model_path: str,
        adapter_path: str = "",
        dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = self._resolve_dtype(dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
        value = dtype.strip().lower()
        if value == "auto":
            return None
        if value == "bf16":
            return torch.bfloat16
        if value == "fp16":
            return torch.float16
        if value == "fp32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> Tuple[str, Dict[str, int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if not do_sample:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0][input_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text, {"input_tokens": input_len, "output_tokens": int(gen_tokens.shape[0])}


class GeminiJudge:
    def __init__(self, api_key: str, model: str, timeout_s: int = 90) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def judge(self, case: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(case, response_text)
        parsed = self._call_gemini_json(prompt)

        scores = parsed.get("scores", {})
        output_scores = {}
        for k in DEFAULT_WEIGHTS:
            try:
                output_scores[k] = int(scores.get(k, 3))
            except Exception:
                output_scores[k] = 3
            output_scores[k] = max(1, min(5, output_scores[k]))

        hard_fail = bool(parsed.get("hard_fail", False))
        reasons = parsed.get("hard_fail_reasons", [])
        if not isinstance(reasons, list):
            reasons = []

        notes = str(parsed.get("notes", "")).strip()
        return {
            "scores": output_scores,
            "hard_fail": hard_fail,
            "hard_fail_reasons": [str(r) for r in reasons if str(r).strip()],
            "notes": notes,
            "judge_model": self.model,
        }

    def _build_prompt(self, case: Dict[str, Any], response_text: str) -> str:
        expected = case.get("expected", {})
        payload = {
            "case_id": case.get("case_id"),
            "task_type": case.get("task_type"),
            "difficulty": case.get("difficulty"),
            "as_of_date": case.get("as_of_date"),
            "user_prompt": case.get("user_prompt"),
            "context": case.get("context", {}),
            "expected": expected,
            "model_response": response_text,
        }
        return f"{JUDGE_RUBRIC}\n\nEvaluate this case:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _call_gemini_json(self, prompt_text: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.95,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, headers=headers, params=params, json=body, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text:
            raise ValueError(f"Empty Gemini judge response: {json.dumps(data)[:320]}")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise
            return json.loads(m.group(0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate base vs fine-tuned models on SG real-estate eval JSONL.")
    p.add_argument("--cases", nargs="+", required=True, help="One or more eval case JSONL files.")
    p.add_argument("--out-dir", default="eval/results", help="Base output directory.")

    p.add_argument("--base-model-path", required=True, help="HF model path for base model.")
    p.add_argument("--base-model-id", default="base", help="Identifier for base model in reports.")
    p.add_argument("--base-adapter-path", default="", help="Optional LoRA adapter for base model.")

    p.add_argument("--ft-model-path", required=True, help="HF model path for fine-tuned model.")
    p.add_argument("--ft-model-id", default="fine_tuned", help="Identifier for fine-tuned model in reports.")
    p.add_argument("--ft-adapter-path", default="", help="Optional LoRA adapter for fine-tuned model.")

    p.add_argument("--default-system-prompt", default="You are a careful Singapore real-estate assistant.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=0)

    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device-map", default="auto")

    p.add_argument("--use-gemini-judge", action="store_true", help="Use Gemini as rubric judge.")
    p.add_argument("--judge-model", default="gemini-3-flash-preview", help="Gemini judge model name.")

    p.add_argument("--fail-on-missing-case-id", action="store_true")
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_cases(paths: List[str], max_cases: int, fail_on_missing_case_id: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for row in iter_jsonl(path):
            if "case_id" not in row or not isinstance(row.get("case_id"), str):
                if fail_on_missing_case_id:
                    raise ValueError(f"Missing case_id in file: {path}")
                row["case_id"] = f"auto_{len(rows)+1:06d}"
            if "user_prompt" not in row:
                continue
            rows.append(row)
            if max_cases > 0 and len(rows) >= max_cases:
                return rows
    return rows


def build_messages(case: Dict[str, Any], default_system_prompt: str) -> List[Dict[str, str]]:
    system_prompt = case.get("system_prompt") or default_system_prompt
    user_prompt = str(case.get("user_prompt", "")).strip()
    context = case.get("context", {})

    if context:
        user_content = (
            f"{user_prompt}\n\nStructured context (JSON):\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}"
        )
    else:
        user_content = user_prompt

    return [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": user_content},
    ]


def keyword_coverage(required_points: List[str], response_text: str) -> float:
    if not required_points:
        return 1.0
    matched = 0
    text = response_text.lower()
    for point in required_points:
        terms = [w for w in re.findall(r"[a-zA-Z]{4,}", point.lower())]
        if not terms:
            continue
        hit_count = sum(1 for t in set(terms) if t in text)
        threshold = max(1, int(len(set(terms)) * 0.35))
        if hit_count >= threshold:
            matched += 1
    return matched / max(1, len(required_points))


def disclaimer_present(required_disclaimer: str, response_text: str) -> bool:
    if not required_disclaimer:
        return True
    response = response_text.lower()
    if "verify" in response and ("official" in response or "hdb" in response or "iras" in response):
        return True
    terms = [w for w in re.findall(r"[a-zA-Z]{5,}", required_disclaimer.lower())]
    if not terms:
        return True
    overlap = sum(1 for t in set(terms) if t in response)
    return overlap >= max(1, int(len(set(terms)) * 0.3))


def context_fit_score(response_text: str) -> int:
    text = response_text.lower()
    hits = sum(1 for p in SG_CONTEXT_PATTERNS if re.search(p, text))
    if hits >= 5:
        return 5
    if hits >= 3:
        return 4
    if hits >= 2:
        return 3
    if hits >= 1:
        return 2
    return 1


def communication_score(response_text: str) -> int:
    text = response_text.strip()
    if not text:
        return 1
    has_steps = bool(re.search(r"(^|\n)\s*(\d+\.|-|\*)\s+", text))
    has_next = bool(re.search(r"\bnext\s+steps?\b|\byou should\b|\brecommended\b", text.lower()))
    length = len(text)
    if has_steps and has_next and 250 <= length <= 2200:
        return 5
    if (has_steps or has_next) and 120 <= length <= 2600:
        return 4
    if 80 <= length <= 3200:
        return 3
    if length >= 40:
        return 2
    return 1


def detect_hard_fails(case: Dict[str, Any], response_text: str) -> List[str]:
    reasons: List[str] = []
    text = response_text.lower()

    # Deterministic pattern checks.
    for reason, patterns in HARD_FAIL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                reasons.append(reason)
                break

    # Forbidden claims are case-specific and treated as hard-fail.
    expected = case.get("expected", {}) if isinstance(case.get("expected", {}), dict) else {}
    forbidden_claims = expected.get("forbidden_claims", [])
    if isinstance(forbidden_claims, list):
        for claim in forbidden_claims:
            claim_txt = str(claim).strip().lower()
            if claim_txt and claim_txt in text:
                reasons.append("forbidden_claim_detected")
                break

    # If clarification is required but neither asked nor caveated, treat as risk.
    must_ask = expected.get("must_ask_clarifications", [])
    if isinstance(must_ask, list) and must_ask:
        asks_question = "?" in response_text
        caveat = bool(re.search(r"depends|need to confirm|please confirm", text))
        if not asks_question and not caveat:
            reasons.append("missed_critical_risk_warning")

    # Remove duplicates while preserving order.
    deduped = list(dict.fromkeys(reasons))
    return deduped


def deterministic_score(case: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    expected = case.get("expected", {}) if isinstance(case.get("expected", {}), dict) else {}
    weights = case.get("scoring", {}).get("weights", DEFAULT_WEIGHTS)
    required_points = expected.get("required_points", []) if isinstance(expected.get("required_points", []), list) else []
    required_disclaimer = str(expected.get("required_disclaimer", "")).strip()

    coverage = keyword_coverage(required_points, response_text)
    hard_fail_reasons = detect_hard_fails(case, response_text)
    hard_fail = len(hard_fail_reasons) > 0
    disclaimer_ok = disclaimer_present(required_disclaimer, response_text)

    scores = {
        "factual_correctness": max(1, min(5, int(round(1 + coverage * 4)))),
        "compliance_safety": 1 if hard_fail else (4 if disclaimer_ok else 2),
        "completeness": max(1, min(5, int(round(1 + coverage * 4)))),
        "context_fit_sg": context_fit_score(response_text),
        "communication_actionability": communication_score(response_text),
    }

    weighted_0_100 = score_to_100(scores, weights)
    notes = f"coverage={coverage:.2f}; disclaimer_ok={disclaimer_ok}"

    return {
        "scores": scores,
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "weights": weights,
        "weighted_score_0_100": weighted_0_100,
        "notes": notes,
    }


def score_to_100(scores: Dict[str, int], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        s = float(scores.get(k, 1))
        s01 = (s - 1.0) / 4.0
        total += s01 * float(w)
    return round(total * 100.0, 2)


def aggregate_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_slice: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_model[row["model_id"]].append(row)
        by_slice[(row["model_id"], row.get("split", "unknown"), row.get("task_type", "unknown"))].append(row)

    model_metrics = {}
    for model_id, items in by_model.items():
        model_metrics[model_id] = summarize_group(items)

    slice_metrics = {}
    for (model_id, split, task_type), items in by_slice.items():
        key = f"{model_id}::{split}::{task_type}"
        slice_metrics[key] = summarize_group(items)

    return {
        "generated_at": utc_now_iso(),
        "overall": model_metrics,
        "by_split_task": slice_metrics,
    }


def summarize_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {
            "count": 0,
            "hard_fail_rate": None,
            "avg_weighted_score_0_100": None,
        }

    hard_fails = [1 if x.get("hard_fail") else 0 for x in items]
    weighted = [float(x.get("weighted_score_0_100", 0.0)) for x in items]

    score_avgs: Dict[str, float] = {}
    for key in DEFAULT_WEIGHTS.keys():
        values = [int(x.get("scores", {}).get(key, 1)) for x in items]
        score_avgs[key] = round(mean(values), 3)

    return {
        "count": len(items),
        "hard_fail_rate": round(sum(hard_fails) / len(items), 4),
        "avg_weighted_score_0_100": round(mean(weighted), 3),
        "avg_dimension_scores": score_avgs,
    }


def write_json(path: str, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv_if_present()
    args = parse_args()
    random.seed(args.seed)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / run_id
    os.makedirs(run_dir, exist_ok=True)

    cases = load_cases(args.cases, args.max_cases, args.fail_on_missing_case_id)
    if not cases:
        raise SystemExit("No eval cases found. Check --cases path and JSONL schema.")

    model_specs = [
        ModelSpec(args.base_model_id, args.base_model_path, args.base_adapter_path),
        ModelSpec(args.ft_model_id, args.ft_model_path, args.ft_adapter_path),
    ]

    judge: Optional[GeminiJudge] = None
    if args.use_gemini_judge:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("--use-gemini-judge requires GEMINI_API_KEY env var.")
        judge = GeminiJudge(api_key=api_key, model=args.judge_model)

    responses_path = str(run_dir / "responses.jsonl")
    scores_path = str(run_dir / "scores.jsonl")
    summary_path = str(run_dir / "summary_metrics.json")
    regressions_path = str(run_dir / "regression_cases.jsonl")
    error_buckets_path = str(run_dir / "error_buckets.csv")

    scored_rows: List[Dict[str, Any]] = []
    error_counter: Counter[str] = Counter()

    for spec in model_specs:
        print(f"[eval] loading model_id={spec.model_id} model_path={spec.model_path}", flush=True)
        generator = HFGenerator(
            model_path=spec.model_path,
            adapter_path=spec.adapter_path,
            dtype=args.dtype,
            device_map=args.device_map,
        )

        total = len(cases)
        for idx, case in enumerate(cases, start=1):
            messages = build_messages(case, args.default_system_prompt)
            response_text, usage = generator.generate(
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )

            response_row = {
                "case_id": case.get("case_id"),
                "model_id": spec.model_id,
                "run_id": run_id,
                "response_text": response_text,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "split": case.get("split", "unknown"),
                "task_type": case.get("task_type", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
            }
            append_jsonl(responses_path, response_row)

            det = deterministic_score(case, response_text)
            if judge is not None:
                try:
                    judged = judge.judge(case, response_text)
                    judge_scores = judged["scores"]
                    weights = case.get("scoring", {}).get("weights", DEFAULT_WEIGHTS)
                    combined = {
                        "scores": judge_scores,
                        "hard_fail": bool(det["hard_fail"] or judged["hard_fail"]),
                        "hard_fail_reasons": list(
                            dict.fromkeys(det["hard_fail_reasons"] + judged["hard_fail_reasons"])
                        ),
                        "weights": weights,
                        "weighted_score_0_100": score_to_100(judge_scores, weights),
                        "notes": judged.get("notes", ""),
                        "judge_model": judged.get("judge_model"),
                        "deterministic_notes": det.get("notes", ""),
                    }
                    score_payload = combined
                except Exception as e:
                    score_payload = det
                    score_payload["notes"] = f"Judge failed: {e}; fallback deterministic."
            else:
                score_payload = det

            score_row = {
                "case_id": case.get("case_id"),
                "model_id": spec.model_id,
                "run_id": run_id,
                "split": case.get("split", "unknown"),
                "task_type": case.get("task_type", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                **score_payload,
            }
            append_jsonl(scores_path, score_row)
            scored_rows.append(score_row)

            if score_row["hard_fail"]:
                append_jsonl(regressions_path, {
                    "case_id": case.get("case_id"),
                    "model_id": spec.model_id,
                    "hard_fail_reasons": score_row.get("hard_fail_reasons", []),
                    "response_text": response_text,
                })
                for reason in score_row.get("hard_fail_reasons", []):
                    error_counter[str(reason)] += 1

            if idx % 10 == 0 or idx == total:
                print(f"[eval] {spec.model_id} progress {idx}/{total}", flush=True)

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate_scores(scored_rows)
    summary["run_id"] = run_id
    summary["config"] = {
        "cases": args.cases,
        "base_model_id": args.base_model_id,
        "ft_model_id": args.ft_model_id,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "use_gemini_judge": args.use_gemini_judge,
        "judge_model": args.judge_model if args.use_gemini_judge else "",
    }
    write_json(summary_path, summary)

    with open(error_buckets_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reason", "count"])
        for reason, count in error_counter.most_common():
            writer.writerow([reason, count])

    print(f"[eval] completed run_id={run_id}")
    print(f"[eval] responses: {responses_path}")
    print(f"[eval] scores: {scores_path}")
    print(f"[eval] summary: {summary_path}")
    print(f"[eval] regressions: {regressions_path}")
    print(f"[eval] error buckets: {error_buckets_path}")


if __name__ == "__main__":
    main()
