#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


@dataclass
class TrainConfig:
    model_name_or_path: str
    output_dir: str

    train_file: str
    eval_file: str = ""
    test_file: str = ""

    max_seq_length: int = 2048
    num_train_epochs: float = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    bf16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = False
    seed: int = 42


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_messages_jsonl(path: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Invalid messages format at {path}:{line_no}")
            rows.append({"messages": messages})
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return Dataset.from_list(rows)


def format_chat_row(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Invalid messages format in example.")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Llama 3 8B SFT with TRL from existing messages JSONL.")
    p.add_argument("--config", default="sft_pipeline/trl_config.yaml", help="Path to YAML config.")
    p.add_argument("--dry-run", action="store_true", help="Load and print setup only, no training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    cfg = TrainConfig(**raw)

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    train_ds = read_messages_jsonl(cfg.train_file)
    eval_ds = read_messages_jsonl(cfg.eval_file) if cfg.eval_file else None
    test_ds = read_messages_jsonl(cfg.test_file) if cfg.test_file else None

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": cfg.model_name_or_path,
                    "train_rows": len(train_ds),
                    "eval_rows": len(eval_ds) if eval_ds is not None else 0,
                    "test_rows": len(test_ds) if test_ds is not None else 0,
                    "output_dir": cfg.output_dir,
                }
            )
        )
        return

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        seed=cfg.seed,
        report_to=[],
        eval_strategy="steps" if eval_ds is not None else "no",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=lambda batch: format_chat_row(batch, tokenizer),
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if test_ds is not None:
        test_text_ds = test_ds.map(
            lambda ex: {"text": format_chat_row(ex, tokenizer)},
            remove_columns=test_ds.column_names,
        )
        metrics = trainer.evaluate(eval_dataset=test_text_ds, metric_key_prefix="test")
        with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
