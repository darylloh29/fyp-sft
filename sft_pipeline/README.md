# Singapore Property SFT Pipeline

This folder contains a practical synthetic-data pipeline for creating SFT data from prompt-only JSONL files.

## What it does

1. `generate_teacher.py`
   Reads prompt-only JSONL and calls Gemini to generate assistant answers.
2. `judge_outputs.py`
   Runs a second-pass quality judge with Gemini.
3. `filter_outputs.py`
   Applies deterministic + judge-based gates.
4. `build_sft_dataset.py`
   Produces final SFT JSONL (`messages` format for Llama 3 chat tuning).

## Input format

Each line must look like:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}]}
```

Your file `sg_property_2000_sft_prompts.jsonl` already matches this shape.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r sft_pipeline/requirements.txt
export GEMINI_API_KEY="YOUR_KEY"
```

## Recommended run

Use `gemini-3-flash` if your account supports it; otherwise `gemini-2.5-flash`.

```bash
python sft_pipeline/generate_teacher.py \
  --input sg_property_2000_sft_prompts.jsonl \
  --output data/generated_teacher.jsonl \
  --model gemini-3-flash-preview \
  --workers 6

python sft_pipeline/judge_outputs.py \
  --input data/generated_teacher.jsonl \
  --output data/judged_teacher.jsonl \
  --model gemini-3-flash-preview \
  --workers 6

python sft_pipeline/filter_outputs.py \
  --input data/judged_teacher.jsonl \
  --output data/accepted_teacher.jsonl \
  --min-score 4.0

python sft_pipeline/build_sft_dataset.py \
  --input data/accepted_teacher.jsonl \
  --train-out data/sft_train.jsonl \
  --val-out data/sft_val.jsonl \
  --test-out data/sft_test.jsonl \
  --seed 42
```

## Notes

- The scripts are resumable: existing IDs in output files are skipped.
- Generation and judging both use retry with backoff.
- Deterministic safety checks block obvious policy violations.
- For high-stakes policy answers, do a manual spot-audit before training.

## TRL SFT training (no JSON conversion step)

The training script consumes your existing `messages` JSONL directly (for example `data/sft_train.jsonl`) and applies the tokenizer chat template on the fly.

Install additional training dependencies:

```bash
pip install "transformers>=4.43" "trl>=0.10.1" "peft>=0.12.0" "datasets>=2.20.0" pyyaml accelerate
```

Dry-run config check:

```bash
python sft_pipeline/train_trl.py --config sft_pipeline/trl_config.yaml --dry-run
```

Start training:

```bash
python sft_pipeline/train_trl.py --config sft_pipeline/trl_config.yaml
```

Config file:
- `sft_pipeline/trl_config.yaml`

## Export fine-tuned adapter to GGUF

After training, if you have LoRA adapter files (for example in `outputs/`), run:

```bash
bash sft_pipeline/export_gguf.sh \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapter-dir outputs \
  --out-dir outputs/gguf_export \
  --quant q4_k_m
```

Artifacts:
- merged HF model: `outputs/gguf_export/merged_model`
- f16 GGUF: `outputs/gguf_export/model-f16.gguf`
- quantized GGUF: `outputs/gguf_export/model-q4_k_m.gguf`

Notes:
- If the base model is gated, set `HF_TOKEN` in your shell before running.
- To skip quantization, pass `--quant none`.
