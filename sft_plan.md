# SFT Plan and Execution Log (Real Estate Chatbot)

## Goal
Build a domain-tuned Singapore residential real-estate chatbot by:
1. Creating high-quality supervised chat pairs from prompt-only seeds.
2. Fine-tuning a Llama 3.x Instruct base model with LoRA.
3. Keeping outputs policy-safe, uncertainty-aware, and grounded to official sources.

## Scope
- Domain: Singapore residential property (HDB/private, eligibility, financing, taxes, compliance).
- Data format: chat `messages` JSONL (`system`, `user`, `assistant`).
- Safety constraints: no fabricated legal/tax thresholds, refuse harmful/illegal/discriminatory guidance, encourage official verification.

## Pipeline Used
Implemented in `sft_pipeline/`:
1. `generate_teacher.py`
2. `judge_outputs.py`
3. `filter_outputs.py`
4. `build_sft_dataset.py`
5. `train_trl.py`

## What Was Actually Run
### 1) Seed prompts
- Input seed set: `sg_property_2000_sft_prompts.jsonl`
- Seed count: `2000`
- Format: `messages` with system + user only.

### 2) Teacher generation (Gemini)
- Script: `python sft_pipeline/generate_teacher.py --input sg_property_2000_sft_prompts.jsonl --output data/generated_teacher.jsonl --model gemini-3-flash-preview --workers 6`
- Result file: `data/generated_teacher.jsonl`
- Generated rows: `2000`
- Generation metadata stored per row: stage/model/timestamp/follow-up questions/sources used.

### 3) LLM judging pass
- Script: `python sft_pipeline/judge_outputs.py --input data/generated_teacher.jsonl --output data/judged_teacher.jsonl --model gemini-3-flash-preview --workers 6`
- Result file: `data/judged_teacher.jsonl`
- Judged rows: `2000`
- Judge schema per row: `score (0-5)`, `verdict (accept|revise|reject)`, `issues`, `improvement_hint`.

### 4) Deterministic + judge filtering
- Script: `python sft_pipeline/filter_outputs.py --input data/judged_teacher.jsonl --output data/accepted_teacher.jsonl --min-score 4.0`
- Accepted file: `data/accepted_teacher.jsonl`
- Accepted rows: `1658`
- Rejected rows: `342`

Applied gates include:
- `judge.score >= 4.0`
- verdict must be `accept` (unless `--allow-revise` is used)
- reject banned intent patterns (fraud/tax evasion/falsification/discrimination)
- for policy-heavy prompts, require official links in assistant answer (HDB/CPF/IRAS/URA)
- flag risky hard numbers without caution language (verify/subject to change)

### 5) Dataset build and split
- Script: `python sft_pipeline/build_sft_dataset.py --input data/accepted_teacher.jsonl --train-out data/sft_train.jsonl --val-out data/sft_val.jsonl --test-out data/sft_test.jsonl --seed 42`
- Split logic: randomized 80/10/10
- Final split sizes:
  - train: `1326`
  - val: `165`
  - test: `167`

### 6) SFT training (TRL + LoRA)
- Script: `python sft_pipeline/train_trl.py --config sft_pipeline/trl_config.yaml`
- Training artifacts written to: `outputs/`
- Checkpoints present: `checkpoint-100`, `checkpoint-168`
- Final adapter present: `outputs/adapter_model.safetensors`

Hyperparameters used (from config/artifacts):
- epochs: `4`
- lr: `1e-4`
- batch size: `4`
- grad accumulation: `8`
- max seq length: `2048`
- LoRA: `r=32`, `alpha=64`, `dropout=0.05`
- bf16: `true`
- gradient checkpointing: `true`
- seed: `42`

Base model note:
- `sft_pipeline/trl_config.yaml` currently shows `meta-llama/Meta-Llama-3-8B-Instruct`.
- Saved adapter metadata in `outputs/adapter_config.json` records `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- For reproducibility, treat `outputs/adapter_config.json` as source-of-truth for the run that produced current artifacts.

### 7) Test split evaluation during training
- File: `outputs/test_metrics.json`
- Recorded metrics:
  - `test_loss`: `0.51953`
  - `test_runtime`: `8.1343s`
  - `test_samples_per_second`: `20.53`
  - `test_steps_per_second`: `5.163`

## Data and Artifact Inventory
- Seed prompts: `sg_property_2000_sft_prompts.jsonl`
- Generated teacher data: `data/generated_teacher.jsonl`
- Judged data: `data/judged_teacher.jsonl`
- Accepted data: `data/accepted_teacher.jsonl`
- SFT splits: `data/sft_train.jsonl`, `data/sft_val.jsonl`, `data/sft_test.jsonl`
- Trained adapter + checkpoints: `outputs/`

## Repro Commands (End-to-End)
```bash
# 1) Generate assistant answers
python sft_pipeline/generate_teacher.py \
  --input sg_property_2000_sft_prompts.jsonl \
  --output data/generated_teacher.jsonl \
  --model gemini-3-flash-preview \
  --workers 6

# 2) Judge quality
python sft_pipeline/judge_outputs.py \
  --input data/generated_teacher.jsonl \
  --output data/judged_teacher.jsonl \
  --model gemini-3-flash-preview \
  --workers 6

# 3) Filter accepted set
python sft_pipeline/filter_outputs.py \
  --input data/judged_teacher.jsonl \
  --output data/accepted_teacher.jsonl \
  --min-score 4.0

# 4) Build train/val/test JSONL
python sft_pipeline/build_sft_dataset.py \
  --input data/accepted_teacher.jsonl \
  --train-out data/sft_train.jsonl \
  --val-out data/sft_val.jsonl \
  --test-out data/sft_test.jsonl \
  --seed 42

# 5) Train LoRA adapter
python sft_pipeline/train_trl.py --config sft_pipeline/trl_config.yaml
```

## Known Gaps / Next Iteration
1. `eval/results/` currently has no persisted base-vs-finetuned comparison outputs; run `sft_pipeline/evaluate.py` to quantify behavior gains and regressions.
2. Harmonize base model reference in `trl_config.yaml` with the actually used model in adapter metadata to avoid future reproducibility ambiguity.
3. Add manual spot-audit on accepted samples for highest-risk policy topics before next training cycle.
