#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sft_pipeline/export_gguf.sh \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --adapter-dir outputs \
    --out-dir outputs/gguf_export \
    --quant q4_k_m

What this does:
1) Merge LoRA adapter into the base HF model.
2) Convert merged model to f16 GGUF with llama.cpp.
3) Quantize GGUF (optional; defaults to q4_k_m).

Arguments:
  --base-model   Required. HF model id or local HF model path.
  --adapter-dir  LoRA adapter dir (default: outputs).
  --out-dir      Export directory (default: outputs/gguf_export).
  --llama-cpp    llama.cpp checkout path (default: <out-dir>/llama.cpp).
  --quant        Quant type for llama-quantize (default: q4_k_m).
                 Use "none" to skip quantization.
  -h, --help     Show this help.

Prerequisites:
  - Python packages: transformers, peft, torch, safetensors
  - Build tools for llama.cpp: git, cmake, c++ compiler
  - If base model is gated: HF_TOKEN exported in shell
EOF
}

BASE_MODEL=""
ADAPTER_DIR="outputs"
OUT_DIR="outputs/gguf_export"
LLAMA_CPP_DIR=""
QUANT="q4_k_m"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="${2:-}"
      shift 2
      ;;
    --adapter-dir)
      ADAPTER_DIR="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --llama-cpp)
      LLAMA_CPP_DIR="${2:-}"
      shift 2
      ;;
    --quant)
      QUANT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_MODEL" ]]; then
  echo "Missing required --base-model" >&2
  usage
  exit 1
fi

if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" || ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
  echo "Adapter files not found in ${ADAPTER_DIR}" >&2
  echo "Expected: adapter_config.json and adapter_model.safetensors" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
MERGED_DIR="${OUT_DIR}/merged_model"
GGUF_F16="${OUT_DIR}/model-f16.gguf"

if [[ -z "${LLAMA_CPP_DIR}" ]]; then
  LLAMA_CPP_DIR="${OUT_DIR}/llama.cpp"
fi

echo "[1/4] Merging LoRA adapter into base model..."
python - <<PY
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "${BASE_MODEL}"
adapter = "${ADAPTER_DIR}"
out = "${MERGED_DIR}"

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.save_pretrained(out)
print(f"Merged model saved to: {out}")
PY

echo "[2/4] Preparing llama.cpp..."
if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
  git clone https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_DIR}"
fi

cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build"
cmake --build "${LLAMA_CPP_DIR}/build" -j

CONVERTER="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANT_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

if [[ ! -f "${CONVERTER}" ]]; then
  echo "Could not find converter script: ${CONVERTER}" >&2
  exit 1
fi
if [[ ! -x "${QUANT_BIN}" && "${QUANT}" != "none" ]]; then
  echo "Could not find quantizer binary: ${QUANT_BIN}" >&2
  exit 1
fi

echo "[3/4] Converting merged model to f16 GGUF..."
python "${CONVERTER}" "${MERGED_DIR}" --outfile "${GGUF_F16}" --outtype f16

if [[ "${QUANT}" == "none" ]]; then
  echo "[4/4] Quantization skipped (--quant none)."
  echo "Done: ${GGUF_F16}"
  exit 0
fi

GGUF_Q="${OUT_DIR}/model-${QUANT}.gguf"
echo "[4/4] Quantizing GGUF (${QUANT})..."
"${QUANT_BIN}" "${GGUF_F16}" "${GGUF_Q}" "${QUANT}"

echo "Done:"
echo "  f16:   ${GGUF_F16}"
echo "  quant: ${GGUF_Q}"
