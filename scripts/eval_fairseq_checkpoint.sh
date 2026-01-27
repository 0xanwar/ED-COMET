#!/usr/bin/env bash
set -euo pipefail

# Allow KEY=VALUE args to act as env overrides.
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    export "$kv"
  fi
done

# Evaluate a fairseq checkpoint and append ROUGE to a log file.

FAIRSEQ_GENERATE=${FAIRSEQ_GENERATE:-fairseq-generate}
DATA_BIN=${DATA_BIN:-"./data-bin/atomic_bart"}
CKPT=${CKPT:-""}
SPLIT=${SPLIT:-"valid"}
OUT_DIR=${OUT_DIR:-"./results_fairseq"}
LOG_FILE=${LOG_FILE:-"$OUT_DIR/rouge_log.jsonl"}
BEAM=${BEAM:-1}
MAX_LEN_B=${MAX_LEN_B:-24}

if [[ -z "$CKPT" ]]; then
  echo "Set CKPT=/path/to/checkpoint.pt"
  exit 1
fi

mkdir -p "$OUT_DIR"

$FAIRSEQ_GENERATE "$DATA_BIN" \
  --path "$CKPT" \
  --source-lang source --target-lang target \
  --gen-subset "$SPLIT" \
  --beam "$BEAM" --max-len-b "$MAX_LEN_B" --max-len-a 0 \
  --results-path "$OUT_DIR"

python scripts/compute_rouge_from_fairseq.py \
  --generate-file "$OUT_DIR/generate-$SPLIT.txt" \
  --checkpoint "$CKPT" \
  --split "$SPLIT" \
  | tee -a "$LOG_FILE"
