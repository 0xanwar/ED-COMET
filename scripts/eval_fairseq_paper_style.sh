#!/usr/bin/env bash
set -euo pipefail

# Allow KEY=VALUE args to act as env overrides.
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    export "$kv"
  fi
done

# Paper-style evaluation (system_eval) for a fairseq checkpoint.

FAIRSEQ_INTERACTIVE=${FAIRSEQ_INTERACTIVE:-fairseq-interactive}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA_BIN=${DATA_BIN:-"./data-bin/atomic_bart"}
CKPT=${CKPT:-""}
SYSTEM_EVAL_DIR=${SYSTEM_EVAL_DIR:-"./system_eval"}
OUT_DIR=${OUT_DIR:-"./results_fairseq"}
LOG_FILE=${LOG_FILE:-"$OUT_DIR/paper_eval_log.jsonl"}
BEAM=${BEAM:-1}
MAX_LEN_B=${MAX_LEN_B:-24}
FORCE_BUILD=${FORCE_BUILD:-0}

if [[ -z "$CKPT" ]]; then
  echo "Set CKPT=/path/to/checkpoint.pt"
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$SYSTEM_EVAL_DIR/results"

TEST_SOURCE="$OUT_DIR/test.source"
if [[ ! -f "$TEST_SOURCE" || "$FORCE_BUILD" -eq 1 ]]; then
  TEST_SOURCE="$TEST_SOURCE" SYSTEM_EVAL_DIR="$SYSTEM_EVAL_DIR" $PYTHON_BIN - <<'PY'
from pathlib import Path
import os

system_eval_dir = Path(os.environ.get("SYSTEM_EVAL_DIR", "./system_eval"))
out_path = Path(os.environ.get("TEST_SOURCE", "./results_fairseq/test.source"))
inp = system_eval_dir / "test.tsv"

out_path.parent.mkdir(parents=True, exist_ok=True)
with inp.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as o:
    for line in f:
        head_rel = line.split("\t")[0].strip()
        head, rel = [x.strip() for x in head_rel.split("@@")]
        o.write(f"{head} {rel}\n")
PY
fi

CKPT_BASE=$(basename "$CKPT" .pt)
GEN_FILE="$OUT_DIR/generate-${CKPT_BASE}.txt"
JSONL_FILE="$OUT_DIR/${CKPT_BASE}.jsonl"

$FAIRSEQ_INTERACTIVE "$DATA_BIN" \
  --path "$CKPT" \
  --source-lang source --target-lang target \
  --input "$TEST_SOURCE" \
  --beam "$BEAM" --max-len-b "$MAX_LEN_B" --max-len-a 0 \
  > "$GEN_FILE"

GEN_FILE="$GEN_FILE" JSONL_FILE="$JSONL_FILE" $PYTHON_BIN - <<'PY'
import json
import os
from pathlib import Path

gen_path = Path(os.environ["GEN_FILE"])
out_path = Path(os.environ["JSONL_FILE"])

hyps = {}
for line in gen_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("H-"):
        idx, _, text = line.split("\t", 2)
        hyps[int(idx[2:])] = text

with out_path.open("w", encoding="utf-8") as f:
    for i in range(len(hyps)):
        f.write(json.dumps({"greedy": hyps[i]}) + "\n")
PY

pushd "$SYSTEM_EVAL_DIR" >/dev/null
$PYTHON_BIN automatic_eval.py --input_file "$JSONL_FILE"
popd >/dev/null

RESULT_FILE="$SYSTEM_EVAL_DIR/results/${CKPT_BASE}_scores.jsonl"
RESULT_FILE="$RESULT_FILE" LOG_FILE="$LOG_FILE" CKPT="$CKPT" $PYTHON_BIN - <<'PY'
import json
import os
from pathlib import Path

result_file = Path(os.environ["RESULT_FILE"])
log_file = Path(os.environ["LOG_FILE"])
ckpt = os.environ["CKPT"]

if not result_file.exists():
    raise SystemExit(f"Missing result file: {result_file}")

data = result_file.read_text(encoding="utf-8").splitlines()
if not data:
    raise SystemExit(f"Empty result file: {result_file}")

entry = json.loads(data[0])
entry["checkpoint"] = ckpt
log_file.parent.mkdir(parents=True, exist_ok=True)
with log_file.open("a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")
print(f"Logged: {log_file}")
PY
