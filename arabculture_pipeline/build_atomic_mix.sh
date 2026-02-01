#!/usr/bin/env bash
set -euo pipefail

# Build a mixed training set: ATOMIC replay + upsampled ArabCulture.
# Edit paths below or override via environment variables.

ATOMIC_BART=${ATOMIC_BART:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/atomic_bart"}
ARAB_BART=${ARAB_BART:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/arabculture_atomic_combined/atomic_bart"}
OUT_DIR=${OUT_DIR:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/atomic_mix_arabculture_30"}
REPLAY_DIR=${REPLAY_DIR:-"/home/ahmedjaheen/Abdelrahman/GD-COMET-Replication/ED-COMET/data/atomic_replay_100k"}

UPSAMPLE=${UPSAMPLE:-7}          # ArabCulture multiplier (7 ~= 30%, 11 ~= 40%)
REPLAY_PER_REL=${REPLAY_PER_REL:-11000}
SEED=${SEED:-42}

mkdir -p "$OUT_DIR"
mkdir -p "$REPLAY_DIR"

python - <<PY
import os
import random
from collections import defaultdict

random.seed($SEED)

atomic_src = "$ATOMIC_BART/train.source"
atomic_tgt = "$ATOMIC_BART/train.target"
out_dir = "$REPLAY_DIR"

rels = ["xIntent","xNeed","xEffect","xReact","xWant","xAttr","oEffect","oReact","oWant"]
n_per = int($REPLAY_PER_REL)
buckets = {r: [] for r in rels}
seen = defaultdict(int)

with open(atomic_src, "r", encoding="utf-8") as fs, open(atomic_tgt, "r", encoding="utf-8") as ft:
    for s, t in zip(fs, ft):
        parts = s.strip().split()
        if not parts:
            continue
        rel = parts[-1]
        if rel not in buckets:
            continue
        seen[rel] += 1
        bucket = buckets[rel]
        if len(bucket) < n_per:
            bucket.append((s, t))
        else:
            j = random.randint(1, seen[rel])
            if j <= n_per:
                bucket[j - 1] = (s, t)

with open(os.path.join(out_dir, "train.source"), "w", encoding="utf-8") as osrc, \
     open(os.path.join(out_dir, "train.target"), "w", encoding="utf-8") as otgt:
    for r in rels:
        for s, t in buckets[r]:
            osrc.write(s)
            otgt.write(t)

print("Replay lines:", sum(len(b) for b in buckets.values()))
PY

cat "$REPLAY_DIR/train.source" > "$OUT_DIR/train.source"
cat "$REPLAY_DIR/train.target" > "$OUT_DIR/train.target"

for i in $(seq 1 "$UPSAMPLE"); do
  cat "$ARAB_BART/train.source" >> "$OUT_DIR/train.source"
  cat "$ARAB_BART/train.target" >> "$OUT_DIR/train.target"
done

paste "$OUT_DIR/train.source" "$OUT_DIR/train.target" | shuf > "$OUT_DIR/train.paired.tsv"
cut -f1 "$OUT_DIR/train.paired.tsv" > "$OUT_DIR/train.source"
cut -f2 "$OUT_DIR/train.paired.tsv" > "$OUT_DIR/train.target"
rm "$OUT_DIR/train.paired.tsv"

cp "$ATOMIC_BART/val.source"  "$OUT_DIR/val.source"
cp "$ATOMIC_BART/val.target"  "$OUT_DIR/val.target"
cp "$ATOMIC_BART/test.source" "$OUT_DIR/test.source"
cp "$ATOMIC_BART/test.target" "$OUT_DIR/test.target"

# Optional: keep separate ArabCulture val/test for evaluation.
if [[ -f "$ARAB_BART/val.source" && -f "$ARAB_BART/val.target" ]]; then
  cp "$ARAB_BART/val.source"  "$OUT_DIR/val_arab.source"
  cp "$ARAB_BART/val.target"  "$OUT_DIR/val_arab.target"
fi
if [[ -f "$ARAB_BART/test.source" && -f "$ARAB_BART/test.target" ]]; then
  cp "$ARAB_BART/test.source"  "$OUT_DIR/test_arab.source"
  cp "$ARAB_BART/test.target"  "$OUT_DIR/test_arab.target"
fi

echo "Done. Mixed dataset at $OUT_DIR"
