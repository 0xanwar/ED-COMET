#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    from rouge_score import rouge_scorer
except ImportError as exc:
    raise SystemExit("Missing rouge_score. Install with: pip install rouge-score") from exc


def parse_generate_file(path: Path):
    refs = {}
    hyps = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("T-"):
            parts = line.split("\t")
            if len(parts) >= 2:
                idx = parts[0][2:]
                refs[idx] = parts[1]
        elif line.startswith("H-"):
            parts = line.split("\t")
            if len(parts) >= 3:
                idx = parts[0][2:]
                hyps[idx] = parts[2]
    return refs, hyps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-file", required=True, help="fairseq-generate output file")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path (for logging only)")
    parser.add_argument("--split", default="", help="Split name (for logging only)")
    args = parser.parse_args()

    gen_path = Path(args.generate_file)
    refs, hyps = parse_generate_file(gen_path)
    keys = sorted(set(refs).intersection(hyps))
    if not keys:
        raise SystemExit(f"No aligned pairs found in {gen_path}")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for k in keys:
        scores = scorer.score(refs[k], hyps[k])
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
    n = len(keys)

    result = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "count": n,
        "rouge1": r1 / n,
        "rouge2": r2 / n,
        "rougeL": rl / n,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
