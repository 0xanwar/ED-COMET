#!/usr/bin/env python3
"""Build atomic_bart .source/.target from atomic_full jsonl splits."""
import argparse
import json
import os


RELATIONS = [
    "xIntent",
    "xNeed",
    "xEffect",
    "xReact",
    "xWant",
    "xAttr",
    "oEffect",
    "oReact",
    "oWant",
]


def select_path(input_dir: str, split: str, use_repaired: bool) -> str:
    if use_repaired:
        repaired = os.path.join(input_dir, f"atomic_full.{split}.repaired.jsonl")
        if os.path.exists(repaired):
            return repaired
    return os.path.join(input_dir, f"atomic_full.{split}.jsonl")


def write_split(input_path: str, output_prefix: str) -> None:
    src_path = output_prefix + ".source"
    tgt_path = output_prefix + ".target"
    with open(input_path, "r", encoding="utf-8") as src_f, open(
        src_path, "w", encoding="utf-8"
    ) as out_src, open(tgt_path, "w", encoding="utf-8") as out_tgt:
        for line in src_f:
            if not line.strip():
                continue
            obj = json.loads(line)
            head = obj.get("event", "").strip()
            meta = obj.get("meta", {}) or {}
            tags = meta.get("tags", "")
            prefix = f"{tags} " if tags else ""
            for rel in RELATIONS:
                for tail in obj.get(rel, []) or []:
                    out_src.write(f"{prefix}{head} {rel}\n")
                    out_tgt.write(f"{tail}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-repaired", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "atomic_bart")
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        input_path = select_path(input_dir, split, args.use_repaired)
        if not os.path.exists(input_path):
            raise SystemExit(f"Missing split file: {input_path}")
        output_prefix = os.path.join(output_dir, split)
        write_split(input_path, output_prefix)

    print(f"Wrote atomic_bart to {output_dir}")


if __name__ == "__main__":
    main()
