#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def run(cmd, label: str) -> None:
    print(f"[{label}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end ED-VCR inference pipeline.")
    parser.add_argument("--vcr-jsonl", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ain-model", default="MBZUAI/AIN")
    parser.add_argument("--head-builder", choices=["ain", "question"], default="ain")
    parser.add_argument("--edcomet-model-dir", required=True)
    parser.add_argument("--edcomet-checkpoint", required=True)
    parser.add_argument("--edcomet-data-bin", required=True)
    parser.add_argument("--relations", default="xIntent,xNeed,xReact,xEffect,oReact,oEffect")
    parser.add_argument("--num-tails", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--scorer", choices=["sbert", "jaccard"], default="sbert")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--tags", default="<REGION=MENA> <COUNTRY=EGY>")
    parser.add_argument("--mode", choices=["generate", "score"], default="generate")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_jsonl = out_dir / "vcr_qa.jsonl"
    qa_heads = out_dir / "vcr_qa.heads.jsonl"
    comet_cache = out_dir / "edcomet_cache.jsonl"
    qa_aug = out_dir / "vcr_qa.aug.jsonl"
    qa_preds = out_dir / "vcr_qa.preds.jsonl"

    base_dir = Path(__file__).resolve().parent

    if not args.skip_existing or not qa_jsonl.exists():
        cmd = [
            "python",
            str(base_dir / "preprocess_vcr.py"),
            "--vcr-jsonl",
            args.vcr_jsonl,
            "--images-dir",
            args.images_dir,
            "--out-dir",
            str(out_dir),
        ]
        if args.max_examples:
            cmd += ["--max-examples", str(args.max_examples)]
        run(cmd, "preprocess")

    if not args.skip_existing or not qa_heads.exists():
        if args.head_builder == "ain":
            cmd = [
                "python",
                str(base_dir / "build_heads_from_ain.py"),
                "--input",
                str(qa_jsonl),
                "--output",
                str(qa_heads),
                "--model",
                args.ain_model,
            ]
        else:
            cmd = [
                "python",
                str(base_dir / "build_heads_from_questions.py"),
                "--input",
                str(qa_jsonl),
                "--output",
                str(qa_heads),
            ]
        run(cmd, "heads")

    if not args.skip_existing or not comet_cache.exists():
        cmd = [
            "python",
            str(base_dir / "generate_edcomet_cache_fairseq.py"),
            "--input",
            str(qa_heads),
            "--output",
            str(comet_cache),
            "--model-dir",
            args.edcomet_model_dir,
            "--checkpoint-file",
            args.edcomet_checkpoint,
            "--data-bin",
            args.edcomet_data_bin,
            "--relations",
            args.relations,
            "--num-tails",
            str(args.num_tails),
            "--tags",
            args.tags,
        ]
        run(cmd, "edcomet")

    if not args.skip_existing or not qa_aug.exists():
        cmd = [
            "python",
            str(base_dir / "inject_edcomet.py"),
            "--input",
            str(qa_heads),
            "--output",
            str(qa_aug),
            "--comet-cache",
            str(comet_cache),
            "--relations",
            args.relations,
            "--k",
            str(args.k),
            "--scorer",
            args.scorer,
            "--sbert-model",
            args.sbert_model,
            "--tags",
            args.tags,
        ]
        run(cmd, "inject")

    cmd = [
        "python",
        str(base_dir / "ain_infer_vcr.py"),
        "--input",
        str(qa_aug),
        "--output",
        str(qa_preds),
        "--model",
        args.ain_model,
        "--mode",
        args.mode,
    ]
    run(cmd, "infer")

    print(f"Done. Predictions: {qa_preds}")


if __name__ == "__main__":
    main()
