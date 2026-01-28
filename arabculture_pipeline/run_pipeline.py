#!/usr/bin/env python3
"""Generate ATOMIC-style data from ArabCulture with Jais (heads) + Qwen (tails)."""
import argparse
import gc
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover - runtime import for user env
    raise SystemExit(
        "vLLM is required. Install it (e.g., `pip install vllm`) before running."
    ) from exc

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

GENERIC_PHRASES = {
    "feel happy",
    "feel sad",
    "feel upset",
    "feel angry",
    "feel good",
    "feel bad",
    "be happy",
    "be sad",
    "be upset",
    "be angry",
    "be okay",
    "be fine",
    "feel okay",
    "feel fine",
    "none",
}

COUNTRY_CODE = {
    "Egypt": "EGY",
    "Jordan": "JOR",
    "Palestine": "PSE",
    "KSA": "KSA",
    "Saudi Arabia": "KSA",
    "UAE": "UAE",
    "United Arab Emirates": "UAE",
    "Morocco": "MAR",
    "Tunisia": "TUN",
    "Sudan": "SDN",
}


@dataclass
class Example:
    sample_id: str
    country: str
    region: str
    sub_topic: str
    scenario: str


@dataclass
class HeadResult:
    example: Example
    head: str
    tags: str


@dataclass
class TailResult:
    head_result: HeadResult
    tails: Dict[str, List[str]]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


GENERIC_NORM = {normalize_text(x) for x in GENERIC_PHRASES}


def jaccard_overlap(a: str, b: str) -> float:
    def tokens(s: str) -> set:
        return {t for t in normalize_text(s).split() if t not in {"personx", "persony", "personz"}}

    a_set = tokens(a)
    b_set = tokens(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def extract_json(text: str) -> Optional[dict]:
    # Try brace-balanced extraction first (more robust than regex-only).
    candidates = []
    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    # Fallback to non-greedy regex if no balanced blocks.
    if not candidates:
        candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def format_chat(tokenizer, system: str, user: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return f"{system}\n\n{user}\n"


def load_arabculture(
    countries: Iterable[str],
    keep_country_yes: bool,
    keep_discard_no: bool,
) -> List[Example]:
    examples: List[Example] = []
    for country in countries:
        ds = load_dataset("MBZUAI/ArabCulture", country, split="test")
        for row in ds:
            if keep_discard_no and str(row.get("should_discard", "No")).strip() != "No":
                continue
            if keep_country_yes and str(row.get("relevant_to_this_country", "Yes")).strip() != "Yes":
                continue
            options = row.get("options", {}) or {}
            answer_key = row.get("answer_key", {}) or {}
            arabic_key = str(answer_key.get("arabic_answer_key", "")).strip()
            english_key = str(answer_key.get("english_answer_key", "")).strip()
            arabic_keys = options.get("arabic_keys", []) or []
            english_keys = options.get("english_keys", []) or []
            option_texts = options.get("text", []) or []
            correct_opt = ""
            if arabic_key and arabic_keys and arabic_key in arabic_keys:
                idx = arabic_keys.index(arabic_key)
                if idx < len(option_texts):
                    correct_opt = option_texts[idx]
            elif english_key and english_keys and english_key in english_keys:
                idx = english_keys.index(english_key)
                if idx < len(option_texts):
                    correct_opt = option_texts[idx]
            scenario = row.get("first_statement", "").strip()
            if correct_opt:
                scenario = f"{scenario}. {correct_opt.strip()}"
            examples.append(
                Example(
                    sample_id=str(row.get("sample_id", "")),
                    country=str(row.get("country", country)),
                    region=str(row.get("region", "")),
                    sub_topic=str(row.get("sub_topic", "")),
                    scenario=scenario,
                )
            )
    return examples


def build_tags(country: str, region: str, add_region: bool, add_country: bool) -> str:
    tags = []
    if add_region:
        tags.append("<REGION=MENA>")
    if add_country:
        code = COUNTRY_CODE.get(country, country[:3].upper())
        tags.append(f"<COUNTRY={code}>")
    return " ".join(tags)


def make_llm(model_id: str, tp_size: int, max_model_len: int, gpu_mem_util: float):
    return LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
    )


def generate_heads(
    examples: List[Example],
    model_id: str,
    tp_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    add_region: bool,
    add_country: bool,
    temperature: float,
) -> List[HeadResult]:
    system = (
        "You convert Arabic scenarios into ATOMIC-style English heads."
    )
    user_tpl = (
        "Convert this Arabic scenario into ONE ATOMIC-style head event in English.\n\n"
        "Country: {country}\n"
        "Scenario (MSA): {scenario}\n\n"
        "Return ONLY JSON:\n"
        "{{\"head\":\"PersonX ...\"}}\n\n"
        "Rules:\n"
        "- One event, one sentence.\n"
        "- No stereotypes, no moral judgement.\n"
        "- Include cultural details only if explicitly implied.\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = make_llm(model_id, tp_size, max_model_len, gpu_mem_util)
    prompts = []
    for ex in examples:
        user = user_tpl.format(country=ex.country, scenario=ex.scenario)
        prompts.append(format_chat(tokenizer, system, user))

    params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=128)
    outputs = llm.generate(prompts, params)

    results: List[HeadResult] = []
    for ex, out in zip(examples, outputs):
        text = out.outputs[0].text
        obj = extract_json(text)
        if not obj or "head" not in obj:
            continue
        head = obj["head"].strip()
        if head:
            head = head.strip()
            if not head.lower().startswith("personx"):
                head = "PersonX " + re.sub(r"^personx\s*", "", head, flags=re.I).strip()
        if not head:
            continue
        tags = build_tags(ex.country, ex.region, add_region, add_country)
        results.append(HeadResult(example=ex, head=head, tags=tags))

    del llm
    gc.collect()
    return results


def build_tails_prompt(head: str, tags: str, n_tails: int) -> str:
    return (
        "Generate ATOMIC inferences for the head.\n\n"
        f"HEAD: {head}\n"
        f"TAGS: {tags}\n\n"
        "Return ONLY JSON with keys:\n"
        "xIntent,xNeed,xEffect,xReact,xWant,xAttr,oEffect,oReact,oWant\n\n"
        f"Constraints:\n- Exactly {n_tails} tails per key.\n"
        "- Short (3-10 words), verb phrases preferred.\n"
        "- Distinct tails; no paraphrases of the head.\n"
        "- Avoid generic tails (happy/sad) unless grounded.\n"
        "- No \"none\".\n"
    )


def filter_tails(head: str, tails: Dict[str, List[str]], min_tails: int, overlap: float) -> Optional[Dict[str, List[str]]]:
    cleaned: Dict[str, List[str]] = {}
    for rel in RELATIONS:
        rel_tails = []
        seen_norm = set()
        for t in tails.get(rel, []):
            t = t.strip()
            if not t:
                continue
            if normalize_text(t) in GENERIC_NORM:
                continue
            if jaccard_overlap(head, t) >= overlap:
                continue
            if t.lower() == "none":
                continue
            norm = normalize_text(t)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            if t not in rel_tails:
                rel_tails.append(t)
        if len(rel_tails) < min_tails:
            return None
        cleaned[rel] = rel_tails[:min_tails]
    return cleaned


def filter_tails_partial(head: str, tails: Dict[str, List[str]], overlap: float) -> Dict[str, List[str]]:
    cleaned: Dict[str, List[str]] = {}
    for rel in RELATIONS:
        rel_tails = []
        seen_norm = set()
        for t in tails.get(rel, []) or []:
            t = t.strip()
            if not t:
                continue
            if normalize_text(t) in GENERIC_NORM:
                continue
            if jaccard_overlap(head, t) >= overlap:
                continue
            if t.lower() == "none":
                continue
            norm = normalize_text(t)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            rel_tails.append(t)
        if rel_tails:
            cleaned[rel] = rel_tails
    return cleaned


def generate_tails(
    heads: List[HeadResult],
    model_id: str,
    tp_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    base_tails: int,
    extra_tail: bool,
    overlap: float,
) -> List[TailResult]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = make_llm(model_id, tp_size, max_model_len, gpu_mem_util)

    def run_batch(prompts: List[str], temperature: float) -> List[dict]:
        max_tokens = 700 if temperature <= 0.6 else 300
        params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens)
        outputs = llm.generate(prompts, params)
        parsed = []
        for out in outputs:
            text = out.outputs[0].text
            obj = extract_json(text)
            parsed.append(obj or {})
        return parsed

    prompts = [
        format_chat(
            tokenizer,
            "You are a careful commonsense generator.",
            build_tails_prompt(h.head, h.tags, base_tails),
        )
        for h in heads
    ]

    base_objs = run_batch(prompts, temperature=0.6)
    results: List[TailResult] = []

    good_heads: List[HeadResult] = []
    for head, obj in zip(heads, base_objs):
        if not obj:
            continue
        filtered = filter_tails(head.head, obj, min_tails=base_tails, overlap=overlap)
        if not filtered:
            continue
        results.append(TailResult(head_result=head, tails=filtered))
        good_heads.append(head)

    if extra_tail and good_heads:
        extra_prompts = [
            format_chat(
                tokenizer,
                "You add one more tail per relation.",
                build_tails_prompt(h.head, h.tags, 1),
            )
            for h in good_heads
        ]
        extra_objs = run_batch(extra_prompts, temperature=0.7)
        by_key = {hr.head_result.head.lower(): hr for hr in results}
        for idx, obj in enumerate(extra_objs):
            if not obj:
                continue
            filtered = filter_tails_partial(good_heads[idx].head, obj, overlap=overlap)
            if not filtered:
                continue
            # merge
            key = good_heads[idx].head.lower()
            target = by_key[key].tails
            for rel, rel_tails in filtered.items():
                for t in rel_tails[:1]:
                    if t not in target[rel]:
                        target[rel].append(t)

    del llm
    gc.collect()
    return results


def write_heads(path: str, heads: List[HeadResult]):
    with open(path, "w", encoding="utf-8") as f:
        for item in heads:
            f.write(
                json.dumps(
                    {
                        "head": item.head,
                        "tags": item.tags,
                        "sample_id": item.example.sample_id,
                        "country": item.example.country,
                        "region": item.example.region,
                        "sub_topic": item.example.sub_topic,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_heads(path: str) -> List[HeadResult]:
    heads: List[HeadResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ex = Example(
                sample_id=obj.get("sample_id", ""),
                country=obj.get("country", ""),
                region=obj.get("region", ""),
                sub_topic=obj.get("sub_topic", ""),
                scenario="",
            )
            heads.append(HeadResult(example=ex, head=obj["head"], tags=obj.get("tags", "")))
    return heads


def dedupe_by_head(items: List[TailResult]) -> List[TailResult]:
    seen = set()
    out = []
    for item in items:
        key = item.head_result.head.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def split_data(items: List[TailResult], val_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    val_n = int(n * val_frac)
    test_n = int(n * test_frac)
    test = items[:test_n]
    val = items[test_n : test_n + val_n]
    train = items[test_n + val_n :]
    return train, val, test


def write_jsonl(path: str, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_atomic_bart_split(path_prefix: str, items: List[TailResult]):
    src_path = path_prefix + ".source"
    tgt_path = path_prefix + ".target"
    with open(src_path, "w", encoding="utf-8") as src_f, open(
        tgt_path, "w", encoding="utf-8"
    ) as tgt_f:
        for item in items:
            tags = item.head_result.tags
            head = item.head_result.head
            for rel in RELATIONS:
                for tail in item.tails[rel]:
                    prefix = f"{tags} " if tags else ""
                    src_f.write(f"{prefix}{head} {rel}\n")
                    tgt_f.write(f"{tail}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--head-model", required=True)
    parser.add_argument("--tail-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--keep-country-yes", action="store_true")
    parser.add_argument("--keep-discard-no", action="store_true")
    parser.add_argument("--add-region-tag", action="store_true")
    parser.add_argument("--add-country-tag", action="store_true")
    parser.add_argument("--base-tails", type=int, default=2)
    parser.add_argument("--add-third-tail", action="store_true")
    parser.add_argument("--overlap-threshold", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads-only", action="store_true")
    parser.add_argument("--tails-only", action="store_true")
    parser.add_argument("--heads-file", default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tails_only and not args.heads_file:
        raise SystemExit("--tails-only requires --heads-file")

    if args.tails_only:
        heads = read_heads(args.heads_file)
    else:
        examples = load_arabculture(
            args.countries, args.keep_country_yes, args.keep_discard_no
        )
        heads = generate_heads(
            examples,
            model_id=args.head_model,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            add_region=args.add_region_tag,
            add_country=args.add_country_tag,
            temperature=0.6,
        )
        heads_file = args.heads_file or os.path.join(args.output_dir, "heads.jsonl")
        write_heads(heads_file, heads)

    if args.heads_only:
        summary = {
            "countries": args.countries,
            "heads": len(heads),
            "heads_file": args.heads_file or os.path.join(args.output_dir, "heads.jsonl"),
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return

    tails = generate_tails(
        heads,
        model_id=args.tail_model,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        base_tails=args.base_tails,
        extra_tail=args.add_third_tail,
        overlap=args.overlap_threshold,
    )

    tails = dedupe_by_head(tails)

    train, val, test = split_data(tails, args.val_frac, args.test_frac, args.seed)

    # Save rich JSONL
    def to_row(item: TailResult) -> dict:
        row = {"event": item.head_result.head}
        row.update(item.tails)
        row["meta"] = {
            "sample_id": item.head_result.example.sample_id,
            "country": item.head_result.example.country,
            "region": item.head_result.example.region,
            "sub_topic": item.head_result.example.sub_topic,
            "tags": item.head_result.tags,
        }
        return row

    write_jsonl(os.path.join(args.output_dir, "atomic_full.train.jsonl"), [to_row(x) for x in train])
    write_jsonl(os.path.join(args.output_dir, "atomic_full.val.jsonl"), [to_row(x) for x in val])
    write_jsonl(os.path.join(args.output_dir, "atomic_full.test.jsonl"), [to_row(x) for x in test])

    # Write atomic-bart splits
    bart_dir = os.path.join(args.output_dir, "atomic_bart")
    os.makedirs(bart_dir, exist_ok=True)
    write_atomic_bart_split(os.path.join(bart_dir, "train"), train)
    write_atomic_bart_split(os.path.join(bart_dir, "val"), val)
    write_atomic_bart_split(os.path.join(bart_dir, "test"), test)

    summary = {
        "countries": args.countries,
        "heads": len(heads),
        "tails": len(tails),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
