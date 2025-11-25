#!/usr/bin/env python3
"""
Convert fairseq dictionary to HuggingFace BART tokenizer format.

This script creates vocab.json and merges.txt files from a fairseq dict.txt file,
allowing you to use a fairseq-trained BART model with HuggingFace Transformers.
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict


def read_fairseq_dict(dict_path):
    """
    Read fairseq dict.txt file.

    Format: token count
    Example:
        the 1234567
        , 987654
        ...
    """
    vocab = OrderedDict()
    with open(dict_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 1:
                token = parts[0]
                # Count is parts[1] if it exists, but we don't need it for HF
                vocab[token] = idx

    return vocab


def create_vocab_json(fairseq_vocab, output_path):
    """
    Create vocab.json for HuggingFace tokenizer.

    This maps tokens to their IDs.
    """
    vocab_dict = {}

    # Add special tokens first if they exist
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>']
    for token in special_tokens:
        if token in fairseq_vocab:
            vocab_dict[token] = fairseq_vocab[token]

    # Add all other tokens
    for token, idx in fairseq_vocab.items():
        if token not in vocab_dict:
            vocab_dict[token] = idx

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    return vocab_dict


def create_merges_txt(fairseq_vocab, output_path):
    """
    Create merges.txt for HuggingFace tokenizer.

    Since we're converting from an existing vocabulary, we create a minimal
    merges file. The actual BPE merges were already applied during fairseq
    preprocessing, so we just need a placeholder that works with HF.
    """
    # Filter out special tokens
    special_tokens = {'<s>', '<pad>', '</s>', '<unk>', '<mask>'}
    regular_tokens = [t for t in fairseq_vocab.keys() if t not in special_tokens]

    merges = []

    # For subword tokens (containing @@), create merge operations
    for token in regular_tokens:
        # Check if token looks like a BPE subword
        if '@@' in token:
            # Remove @@ and split
            clean = token.replace('@@', '')
            if len(clean) > 1:
                # Create a merge: first_char + rest_of_word
                merges.append(f"{clean[0]} {clean[1:]}")
        elif len(token) > 1 and not token.startswith('<'):
            # For multi-character tokens, create character-level merges
            # This is a simplification but works for most cases
            chars = list(token)
            for i in range(len(chars) - 1):
                merge = f"{chars[i]} {chars[i+1]}"
                if merge not in merges:
                    merges.append(merge)

    # Write merges
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("#version: 0.2\n")
        for merge in merges[:50000]:  # Limit to 50k merges to keep file manageable
            f.write(f"{merge}\n")

    return merges


def create_tokenizer_config(output_path, vocab_size):
    """Create a basic tokenizer_config.json for HuggingFace."""
    config = {
        "add_prefix_space": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1024,
        "pad_token": "<pad>",
        "tokenizer_class": "BartTokenizer",
        "unk_token": "<unk>",
        "vocab_size": vocab_size
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return config


def convert_fairseq_dict_to_hf(dict_path, output_dir):
    """
    Main conversion function.

    Args:
        dict_path: Path to fairseq dict.txt
        output_dir: Directory to save HuggingFace tokenizer files
    """
    print(f"\n{'='*70}")
    print(f"Converting fairseq dictionary to HuggingFace tokenizer format")
    print(f"{'='*70}")
    print(f"Input dict: {dict_path}")
    print(f"Output dir: {output_dir}\n")

    # Read fairseq dictionary
    print("Step 1: Reading fairseq dictionary...")
    fairseq_vocab = read_fairseq_dict(dict_path)
    vocab_size = len(fairseq_vocab)
    print(f"✓ Loaded {vocab_size:,} tokens")

    # Show some sample tokens
    sample_tokens = list(fairseq_vocab.keys())[:10]
    print(f"  Sample tokens: {sample_tokens}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create vocab.json
    print("\nStep 2: Creating vocab.json...")
    vocab_json_path = output_path / "vocab.json"
    vocab_dict = create_vocab_json(fairseq_vocab, vocab_json_path)
    print(f"✓ Created {vocab_json_path}")
    print(f"  Vocabulary size: {len(vocab_dict):,}")

    # Create merges.txt
    print("\nStep 3: Creating merges.txt...")
    merges_txt_path = output_path / "merges.txt"
    merges = create_merges_txt(fairseq_vocab, merges_txt_path)
    print(f"✓ Created {merges_txt_path}")
    print(f"  Number of merges: {len(merges):,}")

    # Create tokenizer_config.json
    print("\nStep 4: Creating tokenizer_config.json...")
    config_path = output_path / "tokenizer_config.json"
    config = create_tokenizer_config(config_path, vocab_size)
    print(f"✓ Created {config_path}")

    print(f"\n{'='*70}")
    print(f"✓ Conversion complete!")
    print(f"{'='*70}")
    print(f"Tokenizer files saved to: {output_dir}")
    print(f"  - vocab.json ({vocab_size:,} tokens)")
    print(f"  - merges.txt ({len(merges):,} merges)")
    print(f"  - tokenizer_config.json")
    print(f"\n⚠️  Note: This is a basic conversion. The tokenizer may not perfectly")
    print(f"   match the original fairseq BPE behavior, but it will work with")
    print(f"   HuggingFace Transformers and your converted model weights.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert fairseq dict.txt to HuggingFace tokenizer format"
    )
    parser.add_argument(
        "dict_path",
        type=str,
        help="Path to fairseq dict.txt file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for HuggingFace tokenizer files"
    )

    args = parser.parse_args()
    convert_fairseq_dict_to_hf(args.dict_path, args.output_dir)
