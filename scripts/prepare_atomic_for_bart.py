#!/usr/bin/env python3
"""
Convert ATOMIC2020 TSV files to .source and .target format for BART finetuning.

The ATOMIC TSV format has columns: head_event, relation, tail_event
Output format:
- .source: "{head_event} [{relation}]"
- .target: "{tail_event}"

Usage:
    python prepare_atomic_for_bart.py --input_dir /path/to/atomic2020_data-feb2021 --output_dir ./data/atomic_bart
"""

import argparse
import csv
from pathlib import Path


def convert_tsv_to_bart_format(input_tsv_path, output_source_path, output_target_path, add_gen_token=False):
    """
    Convert a single TSV file to .source and .target files.

    Args:
        input_tsv_path: Path to input TSV file (e.g., train.tsv)
        output_source_path: Path to output .source file
        output_target_path: Path to output .target file
        add_gen_token: Whether to add [GEN] token (for demo model compatibility)
    """
    source_lines = []
    target_lines = []

    with open(input_tsv_path, 'r', encoding='utf-8') as f:
        # First, try to detect if there's a header
        first_line = f.readline()
        f.seek(0)

        # Check if file has header by looking at first line
        has_header = 'head' in first_line.lower() or 'relation' in first_line.lower()

        if has_header:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Handle different possible column names
                head_event = row.get('head', row.get('head_event', '')).strip()
                relation = row.get('relation', '').strip()
                tail_event = row.get('tail', row.get('tail_event', '')).strip()

                # Skip empty rows
                if not head_event or not relation or not tail_event:
                    continue

                # Format: "head_event relation" or "head_event relation [GEN]"
                if add_gen_token:
                    source_line = f"{head_event} {relation} [GEN]"
                else:
                    source_line = f"{head_event} {relation}"
                target_line = tail_event

                source_lines.append(source_line)
                target_lines.append(target_line)
        else:
            # No header, assume columns: head, relation, tail
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue

                head_event = row[0].strip()
                relation = row[1].strip()
                tail_event = row[2].strip()

                # Skip empty rows
                if not head_event or not relation or not tail_event:
                    continue

                # Format: "head_event relation" or "head_event relation [GEN]"
                if add_gen_token:
                    source_line = f"{head_event} {relation} [GEN]"
                else:
                    source_line = f"{head_event} {relation}"
                target_line = tail_event

                source_lines.append(source_line)
                target_lines.append(target_line)

    # Write source file
    with open(output_source_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_lines))

    # Write target file
    with open(output_target_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_lines))

    print(f"Converted {input_tsv_path}")
    print(f"  -> {len(source_lines)} examples written to {output_source_path} and {output_target_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert ATOMIC2020 TSV to BART format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing train.tsv, dev.tsv, test.tsv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .source and .target files')
    parser.add_argument('--add_gen_token', action='store_true',
                        help='Add [GEN] token at the end of source (for demo model compatibility)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each split
    for split in ['train', 'dev', 'test']:
        input_tsv = input_dir / f'{split}.tsv'

        # Note: dev.tsv maps to val.source/val.target for the training script
        output_split = 'val' if split == 'dev' else split

        output_source = output_dir / f'{output_split}.source'
        output_target = output_dir / f'{output_split}.target'

        if input_tsv.exists():
            convert_tsv_to_bart_format(input_tsv, output_source, output_target, add_gen_token=args.add_gen_token)
        else:
            print(f"Warning: {input_tsv} not found, skipping...")

    print(f"\n✓ Conversion complete! Data ready at: {output_dir}")
    print(f"\nYou can now use this data for finetuning with:")
    print(f"  --data_dir {output_dir}")


if __name__ == '__main__':
    main()