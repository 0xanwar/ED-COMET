#!/usr/bin/env python3
"""Wrapper to allowlist safe globals before calling fairseq-interactive."""
import argparse
import sys

import torch

if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

from fairseq_cli.interactive import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
