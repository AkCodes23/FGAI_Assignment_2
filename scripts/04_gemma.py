#!/usr/bin/env python
"""Gemma-family system: Gemma 3 1B instruction-tuned (Gemma3ForCausalLM)."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(Path(__file__).with_name("run_decoder.py")), "--model", "gemma", *sys.argv[1:]]))
