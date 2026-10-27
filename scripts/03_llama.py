#!/usr/bin/env python
"""LLaMA-family system: TinyLlama-1.1B-Chat (LlamaForCausalLM, Apache-2.0)."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(Path(__file__).with_name("run_decoder.py")), "--model", "llama", *sys.argv[1:]]))
