#!/usr/bin/env python
"""Run decoder-only systems (LLaMA / Gemma): few-shot prompting and QLoRA adaptation."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from fgai_mt.config import dump_config, load_config, project_path  # noqa: E402
from fgai_mt.data import load_split  # noqa: E402
from fgai_mt.eval_sample import load_eval_rows  # noqa: E402
from fgai_mt.decoder import count_full_checkpoint_parameters, generate_texts, load_model, load_tokenizer, render_prompt, train_qlora  # noqa: E402
from fgai_mt.metrics import corpus_scores, output_diagnostics  # noqa: E402
from fgai_mt.utils import JsonlLogger, get_logger, set_seed, write_json, write_jsonl  # noqa: E402

RESULTS = project_path("results")
TRANSLATIONS = RESULTS / "translations"


def pick_shots(rows, n=3):
    """Deterministic, length-controlled in-context examples from the training split only."""
    candidates = [r for r in rows if 7 <= len(r["en"].split()) <= 18 and 7 <= len(r["hi"].split()) <= 22]
    shots = []
    used = set()
    for row in candidates:
        key = row["en"].lower()
        if key in used:
            continue
        used.add(key)
        shots.append(row)
        if len(shots) == n:
            break
    path = RESULTS / "prompt_shots.json"
    if not path.exists():
        write_json(path, {"selection_rule": "first n training rows with 7-18 source words", "shots": shots})
    return shots


def result_payload(system, variant, scores, diag, timing, params, cfg_used, n_test, extra=None):
    payload = {
        "system": system,
        "variant": variant,
        "metrics": scores,
        "diagnostics": diag,
        "timing": timing,
        "params_total": params,
        "config": cfg_used,
        "n_test": n_test,
    }
    if extra:
        payload.update(extra)
    return payload


def run_prompting(key: str, cfg: dict, split: str = "test", n: int | None = None, shots_n: int = 3,
                  tag: str | None = None, logger=None):
    spec = cfg["decoder"][key]
    dcfg = cfg["decoder"]
    params = count_full_checkpoint_parameters(spec["model_id"], spec["revision"])
    tok = load_tokenizer(spec["model_id"], spec["revision"])
    model = load_model(spec["model_id"], spec["revision"], load_in_4bit=dcfg["qlora"]["load_in_4bit"])
    if split == "test":
        indexed_rows = load_eval_rows(n)
    else:
        rows_for_split = load_split(split)
        indexed_rows = list(enumerate(rows_for_split[:n] if n else rows_for_split))
    test_indices = [i for i, _ in indexed_rows]
    rows = [row for _, row in indexed_rows]
    train_rows = load_split("train")
    shots = pick_shots(train_rows, shots_n) if shots_n else []
    prompts = [render_prompt(tok, row["en"], shots) for row in rows]
    gen = generate_texts(
        model,
        tok,
        prompts,
        max_new_tokens=dcfg["max_new_tokens"],
        batch_size=dcfg["batch_size"],
        temperature=None,
        max_input_tokens=dcfg["max_input_tokens"],
    )
    tag = tag or f"prompt{shots_n}shot"
    srcs = [r["en"] for r in rows]
    refs = [r["hi"] for r in rows]
    hyps = gen["texts"]
    scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
    diag = output_diagnostics(hyps, srcs)
    timing = {
        "inference_seconds": gen["seconds"],
        "sentences_per_second": gen["sentences_per_second"],
        "new_tokens": gen["new_tokens"],
        "tokens_per_second": gen["tokens_per_second"],
        "hit_max_new_tokens": gen["hit_max_new_tokens"],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "quantization": "4-bit NF4" if dcfg["qlora"]["load_in_4bit"] else "bf16",
        "batch_size": dcfg["batch_size"],
        "decoding": "greedy",
    }
    payload = result_payload(key, tag, scores, diag, timing, params, spec, len(hyps),
                             {"shots": shots_n, "split": split,
                              "adaptation": "prompt only; no parameter updates"})
    payload["parameter_count_method"] = "sum of original safetensors tensor shapes before 4-bit quantization"
    write_json(RESULTS / f"metrics_{key}_{tag}.json", payload)
    write_jsonl(TRANSLATIONS / f"{key}_{tag}.jsonl",
                [{"i": i, "src": s, "ref": r, "hyp": h}
                 for i, s, r, h in zip(test_indices, srcs, refs, hyps)])
    if logger:
        logger.info("[%s/%s] %s | %.1fs | %.2f sent/s", key, tag, json.dumps(scores), gen["seconds"],
                    gen["sentences_per_second"])
    del model
    torch.cuda.empty_cache()
    return payload


def run_qlora(key: str, cfg: dict, logger=None, jsonl=None, smoke: bool = False):
    spec = cfg["decoder"][key]
    qcfg = cfg["decoder"]["qlora"]
    out_dir = project_path("models", f"{key}_qlora")
    train_rows = load_split("train")
    limit = qcfg.get("train_limit")
    if limit and len(train_rows) > limit:
        train_rows = random.Random(cfg["seed"]).sample(train_rows, limit)
    summary = train_qlora(
        spec["model_id"],
        train_rows,
        load_split("validation"),
        out_dir,
        revision=spec["revision"],
        load_in_4bit=qcfg["load_in_4bit"],
        r=qcfg["r"],
        alpha=qcfg["alpha"],
        dropout=qcfg["dropout"],
        lr=qcfg["lr"],
        epochs=qcfg["epochs"],
        batch_size=qcfg["batch_size"],
        grad_accum=qcfg["grad_accum"],
        max_seq_len=qcfg["max_seq_len"],
        warmup_ratio=qcfg["warmup_ratio"],
        weight_decay=qcfg["weight_decay"],
        seed=cfg["seed"],
        logger=logger,
        jsonl=jsonl,
        smoke=smoke,
    )
    write_json(project_path("models", f"{key}_qlora", "training_summary.json"), summary)
    return summary


def evaluate_qlora(key: str, cfg: dict, logger=None, n: int | None = None):
    spec = cfg["decoder"][key]
    dcfg = cfg["decoder"]
    base = load_model(spec["model_id"], spec["revision"], load_in_4bit=dcfg["qlora"]["load_in_4bit"])
    adapter = project_path("models", f"{key}_qlora")
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    tok = load_tokenizer(spec["model_id"], spec["revision"])
    rows = load_split("test")
    if n:
        rows = rows[:n]
    prompts = [render_prompt(tok, row["en"], shots=[]) for row in rows]
    gen = generate_texts(model, tok, prompts, max_new_tokens=dcfg["max_new_tokens"], batch_size=dcfg["batch_size"],
                         temperature=None, max_input_tokens=dcfg["max_input_tokens"])
    srcs = [r["en"] for r in rows]
    refs = [r["hi"] for r in rows]
    hyps = gen["texts"]
    scores = corpus_scores(hyps, refs, cfg["evaluation"]["bleu_tokenizer"])
    diag = output_diagnostics(hyps, srcs)
    summary_path = adapter / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    timing = {
        "adaptation_seconds": summary.get("train_seconds"),
        "inference_seconds": gen["seconds"],
        "sentences_per_second": gen["sentences_per_second"],
        "new_tokens": gen["new_tokens"],
        "tokens_per_second": gen["tokens_per_second"],
        "hit_max_new_tokens": gen["hit_max_new_tokens"],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "quantization": "4-bit NF4 + LoRA" if dcfg["qlora"]["load_in_4bit"] else "bf16 + LoRA",
        "batch_size": dcfg["batch_size"],
        "decoding": "greedy",
    }
    payload = result_payload(key, "qlora", scores, diag, timing, summary.get("total_params"), spec, len(hyps),
                             {"trainable_params": summary.get("trainable_params"),
                              "training_peak_vram_gb": summary.get("peak_vram_gb"),
                              "adaptation": {"r": dcfg["qlora"]["r"], "epochs": dcfg["qlora"]["epochs"],
                                             "train_rows": summary.get("train_rows"),
                                             "train_subset_sha256": summary.get("train_subset_sha256"),
                                             "truncated_train_rows": summary.get("truncated_train_rows")}})
    write_json(RESULTS / f"metrics_{key}_qlora.json", payload)
    write_jsonl(TRANSLATIONS / f"{key}_qlora.jsonl",
                [{"i": i, "src": s, "ref": r, "hyp": h} for i, (s, r, h) in enumerate(zip(srcs, refs, hyps))])
    if logger:
        logger.info("[%s/qlora] %s | %.1fs", key, json.dumps(scores), gen["seconds"])
    del model, base
    torch.cuda.empty_cache()
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "gemma"], required=True)
    ap.add_argument("--mode", choices=["prompt", "qlora", "all"], default="prompt")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger(f"decoder_{args.model}", project_path("logs", f"decoder_{args.model}.log"))
    jsonl = JsonlLogger(project_path("logs", f"decoder_{args.model}.jsonl"))
    dump_config(cfg, project_path("logs", f"config_used_{args.model}.yaml"))

    if args.smoke:
        run_prompting(args.model, cfg, n=8, shots_n=args.shots, logger=log)
        if args.mode in ("qlora", "all"):
            run_qlora(args.model, cfg, logger=log, jsonl=jsonl, smoke=True)
            evaluate_qlora(args.model, cfg, logger=log, n=8)
        return 0

    if args.mode in ("prompt", "all"):
        run_prompting(args.model, cfg, shots_n=args.shots, logger=log)
    if args.mode in ("qlora", "all"):
        run_qlora(args.model, cfg, logger=log, jsonl=jsonl)
        evaluate_qlora(args.model, cfg, logger=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
