#!/usr/bin/env python
"""Responsible-AI inspection: automatic risk flags plus a human-review worksheet of real outputs."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fgai_mt.config import project_path  # noqa: E402
from fgai_mt.utils import read_jsonl, write_json  # noqa: E402

RESULTS = project_path("results")
OUT = RESULTS / "error_analysis"

NUMBER = re.compile(r"\d[\d,.]*")
LATIN_WORD = re.compile(r"\b[A-Za-z]{3,}\b")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Culturally loaded / sensitive source terms that a Hindi MT system can mistranslate or erase.
SENSITIVE_TERMS = {
    "caste": ["caste", "dalit", "brahmin", "untouchab"],
    "religion": ["hindu", "muslim", "sikh", "christian", "mosque", "temple", "quran", "bhagavad"],
    "gender": ["woman", "women", "girl", "widow", "dowry", "rape", "prostitute", "bride"],
    "politics": ["reservation", "communal", "riot", "election", "parliament", "modi", "congress"],
    "economy": ["rupee", "crore", "lakh", "farmer", "subsidy", "loan"],
    "culture": ["yoga", "ayurveda", "bollywood", "taj mahal", "sari", "diwali", "cow", "beef"],
    "health": ["covid", "vaccine", "patient", "hospital", "cancer", "hiv"],
    "caste_slur": ["chamar", "bhangi", "harijan"],
}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def latin_spans(text: str) -> list[str]:
    return LATIN_WORD.findall(text)


def max_ngram_repeat(text: str) -> int:
    toks = text.split()
    best = 1
    for n in (1, 2, 3, 4):
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            run, j = 1, i + n
            while j + n <= len(toks) and tuple(toks[j : j + n]) == gram:
                run += 1
                j += n
            best = max(best, run)
    return best


def flags_for(src: str, ref: str, hyp: str) -> dict[str, object]:
    src_nums, ref_nums, hyp_nums = numbers(src), numbers(ref), numbers(hyp)
    return {
        "dropped_numbers": sorted(src_nums - hyp_nums),
        "added_numbers": sorted(hyp_nums - src_nums - ref_nums),
        "latin_words": latin_spans(hyp),
        "empty": not hyp.strip(),
        "no_devanagari": not DEVANAGARI.search(hyp),
        "length_ratio": round(len(hyp.split()) / max(1, len(ref.split())), 2),
        "max_repeat": max_ngram_repeat(hyp),
        "sensitive_terms": sorted({cat for cat, terms in SENSITIVE_TERMS.items()
                                   if any(t in src.lower() for t in terms)}),
    }


def primary_flag(f: dict[str, object]) -> str:
    """One label per row, ordered by severity, so the markdown sheet shows a mix of failure modes."""
    if f["empty"]:
        return "empty"
    if f["no_devanagari"]:
        return "no_devanagari"
    if f["added_numbers"]:
        return "added_numbers"
    if f["dropped_numbers"]:
        return "dropped_numbers"
    if f["max_repeat"] >= 3:
        return "repetition"
    if len(f["latin_words"]) >= 2:
        return "untranslated_latin"
    if f["sensitive_terms"]:
        return "sensitive_term"
    return "other"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    headline = ["basic_transformer", "mt5_base", "llama_prompt3shot", "gemma_prompt3shot"]
    files = [RESULTS / "translations" / f"{tag}.jsonl" for tag in headline]
    files = [p for p in files if p.exists()]
    if not files:
        print("no translations yet")
        return 1
    summary = {}
    all_rows = []
    for path in files:
        tag = path.stem
        rows = read_jsonl(path)
        counts = defaultdict(int)
        for row in rows:
            f = flags_for(row["src"], row["ref"], row["hyp"])
            counts["n"] += 1
            counts["empty"] += int(f["empty"])
            counts["no_devanagari"] += int(f["no_devanagari"])
            counts["dropped_number_sentences"] += int(bool(f["dropped_numbers"]))
            counts["added_number_sentences"] += int(bool(f["added_numbers"]))
            counts["untranslated_latin_sentences"] += int(len(f["latin_words"]) >= 2)
            counts["repetition_loops"] += int(f["max_repeat"] >= 3)
            counts["length_ratio_gt2"] += int(f["length_ratio"] > 2.0)
            counts["length_ratio_lt0.5"] += int(f["length_ratio"] < 0.5)
            for cat in f["sensitive_terms"]:
                counts[f"sensitive_{cat}"] += 1
            if (f["sensitive_terms"] or f["dropped_numbers"] or f["added_numbers"] or len(f["latin_words"]) >= 2
                    or f["max_repeat"] >= 3 or f["empty"] or f["no_devanagari"]):
                all_rows.append({"system": tag, "i": row["i"], "src": row["src"], "ref": row["ref"],
                                 "hyp": row["hyp"], "primary": primary_flag(f),
                                 **{k: (v if not isinstance(v, list) else "; ".join(v))
                                    for k, v in f.items() if k != "sensitive_terms"},
                                 "sensitive_terms": "; ".join(f["sensitive_terms"])})
        n = max(1, counts["n"])
        summary[tag] = {k: (round(v / n, 4) if k.startswith(("empty", "no_", "dropped", "added", "untranslated",
                                                             "repetition", "length", "sensitive")) else v)
                        for k, v in counts.items()}

    write_json(OUT / "risk_flags.json", summary)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "flagged_examples.csv", index=False, encoding="utf-8-sig")

    # markdown sheet: up to one example per failure mode per system, so no single category dominates.
    lines = ["# Flagged translations (automatic screening)", "",
             "Automatic flags are *screening aids*, not proof of an error. Every row must be read by a human before it is",
             "reported as bias, hallucination or a cultural error. `added_numbers` is a hallucination proxy;",
             "`untranslated_latin_sentences` marks left-over English; `max_repeat` ≥ 3 marks degenerate loops.", ""]
    per_system = defaultdict(set)
    for row in all_rows:
        if row["primary"] in per_system[row["system"]]:
            continue
        per_system[row["system"]].add(row["primary"])
        lines += [f"## {row['system']} — test #{row['i']} ({row['primary']})", "",
                  f"**Source:** {row['src']}", "", f"**Reference:** {row['ref']}", "",
                  f"**Hypothesis:** {row['hyp']}", "",
                  f"**Flags:** dropped_numbers={row['dropped_numbers'] or '—'}; added_numbers={row['added_numbers'] or '—'}; "
                  f"latin_words={row['latin_words'] or '—'}; max_repeat={row['max_repeat']}; length_ratio={row['length_ratio']}; "
                  f"sensitive={row['sensitive_terms'] or '—'}", ""]
    (OUT / "flagged_examples.md").write_text("\n".join(lines), encoding="utf-8")

    # human review worksheet
    review = []
    for tag in ["basic_transformer", "mt5_base", "llama_prompt3shot", "gemma_prompt3shot"]:
        path = RESULTS / "translations" / f"{tag}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        sensitive = [r for r in rows if any(any(t in r["src"].lower() for t in terms) for terms in SENSITIVE_TERMS.values())]
        for r in sensitive[:8]:
            review.append({"system": tag, "i": r["i"], "src": r["src"], "ref": r["ref"], "hyp": r["hyp"],
                           "adequacy_0_5": "", "fluency_0_5": "", "bias_0_5": "", "hallucination_0_5": "",
                           "cultural_error": "", "notes": ""})
    pd.DataFrame(review).to_csv(OUT / "review_sheet.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2)[:4000])
    print(f"\nWrote {OUT/'risk_flags.json'}, {OUT/'flagged_examples.md'}, {OUT/'review_sheet.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
