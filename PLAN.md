# FGAI Assignment 3 — Reduced-scope plan

**Source:** `IA3_ICT4403_FGAI_Final.pdf`  
**Project folder:** `Assignment 2/`

## Objective

Cover the assignment requirements with four English → Hindi systems, a fixed small evaluation sample, reproducible measurements, honest limitations, and a concise report. Reuse completed work; avoid large new sweeps.

## Completed systems and data

| System | Method and bounded scope | Result |
|---|---|---|
| Basic Transformer | From-scratch encoder–decoder; 12,000-row deterministic subset of existing 60,000-pair train split | 15.16M parameters; best validation checkpoint at epoch 22; 463.9s train |
| mT5-small | Existing checkpoint; one completed epoch of planned three on 60,000 pairs | 300.18M parameters; 4,177.8s train |
| TinyLlama 1.1B Chat | Three-shot LLaMA-family prompt; no gradient updates; not official Meta LLaMA | 1,100,048,384 full parameters |
| Gemma 3 1B IT | Three-shot prompt using ungated mirror; no gradient updates; Gemma terms apply | 999,885,952 full parameters |

The shared test sample has 200 unique rows, seed 42, from the 2,507-row test split. All headline translations align by row ID, source, and reference. The 20-row temperature study uses the same first 20 sample IDs at 0.2, 0.7, and 1.0.

## Completed measurements and artifacts

- BLEU, chrF, TER, full parameter count, training/adaptation time where applicable, inference time, throughput, and confidence intervals are in `results/summary_table.*` and per-system metric JSON.
- The comparison reports pretraining, data volume, tokenizer, size, and adaptation as confounders.
- Actual outputs, automated risk flags, practical mitigations, temperature results, and plots are saved under `results/` and summarized in `README.md` and `docs/report.md`.
- Final student-specific cover files are prepared locally for LMS submission; they are excluded from the public GitHub package.
- `scripts/08_check_artifacts.py` passed: all four translation sets have 200 aligned rows; sample seed 42; no split overlap found; temperature study has 20 matching rows for both decoders and all three temperatures.
- Decoder parameter counts were corrected from original local safetensors tensor shapes; `scripts/14_sync_parameter_counts.py` performs this metadata-only correction without model inference.

## Guardrails and limitations

No new decoder training, broad sweep, infrastructure, LMS upload, payment, or deletion. The target repository is public, so the public allowlist excludes student-specific cover files, identity-bearing details and generator scripts, and absolute local paths. Conditional publication is authorized after the final read-only recheck; publish only the sanitized allowlist and preserve target history.

## Current execution

1. Fixed sample, existing checkpoints, and completed training artifacts: done.
2. Four-system shared evaluation, temperature study, plots, and ethics screen: done.
3. Result tables, README/report, traceability, and final named cover-sheet DOCX/PDF: updated locally. The named PDF was exported from the DOCX in Word using File > Save As > PDF; Word visual inspection shows a one-page layout without clipping, and PDF text checks confirm required fields and no placeholders.
4. Independent non-training review: complete. The reviewer confirmed the sanitized manifest, privacy scan, shared outputs, temperature artifacts, metrics, provenance, and cover PDF text; the remaining stale-status wording has been corrected locally.
