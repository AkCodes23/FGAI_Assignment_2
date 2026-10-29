# Reduced-scope GitHub publication manifest

Target supplied by the student: `https://github.com/AkCodes23/FGAI_Assignment_2.git`.

This sanitized allowlist describes the published handoff package. The 96-file refresh is on GitHub `main` at content commit [da390882d3b8c7d1689ccfb95fc58c67d5e7611a](https://github.com/AkCodes23/FGAI_Assignment_2/commit/da390882d3b8c7d1689ccfb95fc58c67d5e7611a), verified on 2026-10-29. Student identifiers, workstation-specific paths, and identity-bearing cover materials remain excluded from this sanitized package.

## Saved overlap-audit record

The counts below are copied from `data/processed/dataset_manifest.json`. The artifact gate validates these saved values and the corresponding recorded audit fields; the raw split TSVs are excluded from the sanitized package, so the gate does not recompute source overlap. The EDA summary was generated locally from the processed TSVs after checking their hashes. In the sanitized package, the gate checks the saved EDA counts, pair-quality audit, cleaning metadata, and source hashes against the shipped manifest and `data_stats.json`, and verifies the saved figure plus EDA references in the notebook and documentation; it does not recompute EDA from the excluded TSVs.

| Audit field | Saved overlap count |
|---|---:|
| `train_vs_val_source_overlap` | 0 |
| `train_vs_test_source_overlap` | 0 |
| `val_vs_test_source_overlap` | 0 |

## Include

- Project guidance and setup: `README.md`, `PLAN.md`, `STATUS.md`, `.gitignore`, `.dockerignore`, `Dockerfile`, `requirements.txt`, `configs/default.yaml`.
- Implementation and reproducibility: `src/fgai_mt/`, `scripts/`, `tests/`, `notebooks/01_results_walkthrough.ipynb`.
- Dataset provenance only: `data/processed/dataset_manifest.json`.
- Public documentation: `docs/report.md`, `docs/requirements_traceability.md`, and `docs/publication_manifest.md`. Student-specific cover documents and `docs/submission_details.md` remain local for LMS submission.
- Current evaluation: `results/README.md`, `results/artifact_check.json`, `results/analysis_facts.json`, `results/basic_train_subset.json`, `results/confidence_intervals.json`, `results/data_stats.json`, `results/eda_summary.json`, `results/evaluation_sample.json`, `results/metrics_basic_transformer.json`, `results/metrics_mt5_base.json`, `results/metrics_llama_prompt3shot.json`, `results/metrics_gemma_prompt3shot.json`, `results/model_provenance.json`, `results/paired_bootstrap.json`, `results/prompt_shots.json`, `results/summary_table.csv`, `results/summary_table.md`, `results/tokenizer_fertility.json`, `results/qualitative_samples.csv`, `results/qualitative_samples.md`.
- Generated outputs: `results/translations/` (the four current systems), `results/temperature/` (current 20-row outputs and metrics), `results/error_analysis/flagged_examples.csv`, `results/error_analysis/flagged_examples.md`, `results/error_analysis/review_sheet.csv`, `results/error_analysis/risk_flags.json`, and `results/figures/bleu_by_system.png`, `results/figures/bleu_vs_params.png`, `results/figures/temperature_analysis.png`, `results/figures/dataset_eda.png`.
- Reproduction evidence: `logs/reduced_pipeline.log`, `logs/train_basic_reduced.jsonl`, `logs/train_mt5.jsonl`.

## Exclude

- Student-specific cover documents and identity-bearing details: `docs/submission_details.md`, the named submission DOCX/PDF, `scripts/09_make_submission_doc.py`, and `scripts/13_make_submission_pdf.py`. These remain local for LMS submission.
- `scripts/run_mt5_fresh3_background.ps1`: local-only training launcher; excluded from the publication allowlist.

- `.venv/`, `.hf/`, `.uv-cache/`, `__pycache__/`, `*.pyc`, `.env`, caches, temporary files, model weights/checkpoints (`models/`), and generated local runtime output (`outputs/`).
- Raw corpus files and split TSVs (`data/raw/`, `data/processed/*.tsv`); the dataset source and reproducible preparation code are provided instead.
- The source assignment PDF, archived templates with placeholders, `docs/legacy_fullscope/`, `results/legacy_fulltest/`, backup-scale error-analysis artifacts, the historical `results/figures/inference_time.png`, and unrelated install/debug logs.

## Compliance note

The assignment brief specifies a LLaMA decoder-only architecture and permits smaller variants. The evaluated checkpoint is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, a LLaMA-family `LlamaForCausalLM` model; it is not represented as an official Meta checkpoint. The documentation and model provenance record this distinction. Gemma's mirror source and applicable Gemma terms are also disclosed. The current named PDF was regenerated from the DOCX in Microsoft Word with SaveAs2 (FileFormat=17); a local-only word_saveas_audit.txt records the call. The PDF metadata identifies Microsoft Word for Microsoft 365. Independent review confirmed a one-page A4 layout, three working links, required fields, and score-table agreement with results/summary_table.csv. The named cover files and local audit remain excluded from the public package.
