# GitHub publication manifest

The currently published GitHub baseline is 96 files on main at [dc20c77c1a9d2149365c71b9149611ab24ce4e14](https://github.com/AkCodes23/FGAI_Assignment_2/commit/dc20c77c1a9d2149365c71b9149611ab24ce4e14), verified on 2026-10-29. This local proposal updates the report, README, results notebook, figure index, tokenizer-fertility reproduction, generated figures, and focused figure-reproduction dependency lock; its allowlist contains 111 files. The proposal has not been pushed and is awaiting review and the student's approval.

## Saved overlap-audit record

The counts below are copied from data/processed/dataset_manifest.json. The artifact gate validates these saved values and recorded audit fields; raw split TSVs are excluded, so the gate does not recompute source overlap. The EDA summary was generated locally from the processed TSVs after their hashes were checked. The published package contains the summary and figure but not the raw corpus files.

| Audit field | Saved overlap count |
|---|---:|
| train_vs_val_source_overlap | 0 |
| train_vs_test_source_overlap | 0 |
| val_vs_test_source_overlap | 0 |

## Include

- Project guidance and setup: README.md, PLAN.md, STATUS.md, .gitignore, .dockerignore, Dockerfile, requirements.txt, requirements-report-figures.txt, configs/default.yaml.
- Implementation and reproducibility: src/fgai_mt/, scripts/, tests/, notebooks/01_results_walkthrough.ipynb.
- Dataset provenance only: data/processed/dataset_manifest.json.
- Public documentation: docs/report.md, docs/requirements_traceability.md, and docs/publication_manifest.md. Cover files, identity-bearing details, and docs/submission_details.md are excluded.
- Current evaluation: results/README.md, results/artifact_check.json, results/analysis_facts.json, results/basic_train_subset.json, results/confidence_intervals.json, results/data_stats.json, results/eda_summary.json, results/evaluation_sample.json, results/metrics_basic_transformer.json, results/metrics_mt5_base.json, results/metrics_llama_prompt3shot.json, results/metrics_gemma_prompt3shot.json, results/model_provenance.json, results/paired_bootstrap.json, results/prompt_shots.json, results/report_figure_data.json, results/summary_table.csv, results/summary_table.md, results/tokenizer_fertility.json, results/qualitative_samples.csv, results/qualitative_samples.md.
- Current translations and diagnostics: results/translations/ (four evaluated systems), results/temperature/ (the 20-row study), and results/error_analysis/ (flags and review worksheet).
- Current figures: results/figures/dataset_eda.png, pipeline_flow.png, pipeline_flow.svg, architecture_comparison.png, architecture_comparison.svg, training_curves.png, score_intervals.png, quality_cost.png, sentence_chrf_distribution.png, chrf_by_source_length.png, paired_bleu_bootstrap.png, tokenizer_fertility.png, risk_flag_heatmap.png, and temperature_analysis.png.
- Figure generation: scripts/16_report_figures.py. It derives figure data from saved results and published logs; no model checkpoints are needed, and it does not run training or inference.
- Encoder–decoder tokenizer fertility: scripts/16a_encoder_tokenizer_fertility.py and results/tokenizers/spm_basic.model. The script verifies the hash of data/processed/test.tsv, which is local-only and remains excluded, and uses the pinned mT5 tokenizer revision for tokenization only.
- Reproduction evidence: logs/reduced_pipeline.log, logs/train_basic_reduced.jsonl, and logs/train_mt5.jsonl.

## Exclude

- Student-specific cover documents and identity-bearing details: docs/submission_details.md, the named submission DOCX/PDF, scripts/09_make_submission_doc.py, and scripts/13_make_submission_pdf.py. These remain local for LMS submission.
- scripts/run_mt5_fresh3_background.ps1, a local-only training launcher.
- .venv/, .hf/, .uv-cache/, __pycache__/, *.pyc, .env, caches, temporary files, model weights/checkpoints (models/), and generated local runtime output (outputs/).
- Raw corpus files and split TSVs (data/raw/ and data/processed/*.tsv); the dataset source and reproducible preparation code are provided instead.
- The source assignment PDF, archived templates with placeholders, docs/legacy_fullscope/, results/legacy_fulltest/, backup-scale error-analysis artifacts, unused historical figures, and unrelated install/debug logs.
- local_only_backups/, including the user review preview. It is a local artifact and is not part of the package.

## Compliance note

TinyLlama/TinyLlama-1.1B-Chat-v1.0 is a LLaMA-family LlamaForCausalLM checkpoint, not an official Meta LLaMA checkpoint. The official Gemma checkpoint is loaded from an Unsloth mirror; Gemma Terms of Use apply. These distinctions and the resolved model revisions are in results/model_provenance.json. The current report and generated figures use saved data; the headline metric files and translations are not modified.
