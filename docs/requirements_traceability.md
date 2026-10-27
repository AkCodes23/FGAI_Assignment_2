# Assignment 3 requirements traceability — ICT4403

Source brief: `IA3_ICT4403_FGAI_Final.pdf`.

| # | Assignment requirement | Evidence in this submission | Status |
|---:|---|---|---|
| 1 | Select a language pair and public parallel corpus | English → Hindi IIT Bombay corpus, source, citation, and CC BY-NC 4.0 license in `README.md` and `results/model_provenance.json` | done |
| 2 | Describe and preprocess data; preserve train/validation/test splits | `src/fgai_mt/data.py`, `data/processed/dataset_manifest.json`, `results/data_stats.json`; audit reports zero exact normalized source-sentence overlap | done |
| 3 | Implement and train a basic Transformer | `src/fgai_mt/model_basic.py`, `scripts/01_train_transformer.py`, `models/basic_transformer/`; 12,000-row subset, epoch-22 best checkpoint, logs and metrics | done with bounded-data limitation |
| 4 | Fine-tune mT5 | `models/mt5_base/`, `logs/train_mt5.jsonl`, `results/metrics_mt5_base.json`; one completed epoch of planned three on 60,000 pairs | done with one-epoch limitation |
| 5 | Prompt or fine-tune a LLaMA-family model | TinyLlama 1.1B 3-shot prompt; `results/translations/llama_prompt3shot.jsonl`, `results/metrics_llama_prompt3shot.json`, provenance | done; not official Meta LLaMA |
| 6 | Prompt or fine-tune Gemma | Gemma 3 1B 3-shot prompt; `results/translations/gemma_prompt3shot.jsonl`, `results/metrics_gemma_prompt3shot.json`, provenance | done; mirror used and Gemma terms apply |
| 7 | Evaluate every system on the same held-out rows | `results/evaluation_sample.json`; `results/artifact_check.json` verifies 200 rows per system, matching IDs/source/reference; checker passed | done |
| 8 | Report BLEU and chrF | Per-system metrics and `results/summary_table.csv` / `.md`; SacreBLEU 13a, chrF, and TER | done |
| 9 | Report training/adaptation and inference time | Metric JSON and summary table; 463.9s scratch training, 4,177.8s mT5 training; inference timing for all four; prompt-only systems have no training time | done |
| 10 | Report parameter counts | Full unquantized counts in metrics and summary: 15.16M, 300.18M, 1,100,048,384, and 999,885,952; decoder counts from safetensors headers | done |
| 11 | Compare models and discuss confounders | README/report compare architecture, pretraining, data volume, tokenizer, model size, and adaptation; plots in `results/figures/` | done with confounders disclosed |
| 12 | Discuss output quality, responsible use, risks, and mitigations | Actual outputs in `results/qualitative_samples.md`; automated flags and review worksheet in `results/error_analysis/`; mitigations in README/report | done with limitation: no bilingual human adequacy study |
| 13 | Analyze sampling temperature | `results/temperature/temperature_results.json` and six generated output files; 20 shared rows at temperatures 0.2, 0.7, and 1.0 for each decoder; checker passed | done |
| 14 | Provide README with data, method, results, analysis, temperature, and conclusion | `README.md`; no result markers remain | done |
| 15 | Provide source/notebook, dependencies, configs, metrics, plots, and outputs | `src/`, `scripts/`, `notebooks/01_results_walkthrough.ipynb`, `requirements.txt`, `configs/`, `results/` | done |
| 16 | Provide dataset source link | README, report, cover sheet | done |
| 17 | Create the named Word document with student name, registration number, assignment title, developed-system title, dataset name/link, and GitHub link | Student-specific named DOCX exists locally; required fields were verified | done locally; excluded from public package because it contains student identifiers |
| 18 | Provide a PDF copy of the named Word document | Matching student-specific PDF was exported from the DOCX in Word using File > Save As > PDF; one page, required fields present, no placeholders | done locally; excluded from public package because it contains student identifiers |
| 19 | Upload implementation, supporting files, results, plots, generated samples, and dataset source link to GitHub | Student-provided target: https://github.com/AkCodes23/FGAI_Assignment_2.git; local project contains the source, results, plots, and samples | published to https://github.com/AkCodes23/FGAI_Assignment_2 on main |
| 20 | Upload the final PDF to the specified LMS section | Not performed | pending separate user action |

## Reproducibility and limits

- The reduced pipeline log is `logs/reduced_pipeline.log`; it completed with exit code 0 and used existing checkpoints/cache.
- The artifact checker command was `python scripts/08_check_artifacts.py`; saved output is `results/artifact_check.json` and reports `pass: true`.
- The sample and deterministic training subset record IDs, seeds, and source hashes. Earlier full-test outputs remain archived under `results/legacy_fulltest/`.
- `scripts/14_sync_parameter_counts.py` reads local cached safetensors headers only. It corrects quantized decoder storage counts to full original model parameter counts and performs no model inference or training.
- The 200-row scores and 20-row temperature results are descriptive. Automatic flags are screening signals; no bilingual human adequacy study was completed.
- TinyLlama is a LLaMA-family substitute, not an official Meta checkpoint. Gemma uses an ungated mirror; Gemma terms apply.
- The supplied GitHub target is https://github.com/AkCodes23/FGAI_Assignment_2. The public package excludes the student-specific DOCX/PDF cover documents and docs/submission_details.md; those remain local for LMS submission.
