# Results index

The headline metrics compare four systems on the same fixed 200-row sample from the official 2,507-row test split. The sample IDs, seed 42, and source-file hash are in evaluation_sample.json. The full test split size is corpus metadata; the headline evaluation contains 200 rows.

| Artifact | Produced by | Contents |
|---|---|---|
| metrics_<system>.json | training/inference scripts | BLEU, chrF, TER, diagnostics, timing, parameter count, and configuration |
| translations/<system>.jsonl | inference scripts | 200 rows per headline system, each with test row ID, source, reference, and hypothesis |
| summary_table.csv / .md | 05_evaluate.py | Four-system comparison and inference settings |
| confidence_intervals.json | 05_evaluate.py | Sentence-bootstrap 95% BLEU intervals |
| paired_bootstrap.json | 05_evaluate.py | Six paired BLEU comparisons |
| temperature/temperature_results.json | 06_temperature.py | Two prompt-only systems × three temperatures on the first 20 sample rows |
| tokenizer_fertility.json | `scripts/05c_tokenizer_fertility.py` and `scripts/16a_encoder_tokenizer_fertility.py` | Hindi reference tokenization for all four systems on the full 2,507-row test split |
| report_figure_data.json | scripts/16_report_figures.py | Sentence-level chrF means and intervals, length buckets, paired results, source hashes |
| eda_summary.json / figures/dataset_eda.png | scripts/15_eda.py | Hash-checked corpus EDA for the processed training subset and full validation/test splits |
| error_analysis/risk_flags.json | 07_ethics_analysis.py | Heuristic screening rates for the four headline outputs |
| qualitative_samples.md / .csv | 12_qualitative_samples.py | Fixed sample positions with all four outputs |
| artifact_check.json | 08_check_artifacts.py | Shared sample alignment, saved EDA audit, figure and notebook references, and temperature coverage |

## Current figure set

The reproducible generator `scripts/16_report_figures.py` reads the saved summary table, bootstrap results, published logs, tokenization audit, risk flags, temperature outputs, and aligned translation JSONL files. It writes derived numbers to report_figure_data.json and creates figures without model checkpoints:

- pipeline_flow.png and pipeline_flow.svg: one cleaned dataset path, four systems, one paired 200-row sample.
- dataset_eda.png: split sizes, sentence lengths, and target-script composition.
- architecture_comparison.png and architecture_comparison.svg: encoder–decoder and decoder-only prompt paths.
- training_curves.png: saved train and validation loss checkpoints.
- score_intervals.png: corpus BLEU and mean sentence-level chrF intervals.
- quality_cost.png: corpus chrF against inference time and parameter count.
- sentence_chrf_distribution.png: sentence-level chrF distributions on the shared sample.
- chrf_by_source_length.png: mean sentence chrF for four source-length buckets.
- paired_bleu_bootstrap.png: paired BLEU-difference intervals.
- tokenizer_fertility.png: Hindi reference tokenization for all four systems.
- risk_flag_heatmap.png: heuristic warnings by system.
- temperature_analysis.png: BLEU, chrF, and Distinct-2 for 20 rows.

The encoder–decoder fertility script tokenizes the local-only `data/processed/test.tsv` using the included Basic Transformer SentencePiece model and pinned mT5 tokenizer revision. It verifies the TSV hash and updates `tokenizer_fertility.json`; tokenization does not run training or inference. The report embeds and interprets each current figure. The notebook figure gallery uses the same image files; its saved image outputs are refreshed without executing training or inference cells.

## Reading order

1. summary_table.md for headline results.
2. confidence_intervals.json and paired_bootstrap.json for sample uncertainty.
3. translations/ and qualitative_samples.md for actual outputs.
4. temperature/temperature_results.json and temperature_analysis.png for the limited decoding study.
5. eda_summary.json and dataset_eda.png for corpus properties.
6. error_analysis/ for heuristic flags and the review worksheet.
7. report_figure_data.json and the generator for derived figure data and input hashes.
8. artifact_check.json for the final saved-artifact checks.

The temperature study is separate from the headline comparison. Automatic flags are screening signals, not human translation ratings. Older ablation or full-test diagnostics do not form part of the four-system headline results.
