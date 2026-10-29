# Results index

The headline metrics compare four systems on the same fixed 200-row sample from the official 2,507-row test split. The sample IDs, seed (42), and source-file hash are in `evaluation_sample.json`. The official split sizes shown in data manifests describe the corpus; they are not the number of headline evaluation rows.

| Artifact | Produced by | Contents |
|---|---|---|
| `metrics_<system>.json` | training/inference scripts | BLEU, chrF, TER, diagnostics, timing, parameter count, configuration |
| `translations/<system>.jsonl` | inference scripts | 200 rows per headline system, with original test row ID, source, reference, and hypothesis |
| `summary_table.csv` / `.md` | `05_evaluate.py` | Four-system comparison and timing settings |
| `confidence_intervals.json` | `05_evaluate.py` | Sentence-bootstrap 95% BLEU intervals for the four systems |
| `paired_bootstrap.json` | `05_evaluate.py` | Six pairwise BLEU comparisons using paired resamples |
| `temperature/temperature_results.json` | `06_temperature.py` | Two prompt-only systems × three temperatures on the first 20 fixed sample rows |
| `temperature/*.json` and `*.jsonl` | `06_temperature.py` | Per-system/per-temperature scores and the corresponding 20 generated outputs |
| `eda_summary.json` / `figures/dataset_eda.png` | `scripts/15_eda.py` | Hash-checked descriptive EDA for the processed train subset and full validation/test splits: missing/duplicate pairs, sentence lengths, and Hindi-target script composition |
| `error_analysis/risk_flags.json` | `07_ethics_analysis.py` | Heuristic screening counts for the four headline outputs |
| `error_analysis/flagged_examples.md`, `review_sheet.csv` | `07_ethics_analysis.py` | Flagged real outputs and a human-review worksheet |
| `qualitative_samples.md` / `.csv` | `12_qualitative_samples.py` | Eight fixed sample positions with all four model outputs |
| `figures/bleu_by_system.png`, `bleu_vs_params.png`, `temperature_analysis.png` | evaluation scripts | Headline score, parameter comparison, and temperature study |
| `analysis_facts.json` | `10_collect_facts.py` | Machine-readable headline metrics, temperature results, and output-screening facts |
| `artifact_check.json` | `08_check_artifacts.py` | Fixed 200-row sample and four-system alignment; saved overlap/EDA audit records; EDA sources, distributions, figure and notebook links; 20-row temperature coverage |
| `model_provenance.json` | recorded from model/data sources | Resolved revisions, source links, and licensing notes |
| `environment.json` | recorded locally | Python, package, CUDA, and GPU versions |
| `legacy_fulltest/` | preserved earlier run | Superseded full-test basic-Transformer results; not included in the four-system headline table |

## Reading order

1. `summary_table.md` for headline metrics.
2. `confidence_intervals.json` and `paired_bootstrap.json` for sample uncertainty and paired comparisons.
3. `qualitative_samples.md` and `translations/` to inspect actual outputs.
4. `temperature/temperature_results.json` and its plot for the limited decoding study.
5. `eda_summary.json` and `figures/dataset_eda.png` for corpus-level descriptive EDA; the headline metric sample remains 200 rows.
6. `error_analysis/` for heuristic flags and review examples.
7. `artifact_check.json` for the final shared-sample alignment and saved-artifact checks.

Prompt-only decoder runs have no training/adaptation time. The temperature study is separate from the headline comparison; no prompt was chosen using the headline test sample. Older ablation/diagnostic files, if present, are not part of the headline results.

`figures/inference_time.png` is a preserved plot from the superseded full-test run. It contains only the old basic-Transformer throughput point and is not cited as a current comparison. Current inference timings are in the four-row headline table.
