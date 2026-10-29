# STATUS — FGAI Assignment 3

Updated: 2026-10-29 (Asia/Kolkata)

## Reduced-scope execution

| Work | State | Evidence |
|---|---|---|
| Assignment requirements and source PDF | done | `IA3_ICT4403_FGAI_Final.pdf`; `docs/requirements_traceability.md` |
| Dataset processing and split audit | done | `data/processed/dataset_manifest.json`; `results/data_stats.json`; zero exact normalized source-sentence overlap across train/validation/test |
| Exploratory data analysis | done locally | `scripts/15_eda.py`; `results/eda_summary.json`; `results/figures/dataset_eda.png`; hash-verified 60,000-row train subset plus full validation/test; headline metrics remain on 200 rows |
| Fixed evaluation sample | done | `results/evaluation_sample.json`: 200 unique test IDs, seed 42, source hash |
| Basic Transformer | done with limitation | 12,000 train rows; best checkpoint epoch 22; 463.9s; BLEU 0.262, chrF 9.624, TER 145.377 |
| mT5-small | done | Three completed epochs of three planned on 60,000 pairs; 12,379.4s; best validation loss 2.2517; 200-row BLEU 7.581, chrF 29.335, TER 82.141; epoch-one checkpoint/history and 2026-10-27 logs archived locally |
| TinyLlama prompt | done with limitation | 3-shot, prompt-only, not official Meta LLaMA; BLEU 0.358, chrF 13.663, TER 118.142 |
| Gemma prompt | done with limitation | Gemma 3 1B mirror, prompt-only, Gemma terms apply; BLEU 10.952, chrF 35.566, TER 84.998 |
| Parameter-count consistency | done | Full counts: TinyLlama 1,100,048,384; Gemma 999,885,952; `scripts/14_sync_parameter_counts.py` counts original safetensors shapes before quantization |
| Temperature, plots, examples, and risk screen | done | `results/temperature/`, `results/figures/`, `results/qualitative_samples.md`, `results/error_analysis/` |
| Artifact/alignment gate | pass (2026-10-29; exit 0) | `results/artifact_check.json` reports `pass: true`; four systems aligned on 200 rows, EDA/hash checks pass, and temperature coverage is exactly 20 rows at each expected setting |
| Assignment documentation and final cover sheet | complete locally | README, report, traceability, submission details, and named DOCX/PDF are prepared; student-specific cover files stay local |
| Independent requirements/metric check | passed (2026-10-29) | Final pre-push verification passed against the sanitized 96-file allowlist; independent isolated artifact gate exited 0 with pass=true |
| Cover PDF workflow and DOCX visual check | verified locally and independently | Regenerated in Word on 2026-10-29 with SaveAs2 (FileFormat=17); local-only word_saveas_audit.txt records the export. Independent review verified one A4 page, three working links, populated required fields, and score-table agreement with results/summary_table.csv. The named cover files remain local and excluded from the public package. |
| GitHub publication | refresh published (2026-10-29) | `main` includes the sanitized 96-file refresh at content commit [da390882d3b8c7d1689ccfb95fc58c67d5e7611a](https://github.com/AkCodes23/FGAI_Assignment_2/commit/da390882d3b8c7d1689ccfb95fc58c67d5e7611a), verified 2026-10-29 |
| LMS submission | pending | Final named PDF remains local; user must submit it through the specified LMS |

## Verification notes

Final independent review passed on 2026-10-29. It confirms the mT5 history and current 200-row metrics report three completed epochs. The preserved, excluded logs/infer_mt5.log still has a stale epochs=1 field on a 2026-10-29 metrics line; current run evidence is in the training history and metric JSON. The fresh3 launcher callback exit code remains unverified locally. The named cover DOCX/PDF were regenerated in Word via SaveAs2 (FileFormat=17) and independently checked; the local-only export log and cover files remain excluded from the public package.
## Key limitations

The test sample has 200 rows. mT5 completed all three planned epochs; the basic Transformer used 12,000 training pairs and has frequent repetition flags. TinyLlama is a LLaMA-family substitute, not an official Meta LLaMA checkpoint. Gemma weights came from an ungated mirror and remain subject to Gemma terms. Decoder runs are prompt-only. Automatic output flags are not semantic judgments; no bilingual human adequacy study or systematic bias audit was completed.

The effective staged-run config is preserved at `logs/config_used_mt5_fresh3_20261029.yaml`; the prior 2026-10-27 config was overwritten before this run and was not recovered.

## Next action

Final pre-push independent verification passed on 2026-10-29 against the sanitized 96-file allowlist; the independent isolated artifact gate exited 0 with pass=true. The sanitized refresh is published on GitHub `main` at content commit [da390882d3b8c7d1689ccfb95fc58c67d5e7611a](https://github.com/AkCodes23/FGAI_Assignment_2/commit/da390882d3b8c7d1689ccfb95fc58c67d5e7611a), verified 2026-10-29. The local artifact gate was rerun after these wording edits on 2026-10-29 and `results/artifact_check.json` records pass=true. LMS submission remains the student's action. The authorized mT5 three-epoch training and fixed 200-row inference completed locally. No LMS submission was made.
