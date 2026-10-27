# STATUS — FGAI Assignment 3

Updated: 2026-10-27 (Asia/Kolkata)

## Reduced-scope execution

| Work | State | Evidence |
|---|---|---|
| Assignment requirements and source PDF | done | `IA3_ICT4403_FGAI_Final.pdf`; `docs/requirements_traceability.md` |
| Dataset processing and split audit | done | `data/processed/dataset_manifest.json`; `results/data_stats.json`; zero exact normalized source-sentence overlap across train/validation/test |
| Fixed evaluation sample | done | `results/evaluation_sample.json`: 200 unique test IDs, seed 42, source hash |
| Basic Transformer | done with limitation | 12,000 train rows; best checkpoint epoch 22; 463.9s; BLEU 0.262, chrF 9.624, TER 145.377 |
| mT5-small | done with limitation | One completed epoch of planned three on 60,000 pairs; 4,177.8s; BLEU 4.549, chrF 23.832, TER 87.680 |
| TinyLlama prompt | done with limitation | 3-shot, prompt-only, not official Meta LLaMA; BLEU 0.358, chrF 13.663, TER 118.142 |
| Gemma prompt | done with limitation | Gemma 3 1B mirror, prompt-only, Gemma terms apply; BLEU 10.952, chrF 35.566, TER 84.998 |
| Parameter-count consistency | done | Full counts: TinyLlama 1,100,048,384; Gemma 999,885,952; `scripts/14_sync_parameter_counts.py` counts original safetensors shapes before quantization |
| Temperature, plots, examples, and risk screen | done | `results/temperature/`, `results/figures/`, `results/qualitative_samples.md`, `results/error_analysis/` |
| Artifact/alignment gate | pass | `results/artifact_check.json`; `logs/reduced_pipeline.log`; all four systems 200 aligned rows; no split overlap; 20-row/three-temperature study for both decoders |
| Assignment documentation and final cover sheet | complete locally | README, report, traceability, submission details, and named DOCX/PDF are prepared; student-specific cover files stay local |
| Independent requirements/metric check | complete | Reviewer confirmed sample alignment, temperature outputs, metrics, provenance, PDF text, and the sanitized publication allowlist; the only remaining finding was stale status wording, corrected locally |
| Cover PDF workflow and DOCX visual check | done locally | Named DOCX visually fits on one page in Word; named PDF exported with Word File > Save As > PDF, one page with required fields and no placeholders |

## Key limitations

The test sample has 200 rows. mT5 completed one epoch of three planned; the basic Transformer used 12,000 training pairs and has frequent repetition flags. TinyLlama is a LLaMA-family substitute, not an official Meta LLaMA checkpoint. Gemma weights came from an ungated mirror and remain subject to Gemma terms. Decoder runs are prompt-only. Automatic output flags are not semantic judgments; no bilingual human adequacy study or systematic bias audit was completed.

## Next action

No model training or inference was run for the final document checks. The sanitized allowlist passed independent review, and the supplied public target was confirmed empty before publication. LMS upload remains a separate user action.
