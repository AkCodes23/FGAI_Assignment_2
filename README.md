# BhashaBridge-Hindi: English-to-Hindi Translation System

This project compares four machine-translation systems on English→Hindi. The headline comparison uses the same precommitted 200-row sample from the IIT Bombay test split for every system. The sample seed, original row IDs, and test-file hash are recorded in [`results/evaluation_sample.json`](results/evaluation_sample.json). The student-supplied repository target is [AkCodes23/FGAI_Assignment_2](https://github.com/AkCodes23/FGAI_Assignment_2.git).

## Objective

Implement a Transformer encoder–decoder from scratch, fine-tune mT5, and use prompting with one LLaMA-family model and one Gemma model. Compare BLEU, chrF, training or adaptation time, inference time, parameter count, and output behaviour. Discuss architecture and model-size confounders, sampling temperature, and responsible use.

## Dataset

The IIT Bombay English–Hindi Parallel Corpus (CFILT) is distributed under **CC BY-NC 4.0**. The official source is [CFILT](https://www.cfilt.iitb.ac.in/iitb_parallel/); the downloaded mirror is [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi). Provider train/validation/test membership is preserved. Text cleanup uses Unicode NFC, whitespace and control-character cleanup, malformed-pair and Hindi-script checks, length-ratio and length limits. Training duplicates are removed. The leakage audit uses NFKC, case folding, punctuation removal, and whitespace folding for exact normalized source matches; it found zero overlap. See [`data/processed/dataset_manifest.json`](data/processed/dataset_manifest.json) for counts and hashes.

The existing shared training file contains 60,000 pairs, selected with seed 42. The basic Transformer was retrained on a deterministic 12,000-row subset of that file; mT5 used the 60,000-row file for one epoch. The decoder-only models use three training examples in their prompts and receive no gradient updates. The headline test set is a uniform 200-row sample drawn once from the official 2,507-row test split. It is reused unchanged for all four systems. No system or prompt was selected using these test scores.

## Systems

| System | Model | Method | Parameters |
|---|---|---|---:|
| Basic Transformer | 6 encoder + 6 decoder layers, 256 hidden size, 8 heads | From-scratch training; validation-loss checkpoint | 15.16 M |
| mT5 | `google/mt5-small` | Full fine-tuning; 1 epoch completed | 300.18 M |
| LLaMA family | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 3-shot prompting, greedy headline decoding | 1.10 B |
| Gemma | `unsloth/gemma-3-1b-it` | 3-shot prompting, greedy headline decoding | 1.00 B |

TinyLlama is a genuine `LlamaForCausalLM` checkpoint under Apache-2.0; it is not an official Meta LLaMA release. Gemma uses an ungated mirror of the Gemma 3 1B instruction-tuned weights; Gemma’s terms still apply. The exact model revisions and provenance notes are in [`results/model_provenance.json`](results/model_provenance.json). Both decoder-only runs are prompt-only; QLoRA was not run in this reduced scope.

## Results

Metrics are corpus SacreBLEU (13a), chrF, and TER. BLEU confidence intervals use 400 sentence-level bootstrap resamples. Inference time measures generation on the 200 examples and excludes model loading. Prompt-only systems have no training time.

**Headline test-set results (200 fixed random rows; seed 42).** BLEU is SacreBLEU 13a; the sample is drawn from the official IITB test split.
chrF uses character order 6 and word order 0; TER is lower-is-better. The 95 % interval is a
sentence-level bootstrap (400 resamples, fixed seed). `Train/adapt` is wall-clock seconds; `—` means the
row did not train (prompt-only). `Batch` and `Decoding` are the inference settings actually used, because
throughput is only comparable when those are stated.

| System | Arch | Params (M) | Train/adapt (s) | Inference (s) | Sent/s | Batch | Decoding | BLEU | 95% CI | chrF | TER |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Basic Transformer (from scratch) | enc-dec | 15.2 | 463.9 | 15.07 | 13.27 | 32 | beam=4, length_penalty=0.6 | 0.26 | [0.19, 0.33] | 9.62 | 145.4 |
| mT5-small (fine-tuned) | enc-dec | 300.2 | 4177.8 | 134.26 | 1.49 | 4 | beam=4 | 4.55 | [3.47, 5.60] | 23.83 | 87.7 |
| TinyLlama 1.1B (3-shot prompt) | decoder-only | 1100.0 | — | 389.144 | 0.51 | 12 | greedy | 0.36 | [0.26, 0.45] | 13.66 | 118.1 |
| Gemma 3 1B (3-shot prompt) | decoder-only | 999.9 | — | 250.85 | 0.80 | 12 | greedy | 10.95 | [9.35, 12.38] | 35.57 | 85.0 |

**Best BLEU:** `gemma_prompt3shot` at 10.95 BLEU / 35.57 chrF / 85.0 TER.


On this fixed sample, Gemma has the highest point scores (10.95 BLEU, 35.57 chrF, 85.0 TER), followed by mT5 (4.55, 23.83, 87.7). TinyLlama and the from-scratch Transformer score 0.36 and 0.26 BLEU. The basic model is fastest here at 13.27 sentences/s; Gemma reaches 0.80, mT5 1.49, and TinyLlama 0.51 with different batch sizes and decoding settings. These are system-level results, not a controlled architecture experiment: pretraining, data volume, tokenizer, model size, and adaptation all differ. The decoder parameter counts are full unquantized counts; 4-bit loading changes storage, not model size.

Plots: [BLEU by system](results/figures/bleu_by_system.png), [BLEU vs full parameter count](results/figures/bleu_vs_params.png), and [temperature analysis](results/figures/temperature_analysis.png).

## Sampling temperature

The decoder prompts are also evaluated on 20 fixed rows from the shared sample at temperatures 0.2, 0.7, and 1.0. [`results/temperature/temperature_results.json`](results/temperature/temperature_results.json) records the row IDs and metrics; JSONL files contain the generated examples. Greedy decoding remains the setting used in the headline table.

| System | Temperature | BLEU | chrF | Distinct-2 |
|---|---:|---:|---:|---:|
| TinyLlama (3-shot) | 0.2 | 0.408 | 13.861 | 0.648 |
| TinyLlama (3-shot) | 0.7 | 0.568 | 15.531 | 0.900 |
| TinyLlama (3-shot) | 1.0 | 0.559 | 15.436 | 0.980 |
| Gemma 3 1B (3-shot) | 0.2 | 12.462 | 38.772 | 0.949 |
| Gemma 3 1B (3-shot) | 0.7 | 10.259 | 35.902 | 0.938 |
| Gemma 3 1B (3-shot) | 1.0 | 8.826 | 34.734 | 0.933 |

On these 20 rows, higher temperature raised TinyLlama's distinct-2 while its BLEU remained below 0.6; Gemma's BLEU and chrF were highest at 0.2 and declined at 0.7 and 1.0. This small descriptive study does not establish a general decoding optimum and was not used to select the headline prompt. See [temperature plot](results/figures/temperature_analysis.png).

## Responsible use and qualitative examples

The project screens real outputs for empty or non-Devanagari text, number changes, untranslated English, repetition, and unusually short or long outputs. These are review flags, not proof of a semantic error. Examples and a human-review worksheet are in [`results/error_analysis/`](results/error_analysis/); fixed qualitative examples are in [`results/qualitative_samples.md`](results/qualitative_samples.md). The report discusses bias, hallucination, cultural or sensitive-content risks, and mitigations.

On 200 outputs, the automated screen found no empty translations. Repetition-loop flags occurred in 47.0% of basic Transformer outputs, 9.5% of mT5, 22.5% of TinyLlama, and 0.5% of Gemma outputs. Gemma also had no Devanagari in 5.5% and untranslated Latin text in 14.0%. These are heuristic review flags, not human judgments. In test row 1126, mT5 changes Australian airline passengers to railway passengers in the United States and omits the take-off/landing context. Row 356 shows variation in a person’s name and gendered wording; row 899 contains a statement about unwanted sexual attention, where meaning changes could harm the speaker. Exact sources, references, and outputs are in [`results/qualitative_samples.md`](results/qualitative_samples.md); flagged examples and a human-review worksheet are in [`results/error_analysis/`](results/error_analysis/).

Bias and cultural misrepresentation are relevant risks, but no systematic human bias audit was performed. For sensitive or consequential use, require qualified bilingual review; check names, numbers, negation, and context; flag suspicious repetition or language/script mismatches; and route uncertain cases to a human. Do not use these models as the sole source for medical, legal, or public decisions. No bilingual human adequacy study was performed.

## Reproduce the reduced evaluation

Verified environment: Windows 11, Python 3.13, PyTorch 2.9.1 with CUDA 12.8, RTX 5050 Laptop GPU (8 GB). From the repository root, install the pinned dependencies from `requirements.txt`, set `PYTHONPATH=src` and `HF_HOME=.hf`, and make the trained checkpoints available locally. The model caches are not included in the source tree.

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:HF_HOME = "$PWD\.hf"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts\00b_select_eval_sample.py
.\.venv\Scripts\python.exe scripts\02b_infer_mt5.py
.\.venv\Scripts\python.exe scripts\01b_infer_transformer.py
.\.venv\Scripts\python.exe scripts\03_llama.py --mode prompt --shots 3
.\.venv\Scripts\python.exe scripts\04_gemma.py --mode prompt --shots 3
.\.venv\Scripts\python.exe scripts\05_evaluate.py
.\.venv\Scripts\python.exe scripts\06_temperature.py --limit 20
.\.venv\Scripts\python.exe scripts\07_ethics_analysis.py
.\.venv\Scripts\python.exe scripts\14_sync_parameter_counts.py
.\.venv\Scripts\python.exe scripts\10_collect_facts.py
.\.venv\Scripts\python.exe scripts\11_fill_results_table.py
.\.venv\Scripts\python.exe scripts\12_qualitative_samples.py
.\.venv\Scripts\python.exe scripts\08_check_artifacts.py
```

Or run `scripts/run_all.ps1`. It uses existing checkpoints and cached models; it does not redownload the dataset, train models, or run a full-test sweep. The exact baseline training command and subset manifest are recorded in `logs/config_used_basic_reduced.yaml` and `results/basic_train_subset.json`; the mT5 training log is `logs/train_mt5.jsonl`.

## Conclusion

Gemma led this 200-row sample, but its 10.95 BLEU and observed untranslated-text flags still require human checking. mT5 scored 4.55 BLEU after one completed epoch; the scratch Transformer used 12,000 training examples and showed frequent repetition; TinyLlama is a LLaMA-family substitute, not an official Meta LLaMA model. Results are limited by the small sample, one training seed, unequal model/data scale, and heuristic-only output screening. Treat this as an assignment comparison, not a production translation benchmark. The student-specific named cover DOCX and Word-exported PDF are prepared locally for LMS submission and are excluded from the public GitHub package because they contain student identifiers.

The named DOCX and PDF were exported and checked locally in Word; the PDF is one page, contains the required fields, and has no placeholders. They are excluded from the public repository because they contain student identifiers. See [`docs/report.md`](docs/report.md) for the method, results, limitations, and references, and [`docs/requirements_traceability.md`](docs/requirements_traceability.md) for the assignment checklist. The GitHub URL above was supplied by the student.
