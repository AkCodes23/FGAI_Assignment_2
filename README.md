# English–Hindi machine translation: four-system comparison

This assignment compares four systems on one shared evaluation sample. The full analysis, exact example translations, and figure interpretations are in the [technical report](docs/report.md).

## 1. Problem statement and objective

Translate English sentences into Hindi while preserving meaning, names, numbers, and sentence structure. The comparison covers a small Transformer trained from scratch, fine-tuned mT5, and two decoder-only instruction models evaluated with three examples in the prompt. [Read the problem definition and scope](docs/report.md#1-problem-and-objective).

## 2. Dataset description and source

The data is the IIT Bombay English–Hindi Parallel Corpus from [CFILT, IIT Bombay](https://www.cfilt.iitb.ac.in/iitb_parallel/), distributed under CC BY-NC 4.0. The [Hugging Face dataset mirror](https://huggingface.co/datasets/cfilt/iitb-english-hindi) provides the same corpus. The processed splits contain 60,000 training pairs, 520 validation pairs, and a 2,507-example test set. The **200-row sample drawn once from the official 2,507-row test split** uses seed 42 and the same row IDs, sources, and references for all four systems.

![Dataset preparation and shared evaluation flow](results/figures/pipeline_flow.png)

The diagram shows one cleaning and split path feeding four systems and one paired 200-row sample. Each system is therefore scored against the same 200 references. [Dataset counts, cleaning, and EDA](docs/report.md#2-dataset-and-evaluation-sample).

![Dataset exploratory data analysis](results/figures/dataset_eda.png)

The test split has median source and target lengths of 18 and 20 whitespace words. The report explains the full split-level distributions and script composition.

## 3. Models implemented

| System | Architecture | Parameters | Task adaptation |
|---|---|---:|---|
| Basic Transformer | Encoder–decoder | 15.2M | Trained from scratch on 12,000 pairs |
| mT5-small | Encoder–decoder | 300.2M | Fine-tuned on 60,000 pairs |
| TinyLlama 1.1B | Decoder-only | 1,100.0M | Three-shot prompt; no local weight updates |
| Gemma 3 1B | Decoder-only | 999.9M | Three-shot prompt; no local weight updates |

TinyLlama is a LLaMA-family checkpoint, not an official Meta LLaMA checkpoint. Gemma was loaded from a mirror; Gemma terms apply. [Architecture, pretraining, tokenization, and prompt details](docs/report.md#3-systems-and-adaptation).

![Encoder–decoder path and decoder-only three-shot prompt structure](results/figures/architecture_comparison.png)

The diagram contrasts source encoding followed by target decoding with a causal prompt. The plotted parameter counts range from 15.2M for the scratch model to about 1.1B for TinyLlama; the decoder-only runs use three fixed examples.

## 4. Execution steps and dependencies

The recorded environment used Windows 11, Python 3.13, PyTorch 2.9.1, CUDA 12.8, and the packages listed in [requirements.txt](requirements.txt), including Transformers, Hugging Face Datasets, SentencePiece, SacreBLEU, pandas, NumPy, and Matplotlib. Install the CUDA-compatible PyTorch build before the remaining pinned packages.

From the project root, prepare the virtual environment and dependencies:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt

The saved-data evaluation command is:

    powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1

It requires the processed splits, trained checkpoints, and model cache. It selects the sample, runs inference and scoring, evaluates 20 temperature rows, produces screening and EDA artifacts, and runs the artifact gate. It sets offline mode and uses existing checkpoints. For the data preparation and training commands, see [execution details](docs/report.md#4-execution-and-dependencies).

## 5. Results

Scores below are corpus BLEU (SacreBLEU 13a), corpus chrF, and TER; lower TER is better. All are measured on the same 200 rows.

| System | BLEU (95% CI) | chrF | TER | Inference seconds |
|---|---:|---:|---:|---:|
| Basic Transformer | 0.262 (0.190–0.326) | 9.624 | 145.377 | 15.07 |
| mT5-small | 7.581 (6.376–8.753) | 29.335 | 82.141 | 128.95 |
| TinyLlama 1.1B | 0.358 (0.264–0.448) | 13.663 | 118.142 | 389.144 |
| Gemma 3 1B | 10.952 (9.345–12.383) | 35.566 | 84.998 | 250.85 |

![BLEU and mean sentence-level chrF with 95% bootstrap intervals](results/figures/score_intervals.png)

Gemma has the highest corpus BLEU at 10.952, while mT5 is second at 7.581. In the separate sentence-level chrF summary, Gemma’s mean is 34.180 (95% interval 32.169–36.235) and mT5’s is 29.486 (27.759–31.419). [All result details and uncertainty](docs/report.md#6-results).

## 6. Model comparison and evaluation

The four systems vary in model size, pretraining, tokenizer, and adaptation. On the same 2,507 Hindi test references, tokenizer fertility is 1.586 pieces per whitespace word for the Basic Transformer, 2.058 for mT5, 5.55 for TinyLlama, and 1.364 for Gemma. The two decoder prompt models did not receive task-specific weight updates. The report includes a direct comparison of the encoder–decoder and decoder-only systems, with their scores, adaptation costs, inference time, throughput, and tokenizer measurements.

![Corpus chrF by inference time and parameter count](results/figures/quality_cost.png)

Gemma has the highest corpus chrF at 35.566 and takes 250.85 seconds for 200 sentences; mT5 scores 29.335 chrF in 128.95 seconds. The scratch Transformer scores 9.624 chrF in 15.07 seconds, while TinyLlama scores 13.663 in 389.144 seconds. These are observed run times, not controlled hardware-efficiency estimates. Automated screening found repetition in 47.0% of scratch-model outputs and untranslated Latin text in 14.0% of Gemma outputs; these flags require output review.

The report compares pretraining, vocabulary, tokenizer fertility, and task adaptation numerically. It also shows exact source, reference, and four output records for rows 1126, 356, and 899, and discusses observed name, number, gender, repetition, and sensitive-content errors. [Model comparison, evaluation, and responsible-use analysis](docs/report.md#7-model-comparison-and-evaluation) and [sampled error review](docs/report.md#9-responsible-use-and-sampled-errors).

## 7. Sampling-temperature analysis

The decoder models were also evaluated on the same 20 rows at temperatures 0.2, 0.7, and 1.0. This small study is descriptive; the 20 rows are the first 20 rows of the fixed 200-row sample.

| System | T | BLEU | chrF | Distinct-2 |
|---|---:|---:|---:|---:|
| TinyLlama | 0.2 | 0.408 | 13.861 | 0.6481 |
| TinyLlama | 0.7 | 0.568 | 15.531 | 0.8998 |
| TinyLlama | 1.0 | 0.559 | 15.436 | 0.9798 |
| Gemma 3 1B | 0.2 | 12.462 | 38.772 | 0.9490 |
| Gemma 3 1B | 0.7 | 10.259 | 35.902 | 0.9375 |
| Gemma 3 1B | 1.0 | 8.826 | 34.734 | 0.9330 |

![Temperature scores and diversity](results/figures/temperature_analysis.png)

Gemma’s BLEU falls from 12.462 at 0.2 to 8.826 at 1.0. TinyLlama’s Distinct-2 rises from 0.6481 to 0.9798, while its BLEU remains below 0.6. [Temperature analysis](docs/report.md#8-sampling-temperature-analysis).

## 8. Conclusion

Gemma 3 1B is the best system by corpus BLEU and chrF on this evaluation sample. Its output still needs human review: the saved sample contains a 14.0% untranslated-Latin flag rate. mT5 is second on BLEU and faster at inference; model selection depends on the required quality and latency. The [technical report](docs/report.md#11-conclusion) gives the limitations and supporting comparisons.
