# English–Hindi Machine Translation: Reduced-Scope Evaluation

**Foundations of Generative AI (ICT4403), Assignment 3**

## 1. Objective

This work implements a small Transformer encoder–decoder from scratch, fine-tunes mT5, and evaluates two decoder-only models with few-shot prompting. The comparison reports translation quality, training or adaptation cost, inference time, and parameter count. The analysis covers architecture confounders, temperature, and responsible use.

## 2. Dataset and evaluation sample

The [IIT Bombay English–Hindi Parallel Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/) is used through the Hugging Face mirror [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi). The corpus is licensed CC BY-NC 4.0. The provider’s train, validation, and test memberships are preserved. Text cleanup applies Unicode NFC, whitespace and control-character cleanup, Hindi-script checks, length-ratio and length filters, and training deduplication. Leakage checks use Unicode NFKC, case folding, punctuation removal, and whitespace folding for exact normalized source matching; the audit found zero overlaps. Counts and SHA-256 hashes are in `data/processed/dataset_manifest.json` and `results/data_stats.json`.

The processed training file contains 60,000 examples selected with seed 42. The corrected from-scratch model uses a deterministic 12,000-row subset of that file; mT5 uses the 60,000-row file. The official validation set has 520 examples. The official test set has 2,507 examples, from which a uniform random sample of 200 is selected once with seed 42. Its original row indices and test-file hash are saved in `results/evaluation_sample.json` before comparison. All four headline systems use those exact rows in the same order. No test result was used to select a checkpoint or prompt.

### Exploratory data analysis

The EDA figure and machine-readable summary are `results/figures/dataset_eda.png` and `results/eda_summary.json`; `scripts/15_eda.py` generates them from the saved processed TSVs. Before reading each split, the script checks its SHA-256 against `data/processed/dataset_manifest.json`, then confirms that the recomputed length, uniqueness, and Devanagari-ratio statistics match `results/data_stats.json`. The training view is the fixed 60,000-row shared subset, while validation (520) and test (2,507) are the full provider splits. The manifest records 1,659,083 raw training rows, 51,232 rows dropped during cleaning, and 258,395 duplicate training pairs removed before the 60,000-row subset was selected; validation and test each record zero cleaning drops.

There are no blank source/target pairs in the three processed files. Counting exact normalized pair repetitions with the same NFKC/case-fold/punctuation/whitespace key gives 0 additional rows in train, 0 in validation, and 1 in test. That within-split test duplicate is distinct from the cross-split source leakage audit, which reports zero overlap. Source and target unique-sentence counts are 57,503/56,884 in train, 520/520 in validation, and 2,506/2,506 in test.

Median whitespace-token lengths (source/target) are 11/11 for train, 16/16 for validation, and 18/20 for test. Median Unicode-code-point lengths are 59/57, 97.5/82, and 107/104 respectively. Devanagari accounts for about 99.15%, 100.00%, and 99.53% of target alphabetic characters in train, validation, and test; the mean per-sentence Devanagari-letter ratios are 0.9881, 1.0000, and 0.9958. The plots show the length distributions and the small Latin/other-script remainder. These are descriptive properties of the saved corpus files: the training statistics cover only the shared 60,000-row subset, the headline metric sample remains 200 rows from the full 2,507-row test split, and length/script checks do not measure translation adequacy, corpus representativeness, or systematic bias.

![Dataset exploratory data analysis: source and target length distributions and target-script composition for train, validation, and test splits.](../results/figures/dataset_eda.png)

*Figure 1. Processed split length distributions and Hindi-target alphabetic script composition. The small Latin/other-script remainder is labelled above each bar.*

## 3. Systems

### Basic Transformer

The encoder–decoder is implemented in `src/fgai_mt/model_basic.py`: six encoder and six decoder layers, 256-dimensional embeddings, eight heads, and an 8,000-piece SentencePiece tokenizer. Embeddings are scaled by √d_model and final layer normalisation is applied. The model has approximately 15.16 million parameters. A bounded run uses at most 30 epochs on 12,000 training rows and stops after four consecutive validation-loss misses. The checkpoint with the lowest validation loss is evaluated.

### mT5-small

`google/mt5-small` has approximately 300.18 million parameters. A fresh run from the cached base completed all three planned epochs on the 60,000-row training file (5,625 optimizer steps). Validation loss decreased from 2.4672 after epoch 1 to 2.3268 after epoch 2 and 2.2517 after epoch 3. Measured training time was 12,379.4 seconds (3 hours, 26 minutes, 19.4 seconds; 4,126.5 seconds per epoch). The promoted epoch-3 best checkpoint uses the prefix "translate English to Hindi:" and beam search. The earlier epoch-1 checkpoint and its 2026-10-27 training records are preserved locally as separate evidence.

### Decoder-only models

TinyLlama 1.1B is a LLaMA-family `LlamaForCausalLM` model under Apache-2.0. It is not an official Meta LLaMA checkpoint. The official Meta repositories were gated, so they were not used. Gemma 3 1B instruction-tuned weights are loaded from the ungated Unsloth mirror; Gemma’s terms still apply. Exact revisions are in `results/model_provenance.json`.

Both models use the same three fixed English–Hindi examples from the training split and greedy decoding for the headline comparison. They are prompt-only: no QLoRA or other decoder training is included. This satisfies the assignment’s prompting-or-fine-tuning option while keeping the remaining work small.

## 4. Metrics and comparison protocol

SacreBLEU uses the 13a tokenizer; chrF uses character order 6 and word order 0. TER is included as a lower-is-better secondary metric. BLEU intervals use 400 sentence-level bootstrap resamples. Inference time is generation-only wall-clock over the same 200 examples, excluding model loading. Training time is the measured wall-clock for completed training; prompting rows have no training time. Parameter count is the full model’s parameter count, even when decoder weights are loaded in 4-bit form.

The result table and paired comparisons are generated from saved JSONL translations. Timing is hardware-dependent and the bootstrap interval measures sample uncertainty, not run-to-run training variation.

## 5. Results

**Headline test-set results (200 fixed random rows; seed 42).** BLEU is SacreBLEU 13a; the sample is drawn from the official IITB test split.
chrF uses character order 6 and word order 0; TER is lower-is-better. The 95 % interval is a
sentence-level bootstrap (400 resamples, fixed seed). `Train/adapt` is wall-clock seconds; `—` means the
row did not train (prompt-only). `Batch` and `Decoding` are the inference settings actually used, because
throughput is only comparable when those are stated.

| System | Arch | Params (M) | Train/adapt (s) | Inference (s) | Sent/s | Batch | Decoding | BLEU | 95% CI | chrF | TER |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Basic Transformer (from scratch) | enc-dec | 15.2 | 463.9 | 15.07 | 13.27 | 32 | beam=4, length_penalty=0.6 | 0.26 | [0.19, 0.33] | 9.62 | 145.4 |
| mT5-small (fine-tuned) | enc-dec | 300.2 | 12379.4 | 128.95 | 1.55 | 4 | beam=4 | 7.58 | [6.38, 8.75] | 29.34 | 82.1 |
| TinyLlama 1.1B (3-shot prompt) | decoder-only | 1100.0 | — | 389.144 | 0.51 | 12 | greedy | 0.36 | [0.26, 0.45] | 13.66 | 118.1 |
| Gemma 3 1B (3-shot prompt) | decoder-only | 999.9 | — | 250.85 | 0.80 | 12 | greedy | 10.95 | [9.35, 12.38] | 35.57 | 85.0 |

**Best BLEU:** `gemma_prompt3shot` at 10.95 BLEU / 35.57 chrF / 85.0 TER.


Gemma leads the point estimates on the fixed sample (10.952 BLEU, 35.566 chrF, 84.998 TER); mT5 follows at 7.581 BLEU, 29.335 chrF, and 82.141 TER. TinyLlama scores 0.358 BLEU and the from-scratch Transformer 0.262. The basic model is fastest at 13.27 sentences/s, followed by mT5 at 1.55, Gemma at 0.80, and TinyLlama at 0.51; batch sizes and decoding differ, so throughput is descriptive rather than a controlled speed ranking. The basic model's low score and high repetition flag, and TinyLlama's low BLEU, are material weaknesses. Gemma's stronger scores do not remove its untranslated-text flags.

Test row 1126 is one fixed qualitative example: mT5 repeats the phone/tablet shutdown instruction and omits the take-off/landing context. Row 356 concerns a named person and family relationship; outputs vary the name and gendered wording. Row 899 contains a statement about unwanted sexual attention, where a mistranslation could change the meaning of sensitive content. These examples illustrate hallucination, entity, gender, and sensitive-content risks; they are not a human adequacy rating. `results/qualitative_samples.md` contains exact sources, references, and outputs from all four systems. Current plots are saved in `results/figures/` (BLEU by system, BLEU vs full parameter count, and temperature analysis).

These models do not isolate architecture. The scratch Transformer has 15.16M parameters and 12,000 training rows; mT5 has about 300.18M and used 60,000 rows for three completed epochs; the prompted decoder models have about 1.10B and 1.00B full parameters and no gradient updates. Pretraining, tokenization, training volume, and adaptation differ. The comparison supports observations about these four systems under the stated setup, not a causal claim that one architecture is inherently better. Decoder counts use original safetensors tensor shapes before 4-bit quantization.


## 6. Sampling-temperature analysis

The same three-shot decoder prompts are sampled on the first 20 row IDs from the fixed 200-row test sample. Temperatures are 0.2, 0.7, and 1.0 with fixed seed 42 and top-p 0.9. BLEU and chrF measure reference similarity; distinct-2 and output diagnostics show variation and failure behaviour. These 20 examples are a small descriptive study, not a precise quality estimate. The exact outputs and settings are stored under `results/temperature/`.

| Model | Temperature | BLEU | chrF | Distinct-2 |
|---|---:|---:|---:|---:|
| TinyLlama (3-shot) | 0.2 | 0.408 | 13.861 | 0.648 |
| TinyLlama (3-shot) | 0.7 | 0.568 | 15.531 | 0.900 |
| TinyLlama (3-shot) | 1.0 | 0.559 | 15.436 | 0.980 |
| Gemma 3 1B (3-shot) | 0.2 | 12.462 | 38.772 | 0.949 |
| Gemma 3 1B (3-shot) | 0.7 | 10.259 | 35.902 | 0.938 |
| Gemma 3 1B (3-shot) | 1.0 | 8.826 | 34.734 | 0.933 |

Across the same 20 row IDs, TinyLlama's distinct-2 increased with temperature while BLEU stayed below 0.6. Gemma's reference scores were highest at 0.2 and declined at higher temperatures. This small descriptive sample does not establish a universal decoding optimum and was not used to select the headline prompt.

## 7. Qualitative and responsible-AI analysis

`results/qualitative_samples.md` shows fixed examples shared by all four systems. `results/error_analysis/risk_flags.json` and `flagged_examples.md` screen outputs for empty text, missing Devanagari, number changes, untranslated English, repetition, and abnormal length. Automatic flags are screening signals; they do not prove a translation is wrong. The review worksheet is available for a Hindi speaker to assess adequacy, fluency, bias, hallucination, and cultural errors.

Potential risks include changing names or numbers, adding unsupported details, erasing or stereotyping caste, religious, gender, political, or cultural content, and failing on health or other sensitive text. Mitigations are to require qualified human review for sensitive or consequential translations, preserve named entities and quantities, flag low-confidence or script-mismatched output, and avoid using these small models as the sole source for legal, medical, or public decisions. A production system would also need broader dialect and domain evaluation, privacy controls, and an escalation path.

The 200-row automated screen found zero empty outputs. Repetition-loop flags occurred in 47.0% of basic Transformer outputs, 6.5% of mT5, 22.5% of TinyLlama, and 0.5% of Gemma outputs. mT5 also had untranslated Latin text in 1.0% of outputs. Gemma had no Devanagari in 5.5% of outputs and untranslated Latin spans in 14.0%. Other number-change flags are recorded in `results/error_analysis/risk_flags.json`. These heuristics can miss semantic errors or flag acceptable text; they are not human quality ratings.

Mitigations are qualified bilingual review for consequential or sensitive text, checking names and quantities, script/language and repetition checks, and escalation of suspicious cases. Do not rely on these systems alone for medical, legal, or public decisions. The project includes a human-review worksheet and real sample outputs, but no independent human adequacy study was completed.

## 8. Limitations and conclusion

The fixed 200-example test sample is intentionally small; its scores are useful for assignment comparison, not for claiming benchmark-level quality. The model runs use one training seed. mT5 completed all three planned epochs. The basic Transformer and mT5 used different training volumes, and the decoder-only systems received prompt examples instead of gradient updates. TinyLlama and Gemma are documented family representatives, not the gated official Meta and Google repositories. Automatic metrics do not replace bilingual human review.

On the fixed 200-row sample, Gemma is the strongest of these four systems by the reported automatic metrics, while its untranslated-text flags and all systems' qualitative errors show that none can be treated as reliable without review. mT5 completed all three planned epochs, the scratch Transformer used 12,000 examples, and the decoder systems were prompt-only. This is a small assignment comparison, not a production benchmark. A broader test set, repeated training seeds, and bilingual human evaluation would be needed for stronger conclusions.

## References

- Kunchukuttan, Mehta & Bhattacharyya (2018). *The IIT Bombay English-Hindi Parallel Corpus*. LREC.
- Vaswani et al. (2017). *Attention Is All You Need*.
- Xue et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*.
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*.
- Gemma Team (2025). *Gemma 3 Technical Report*.
- Post (2018). *A Call for Clarity in Reporting BLEU Scores*.
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation*.
