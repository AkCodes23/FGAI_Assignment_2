| System | Arch | Params (M) | Train/adapt (s) | Inference (s) | Sent/s | Batch | Decoding | BLEU | 95% CI | chrF | TER |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Basic Transformer (from scratch) | enc-dec | 15.2 | 463.9 | 15.07 | 13.27 | 32 | beam=4, length_penalty=0.6 | 0.26 | [0.19, 0.33] | 9.62 | 145.4 |
| mT5-small (fine-tuned) | enc-dec | 300.2 | 4177.8 | 134.26 | 1.49 | 4 | beam=4 | 4.55 | [3.47, 5.60] | 23.83 | 87.7 |
| TinyLlama 1.1B (3-shot prompt) | decoder-only | 1100.0 | — | 389.144 | 0.51 | 12 | greedy | 0.36 | [0.26, 0.45] | 13.66 | 118.1 |
| Gemma 3 1B (3-shot prompt) | decoder-only | 999.9 | — | 250.85 | 0.80 | 12 | greedy | 10.95 | [9.35, 12.38] | 35.57 | 85.0 |
