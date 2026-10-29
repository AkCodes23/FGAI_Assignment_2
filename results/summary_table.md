| System | Arch | Params (M) | Train/adapt (s) | Inference (s) | Sent/s | Batch | Decoding | BLEU | 95% CI | chrF | TER |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| Basic Transformer (from scratch) | enc-dec | 15.2 | 463.9 | 15.07 | 13.27 | 32 | beam=4, length_penalty=0.6 | 0.26 | [0.19, 0.33] | 9.62 | 145.4 |
| mT5-small (fine-tuned) | enc-dec | 300.2 | 12379.4 | 128.95 | 1.55 | 4 | beam=4 | 7.58 | [6.38, 8.75] | 29.34 | 82.1 |
| TinyLlama 1.1B (3-shot prompt) | decoder-only | 1100.0 | — | 389.144 | 0.51 | 12 | greedy | 0.36 | [0.26, 0.45] | 13.66 | 118.1 |
| Gemma 3 1B (3-shot prompt) | decoder-only | 999.9 | — | 250.85 | 0.80 | 12 | greedy | 10.95 | [9.35, 12.38] | 35.57 | 85.0 |
