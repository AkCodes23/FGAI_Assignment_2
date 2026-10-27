"""MT metrics: sacrebleu BLEU/chrF, bootstrap confidence intervals, output diagnostics."""
from __future__ import annotations

import math
import random
import re
import statistics
from typing import Any, Sequence

import numpy as np
import sacrebleu

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def corpus_scores(hyps: Sequence[str], refs: Sequence[str], bleu_tokenizer: str = "13a") -> dict[str, float]:
    assert len(hyps) == len(refs), (len(hyps), len(refs))
    bleu = sacrebleu.corpus_bleu(list(hyps), [list(refs)], tokenize=bleu_tokenizer)
    chrf = sacrebleu.corpus_chrf(list(hyps), [list(refs)], char_order=6, word_order=0)
    chrf_pp = sacrebleu.corpus_chrf(list(hyps), [list(refs)], char_order=6, word_order=2)
    ter = sacrebleu.corpus_ter(list(hyps), [list(refs)])
    return {
        "bleu": round(bleu.score, 3),
        "bleu_precision_1": round(bleu.precisions[0], 2),
        "bleu_precision_2": round(bleu.precisions[1], 2),
        "bleu_precision_3": round(bleu.precisions[2], 2),
        "bleu_precision_4": round(bleu.precisions[3], 2),
        "bleu_bp": round(bleu.bp, 3),
        "chrf": round(chrf.score, 3),
        "chrf_pp": round(chrf_pp.score, 3),
        "ter": round(ter.score, 3),
    }


def _bleu_stat_matrix(hyps: Sequence[str], refs: Sequence[str], tokenizer: str = "13a"):
    """Per-sentence BLEU statistics, so bootstrap resamples can be scored by summing rows.

    Re-tokenising the full corpus 400 times is ~0.4 s each; summing cached statistics is
    microseconds and gives exactly the same corpus BLEU.
    """
    bleu = sacrebleu.BLEU(tokenize=tokenizer)
    stats = bleu._extract_corpus_statistics(list(hyps), [list(refs)])
    return bleu, np.asarray(stats, dtype=np.float64)


def _score_stats(bleu: sacrebleu.BLEU, stats: np.ndarray) -> float:
    return bleu._compute_score_from_stats(stats.tolist()).score


def bootstrap_ci(
    hyps: Sequence[str],
    refs: Sequence[str],
    metric: str = "bleu",
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Sentence-level bootstrap resampling -> 95% CI for a corpus-level metric."""
    rng = random.Random(seed)
    n = len(hyps)
    if metric == "bleu":
        bleu, matrix = _bleu_stat_matrix(hyps, refs)
        values = [_score_stats(bleu, matrix[[rng.randrange(n) for _ in range(n)]].sum(axis=0))
                  for _ in range(n_samples)]
    elif metric == "chrf":
        values = []
        for _ in range(n_samples):
            idx = [rng.randrange(n) for _ in range(n)]
            h = [hyps[i] for i in idx]
            r = [refs[i] for i in idx]
            values.append(sacrebleu.corpus_chrf(h, [r]).score)
    else:  # pragma: no cover
        raise ValueError(metric)
    values.sort()
    return {
        "mean": round(statistics.fmean(values), 3),
        "lo": round(values[int(0.025 * n_samples)], 3),
        "hi": round(values[int(0.975 * n_samples) - 1], 3),
        "n_samples": n_samples,
    }


def paired_bootstrap_bleu(
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    refs: Sequence[str],
    n_samples: int = 400,
    seed: int = 7,
) -> dict[str, float]:
    """Paired sentence bootstrap of BLEU(A) - BLEU(B) on identical test items."""
    assert len(hyps_a) == len(hyps_b) == len(refs)
    rng = random.Random(seed)
    n = len(refs)
    bleu, mat_a = _bleu_stat_matrix(hyps_a, refs)
    _, mat_b = _bleu_stat_matrix(hyps_b, refs)
    deltas = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(_score_stats(bleu, mat_a[idx].sum(axis=0)) - _score_stats(bleu, mat_b[idx].sum(axis=0)))
    deltas.sort()
    return {
        "delta_bleu": round(statistics.fmean(deltas), 3),
        "ci_lo": round(deltas[int(0.025 * n_samples)], 3),
        "ci_hi": round(deltas[int(0.975 * n_samples) - 1], 3),
        "p_delta_le_0": round(sum(1 for d in deltas if d <= 0) / n_samples, 3),
    }


def output_diagnostics(hyps: Sequence[str], srcs: Sequence[str]) -> dict[str, Any]:
    """Signals used in the responsible-AI section: empty, copy, script and length behaviour."""
    n = max(1, len(hyps))
    empty = sum(1 for h in hyps if not h.strip())
    latin_only = sum(1 for h in hyps if LATIN.search(h) and not DEVANAGARI.search(h))
    no_devanagari = sum(1 for h in hyps if not DEVANAGARI.search(h))
    identical_to_source = sum(1 for h, s in zip(hyps, srcs) if h.strip() == s.strip())
    ratios = [len(h.split()) / max(1, len(s.split())) for h, s in zip(hyps, srcs)]
    return {
        "empty_outputs": empty,
        "empty_rate": round(empty / n, 4),
        "no_devanagari": no_devanagari,
        "no_devanagari_rate": round(no_devanagari / n, 4),
        "latin_only": latin_only,
        "latin_only_rate": round(latin_only / n, 4),
        "copied_source": identical_to_source,
        "length_ratio_mean": round(statistics.fmean(ratios), 3),
        "length_ratio_median": round(statistics.median(ratios), 3),
        "length_ratio_gt2": sum(1 for r in ratios if r > 2.0),
        "length_ratio_lt0.5": sum(1 for r in ratios if r < 0.5),
    }


def geometric_mean(values: Sequence[float]) -> float:
    vals = [max(v, 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
