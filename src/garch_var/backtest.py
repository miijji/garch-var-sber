"""Coverage tests with stable boundary likelihoods."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2


def _ll(successes: int, total: int, probability: float) -> float:
    return float(xlogy(successes, probability) + xlogy(total - successes, 1 - probability))


def coverage_tests(hits, alpha: float) -> dict:
    raw = np.asarray(hits)
    if raw.ndim != 1 or len(raw) == 0 or not np.isin(raw, [0, 1]).all():
        raise ValueError("Нужна непустая последовательность индикаторов 0/1 без пропусков.")
    if not 0 < alpha < 1:
        raise ValueError("alpha должна быть между 0 и 1.")
    x = raw.astype(int)
    n, k = len(x), int(x.sum())
    lr_uc = max(0., 2 * (_ll(k, n, k / n) - _ll(k, n, alpha)))
    counts = np.zeros((2, 2), dtype=int)
    for previous, current in zip(x[:-1], x[1:]):
        counts[previous, current] += 1
    n0, n1 = counts.sum(axis=1)
    # If one prior state is unobserved, the asymptotic independence test is uninformative.
    lr_ind = p_ind = lr_cc = p_cc = float("nan")
    if n0 > 0 and n1 > 0:
        total = int(counts.sum())
        successes = int(counts[:, 1].sum())
        separate = sum(_ll(int(counts[i, 1]), int(counts[i].sum()), counts[i, 1] / counts[i].sum()) for i in range(2))
        lr_ind = max(0., 2 * (separate - _ll(successes, total, successes / total)))
        p_ind = float(chi2.sf(lr_ind, 1))
        lr_cc = lr_uc + lr_ind
        p_cc = float(chi2.sf(lr_cc, 2))
    return {"n": n, "exceptions": k, "expected": n * alpha, "exception_rate": k / n,
            "kupiec_lr": lr_uc, "kupiec_p": float(chi2.sf(lr_uc, 1)),
            "binomial_exact_p": float(binomtest(k, n, alpha).pvalue),
            "independence_lr": lr_ind, "independence_p": p_ind,
            "conditional_coverage_lr": lr_cc, "conditional_coverage_p": p_cc,
            "n00": int(counts[0, 0]), "n01": int(counts[0, 1]), "n10": int(counts[1, 0]), "n11": int(counts[1, 1])}


def summarize(predictions: pd.DataFrame, levels=(0.95, 0.99)) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=False):
        if group.index.has_duplicates or not group.index.is_monotonic_increasing:
            raise ValueError("Прогнозы каждой модели должны иметь уникальные отсортированные даты.")
        for level in levels:
            tag = str(int(round(level * 100)))
            columns = ["actual_pct", f"q_{tag}_pct"]
            if not np.isfinite(group[columns].to_numpy()).all():
                raise ValueError("Нельзя удалять неудачные прогнозы из бэктеста молча.")
            hits = group.actual_pct < group[f"q_{tag}_pct"]
            result = coverage_tests(hits, 1 - level)
            result.update(model=model, confidence=level,
                          realized_tail_loss_pct=float(-group.loc[hits, "actual_pct"].mean()),
                          mean_predicted_es_on_hits_pct=float(group.loc[hits, f"es_{tag}_loss_pct"].mean()) if f"es_{tag}_loss_pct" in group else float("nan"),
                          quantile_loss=float(np.mean((1 - level - (group.actual_pct < group[f"q_{tag}_pct"]).astype(float)) * (group.actual_pct - group[f"q_{tag}_pct"]))))
            rows.append(result)
    return pd.DataFrame(rows)
