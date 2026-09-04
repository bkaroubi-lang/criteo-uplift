"""Uplift evaluation: Qini curve, Qini coefficient, uplift-by-decile.

All functions operate on evaluation-set arrays and are estimator-agnostic:
they take a CATE score per user plus observed (treatment, outcome).

Qini construction (Radcliffe 2007 / Diemert et al. 2018 convention)
-------------------------------------------------------------------
Rank users by descending CATE score. Targeting the top-k users, the Qini
value credits the incremental outcomes relative to a control group rescaled
to the treated count:

    q(k) = Y_t(k) - Y_c(k) * n_t(k) / n_c(k)

where Y_t, Y_c are cumulative outcomes and n_t, n_c cumulative counts among
the top-k. The random-targeting baseline is the straight line from (0,0) to
(N, q(N)). The Qini coefficient is the area between the curve and that line,
normalised by the area of the "perfect" reference so it is comparable across
outcomes and samples.
"""
from __future__ import annotations

import numpy as np

# numpy>=2 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def qini_curve(score: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
    """Return (x_fraction, qini_values) with a leading (0, 0) point."""
    order = np.argsort(-score, kind="mergesort")  # stable, ties keep input order
    t = treatment[order].astype(np.float64)
    y = outcome[order].astype(np.float64)

    n_t = np.cumsum(t)
    n_c = np.cumsum(1.0 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1.0 - t))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c > 0, n_t / n_c, 0.0)
    q = y_t - y_c * ratio

    n = len(score)
    x = np.arange(1, n + 1) / n
    x = np.concatenate([[0.0], x])
    q = np.concatenate([[0.0], q])
    return x, q


def qini_coefficient(score: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """Normalised Qini coefficient (area vs random baseline / |area of curve endpoint|).

    Returns a scale-free number where >0 means the score ranks users better than
    random targeting, ~0 means no usable heterogeneity, <0 means anti-targeting.
    """
    x, q = qini_curve(score, treatment, outcome)
    q_end = q[-1]
    area_model = _trapz(q, x)                   # area under Qini curve
    area_random = 0.5 * q_end * x[-1]           # triangle under random line
    # Normalise by the random-line area magnitude so the number is comparable
    # across outcomes with different base rates; guard tiny/zero endpoints.
    denom = abs(area_random) if abs(area_random) > 1e-12 else 1.0
    return float((area_model - area_random) / denom)


def uplift_by_decile(score, treatment, outcome, n_bins: int = 10):
    """Observed uplift within score-ranked bins (bin 1 = highest predicted uplift).

    Returns a list of dicts with observed treated/control rates and their diff.
    A well-calibrated ranker shows monotonically decreasing observed uplift.
    """
    order = np.argsort(-score, kind="mergesort")
    t = treatment[order]
    y = outcome[order]
    bins = np.array_split(np.arange(len(score)), n_bins)
    rows = []
    for b, idx in enumerate(bins, start=1):
        tt, yy = t[idx], y[idx]
        treated = yy[tt == 1]
        control = yy[tt == 0]
        r_t = treated.mean() if len(treated) else np.nan
        r_c = control.mean() if len(control) else np.nan
        rows.append(
            {
                "decile": b,
                "n": len(idx),
                "n_treated": int((tt == 1).sum()),
                "rate_treated": r_t,
                "rate_control": r_c,
                "observed_uplift": r_t - r_c,
            }
        )
    return rows
