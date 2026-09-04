"""Print paste-ready article numbers from a completed run's result tables.

    python -m src.report_numbers --sample dev --outcome visit
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C


def targeting_stat(tag: str) -> None:
    """From the best estimator's CATE, report how much lift the top-k% captures."""
    cate = pd.read_parquet(C.TAB_DIR / f"cate_eval_{tag}.parquet")
    qini = pd.read_csv(C.TAB_DIR / f"qini_coefficients_{tag}.csv")
    best = qini.sort_values("qini_coefficient", ascending=False)["estimator"].iloc[0]
    dec = pd.read_csv(C.TAB_DIR / f"uplift_deciles_{tag}.csv")
    d = dec[dec.estimator == best].sort_values("decile")
    # cumulative share of positive observed uplift captured by top-k deciles
    up = d["observed_uplift"].clip(lower=0).to_numpy()
    total = up.sum()
    cum = np.cumsum(up) / total if total > 0 else np.zeros_like(up)
    print(f"\n== TARGETING (best ranker = {best}) ==")
    for k in (1, 2, 3, 5):
        print(f"  top {k*10:>2}% of users -> {cum[k-1]*100:5.1f}% of positive incremental outcomes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="dev")
    ap.add_argument("--outcome", default=C.PRIMARY_OUTCOME)
    a = ap.parse_args()
    tag = f"{a.sample}_{a.outcome}"

    print("== QINI COEFFICIENTS ==")
    print(pd.read_csv(C.TAB_DIR / f"qini_coefficients_{tag}.csv").to_string(index=False))

    print("\n== CATE SUMMARY ==")
    print(pd.read_csv(C.TAB_DIR / f"cate_summary_{tag}.csv").round(5).to_string(index=False))

    print("\n== ESTIMATOR AGREEMENT (Spearman) ==")
    corr = pd.read_csv(C.TAB_DIR / f"estimator_agreement_spearman_{tag}.csv", index_col=0)
    print(corr.round(3).to_string())
    off = corr.where(~np.eye(len(corr), dtype=bool))
    print(f"  pairwise Spearman range: [{off.min().min():.3f}, {off.max().max():.3f}]")

    targeting_stat(tag)

    try:
        pl = pd.read_csv(C.TAB_DIR / f"placebo_{tag}.csv")
        print("\n== PLACEBO ==")
        print(pl.round(4).to_string(index=False))
    except FileNotFoundError:
        print("\n(placebo table not found — run src.placebo)")


if __name__ == "__main__":
    main()
