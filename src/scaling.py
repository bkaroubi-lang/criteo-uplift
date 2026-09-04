"""Qini vs. training-sample-size sweep — turns the §7 scaling story into a figure.

Fits every estimator at a range of training sizes and scores each on a *fixed*
held-out evaluation set, so Qini values are comparable across sizes. The visual
payoff: the fan of estimator curves at small n converges as n grows.

    python -m src.scaling --outcome visit

Outputs:
    results/tables/scaling_<outcome>.csv
    results/figures/scaling_qini_<outcome>.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .estimators import build_estimators, fit_effect
from .uplift_metrics import qini_coefficient

SIZES = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]


def run(outcome: str, cv: int, cf_estimators: int, eval_n: int) -> None:
    df = pd.read_parquet(C.FULL_PARQUET)
    X = df[C.FEATURES].to_numpy(np.float32)
    T = df[C.TREATMENT].to_numpy(np.int8)
    Y = df[outcome].to_numpy(np.int8)

    # fixed evaluation set (stratified on treatment); remainder is the train pool
    X_pool, X_ev, T_pool, T_ev, Y_pool, Y_ev = train_test_split(
        X, T, Y, test_size=eval_n, random_state=C.RANDOM_STATE, stratify=T
    )
    print(f"[scaling] eval={len(X_ev):,} pool={len(X_pool):,} outcome={outcome}")

    rng = np.random.RandomState(C.RANDOM_STATE)
    rows = []
    for n in SIZES:
        if n > len(X_pool):
            continue
        sel = rng.choice(len(X_pool), n, replace=False)
        Xn, Tn, Yn = X_pool[sel], T_pool[sel], Y_pool[sel]
        ests = build_estimators(cv=cv, cf_estimators=cf_estimators, random_state=C.RANDOM_STATE)
        for name, est in ests.items():
            t0 = time.time()
            tau = fit_effect(name, est, Xn, Tn, Yn, X_ev)
            qc = qini_coefficient(tau, T_ev, Y_ev)
            rows.append({"n_train": n, "estimator": name, "qini": qc,
                         "fit_seconds": round(time.time() - t0, 1)})
            print(f"  n={n:>9,} {name:14s} qini={qc:+.4f} ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(C.TAB_DIR / f"scaling_{outcome}.csv", index=False)

    # spread (max-min Qini across estimators) per size, for annotation
    spread = out.groupby("n_train")["qini"].agg(lambda s: s.max() - s.min())

    plt.figure(figsize=(7.5, 5.5))
    for name, g in out.groupby("estimator"):
        g = g.sort_values("n_train")
        plt.plot(g["n_train"], g["qini"], marker="o", lw=1.8, label=name)
    plt.xscale("log")
    plt.xlabel("Training sample size (log scale)")
    plt.ylabel("Qini coefficient (fixed 1M eval set)")
    plt.title(f"Estimator disagreement shrinks with data — {outcome}\n"
              f"Qini spread: {spread.iloc[0]:.2f} at {SIZES[0]//1000}k "
              f"-> {spread.iloc[-1]:.2f} at {list(spread.index)[-1]//1_000_000}M")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"scaling_qini_{outcome}.png", dpi=140)
    plt.close()
    print(f"[scaling] wrote scaling_qini_{outcome}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=C.OUTCOMES, default=C.PRIMARY_OUTCOME)
    ap.add_argument("--cv", type=int, default=2)
    ap.add_argument("--cf-estimators", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=1_000_000)
    a = ap.parse_args()
    run(a.outcome, a.cv, a.cf_estimators, a.eval_n)


if __name__ == "__main__":
    main()
