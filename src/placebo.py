"""Negative-control / placebo test.

Because treatment is randomized, permuting the treatment label destroys any
real treatment effect while preserving the marginal treated/control rates. A
trustworthy CATE estimator should then detect *no* usable heterogeneity: its
Qini coefficient on placebo labels should sit around zero.

We build a null distribution of Qini coefficients over several permutations and
compare it against the real-label Qini for the same estimator(s). If the real
Qini clearly exceeds the placebo band, the detected heterogeneity is signal,
not an artefact of the estimator's flexibility.

Usage
-----
    python -m src.placebo --sample dev --outcome visit --n-perm 20
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


# Default estimators for the placebo (flexible ones are the interesting stress test)
PLACEBO_ESTIMATORS = ["Causal Forest", "DR-learner", "T-learner"]


def _subset(estimators: dict, names: list[str]) -> dict:
    return {k: estimators[k] for k in names if k in estimators}


def run(sample: str, outcome: str, n_perm: int, cv: int, cf_estimators: int,
        train_cap: int | None, eval_cap: int, names: list[str]) -> None:
    tag = f"{sample}_{outcome}"
    path = C.DEV_PARQUET if sample == "dev" else C.FULL_PARQUET
    df = pd.read_parquet(path)
    X = df[C.FEATURES].to_numpy(np.float32)
    T = df[C.TREATMENT].to_numpy(np.int8)
    Y = df[outcome].to_numpy(np.int8)

    X_tr, X_ev, T_tr, T_ev, Y_tr, Y_ev = train_test_split(
        X, T, Y, test_size=0.3, random_state=C.RANDOM_STATE, stratify=T
    )
    if train_cap and len(X_tr) > train_cap:
        rng = np.random.RandomState(C.RANDOM_STATE)
        sel = rng.choice(len(X_tr), train_cap, replace=False)
        X_tr, T_tr, Y_tr = X_tr[sel], T_tr[sel], Y_tr[sel]
    if eval_cap and len(X_ev) > eval_cap:
        rng = np.random.RandomState(C.RANDOM_STATE + 1)
        sel = rng.choice(len(X_ev), eval_cap, replace=False)
        X_ev, T_ev, Y_ev = X_ev[sel], T_ev[sel], Y_ev[sel]

    rng = np.random.RandomState(C.RANDOM_STATE)

    # 1) real-label Qini per estimator
    real = {}
    for name, est in _subset(build_estimators(cv, cf_estimators, C.RANDOM_STATE), names).items():
        tau = fit_effect(name, est, X_tr, T_tr, Y_tr, X_ev)
        real[name] = qini_coefficient(tau, T_ev, Y_ev)
        print(f"[real] {name}: qini={real[name]:.4f}")

    # 2) placebo null distribution: permute treatment on the TRAIN set
    placebo = {name: [] for name in real}
    for p in range(n_perm):
        t0 = time.time()
        T_perm = rng.permutation(T_tr)                 # break W<->(X,Y) link
        ests = _subset(build_estimators(cv, cf_estimators, C.RANDOM_STATE + p), names)
        for name, est in ests.items():
            tau = fit_effect(name, est, X_tr, T_perm, Y_tr, X_ev)
            # evaluate against the REAL eval labels: a placebo-trained ranker
            # should not sort real uplift, so Qini ~ 0
            placebo[name].append(qini_coefficient(tau, T_ev, Y_ev))
        print(f"[placebo] perm {p+1}/{n_perm} done ({time.time()-t0:.1f}s)")

    # ---- Save -----------------------------------------------------------
    rows = []
    for name in real:
        arr = np.array(placebo[name])
        rows.append({
            "estimator": name,
            "real_qini": real[name],
            "placebo_mean": arr.mean(),
            "placebo_std": arr.std(),
            "placebo_p95": np.quantile(arr, 0.95),
            "z_vs_placebo": (real[name] - arr.mean()) / (arr.std() + 1e-9),
            "n_perm": n_perm,
        })
    out = pd.DataFrame(rows)
    out.to_csv(C.TAB_DIR / f"placebo_{tag}.csv", index=False)
    print("\n[placebo] summary:\n", out.to_string(index=False))

    # ---- Figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(real):
        arr = np.array(placebo[name])
        ax.scatter(np.full_like(arr, i, dtype=float) + rng.uniform(-0.06, 0.06, len(arr)),
                   arr, s=18, alpha=0.6, color="0.5",
                   label="Placebo perms" if i == 0 else None)
        ax.scatter([i], [real[name]], s=90, marker="D", color="crimson", zorder=3,
                   label="Real labels" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(real)))
    ax.set_xticklabels(list(real.keys()), fontsize=9)
    ax.set_ylabel("Qini coefficient")
    ax.set_title(f"Placebo test: real vs permuted-treatment Qini — {tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"placebo_{tag}.png", dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["dev", "full"], default="dev")
    ap.add_argument("--outcome", choices=C.OUTCOMES, default=C.PRIMARY_OUTCOME)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--cf-estimators", type=int, default=200)
    ap.add_argument("--train-cap", type=int, default=None)
    ap.add_argument("--eval-cap", type=int, default=1_000_000)
    ap.add_argument("--estimators", default=",".join(PLACEBO_ESTIMATORS),
                    help="comma-separated estimator names to test")
    args = ap.parse_args()
    names = [s.strip() for s in args.estimators.split(",") if s.strip()]
    run(args.sample, args.outcome, args.n_perm, args.cv, args.cf_estimators,
        args.train_cap, args.eval_cap, names)


if __name__ == "__main__":
    main()
