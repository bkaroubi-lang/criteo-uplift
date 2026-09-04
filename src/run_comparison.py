"""Head-to-head CATE estimator comparison on the Criteo uplift data.

Usage
-----
    python -m src.run_comparison --sample dev  --outcome visit
    python -m src.run_comparison --sample full --outcome visit --cv 2 --cf-estimators 200 --train-cap 4000000

Produces (tagged by <sample>_<outcome>):
    results/tables/qini_coefficients_*.csv
    results/tables/cate_summary_*.csv
    results/tables/estimator_agreement_spearman_*.csv
    results/tables/uplift_deciles_*.csv
    results/tables/cate_eval_*.parquet          (per-user CATE from every estimator)
    results/figures/qini_curves_*.png
    results/figures/cate_distributions_*.png
    results/figures/estimator_agreement_*.png
    results/figures/uplift_deciles_*.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from . import config as C
from .estimators import build_estimators, fit_effect
from .uplift_metrics import qini_coefficient, qini_curve, uplift_by_decile


def load(sample: str) -> pd.DataFrame:
    path = C.DEV_PARQUET if sample == "dev" else C.FULL_PARQUET
    print(f"[load] {path.name}")
    return pd.read_parquet(path)


def ate(treatment, outcome) -> float:
    """Difference-in-means ATE (unbiased under randomization)."""
    t = treatment == 1
    return float(outcome[t].mean() - outcome[~t].mean())


def run(sample: str, outcome: str, cv: int, cf_estimators: int,
        train_cap: int | None, eval_cap: int) -> None:
    tag = f"{sample}_{outcome}"
    df = load(sample)
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

    ate_val = ate(T_ev, Y_ev)
    print(f"[data] train={len(X_tr):,} eval={len(X_ev):,} | eval ATE({outcome})={ate_val:.5f}")

    estimators = build_estimators(cv=cv, cf_estimators=cf_estimators, random_state=C.RANDOM_STATE)

    cate = {}            # name -> CATE vector on eval
    qini_rows = []
    timings = {}
    for name, est in estimators.items():
        t0 = time.time()
        print(f"[fit] {name} ...", flush=True)
        tau = fit_effect(name, est, X_tr, T_tr, Y_tr, X_ev)
        dt = time.time() - t0
        timings[name] = dt
        cate[name] = tau
        qc = qini_coefficient(tau, T_ev, Y_ev)
        qini_rows.append({"estimator": name, "qini_coefficient": qc,
                          "cate_mean": float(tau.mean()), "cate_std": float(tau.std()),
                          "fit_seconds": round(dt, 1)})
        print(f"      qini={qc:.4f}  mean_cate={tau.mean():.5f}  ({dt:.1f}s)")

    # ---- Tables ---------------------------------------------------------
    qini_df = pd.DataFrame(qini_rows).sort_values("qini_coefficient", ascending=False)
    qini_df.insert(1, "ate_reference", ate_val)
    qini_df.to_csv(C.TAB_DIR / f"qini_coefficients_{tag}.csv", index=False)

    summ = []
    for name, tau in cate.items():
        qs = np.quantile(tau, [0.05, 0.25, 0.5, 0.75, 0.95])
        summ.append({"estimator": name, "mean": tau.mean(), "std": tau.std(),
                     "p05": qs[0], "p25": qs[1], "median": qs[2], "p75": qs[3], "p95": qs[4],
                     "pct_negative": float((tau < 0).mean())})
    pd.DataFrame(summ).to_csv(C.TAB_DIR / f"cate_summary_{tag}.csv", index=False)

    # Pairwise Spearman rank correlation of CATE estimates
    names = list(cate.keys())
    corr = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = spearmanr(cate[names[i]], cate[names[j]]).correlation
            corr[i, j] = corr[j, i] = rho
    corr_df = pd.DataFrame(corr, index=names, columns=names)
    corr_df.to_csv(C.TAB_DIR / f"estimator_agreement_spearman_{tag}.csv")

    # Uplift-by-decile per estimator (long format)
    dec_rows = []
    for name, tau in cate.items():
        for r in uplift_by_decile(tau, T_ev, Y_ev):
            r["estimator"] = name
            dec_rows.append(r)
    pd.DataFrame(dec_rows).to_csv(C.TAB_DIR / f"uplift_deciles_{tag}.csv", index=False)

    # Per-user CATE matrix (for reuse / drill-down)
    pd.DataFrame(cate).to_parquet(C.TAB_DIR / f"cate_eval_{tag}.parquet", index=False)

    # ---- Figures --------------------------------------------------------
    _fig_qini(cate, T_ev, Y_ev, tag)
    _fig_cate_dist(cate, tag)
    _fig_agreement(corr_df, tag)
    _fig_deciles(dec_rows, tag)

    print(f"\n[done] {tag}. Qini ranking:")
    print(qini_df[["estimator", "qini_coefficient", "fit_seconds"]].to_string(index=False))


def _fig_qini(cate, T_ev, Y_ev, tag):
    plt.figure(figsize=(7, 5.5))
    for name, tau in cate.items():
        x, q = qini_curve(tau, T_ev, Y_ev)
        plt.plot(x, q, label=name, lw=1.8)
    # random baseline uses any estimator's endpoint (same total)
    any_tau = next(iter(cate.values()))
    x, q = qini_curve(any_tau, T_ev, Y_ev)
    plt.plot([0, 1], [0, q[-1]], "k--", lw=1, label="Random targeting")
    plt.xlabel("Fraction of population targeted (ranked by predicted uplift)")
    plt.ylabel("Cumulative incremental outcomes (Qini)")
    plt.title(f"Qini curves — {tag}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"qini_curves_{tag}.png", dpi=140)
    plt.close()


def _fig_cate_dist(cate, tag):
    plt.figure(figsize=(7, 5))
    for name, tau in cate.items():
        plt.hist(tau, bins=80, histtype="step", lw=1.6, density=True, label=name)
    plt.xlabel("Estimated CATE (individual treatment effect)")
    plt.ylabel("Density")
    plt.title(f"CATE distribution by estimator — {tag}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"cate_distributions_{tag}.png", dpi=140)
    plt.close()


def _fig_agreement(corr_df, tag):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_df.values, vmin=-0.2, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(corr_df)))
    ax.set_yticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman rank correlation of CATE")
    ax.set_title(f"Estimator agreement — {tag}")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"estimator_agreement_{tag}.png", dpi=140)
    plt.close()


def _fig_deciles(dec_rows, tag):
    df = pd.DataFrame(dec_rows)
    plt.figure(figsize=(7, 5))
    for name, g in df.groupby("estimator"):
        plt.plot(g["decile"], g["observed_uplift"], marker="o", lw=1.5, label=name)
    plt.axhline(0, color="k", lw=0.7)
    plt.gca().invert_xaxis()  # decile 1 (highest predicted) on the right
    plt.xlabel("Score decile (1 = highest predicted uplift)")
    plt.ylabel("Observed uplift (treated - control rate)")
    plt.title(f"Observed uplift by predicted-uplift decile — {tag}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"uplift_deciles_{tag}.png", dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["dev", "full"], default="dev")
    ap.add_argument("--outcome", choices=C.OUTCOMES, default=C.PRIMARY_OUTCOME)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--cf-estimators", type=int, default=400)
    ap.add_argument("--train-cap", type=int, default=None,
                    help="cap training rows (recommended for full-data causal forest)")
    ap.add_argument("--eval-cap", type=int, default=1_000_000)
    args = ap.parse_args()
    run(args.sample, args.outcome, args.cv, args.cf_estimators, args.train_cap, args.eval_cap)


if __name__ == "__main__":
    main()
