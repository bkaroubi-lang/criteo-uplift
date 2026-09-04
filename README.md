# From ATE to CATE: Benchmarking Causal-ML Estimators on 14M Randomized Ad Exposures

A methods-comparison study on the **Criteo Uplift Modeling** dataset — ~14M users
from a genuine randomized incrementality test. The question is not *"does
advertising work"* (the ATE settles that) but the one a growth or underwriting
team actually asks:

> **Which users should we target, and does the choice of CATE estimator
> materially change the answer?**

📄 **Read the write-up:** [`article/uplift_article.md`](article/uplift_article.md)
· [PDF](article/uplift_article.pdf) · [LinkedIn carousel](article/uplift_carousel.pdf)

🌐 **Live page:** the self-contained [`index.html`](index.html) renders the full
article — enable **GitHub Pages** (Settings → Pages → *Deploy from branch* →
`main` / `/root`) to serve it at `https://<user>.github.io/<repo>/`.

---

## Results at a glance

Six CATE estimators — S-, T-, X-learner, DR-learner, R-learner, and a causal
forest (GRF) — on identical data and **identical HistGradientBoosting base
learners**, so any divergence is attributable to the estimator's meta-structure.

![Estimator disagreement shrinks with data](results/figures/scaling_qini_visit.png)

**Headline findings (outcome = `visit`):**

- **Targeting is the business case.** The top 10% of users by predicted uplift
  carry **~82%** of the incremental visits (top 30% → 95%). Half the audience is
  dead weight — invisible if you only read the ATE.
- **Estimator choice is a small-sample problem.** At 500k rows the six methods
  span a **2.6×** Qini range and disagree on whether 4%–44% of users are even
  *helped*; at 4M rows the spread collapses to **1.3×** and the disagreement to
  1–3%.
- **The placebo test catches a fake — then clears it.** Permuting the (randomized)
  treatment label must destroy heterogeneity. On 500k the DR-learner failed this
  negative control (z = 1.4, indistinguishable from noise); on 4M it passed by
  **z = 7.0**. It was never overfitting — it was under-powered.

| Estimator | Qini @ 500k | Qini @ 4M | % "harmed" @500k → @4M |
|:--|--:|--:|:--:|
| S-learner | 0.539 | **0.758** | 4% → 1% |
| Causal Forest | 0.561 | 0.701 | 28% → 23% |
| R-learner | 0.315 | 0.675 | 44% → 3% |
| X-learner | 0.417 | 0.668 | 15% → 3% |
| DR-learner | 0.215 | 0.646 | 44% → 2% |
| T-learner | 0.351 | 0.578 | 35% → 15% |

The takeaway: the *regularized* estimators (S-learner, causal forest) were
trustworthy at both scales; the doubly-robust / DML methods needed volume. At
small scale, prefer regularized methods **and run the placebo check.**

---

## The three questions

1. **Targeting.** How much smaller can a targeted population be than "treat
   everyone" while retaining most of the aggregate lift? → Qini / uplift curves.
2. **Estimator comparison.** How much do heterogeneous-treatment-effect estimates
   disagree across the six methods on identical data?
3. **Robustness (negative control).** Since treatment is randomized, permuting the
   label must destroy heterogeneity — a trustworthy estimator's Qini should
   collapse to ~0 on placebo labels.

## Repository layout

```
Criteo/
  src/
    config.py            paths, columns, experiment constants
    prepare_data.py      CSV.gz -> compact parquet + stratified 500k dev sample
    uplift_metrics.py    Qini curve, Qini coefficient, uplift-by-decile
    estimators.py        the 6-estimator roster on shared HistGBM base learners
    run_comparison.py    fit all, evaluate, emit tables + figures
    scaling.py           Qini vs training-sample-size sweep
    placebo.py           negative-control / permutation test
    report_numbers.py    paste-ready numbers from the result tables
    build_html.py        article Markdown -> self-contained HTML
    build_carousel.py    6-slide LinkedIn carousel
  results/
    figures/  tables/    generated outputs (tagged <sample>_<outcome>)
  article/
    uplift_article.md    the write-up      (+ .html / .pdf renders)
    uplift_carousel.pdf  LinkedIn carousel
  environment.yml        reproducible conda env
  DATA.md                dataset card + how to obtain the data (not committed)
```

## Data

The raw dataset (**~297 MB**, 13,979,592 rows) is **not committed** — it exceeds
GitHub's file limit and is licensed **CC-BY-NC-SA-4.0** (non-commercial). Download
it yourself and drop it in the repo root as `criteo-research-uplift-v2.1.csv.gz`:

- Source: [Criteo AI Lab — Uplift Modeling dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
  (mirrored on the Hugging Face Hub as `criteo/criteo-research-uplift-v2.1`).
- See [`DATA.md`](DATA.md) for the full field description and citation.

## Environment

**Recommended — [uv](https://docs.astral.sh/uv/)** (fetches Python 3.11 itself, no conda needed):

```bash
uv sync
uv run python -m src.prepare_data
```

**Alternative — conda:**

```bash
conda env create -f environment.yml   # creates the `criteo` env (Python 3.11)
conda activate criteo
```

Key pins (what the figures were produced with): `econml 0.16`, `scikit-learn 1.6.1`,
`pandas 3.0`, `numpy 2.4`.

## Reproduce

```bash
# 0. one-time data prep (writes data/processed/*.parquet)
python -m src.prepare_data

# 1. development run (500k stratified sample) — minutes
python -m src.run_comparison --sample dev  --outcome visit
python -m src.placebo        --sample dev  --outcome visit --n-perm 20

# 2. full-data runs (cap causal-forest training for tractability)
python -m src.run_comparison --sample full --outcome visit \
    --cv 2 --cf-estimators 200 --train-cap 4000000 --eval-cap 1000000
python -m src.run_comparison --sample full --outcome conversion \
    --cv 2 --cf-estimators 200 --train-cap 4000000 --eval-cap 1000000

# 3. full-data placebo (the z=1.4 -> z=7.0 flip) and the scaling figure
python -m src.placebo --sample full --outcome visit --n-perm 10 \
    --train-cap 4000000 --eval-cap 1000000 --estimators "DR-learner,R-learner,T-learner"
python -m src.scaling --outcome visit

# 4. render the article + carousel
python -m src.build_html
python -m src.build_carousel
```

## Method notes

- **Treatment** is the randomized `treatment` flag (intent-to-treat), *not*
  `exposure` (which is post-randomization and would reintroduce confounding).
- **Propensity is ~0.85 everywhere** by design; the propensity models recover it
  empirically, so DR/DML orthogonalization is well-behaved.
- **Primary outcome is `visit`** (~4.7%, enough events for stable heterogeneity);
  `conversion` (~0.29%) is a rare-event stress test in the article appendix.
- The **Qini coefficient** is normalised against the random-targeting baseline so
  it is comparable across outcomes and sample sizes.

## License

- **Code** (everything in `src/` and the renders): [MIT](LICENSE).
- **Data**: CC-BY-NC-SA-4.0, © Criteo — not included here; obtain from the source
  above under its own terms.

## Citation

> Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). *A Large Scale
> Benchmark for Uplift Modeling.* AdKDD & TargetAd Workshop, KDD 2018.
