"""Shared configuration: paths, columns, constants."""
from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_GZ = ROOT / "criteo-research-uplift-v2.1.csv.gz"
PROC_DIR = ROOT / "data" / "processed"
FULL_PARQUET = PROC_DIR / "criteo_full.parquet"
DEV_PARQUET = PROC_DIR / "criteo_dev.parquet"

RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"

for _d in (PROC_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Columns -------------------------------------------------------------
FEATURES = [f"f{i}" for i in range(12)]          # f0..f11 dense floats
TREATMENT = "treatment"                          # 1 = eligible for ads, 0 = held out (randomized)
EXPOSURE = "exposure"                             # effectively exposed (post-treatment; NOT used as W)
OUTCOMES = ["visit", "conversion"]               # binary labels

# --- Experiment facts (from dataset README) ------------------------------
N_ROWS = 13_979_592
TREATMENT_RATIO = 0.85                            # P(treatment=1); randomized => ~constant propensity
DEV_N = 500_000                                   # development subsample size
RANDOM_STATE = 20240714

# Primary analysis outcome. `visit` (~4.7%) has enough events for stable
# heterogeneity estimation; `conversion` (~0.29%) is included as a stress test.
PRIMARY_OUTCOME = "visit"
