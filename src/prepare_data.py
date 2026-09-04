"""Convert the raw Criteo CSV.gz into compact parquet + a stratified dev sample.

Run once:  python -m src.prepare_data

Outputs
-------
data/processed/criteo_full.parquet : all ~14M rows, float32 features, int8 labels
data/processed/criteo_dev.parquet  : stratified 500k subsample for development
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C


def _dtypes() -> dict:
    d = {f: "float32" for f in C.FEATURES}
    d.update({C.TREATMENT: "int8", C.EXPOSURE: "int8", "visit": "int8", "conversion": "int8"})
    return d


def build_full() -> pd.DataFrame:
    """Stream the gzip CSV in chunks, downcast, and write a single parquet file."""
    t0 = time.time()
    print(f"[prepare] reading {C.RAW_CSV_GZ.name} in chunks ...")
    chunks = []
    reader = pd.read_csv(C.RAW_CSV_GZ, dtype=_dtypes(), chunksize=2_000_000)
    for i, ch in enumerate(reader):
        chunks.append(ch)
        print(f"  chunk {i}: {len(ch):,} rows (cum {sum(len(c) for c in chunks):,})")
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(C.FULL_PARQUET, index=False)
    print(f"[prepare] wrote {C.FULL_PARQUET} ({len(df):,} rows) in {time.time()-t0:.1f}s")
    return df


def build_dev(df: pd.DataFrame) -> None:
    """Stratified subsample on (treatment x primary outcome) to preserve rare cells."""
    strata = df[C.TREATMENT].astype(str) + "_" + df[C.PRIMARY_OUTCOME].astype(str)
    frac = C.DEV_N / len(df)
    dev = (
        df.groupby(strata, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=C.RANDOM_STATE))
        .reset_index(drop=True)
    )
    dev.to_parquet(C.DEV_PARQUET, index=False)
    print(f"[prepare] wrote {C.DEV_PARQUET} ({len(dev):,} rows)")
    # sanity: treatment ratio and outcome rates preserved
    print(f"  treatment ratio dev={dev[C.TREATMENT].mean():.3f} full={df[C.TREATMENT].mean():.3f}")
    for o in C.OUTCOMES:
        print(f"  {o:11s} rate dev={dev[o].mean():.5f} full={df[o].mean():.5f}")


def main() -> None:
    df = build_full()
    build_dev(df)


if __name__ == "__main__":
    main()
