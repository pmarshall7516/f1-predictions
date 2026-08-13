#!/usr/bin/env python3
"""Print a short audit of the local Parquet data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import IN_RACE_PATH, PRE_RACE_PATH


def _print_dataset(label: str, path: Path) -> None:
    print(f"\n{label}: {path}")
    if not path.exists():
        print("  MISSING — build locally with scripts/build_datasets.py")
        return

    frame = pd.read_parquet(path)
    print(f"  shape={frame.shape}")
    print(f"  years={sorted(frame['year'].dropna().astype(int).unique().tolist())}")
    print(f"  races={frame['race_id'].nunique()} rows_by_year={frame.groupby('year').size().to_dict()}")
    for column in ("driver", "team", "circuit"):
        if column in frame:
            values = sorted(frame[column].dropna().astype(str).unique().tolist())
            print(f"  {column}s ({len(values)}): {', '.join(values)}")
    null_rates = frame.isna().mean().sort_values(ascending=False)
    null_rates = null_rates[null_rates > 0].head(12)
    if len(null_rates):
        print("  highest_null_rates:")
        for column, rate in null_rates.items():
            print(f"    {column}: {rate:.1%}")


def _race_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, columns=["race_id"])
    return set(frame["race_id"].astype(str))


if __name__ == "__main__":
    _print_dataset("pre-race", PRE_RACE_PATH)
    _print_dataset("in-race", IN_RACE_PATH)
    pre_ids = _race_ids(PRE_RACE_PATH)
    in_ids = _race_ids(IN_RACE_PATH)
    if pre_ids != in_ids:
        print("\nWarning: the two files do not contain the same races.")
        print(f"  Only in pre-race: {sorted(pre_ids - in_ids)}")
        print(f"  Only in in-race: {sorted(in_ids - pre_ids)}")
