"""Evaluation metrics for finishing-position regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def clip_round_positions(preds, min_pos: float = 1.0, max_pos: float = 20.0) -> np.ndarray:
    """Round continuous predictions to integer positions within a valid range."""
    arr = np.asarray(preds, dtype=float)
    return np.clip(np.rint(arr), min_pos, max_pos)


def hit_rate(y_true, y_pred, k: int) -> float:
    """Fraction of samples where rounded prediction is within k places of true."""
    yt = np.asarray(y_true, dtype=float)
    yp = clip_round_positions(y_pred, max_pos=max(20.0, float(np.nanmax(yt))))
    return float(np.mean(np.abs(yt - yp) <= k))


def top_exact_hit_rate(y_true, y_pred, place: int = 1) -> float:
    """Hit-rate for predicting the exact finishing place among drivers who finished at `place`.

    More useful companion: overall exact-match rate for rounded preds.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = clip_round_positions(y_pred, max_pos=max(20.0, float(np.nanmax(yt))))
    if place == 1:
        # "Top-1 hit": exact position match rate
        return float(np.mean(yt == yp))
    mask = yt <= place
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(yp[mask] <= place))


def regression_metrics(y_true, y_pred, max_pos: float | None = None) -> dict:
    """MAE, RMSE, and Top-1 / Top-3 / Top-10 style hit-rates."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    cap = max_pos if max_pos is not None else max(20.0, float(np.nanmax(yt)))
    yp_round = clip_round_positions(yp, max_pos=cap)

    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "top1_exact": float(np.mean(yt == yp_round)),
        "top3_hit": float(np.mean(np.abs(yt - yp_round) <= 3)),
        "top10_hit": float(np.mean(np.abs(yt - yp_round) <= 10)),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Build a tidy metrics DataFrame from a list of metric dicts with a name key."""
    return pd.DataFrame(rows)
