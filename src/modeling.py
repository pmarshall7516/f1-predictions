"""Shared modeling helpers for the baseline notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder


PRE_RACE_FEATURES = [
    "grid_position",
    "grid_position_missing",
    "driver_roll_finish_5",
    "team_roll_finish_5",
    "track_avg_finish",
    "driver_quali_race_conv",
    "weather_air_temp",
    "weather_track_temp",
    "weather_humidity",
    "weather_rainfall",
    "driver_enc",
    "team_enc",
    "circuit_enc",
]

IN_RACE_FEATURES = [
    "current_position",
    "gap_to_leader",
    "gap_ahead",
    "gap_behind",
    "tyre_age",
    "stint_number",
    "pit_this_lap",
    "already_pitted",
    "safety_car_flag",
    "lap_delta_to_pb",
    "roll_lap_3",
    "pre_race_pred",
]

IN_RACE_FEATURES_ABLATION = [c for c in IN_RACE_FEATURES if c != "pre_race_pred"]

# These fields support the new race-state model. The baseline fields above
# stay unchanged.
IN_RACE_STATE_FEATURES = [
    "current_position",
    "driver_enc",
    "team_enc",
    "circuit_enc",
    "compound_enc",
    "position_delta_from_grid",
    "position_change_last_lap",
    "position_change_last_3",
    "gap_to_leader",
    "gap_ahead",
    "gap_behind",
    "lap_number",
    "lap_fraction",
    "laps_remaining",
    "tyre_age",
    "stint_number",
    "pit_this_lap",
    "pits_so_far",
    "already_pitted",
    "track_status_yellow",
    "track_status_sc",
    "track_status_vsc",
    "track_status_red",
    "track_status_changed",
    "track_status_laps_since_change",
    "sc_laps_last_3",
    "vsc_laps_last_3",
    "interruption_laps_last_5",
    "lap_time_sec",
    "lap_time_vs_field_median",
    "lap_pace_rank",
    "lap_delta_to_pb",
    "roll_lap_3",
    "roll_lap_5",
    "pre_race_pred",
]

# The residual model predicts future movement. It does not use the current
# position or the pre-race prediction as input.
IN_RACE_RESIDUAL_FEATURES = [
    c for c in IN_RACE_STATE_FEATURES if c not in {"current_position", "pre_race_pred"}
]


def encode_categoricals(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[OrdinalEncoder, list[pd.DataFrame]]:
    """Fit ordinal encoders on train categoricals; transform all frames."""
    cols = ["driver", "team", "circuit"] if cols is None else cols
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[cols].astype(str))
    out = []
    for df in (train, *others):
        d = df.copy()
        encoded = enc.transform(d[cols].astype(str))
        for i, col in enumerate(cols):
            d[f"{col}_enc"] = encoded[:, i]
        out.append(d)
    return enc, out


def fill_numeric(train: pd.DataFrame, *others: pd.DataFrame, cols: list[str]) -> list[pd.DataFrame]:
    """Fill NaNs with train medians for numeric feature columns."""
    medians = train[cols].median(numeric_only=True)
    out = []
    for df in (train, *others):
        d = df.copy()
        d[cols] = d[cols].fillna(medians)
        # Any remaining (all-NaN cols) -> 0
        d[cols] = d[cols].fillna(0.0)
        out.append(d)
    return out


def oof_pre_race_predictions(
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str = "finish_position",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, RandomForestRegressor]:
    """Race-grouped OOF preds + final RF fit on full train."""
    X = train[feature_cols]
    y = train[target]
    groups = train["race_id"]

    n_unique = groups.nunique()
    splits = min(n_splits, n_unique)
    gkf = GroupKFold(n_splits=splits)
    oof = np.zeros(len(train), dtype=float)

    for fold_train, fold_val in gkf.split(X, y, groups):
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X.iloc[fold_train], y.iloc[fold_train])
        oof[fold_val] = model.predict(X.iloc[fold_val])

    final = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    final.fit(X, y)
    return oof, final


def add_position_delta_target(
    frame: pd.DataFrame,
    *,
    current_col: str = "current_position",
    finish_col: str = "finish_position",
    target_col: str = "finish_delta",
) -> pd.DataFrame:
    """Add finish position minus current position as the target."""

    out = frame.copy()
    out[target_col] = out[finish_col] - out[current_col]
    return out


def reconstruct_finish_from_delta(
    current_position,
    predicted_delta,
    *,
    min_position: float = 1.0,
    max_position: float = 20.0,
) -> np.ndarray:
    """Convert a movement prediction to a finish-position prediction."""

    result = np.asarray(current_position, dtype=float) + np.asarray(predicted_delta, dtype=float)
    return np.clip(result, min_position, max_position)
