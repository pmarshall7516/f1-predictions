#!/usr/bin/env python3
"""Train and test the race-state models by season."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from xgboost import XGBRegressor

from src.config import IN_RACE_PATH, PRE_RACE_PATH, TEST_YEAR, TRAIN_YEARS, VAL_YEAR
from src.metrics import metrics_by_lap, regression_metrics
from src.modeling import (
    IN_RACE_RESIDUAL_FEATURES,
    IN_RACE_STATE_FEATURES,
    add_position_delta_target,
    encode_categoricals,
    fill_numeric,
    oof_pre_race_predictions,
    reconstruct_finish_from_delta,
)


def _model(params: dict) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        **params,
    )


def _required_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"{label} is missing {missing}. Rebuild processed data with "
            "scripts/build_datasets.py --rebuild."
        )


def run(train_years: list[int], val_year: int, test_year: int) -> None:
    pre = pd.read_parquet(PRE_RACE_PATH)
    laps = pd.read_parquet(IN_RACE_PATH)
    raw_state_columns = [
        c for c in IN_RACE_STATE_FEATURES if not c.endswith("_enc") and c != "pre_race_pred"
    ]
    _required_columns(
        laps,
        raw_state_columns + ["race_id", "driver", "team", "circuit", "compound"],
        "in-race data",
    )

    pre_train = pre[pre["year"].isin(train_years)].copy()
    pre_val = pre[pre["year"] == val_year].copy()
    pre_test = pre[pre["year"] == test_year].copy()
    _, (pre_train, pre_val, pre_test) = encode_categoricals(pre_train, pre_val, pre_test)
    pre_numeric = [c for c in [
        "grid_position",
        "driver_roll_finish_5",
        "team_roll_finish_5",
        "track_avg_finish",
        "driver_quali_race_conv",
        "weather_air_temp",
        "weather_track_temp",
        "weather_humidity",
        "weather_rainfall",
    ] if c in pre_train.columns]
    pre_train, pre_val, pre_test = fill_numeric(
        pre_train,
        pre_val,
        pre_test,
        cols=pre_numeric + ["driver_enc", "team_enc", "circuit_enc"],
    )
    pre_features = pre_numeric + ["driver_enc", "team_enc", "circuit_enc"]
    oof, pre_model = oof_pre_race_predictions(pre_train, pre_features)
    pre_train["pre_race_pred"] = oof
    pre_val["pre_race_pred"] = pre_model.predict(pre_val[pre_features])
    pre_test["pre_race_pred"] = pre_model.predict(pre_test[pre_features])

    predictions = pd.concat(
        [
            pre_train[["race_id", "driver", "pre_race_pred"]],
            pre_val[["race_id", "driver", "pre_race_pred"]],
            pre_test[["race_id", "driver", "pre_race_pred"]],
        ],
        ignore_index=True,
    )
    laps = laps.merge(predictions, on=["race_id", "driver"], how="inner")
    lap_train = laps[laps["year"].isin(train_years)].copy()
    lap_val = laps[laps["year"] == val_year].copy()
    lap_test = laps[laps["year"] == test_year].copy()

    _, (lap_train, lap_val, lap_test) = encode_categoricals(
        lap_train,
        lap_val,
        lap_test,
        cols=["driver", "team", "circuit", "compound"],
    )
    state_fill = [c for c in IN_RACE_STATE_FEATURES if c != "compound_enc"]
    lap_train, lap_val, lap_test = fill_numeric(lap_train, lap_val, lap_test, cols=state_fill + ["compound_enc"])

    direct = _model({})
    direct.fit(lap_train[IN_RACE_STATE_FEATURES], lap_train["finish_position"])

    residual_train = add_position_delta_target(lap_train)
    residual = _model({})
    residual.fit(residual_train[IN_RACE_RESIDUAL_FEATURES], residual_train["finish_delta"])

    rows = []
    for split, frame in (("val", lap_val), ("test", lap_test)):
        direct_pred = direct.predict(frame[IN_RACE_STATE_FEATURES])
        delta_pred = residual.predict(frame[IN_RACE_RESIDUAL_FEATURES])
        residual_pred = reconstruct_finish_from_delta(frame["current_position"], delta_pred)
        for name, pred in (("state direct", direct_pred), ("state residual", residual_pred)):
            rows.append({"model": name, "split": split, **regression_metrics(frame["finish_position"], pred)})
        if split == "test":
            by_lap = metrics_by_lap(frame, residual_pred)
            print("\nResidual model by lap (test):")
            print(by_lap.loc[by_lap["lap_number"] % 5 == 0].round(4).to_string(index=False))

    print("\nChronological state-model results:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-through", type=int, default=max(TRAIN_YEARS))
    parser.add_argument("--validation-year", type=int, default=VAL_YEAR)
    parser.add_argument("--test-year", type=int, default=TEST_YEAR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(list(range(min(TRAIN_YEARS), args.train_through + 1)), args.validation_year, args.test_year)
