"""Pre-race feature construction from FastF1 race results."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config import ROLLING_WINDOW
from src.fastf1_utils import classified_position, weather_summary

logger = logging.getLogger(__name__)


def extract_pre_race_from_session(
    race,
    year: int,
    round_number: int,
    event_name: str,
    circuit_key: str,
    quali=None,
) -> Optional[pd.DataFrame]:
    """Build pre-race rows from an already-loaded race session (and optional quali)."""
    results = race.results
    if results is None or len(results) == 0:
        logger.warning("No results for %s R%s", year, round_number)
        return None

    weather = weather_summary(quali) if quali is not None else weather_summary(race)

    rows = []
    for _, r in results.iterrows():
        finish = classified_position(r)
        if finish is None:
            continue
        grid = r.get("GridPosition")
        if pd.isna(grid):
            # Keep the input empty. Do not use the finish position here.
            # The model fills the value from the training data.
            grid = float("nan")
        rows.append(
            {
                "year": year,
                "round": int(round_number),
                "event_name": event_name,
                "circuit": circuit_key,
                "race_id": f"{year}_R{int(round_number)}",
                "driver": str(r.get("Abbreviation", r.get("DriverNumber", ""))),
                "driver_number": str(r.get("DriverNumber", "")),
                "team": str(r.get("TeamName", "")),
                "grid_position": float(grid),
                "grid_position_missing": int(pd.isna(grid)),
                "finish_position": float(finish),
                "quali_to_race_delta": float(grid) - float(finish),
                **weather,
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


def add_historical_features(pre_race: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add rolling form features using only prior races (chronological)."""
    df = pre_race.sort_values(["year", "round", "driver"]).reset_index(drop=True).copy()

    race_order = (
        df[["year", "round", "race_id"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .reset_index(drop=True)
    )
    race_order["race_seq"] = range(len(race_order))
    df = df.merge(race_order[["race_id", "race_seq"]], on="race_id", how="left")

    df["driver_roll_finish_5"] = float("nan")
    df["team_roll_finish_5"] = float("nan")
    df["track_avg_finish"] = float("nan")
    df["driver_quali_race_conv"] = float("nan")

    for idx, row in df.iterrows():
        prior = df[df["race_seq"] < row["race_seq"]]

        d_hist = prior[prior["driver"] == row["driver"]].tail(window)
        if len(d_hist):
            df.at[idx, "driver_roll_finish_5"] = d_hist["finish_position"].mean()
            df.at[idx, "driver_quali_race_conv"] = d_hist["quali_to_race_delta"].mean()

        # A team has two driver rows per race. First make one team value for
        # each race. Then use the last ``window`` team races.
        t_hist = prior[prior["team"] == row["team"]]
        if len(t_hist):
            t_race_hist = (
                t_hist.groupby("race_seq", as_index=False)["finish_position"]
                .mean()
                .sort_values("race_seq")
                .tail(window)
            )
            if len(t_race_hist):
                df.at[idx, "team_roll_finish_5"] = t_race_hist["finish_position"].mean()

        c_hist = prior[prior["circuit"] == row["circuit"]]
        if len(c_hist):
            d_c = c_hist[c_hist["driver"] == row["driver"]]
            if len(d_c):
                df.at[idx, "track_avg_finish"] = d_c["finish_position"].mean()
            else:
                df.at[idx, "track_avg_finish"] = c_hist["finish_position"].mean()

    return df.drop(columns=["race_seq", "quali_to_race_delta"], errors="ignore")
