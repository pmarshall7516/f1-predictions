"""Build race-state features from FastF1 lap data.

The features use data available at the end of a completed lap. They include
position, movement, pace, tyre, pit, and track-state data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert time values to seconds."""

    def convert(value):
        if pd.isna(value):
            return np.nan
        if hasattr(value, "total_seconds"):
            return float(value.total_seconds())
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    return series.map(convert).astype(float)


def _has_status(value, codes: set[str]) -> int:
    if pd.isna(value):
        return 0
    text = str(value)
    return int(any(code in text for code in codes))


def _lap_status_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add separate flags for track status values."""

    status = df["TrackStatus"] if "TrackStatus" in df.columns else pd.Series(
        np.nan, index=df.index
    )
    df["track_status_code"] = status.astype("string")
    df["track_status_yellow"] = status.map(lambda value: _has_status(value, {"2"})).astype(int)
    df["track_status_sc"] = status.map(lambda value: _has_status(value, {"4"})).astype(int)
    df["track_status_red"] = status.map(lambda value: _has_status(value, {"5"})).astype(int)
    df["track_status_vsc"] = status.map(lambda value: _has_status(value, {"6", "7"})).astype(int)
    df["track_status_interruption"] = (
        df[["track_status_yellow", "track_status_sc", "track_status_red", "track_status_vsc"]]
        .max(axis=1)
        .astype(int)
    )

    # Track status applies to the full lap. Build one value per lap.
    lap_state = (
        df.groupby("LapNumber", sort=True)[
            [
                "track_status_yellow",
                "track_status_sc",
                "track_status_red",
                "track_status_vsc",
                "track_status_interruption",
            ]
        ]
        .max()
        .sort_index()
    )
    state_signature = lap_state.astype(str).agg("|".join, axis=1)
    lap_state["track_status_changed"] = state_signature.ne(state_signature.shift(1)).astype(int)
    run_id = state_signature.ne(state_signature.shift(1)).cumsum()
    lap_state["track_status_laps_since_change"] = lap_state.groupby(run_id).cumcount()
    lap_state["sc_laps_last_3"] = lap_state["track_status_sc"].rolling(3, min_periods=1).sum()
    lap_state["vsc_laps_last_3"] = lap_state["track_status_vsc"].rolling(3, min_periods=1).sum()
    lap_state["interruption_laps_last_5"] = (
        lap_state["track_status_interruption"].rolling(5, min_periods=1).sum()
    )

    return df.merge(
        lap_state.reset_index(),
        on="LapNumber",
        how="left",
        suffixes=("", "_lap"),
    )


def add_race_state_features(
    laps: pd.DataFrame,
    *,
    scheduled_laps: int | None = None,
) -> pd.DataFrame:
    """Build features for one race at the end of each completed lap.

    Use ``Time`` for gaps when it is available. Leave gaps empty when it is
    not available. Do not use driver-only cumulative lap times for gaps.
    """

    required = {"Driver", "LapNumber", "Position"}
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"laps is missing required columns: {sorted(missing)}")

    df = laps.copy().sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    df["lap_number"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["current_position"] = pd.to_numeric(df["Position"], errors="coerce")

    if "LapTime" in df.columns:
        df["lap_time_sec"] = _to_seconds(df["LapTime"])
    else:
        df["lap_time_sec"] = np.nan

    if "Time" in df.columns:
        df["timing_time_sec"] = _to_seconds(df["Time"])
    else:
        df["timing_time_sec"] = np.nan

    df["lap_time_personal_best"] = df.groupby("Driver")["lap_time_sec"].cummin()
    df["lap_delta_to_pb"] = df["lap_time_sec"] - df["lap_time_personal_best"]
    df["roll_lap_3"] = df.groupby("Driver")["lap_time_sec"].transform(
        lambda series: series.rolling(3, min_periods=1).mean()
    )
    df["roll_lap_5"] = df.groupby("Driver")["lap_time_sec"].transform(
        lambda series: series.rolling(5, min_periods=1).mean()
    )

    field_pace = df.groupby("LapNumber")["lap_time_sec"].transform("median")
    df["field_median_lap_time_sec"] = field_pace
    df["lap_time_vs_field_median"] = df["lap_time_sec"] - field_pace
    df["lap_pace_rank"] = df.groupby("LapNumber")["lap_time_sec"].rank(
        method="average", ascending=True
    )

    # A positive value means that the driver gained places.
    previous_position = df.groupby("Driver")["current_position"].shift(1)
    three_laps_ago = df.groupby("Driver")["current_position"].shift(3)
    df["position_change_last_lap"] = previous_position - df["current_position"]
    df["position_change_last_3"] = three_laps_ago - df["current_position"]

    if "GridPosition" in df.columns:
        grid = pd.to_numeric(df["GridPosition"], errors="coerce")
        df["position_delta_from_grid"] = grid - df["current_position"]
    else:
        df["position_delta_from_grid"] = np.nan

    if "PitInTime" in df.columns:
        df["pit_this_lap"] = df["PitInTime"].notna().astype(int)
    else:
        df["pit_this_lap"] = 0
    df["pits_so_far"] = df.groupby("Driver")["pit_this_lap"].cumsum()
    df["already_pitted"] = (df["pits_so_far"] > 0).astype(int)

    if "TyreLife" in df.columns:
        df["tyre_age"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    else:
        df["tyre_age"] = np.nan
    df["stint_number"] = (
        pd.to_numeric(df["Stint"], errors="coerce") if "Stint" in df.columns else np.nan
    )
    df["compound"] = (
        df["Compound"].astype("string") if "Compound" in df.columns else pd.Series(pd.NA, index=df.index)
    )

    # Use the timing value for the same lap. Do not return negative gaps.
    df["gap_to_leader_seconds"] = np.nan
    df["gap_ahead_seconds"] = np.nan
    df["gap_behind_seconds"] = np.nan
    for _, lap_group in df.groupby("LapNumber", sort=True):
        valid = lap_group.dropna(subset=["current_position", "timing_time_sec"])
        if valid.empty:
            continue
        valid = valid.sort_values("current_position")
        leader_time = valid["timing_time_sec"].min()
        by_position = dict(
            zip(valid["current_position"].astype(int), valid["timing_time_sec"].astype(float))
        )
        for idx, row in valid.iterrows():
            position = int(row["current_position"])
            current_time = float(row["timing_time_sec"])
            df.at[idx, "gap_to_leader_seconds"] = max(current_time - leader_time, 0.0)
            if position - 1 in by_position:
                df.at[idx, "gap_ahead_seconds"] = max(
                    current_time - by_position[position - 1], 0.0
                )
            if position + 1 in by_position:
                df.at[idx, "gap_behind_seconds"] = max(
                    by_position[position + 1] - current_time, 0.0
                )

    df = _lap_status_features(df)
    if scheduled_laps is not None and scheduled_laps > 0:
        df["scheduled_laps"] = float(scheduled_laps)
        df["lap_fraction"] = df["lap_number"] / float(scheduled_laps)
        df["laps_remaining"] = (float(scheduled_laps) - df["lap_number"]).clip(lower=0)
    else:
        df["scheduled_laps"] = np.nan
        df["lap_fraction"] = np.nan
        df["laps_remaining"] = np.nan

    return df
