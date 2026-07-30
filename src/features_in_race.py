"""In-race lap-level feature construction from FastF1."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.fastf1_utils import classified_position, is_safety_car_status

logger = logging.getLogger(__name__)


def _lap_time_seconds(series: pd.Series) -> pd.Series:
    """Convert FastF1 timedeltas / strings to float seconds."""
    out = []
    for val in series:
        if pd.isna(val):
            out.append(np.nan)
        elif hasattr(val, "total_seconds"):
            out.append(float(val.total_seconds()))
        else:
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                out.append(np.nan)
    return pd.Series(out, index=series.index, dtype=float)


def extract_in_race_from_session(
    race,
    year: int,
    round_number: int,
    event_name: str,
    circuit_key: str,
) -> Optional[pd.DataFrame]:
    """One sample per driver-lap from an already-loaded race session."""
    results = race.results
    laps = race.laps
    if results is None or laps is None or len(results) == 0 or len(laps) == 0:
        logger.warning("Missing laps/results for %s R%s", year, round_number)
        return None

    finish_map = {}
    team_map = {}
    for _, r in results.iterrows():
        abbr = str(r.get("Abbreviation", ""))
        finish = classified_position(r)
        if finish is None or not abbr:
            continue
        finish_map[abbr] = float(finish)
        team_map[abbr] = str(r.get("TeamName", ""))

    if not finish_map:
        return None

    laps = laps.copy()
    laps["LapTimeSec"] = _lap_time_seconds(laps["LapTime"])
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["CumTime"] = laps.groupby("Driver")["LapTimeSec"].cumsum()

    if "PitInTime" in laps.columns:
        laps["pit_this_lap"] = laps["PitInTime"].notna().astype(int)
    else:
        laps["pit_this_lap"] = 0

    laps["pits_so_far"] = laps.groupby("Driver")["pit_this_lap"].cumsum()
    laps["already_pitted"] = (laps["pits_so_far"] > 0).astype(int)

    if "TrackStatus" in laps.columns:
        laps["safety_car_flag"] = laps["TrackStatus"].map(is_safety_car_status)
    else:
        laps["safety_car_flag"] = 0

    laps["tyre_age"] = laps["TyreLife"] if "TyreLife" in laps.columns else np.nan
    laps["stint_number"] = laps["Stint"] if "Stint" in laps.columns else np.nan

    laps["personal_best"] = laps.groupby("Driver")["LapTimeSec"].cummin()
    laps["lap_delta_to_pb"] = laps["LapTimeSec"] - laps["personal_best"]
    laps["roll_lap_3"] = laps.groupby("Driver")["LapTimeSec"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    gap_rows = []
    for lap_no, lap_group in laps.groupby("LapNumber"):
        g = lap_group.dropna(subset=["Position", "CumTime"]).copy()
        if g.empty:
            continue
        g = g.sort_values("Position")
        leader_cum = g["CumTime"].min()
        pos_to_cum = dict(zip(g["Position"].astype(int), g["CumTime"]))
        for _, row in g.iterrows():
            driver = str(row["Driver"])
            if driver not in finish_map:
                continue
            try:
                pos = int(row["Position"])
            except (TypeError, ValueError):
                continue
            cum = float(row["CumTime"])
            if pd.isna(cum):
                continue
            gap_leader = cum - leader_cum
            gap_ahead = np.nan
            gap_behind = np.nan
            if pos - 1 in pos_to_cum:
                gap_ahead = cum - float(pos_to_cum[pos - 1])
            if pos + 1 in pos_to_cum:
                gap_behind = float(pos_to_cum[pos + 1]) - cum

            gap_rows.append(
                {
                    "year": year,
                    "round": int(round_number),
                    "event_name": event_name,
                    "circuit": circuit_key,
                    "race_id": f"{year}_R{int(round_number)}",
                    "driver": driver,
                    "team": team_map.get(driver, ""),
                    "lap_number": int(lap_no),
                    "current_position": float(pos),
                    "gap_to_leader": float(gap_leader),
                    "gap_ahead": float(gap_ahead) if not pd.isna(gap_ahead) else np.nan,
                    "gap_behind": float(gap_behind) if not pd.isna(gap_behind) else np.nan,
                    "tyre_age": float(row["tyre_age"]) if not pd.isna(row["tyre_age"]) else np.nan,
                    "stint_number": float(row["stint_number"])
                    if not pd.isna(row["stint_number"])
                    else np.nan,
                    "pit_this_lap": int(row["pit_this_lap"]),
                    "already_pitted": int(row["already_pitted"]),
                    "safety_car_flag": int(row["safety_car_flag"]),
                    "lap_delta_to_pb": float(row["lap_delta_to_pb"])
                    if not pd.isna(row["lap_delta_to_pb"])
                    else np.nan,
                    "roll_lap_3": float(row["roll_lap_3"])
                    if not pd.isna(row["roll_lap_3"])
                    else np.nan,
                    "finish_position": finish_map[driver],
                }
            )

    if not gap_rows:
        return None
    return pd.DataFrame(gap_rows)
