"""In-race lap-level feature construction from FastF1."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.race_state_features import add_race_state_features
from src.fastf1_utils import classified_position

logger = logging.getLogger(__name__)


def extract_in_race_from_session(
    race,
    year: int,
    round_number: int,
    event_name: str,
    circuit_key: str,
    scheduled_laps: int | None = None,
) -> Optional[pd.DataFrame]:
    """Return one row per driver and completed lap.

    Each row is a snapshot at the end of a completed lap.
    """
    results = race.results
    laps = race.laps
    if results is None or laps is None or len(results) == 0 or len(laps) == 0:
        logger.warning("Missing laps/results for %s R%s", year, round_number)
        return None

    finish_map = {}
    team_map = {}
    grid_map = {}
    for _, r in results.iterrows():
        abbr = str(r.get("Abbreviation", ""))
        finish = classified_position(r)
        if finish is None or not abbr:
            continue
        finish_map[abbr] = float(finish)
        team_map[abbr] = str(r.get("TeamName", ""))
        grid = pd.to_numeric(pd.Series([r.get("GridPosition")]), errors="coerce").iloc[0]
        grid_map[abbr] = float(grid) if not pd.isna(grid) else np.nan

    if not finish_map:
        return None

    laps = laps.copy()
    laps["GridPosition"] = laps["Driver"].map(grid_map)
    laps = add_race_state_features(laps, scheduled_laps=scheduled_laps)

    gap_rows = []
    for _, row in laps.iterrows():
        driver = str(row["Driver"])
        if driver not in finish_map or pd.isna(row["current_position"]):
            continue
        lap_no = int(row["lap_number"])
        gap_rows.append(
            {
                "year": year,
                "round": int(round_number),
                "event_name": event_name,
                "circuit": circuit_key,
                "race_id": f"{year}_R{int(round_number)}",
                "driver": driver,
                "team": team_map.get(driver, ""),
                "lap_number": lap_no,
                "current_position": float(row["current_position"]),
                "grid_position": row["GridPosition"],
                "position_delta_from_grid": row["position_delta_from_grid"],
                "position_change_last_lap": row["position_change_last_lap"],
                "position_change_last_3": row["position_change_last_3"],
                "gap_to_leader": row["gap_to_leader_seconds"],
                "gap_ahead": row["gap_ahead_seconds"],
                "gap_behind": row["gap_behind_seconds"],
                "tyre_age": row["tyre_age"],
                "stint_number": row["stint_number"],
                "compound": row["compound"],
                "pit_this_lap": int(row["pit_this_lap"]),
                "pits_so_far": int(row["pits_so_far"]),
                "already_pitted": int(row["already_pitted"]),
                "safety_car_flag": int(row["track_status_sc"]),
                "track_status_yellow": int(row["track_status_yellow"]),
                "track_status_sc": int(row["track_status_sc"]),
                "track_status_vsc": int(row["track_status_vsc"]),
                "track_status_red": int(row["track_status_red"]),
                "track_status_changed": int(row["track_status_changed"]),
                "track_status_laps_since_change": row["track_status_laps_since_change"],
                "sc_laps_last_3": row["sc_laps_last_3"],
                "vsc_laps_last_3": row["vsc_laps_last_3"],
                "interruption_laps_last_5": row["interruption_laps_last_5"],
                "lap_time_sec": row["lap_time_sec"],
                "lap_time_vs_field_median": row["lap_time_vs_field_median"],
                "lap_pace_rank": row["lap_pace_rank"],
                "lap_delta_to_pb": row["lap_delta_to_pb"],
                "roll_lap_3": row["roll_lap_3"],
                "roll_lap_5": row["roll_lap_5"],
                "scheduled_laps": row["scheduled_laps"],
                "lap_fraction": row["lap_fraction"],
                "laps_remaining": row["laps_remaining"],
                "finish_position": finish_map[driver],
            }
        )

    if not gap_rows:
        return None
    return pd.DataFrame(gap_rows)
