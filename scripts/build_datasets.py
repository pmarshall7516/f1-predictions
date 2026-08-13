#!/usr/bin/env python3
"""Download FastF1 sessions and write processed baseline Parquets.

Resumes safely: existing race_ids in processed Parquets are skipped.
Uses rate-limit retries against Jolpica's 500 calls/hour cap.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import ALL_YEARS, IN_RACE_PATH, PRE_RACE_PATH, PROCESSED_DIR
from src.fastf1_utils import RateLimitError, enable_cache, get_feature_events, load_session
from src.features_in_race import extract_in_race_from_session
from src.features_pre_race import add_historical_features, extract_pre_race_from_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_datasets")


def _existing_race_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["race_id"])
    return set(df["race_id"].astype(str).unique())


def build(
    years: list[int],
    sleep_between_events: float = 1.5,
    load_quali_weather: bool = False,
    rebuild: bool = False,
) -> None:
    enable_cache()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pre_done = _existing_race_ids(PRE_RACE_PATH)
    in_done = _existing_race_ids(IN_RACE_PATH)
    # Skip a race only when both output files contain it.
    done_ids = set() if rebuild else (pre_done & in_done)
    if done_ids:
        logger.info("Resuming; %s race_ids already present", len(done_ids))

    pre_frames: list[pd.DataFrame] = []
    in_frames: list[pd.DataFrame] = []
    if PRE_RACE_PATH.exists():
        pre_frames.append(pd.read_parquet(PRE_RACE_PATH))
    if IN_RACE_PATH.exists():
        in_frames.append(pd.read_parquet(IN_RACE_PATH))

    for year in years:
        try:
            events = get_feature_events(year)
        except RateLimitError as exc:
            logger.error("%s — will retry later seasons after backoff", exc)
            time.sleep(600)
            try:
                events = get_feature_events(year)
            except RateLimitError:
                logger.error("Still rate-limited for %s; stopping year loop early", year)
                break

        logger.info("Season %s: %s events", year, len(events))
        for _, event in tqdm(events.iterrows(), total=len(events), desc=str(year)):
            round_number = int(event["RoundNumber"])
            race_id = f"{year}_R{round_number}"
            if race_id in done_ids:
                continue

            event_name = str(event.get("EventName", f"Round {round_number}"))
            circuit = str(event.get("Location", event.get("Country", event_name)))
            race = load_session(year, round_number, "R", laps=True)
            if race is None:
                logger.warning("Skipping %s — race session unavailable", race_id)
                time.sleep(sleep_between_events)
                continue

            scheduled_laps = None
            candidate = pd.to_numeric(
                pd.Series([getattr(race, "total_laps", None)]), errors="coerce"
            ).iloc[0]
            if not pd.isna(candidate) and candidate > 0:
                scheduled_laps = int(candidate)

            quali = None
            if load_quali_weather:
                quali = load_session(year, round_number, "Q", laps=False)

            pre = extract_pre_race_from_session(
                race, year, round_number, event_name, circuit, quali=quali
            )
            laps = extract_in_race_from_session(
                race,
                year,
                round_number,
                event_name,
                circuit,
                scheduled_laps=scheduled_laps,
            )

            if pre is not None:
                pre_frames.append(pre)
            if laps is not None:
                in_frames.append(laps)
            if pre is not None or laps is not None:
                done_ids.add(race_id)

            # Persist incrementally so long runs survive interruption
            if pre_frames:
                pre_race = pd.concat(pre_frames, ignore_index=True)
                pre_race = pre_race.drop_duplicates(subset=["race_id", "driver"], keep="last")
                # Historical features recomputed over full table at the end; store raw for now
                pre_race.to_parquet(PRE_RACE_PATH.with_suffix(".raw.parquet"), index=False)
            if in_frames:
                in_race = pd.concat(in_frames, ignore_index=True)
                in_race = in_race.drop_duplicates(
                    subset=["race_id", "driver", "lap_number"], keep="last"
                )
                in_race.to_parquet(IN_RACE_PATH, index=False)

            time.sleep(sleep_between_events)

    if not pre_frames:
        raise RuntimeError("No pre-race rows were built — check network/cache/FastF1.")

    pre_race = pd.concat(pre_frames, ignore_index=True)
    pre_race = pre_race.drop_duplicates(subset=["race_id", "driver"], keep="last")
    pre_race["quali_to_race_delta"] = pre_race["grid_position"] - pre_race["finish_position"]
    hist_cols = [
        "driver_roll_finish_5",
        "team_roll_finish_5",
        "track_avg_finish",
        "driver_quali_race_conv",
        "race_seq",
    ]
    pre_race = pre_race.drop(columns=[c for c in hist_cols if c in pre_race.columns], errors="ignore")
    pre_race = add_historical_features(pre_race)
    pre_race.to_parquet(PRE_RACE_PATH, index=False)
    logger.info("Wrote %s (%s rows)", PRE_RACE_PATH, len(pre_race))

    if not in_frames:
        raise RuntimeError("No in-race lap rows were built.")

    in_race = pd.concat(in_frames, ignore_index=True)
    in_race = in_race.drop_duplicates(subset=["race_id", "driver", "lap_number"], keep="last")
    in_race.to_parquet(IN_RACE_PATH, index=False)
    logger.info("Wrote %s (%s rows)", IN_RACE_PATH, len(in_race))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=ALL_YEARS,
        help="Seasons to download/build (default: 2019–current calendar year)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between events (rate-limit courtesy)",
    )
    parser.add_argument(
        "--quali-weather",
        action="store_true",
        help="Also load qualifying for weather features (uses more API calls)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-extract selected seasons even when their race_ids already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        args.years,
        sleep_between_events=args.sleep,
        load_quali_weather=args.quali_weather,
        rebuild=args.rebuild,
    )
