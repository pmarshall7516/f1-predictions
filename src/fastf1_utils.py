"""FastF1 cache setup and session loading helpers."""

from __future__ import annotations

import logging
import time
from typing import Optional

import fastf1
import pandas as pd

from src.config import CACHE_DIR, SC_TRACK_STATUS_CHARS

logger = logging.getLogger(__name__)


class RateLimitError(RuntimeError):
    """Raised when Jolpica/Ergast rate limit is detected."""


def enable_cache(cache_dir=None) -> None:
    """Enable FastF1 disk cache under data/fastf1_cache."""
    path = cache_dir or CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(path))
    logger.info("FastF1 cache enabled at %s", path)


def _is_rate_limit(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "500 calls" in text or "rate limit" in text or "429" in text


def retry_call(fn, *, desc: str, max_retries: int = 4, base_sleep: float = 3700.0):
    """Retry FastF1 calls when hitting the Jolpica hourly rate limit.

    Jolpica enforces ~500 calls/hour. When exceeded, wait just over an hour
    before retrying rather than thrashing short backoffs.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_rate_limit(exc):
                sleep_s = base_sleep if attempt > 0 else min(base_sleep, 600.0)
                logger.warning(
                    "%s hit rate limit (attempt %s/%s). Sleeping %.0fs…",
                    desc,
                    attempt + 1,
                    max_retries,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            logger.warning("%s failed: %s", desc, exc)
            return None
    logger.error("%s failed after retries: %s", desc, last_exc)
    return None


def get_feature_events(year: int) -> pd.DataFrame:
    """Return race weekends for a season (exclude testing)."""

    def _load():
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        events = schedule.copy()
        events = events[events["RoundNumber"] > 0].copy()
        if "EventFormat" in events.columns:
            events = events[~events["EventFormat"].astype(str).str.lower().eq("testing")]
        return events.reset_index(drop=True)

    result = retry_call(_load, desc=f"schedule {year}")
    if result is None:
        raise RateLimitError(f"Could not load schedule for {year}")
    return result


def load_session(year: int, round_number: int, session_type: str, laps: bool = True):
    """Load a session with retries; return None on non-rate-limit failure."""

    def _load():
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=laps, telemetry=False, weather=True, messages=False)
        return session

    return retry_call(
        _load,
        desc=f"{year} {session_type} R{round_number}",
        max_retries=4,
        base_sleep=3700.0,
    )


def weather_summary(session) -> dict:
    """Aggregate session weather into a compact feature dict."""
    defaults = {
        "weather_air_temp": float("nan"),
        "weather_track_temp": float("nan"),
        "weather_humidity": float("nan"),
        "weather_rainfall": 0.0,
    }
    try:
        weather = session.weather_data
    except Exception:  # noqa: BLE001
        return defaults
    if weather is None or len(weather) == 0:
        return defaults

    rainfall = 0.0
    if "Rainfall" in weather.columns:
        rainfall = float(weather["Rainfall"].astype(float).max())

    def _mean(col: str) -> float:
        if col not in weather.columns:
            return float("nan")
        return float(weather[col].astype(float).mean())

    return {
        "weather_air_temp": _mean("AirTemp"),
        "weather_track_temp": _mean("TrackTemp"),
        "weather_humidity": _mean("Humidity"),
        "weather_rainfall": rainfall,
    }


def classified_position(results_row) -> Optional[float]:
    """Official classified finishing position (includes DNFs as classified)."""
    pos = results_row.get("Position")
    if pd.isna(pos):
        return None
    return float(pos)


def is_safety_car_status(track_status) -> int:
    """Return 1 if TrackStatus indicates SC/VSC (or related) codes."""
    if track_status is None or (isinstance(track_status, float) and pd.isna(track_status)):
        return 0
    text = str(track_status)
    return int(any(ch in text for ch in SC_TRACK_STATUS_CHARS))
