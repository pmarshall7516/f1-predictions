"""Project paths and baseline configuration."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "fastf1_cache"
PROCESSED_DIR = DATA_DIR / "processed"

PRE_RACE_PATH = PROCESSED_DIR / "pre_race.parquet"
IN_RACE_PATH = PROCESSED_DIR / "in_race_laps.parquet"

TRAIN_YEARS = list(range(2019, 2024))  # 2019–2023
VAL_YEAR = 2024
TEST_YEAR = 2025
CURRENT_SEASON = date.today().year
ALL_YEARS = list(range(2019, CURRENT_SEASON + 1))  # Include the current season.

ROLLING_WINDOW = 5

# TrackStatus codes that indicate Safety Car / Virtual Safety Car periods
SC_TRACK_STATUS_CHARS = {"4", "5", "6", "7"}
