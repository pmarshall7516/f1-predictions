# F1 Finishing Position Predictions

Dual-stage baseline for predicting each driver’s classified finishing position in a Formula 1 Grand Prix:

1. **Pre-race** — Random Forest after the grid is set (qualifying / race grid features).
2. **In-race** — XGBoost updates that forecast after each lap, using live race state plus the pre-race prediction as an input feature.

Data comes from [FastF1](https://docs.fastf1.dev/). The baseline uses 2019–2025. The builder can add the current season. Train, validation, and test use this order: **2019–2023 / 2024 / 2025**.

Implementation rationale: [`docs/baseline-notes.md`](docs/baseline-notes.md). Current audit and saved baseline results: [`docs/current-audit.md`](docs/current-audit.md). Modeling and plots: [`baseline-notebook.ipynb`](baseline-notebook.ipynb).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs OpenMP:

```bash
brew install libomp
```

## Build the dataset

Downloads/caches FastF1 sessions and writes processed Parquets under `data/processed/` (gitignored). Resumes safely if interrupted; Jolpica’s ~500 calls/hour limit may cause long waits on a fresh build.

```bash
python scripts/build_datasets.py
```

Useful options:

```bash
python scripts/build_datasets.py --years 2024 2025 --sleep 3

python scripts/build_datasets.py --years 2026 --sleep 3
```

Outputs:

- `data/processed/pre_race.parquet` — one row per driver–race
- `data/processed/in_race_laps.parquet` — one row per driver–lap
- `data/fastf1_cache/` — FastF1 session cache

Check a local build. The command prints the drivers and teams in the data:

```bash
python scripts/audit_dataset.py
```

The new race-state fields are written when you build the data again. Use
`--rebuild` after a schema change.

After the build, compare the two state models:

```bash
python scripts/evaluate_state_model.py
python scripts/evaluate_state_model.py --train-through 2024 --validation-year 2025 --test-year 2026
```

## Run the baseline notebook

```bash
jupyter notebook baseline-notebook.ipynb
```

Run the separate improvement study here:

```bash
jupyter notebook improvement-notebook.ipynb
```

This notebook keeps the baseline notebook unchanged. It creates comparison
charts under `figures/improvement/`.

The notebook loads the Parquets, trains the pre-race RF (with race-grouped OOF predictions), trains the coupled in-race XGBoost (plus an ablation without `pre_race_pred`), and reports metrics tables and plots against naive baselines (grid position / current position).
