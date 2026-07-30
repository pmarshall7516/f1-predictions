# Baseline Implementation Notes (Phase 1)

This document explains the dual-stage baseline for **F1 Grand Prix finishing-position prediction**. It covers data choices, preprocessing, model rationale, evaluation design, and known limitations. **Numeric run results and post-hoc analysis will be appended in a later phase after the notebook has been executed and reviewed.**

## Problem framing

We predict each driver’s classified finishing position in a Formula 1 feature race at two stages:

1. **Pre-race** — immediately after qualifying / once the race grid is known.
2. **In-race** — after each completed lap, updating the forecast with live race state.

The stages are coupled for the baseline by feeding the pre-race model’s prediction (`pre_race_pred`) into the in-race model as an input feature. That mirrors the product story of a static baseline forecast that is revised as the race unfolds, without yet implementing a full Bayesian update stack.

## Data source

**FastF1** (local cache under `data/fastf1_cache/`).

Rationale:

- Lap-level timing, positions, tyres, pits, weather, and track status are all required for the in-race model; FastF1 is the natural single pipeline for that.
- Adding Jolpica, Kaggle dumps, or Open-Meteo in the same baseline pass would multiply ingestion failure modes before the models are proven.
- Session weather from FastF1 stands in for “conditions” features in this baseline (true forecast weather can be added later).

Processed tables written by `scripts/build_datasets.py`:

- `data/processed/pre_race.parquet` — one row per driver–race
- `data/processed/in_race_laps.parquet` — one row per driver–lap

Raw cache and Parquets are gitignored; rebuild locally with:

```bash
.venv/bin/python scripts/build_datasets.py
```

The builder resumes by skipping `race_id`s already present, retries on Jolpica’s **500 calls/hour** rate limit, and loads each feature race once (laps + results) to reduce API traffic. Optional `--quali-weather` loads qualifying for weather features at the cost of extra calls.
## Scope choices that shape the label

| Choice | Decision | Why |
|--------|----------|-----|
| Event type | Feature races only (no sprint-as-sample) | Keeps the target = main-race classified finish; avoids mixing short-sprint strategy into the same label |
| DNFs | Keep official classified position | Matches results tables and keeps a single regression target |
| Task type | Regression on finish position | Aligns with RF / XGBoost regressors and MAE/RMSE interpretability (“off by ~N places”) |

## Chronological split

| Split | Years |
|-------|-------|
| Train | 2019–2023 |
| Validation | 2024 |
| Test | 2025 |

Chronological splitting prevents future races from leaking into training. Rolling historical features (form, track averages, quali→race conversion) are computed using **only prior races** in calendar order.

## Features (lean set)

### Pre-race

- Grid position
- Driver rolling average finish (last 5 prior races)
- Team rolling average finish (last 5)
- Track-specific historical average finish (driver at circuit when available, else circuit mean)
- Driver quali→race conversion history (mean grid−finish over last 5)
- Qualifying-session weather summary (air/track temp, humidity, rainfall flag)
- Encoded driver / team / circuit (ordinal, fit on train only)

### In-race

- Current position
- Gap to leader / car ahead / car behind (from cumulative lap times)
- Tyre age, stint number
- Pit-this-lap and already-pitted indicators
- Safety Car / VSC-related track-status flag
- Lap-time delta to personal best; rolling mean of last 3 lap times
- **`pre_race_pred`** (coupled prior)

## Preprocessing

1. Load processed Parquets; split by `year`.
2. Fit categorical ordinal encoders on **train only**; unknown categories → encoded sentinel.
3. Impute numeric NaNs with **train medians** (remaining all-missing → 0).
4. No aggressive normalization for tree models (RF / XGBoost are scale-insensitive).
5. For ranking-style hit-rates, continuous predictions are rounded and clipped to a valid position range.

## Models and coupling

### Pre-race: Random Forest Regressor

Chosen as a strong tabular baseline that:

- Handles mixed numeric + encoded categorical inputs with little tuning
- Provides interpretable feature importances for the post-qualifying story
- Is robust on the modest sample size of one row per driver–race

### In-race: XGBoost Regressor

Chosen because:

- Lap-level data is much denser; gradient boosting typically fits that regime well
- Missingness and nonlinear interactions (tyre age × position, SC periods, etc.) are common in race state
- Matches the project proposal’s real-time baseline algorithm

### How “updates” works in this baseline

The in-race model does **not** run a Bayesian posterior update. Instead:

1. Generate **out-of-fold (OOF)** pre-race predictions on the training set with **GroupKFold by `race_id`**, so an entire Grand Prix is never in both sides of a fold.
2. Refit the RF on all training races; predict `pre_race_pred` for validation and test.
3. Merge `pre_race_pred` onto lap rows and train XGBoost with it as a feature.
4. Report an **ablation** model trained without `pre_race_pred` to quantify the value of coupling.

This is standard stacking discipline and avoids optimistic leakage from in-sample RF predictions.

## Evaluation

### Metrics

- **MAE** and **RMSE** on continuous predicted position
- **Top-1 exact** match rate after rounding
- **Top-3 / Top-10 hit** rates: fraction of samples within 3 / 10 places of truth after rounding

### Naive baselines

- Pre-race: predict **grid position**
- In-race: predict **current lap position**

Models must beat these to be considered useful beyond “grid ≈ finish” or “current place ≈ finish.”

### Plots (notebook)

1. Feature importance (RF and XGBoost)
2. Predicted vs actual scatter (test) for pre-race and coupled in-race
3. Residual / error distribution
4. MAE vs lap number (naive current-pos vs ablation vs coupled) on test
5. Example race trajectories of predicted finish over laps

## Implementation layout

| Path | Role |
|------|------|
| `src/` | Cache helpers, feature builders, metrics, modeling utilities |
| `scripts/build_datasets.py` | Download/cache FastF1 + write Parquets |
| `baseline-notebook.ipynb` | Load → preprocess → train/eval → tables & plots |
| `docs/baseline-notes.md` | This document (Phase 1) |

## Known limitations (baseline)

- Session weather ≠ true pre-race forecast weather.
- Classified DNF positions inject noise; a Finish-vs-DNF head is deferred.
- Sprint weekends contribute only the feature race; sprint results are ignored as samples.
- Gap features derived from cumulative lap times are an approximation of timing-tower gaps.
- Driver/team encoding treats identity as a category; mid-season transfers / rebrands can blur team strength.
- No explicit modeling of strategy compounds beyond tyre age / stint / pit flags.
- Improved phase (stacking ensembles, LSTM/TFT, Bayesian updates, external standings/weather APIs) is intentionally out of scope here.

## Phase 2 (deferred)

After `baseline-notebook.ipynb` has been run and results reviewed, append:

- Metric tables (val/test) for RF, XGBoost coupled, ablation, and naive baselines
- Interpretation of feature importances and MAE-vs-lap curves
- Failure cases and next-model priorities

*(No results analysis in this Phase 1 document.)*
