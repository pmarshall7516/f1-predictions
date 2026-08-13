# Repository audit

This audit uses the source code and the saved output in
`baseline-notebook.ipynb`. It was made on 2026-08-12.

The processed Parquet files are not in Git. Run this command after a local
data build:

```bash
python scripts/audit_dataset.py
```

The command also warns when the two files contain different races.

## Data

- FastF1 provides the data.
- The builder loads the main race session (`R`).
- A pre-race row uses each driver with a classified result.
- An in-race row uses each driver and each available completed lap.
- The target is the classified finish position.
- The data keeps a driver with a classified DNF position.
- The builder loads qualifying only when you use `--quali-weather`.
- The weather fields use qualifying weather when that option is active.
- Otherwise, the weather fields use race-session weather.
- Driver and team names come from FastF1.
- There is no fixed driver list.

The saved baseline run has these row counts:

| Season | Pre-race rows | In-race rows |
|---:|---:|---:|
| 2019 | 420 | 23,392 |
| 2020 | 340 | 16,951 |
| 2021 | 439 | 23,303 |
| 2022 | 439 | 23,050 |
| 2023 | 439 | 23,983 |
| 2024 | 479 | 26,381 |
| 2025 | 479 | 26,337 |
| **Total** | **3,035** | **163,397** |

The baseline uses this split:

- Train: 2019–2023
- Validation: 2024
- Test: 2025

The saved baseline output has no 2026 rows.

## Current processing

Pre-race processing creates these groups of fields:

- grid position;
- driver, team, and track history;
- driver qualifying-to-race history;
- session weather;
- driver, team, and track codes.

History uses races before the current race. Team history first combines both
team drivers into one value for each race. The rolling window then uses the
last five team races.

If the grid position is empty, the code keeps it empty. It does not use the
finish position as a replacement. The model fills the empty value from the
training data.

In-race processing keeps the baseline fields and adds these fields:

- position change in the last lap and last three laps;
- position change from the grid;
- lap number, lap share, and laps left;
- gaps from the timing value for the same lap;
- lap pace compared with the field;
- best-lap difference and rolling three-lap and five-lap pace;
- tyre compound, tyre age, pit count, and stint number;
- yellow flag, Safety Car, VSC, and red flag fields;
- track-status change and recent interruption counts.

The gap fields are empty when FastF1 does not provide a timing value. The
code does not build a gap from driver-only cumulative lap times.

The state-model script encodes driver, team, track, and compound names with
values learned from the training data. It does not send the text values to
XGBoost.

Damage data is not in the current tables. Add damage from race-control data
as a time-based event. Do not infer damage from the final result.

## Saved baseline results

The metrics use one row for each driver and lap in the in-race data. The
metrics include MAE, RMSE, exact position, and position within three places.

| Model | Split | MAE | RMSE | Exact | Within 3 |
|---|---|---:|---:|---:|---:|
| RF pre-race | validation | 3.0963 | 4.0051 | 9.39% | 66.60% |
| Naive grid | validation | 2.9061 | 4.2471 | 15.45% | 71.82% |
| RF pre-race | test | 3.4704 | 4.3452 | 8.35% | 56.99% |
| Naive grid | test | 3.3445 | 4.8056 | 18.37% | 63.88% |
| XGB coupled | validation | 1.9023 | 2.8879 | 24.07% | 85.38% |
| XGB ablation | validation | 1.8899 | 2.8248 | 24.08% | 85.48% |
| Naive current position | validation | 1.7723 | 2.9839 | 34.05% | 84.54% |
| XGB coupled | test | 2.2973 | 3.3729 | 18.63% | 80.31% |
| XGB ablation | test | 2.1990 | 3.2584 | 19.36% | 82.33% |
| Naive current position | test | 2.2052 | 3.5846 | 30.91% | 79.30% |

The baseline does not win against all simple comparisons. The pre-race model
does not beat the grid model on test MAE or exact position. The in-race
ablation is slightly better than the coupled model. Current position has the
best exact-match rate.

This result supports the next step: predict future movement from the current
race state instead of predicting the finish position from current position
alone.

## 2026 data

The builder now does these actions:

- includes the current calendar year in its default year list;
- skips future events;
- uses FastF1 `Session.total_laps` when it is available;
- adds completed 2026 races with an incremental build.

The builder skips a race only when both output files contain that race. This
allows a later run to complete a partial build.

The builder skips a race only when both output files contain that race. This
allows a later run to complete a partial build.

Use this command to add completed 2026 races:

```bash
python scripts/build_datasets.py --years 2026 --sleep 3
```

Use this command to build all seasons with the new fields:

```bash
python scripts/build_datasets.py --years 2019 2020 2021 2022 2023 2024 2025 2026 --rebuild
```

For a 2026 test, use 2019–2024 for training, 2025 for validation, and
completed 2026 races for the test. Do not tune model settings on the 2026
test data.
