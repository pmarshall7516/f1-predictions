import unittest

import numpy as np
import pandas as pd

from src.metrics import metrics_by_lap
from src.modeling import add_position_delta_target, reconstruct_finish_from_delta
from src.race_state_features import add_race_state_features


class RaceStateFeatureTests(unittest.TestCase):
    def setUp(self):
        self.laps = pd.DataFrame(
            [
                {"Driver": "AAA", "LapNumber": 1, "Position": 1, "GridPosition": 2, "Time": 100.0, "LapTime": 100.0, "TrackStatus": "1", "TyreLife": 1, "Stint": 1, "PitInTime": pd.NaT, "Compound": "SOFT"},
                {"Driver": "BBB", "LapNumber": 1, "Position": 2, "GridPosition": 1, "Time": 102.0, "LapTime": 102.0, "TrackStatus": "1", "TyreLife": 1, "Stint": 1, "PitInTime": pd.NaT, "Compound": "MEDIUM"},
                {"Driver": "AAA", "LapNumber": 2, "Position": 2, "GridPosition": 2, "Time": 205.0, "LapTime": 105.0, "TrackStatus": "4", "TyreLife": 2, "Stint": 1, "PitInTime": pd.NaT, "Compound": "SOFT"},
                {"Driver": "BBB", "LapNumber": 2, "Position": 1, "GridPosition": 1, "Time": 203.0, "LapTime": 101.0, "TrackStatus": "4", "TyreLife": 2, "Stint": 1, "PitInTime": pd.NaT, "Compound": "MEDIUM"},
                {"Driver": "AAA", "LapNumber": 3, "Position": 2, "GridPosition": 2, "Time": 306.0, "LapTime": 101.0, "TrackStatus": "1", "TyreLife": 1, "Stint": 2, "PitInTime": 300.0, "Compound": "MEDIUM"},
                {"Driver": "BBB", "LapNumber": 3, "Position": 1, "GridPosition": 1, "Time": 304.0, "LapTime": 101.0, "TrackStatus": "1", "TyreLife": 3, "Stint": 1, "PitInTime": pd.NaT, "Compound": "MEDIUM"},
            ]
        )

    def test_features_are_end_of_lap_and_gaps_are_nonnegative(self):
        result = add_race_state_features(self.laps, scheduled_laps=10)
        self.assertEqual(result.shape[0], 6)
        self.assertTrue((result["gap_to_leader_seconds"].dropna() >= 0).all())
        self.assertTrue((result["gap_ahead_seconds"].dropna() >= 0).all())
        self.assertTrue((result["gap_behind_seconds"].dropna() >= 0).all())
        self.assertEqual(result.loc[(result.Driver == "BBB") & (result.LapNumber == 2), "position_change_last_lap"].iloc[0], 1)
        self.assertEqual(result.loc[result.LapNumber == 2, "track_status_sc"].sum(), 2)
        self.assertEqual(result.loc[result.LapNumber == 3, "sc_laps_last_3"].iloc[0], 1)
        self.assertAlmostEqual(result.loc[result.LapNumber == 3, "lap_fraction"].iloc[0], 0.3)

    def test_residual_target_reconstructs_finish(self):
        frame = pd.DataFrame({"current_position": [2.0, 8.0], "finish_position": [1.0, 10.0]})
        with_target = add_position_delta_target(frame)
        np.testing.assert_allclose(with_target["finish_delta"], [-1.0, 2.0])
        np.testing.assert_allclose(
            reconstruct_finish_from_delta(frame["current_position"], with_target["finish_delta"]),
            [1.0, 10.0],
        )

    def test_metrics_by_lap_keeps_race_counts(self):
        frame = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R2"],
                "lap_number": [1, 1, 2],
                "finish_position": [1.0, 2.0, 1.0],
            }
        )
        result = metrics_by_lap(frame, [1.0, 3.0, 2.0])
        self.assertEqual(result.loc[result.lap_number == 1, "races"].iloc[0], 1)
        self.assertEqual(result.loc[result.lap_number == 2, "samples"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()

