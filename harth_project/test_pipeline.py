"""Focused checks for window boundaries and feature correctness."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import har_utils as h


class WindowTests(unittest.TestCase):
    def test_constant_features(self):
        features = h.window_features(np.ones((2, 100, 6)))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertAlmostEqual(features.loc[0, 'back_x__mean'], 1)
        self.assertAlmostEqual(features.loc[0, 'back_x__std'], 0)
        self.assertAlmostEqual(features.loc[0, 'back_mag__mean'], np.sqrt(3))
        self.assertAlmostEqual(features.loc[0, 'back__sma'], 3)
        self.assertAlmostEqual(features.loc[0, 'back_x__dominant_hz'], 0)
        self.assertEqual(len(h.feature_columns(features)), 58)
        self.assertEqual(len(features.columns), 82)

    def test_frequency_in_hz(self):
        t = np.arange(100) / 50
        raw = np.tile(np.sin(2 * np.pi * 2 * t)[None, :, None], (1, 1, 6))
        features = h.window_features(raw)
        self.assertAlmostEqual(features.loc[0, 'back_x__dominant_hz'], 2)

    def test_gap_and_mixed_labels(self):
        # 110 samples followed by a gap and another 110. Must make two
        # windows (starts 0 and 110), not bridge the gap at sample 100.
        times = pd.date_range('2020-01-01', periods=220, freq='20ms').to_series().reset_index(drop=True)
        times.iloc[110:] += pd.Timedelta(seconds=3)
        frame = pd.DataFrame(np.ones((220, 6)), columns=h.AXES)
        frame['timestamp'] = times
        frame['label'] = 1
        frame.loc[60:109, 'label'] = 6
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(h, 'ROOT', Path(tmp)), patch.object(h, 'subject_files', return_value=[Path('S001.csv')]), patch.object(h, 'read_subject', return_value=frame):
                features, meta = h.build_windows(2)
                self.assertEqual(meta.start_row.tolist(), [0, 110])
                self.assertEqual(meta.label.tolist(), [1, 1])
                self.assertAlmostEqual(meta.purity.iloc[0], .6)
                self.assertEqual(len(features), 2)
                audit = pd.read_csv(Path(tmp) / 'results/window_audit_2s.csv')
                self.assertEqual(audit.discarded_samples.iloc[0], 20)

    def test_invalid_reading_splits_run(self):
        frame = pd.DataFrame(np.ones((301, 6)), columns=h.AXES)
        frame['timestamp'] = pd.date_range('2020-01-01', periods=301, freq='20ms')
        frame['label'] = 7
        frame.loc[100, 'back_x'] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(h, 'ROOT', Path(tmp)), patch.object(h, 'subject_files', return_value=[Path('S001.csv')]), patch.object(h, 'read_subject', return_value=frame):
                _, meta = h.build_windows(2)
                self.assertEqual(meta.start_row.tolist(), [0, 101, 201])


if __name__ == '__main__':
    unittest.main()
