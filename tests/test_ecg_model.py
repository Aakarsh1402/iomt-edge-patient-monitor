"""Tests for ECG recording loading and windowing (no TensorFlow required)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "arrhythmia"))

from ecg_model import load_ecg_csv, make_windows  # noqa: E402

COLLECTED = os.path.join(ROOT, "arrhythmia", "raw_data", "collected")


def test_bare_column_csv_loads():
    sig = load_ecg_csv(os.path.join(COLLECTED, "normal", "2025-11-25_20-04-09-Aakarsh.csv"))
    assert sig.dtype == np.float32
    assert len(sig) >= 3600
    assert not np.isnan(sig).any()


def test_timestamp_value_csv_loads_and_fills_truncated_last_sample():
    """The exercise recordings end with an empty value; it must not become NaN."""
    sig = load_ecg_csv(os.path.join(COLLECTED, "arrhythmia", "Aakarsh_light_30s_20251201_202022.csv"))
    assert len(sig) == 3600
    assert not np.isnan(sig).any()


def test_windows_tile_the_signal():
    sig = np.arange(10, dtype=np.float32)
    w = make_windows(sig, 4)
    assert w.shape == (2, 4)
    assert w[1].tolist() == [4, 5, 6, 7]


def test_windows_with_stride_overlap():
    w = make_windows(np.arange(10, dtype=np.float32), 4, stride=2)
    assert w.shape == (4, 4)


def test_short_signal_is_zero_padded_to_one_window():
    w = make_windows(np.ones(3, dtype=np.float32), 8)
    assert w.shape == (1, 8)
    assert w[0, :3].tolist() == [1, 1, 1] and w[0, 3:].sum() == 0
