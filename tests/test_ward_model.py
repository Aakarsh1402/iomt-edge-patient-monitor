"""IMU motion classifier: checkpoint loads and recognises the physics generators."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gesture_recognition"))

torch = pytest.importorskip("torch")

from synthetic_signals import make_window  # noqa: E402
from ward_model import CLASSES, WINDOW_SIZE, load_model, predict_window, window_to_tensor  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_window_to_tensor_accepts_both_layouts():
    w = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
    assert tuple(window_to_tensor(w).shape) == (1, 3, WINDOW_SIZE)
    assert tuple(window_to_tensor(w.T).shape) == (1, 3, WINDOW_SIZE)


@pytest.mark.parametrize("class_name", ["Lying", "Walking", "FALL", "SEIZURE", "CPR"])
def test_generated_windows_are_recognised(model, class_name):
    np.random.seed(0)
    hits = 0
    for _ in range(5):
        name, conf, probs = predict_window(model, make_window(CLASSES.index(class_name)))
        assert 0 <= conf <= 1 and abs(probs.sum() - 1) < 1e-4
        hits += name == class_name
    assert hits >= 4, f"{class_name}: {hits}/5"
