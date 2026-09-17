"""Load the ECG arrhythmia classifier and score recordings outside the notebook.

    from ecg_model import ArrhythmiaDetector, load_ecg_csv
    det = ArrhythmiaDetector()                 # bundled ecg_model.keras
    probs = det.score(load_ecg_csv("raw_data/collected/normal/x.csv"))

The bundled `ecg_model.keras` is trained by ecg_train.py on MIT-BIH (DS1
records) plus normal recordings from the AD8232 sensor. Every 10 s window is
baseline-corrected and scaled to unit variance before inference
(`preprocess_windows`), so raw ADC counts from the sensor and millivolts from
MIT-BIH are handled identically - no gain constant needed.

`base_model/` is the older notebook model (raw mV input, no normalisation);
pass it as `model_path` to compare. It is not useful on sensor data.
"""
import os

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_FILE = os.path.join(HERE, "ecg_model.keras")
BASE_MODEL_DIR = os.path.join(HERE, "base_model")

SAMPLE_RATE = 360
WINDOW_SIZE = 10 * SAMPLE_RATE          # 3600 samples
BASELINE_SECONDS = 0.6                  # moving-average window removed as baseline wander


def load_ecg_csv(path):
    """Read one recording as a float32 vector.

    Accepts the two formats the collector produced: a bare column of numbers,
    or `timestamp,value` with a header. A truncated final sample (NaN) is
    common in the collected files and is filled from its neighbour.
    """
    with open(path) as f:
        first = f.readline()
    header = 0 if any(c.isalpha() for c in first) else None
    df = pd.read_csv(path, header=header)
    values = pd.to_numeric(df.iloc[:, -1], errors="coerce").ffill().bfill()
    return values.to_numpy(dtype=np.float32)


def make_windows(signal, window_size=WINDOW_SIZE, stride=None):
    """Slice a 1-D signal into (n, window_size); a short signal is zero-padded to one window."""
    stride = stride or window_size
    if len(signal) < window_size:
        padded = np.zeros(window_size, dtype=np.float32)
        padded[:len(signal)] = signal
        return padded[None, :]
    starts = range(0, len(signal) - window_size + 1, stride)
    return np.stack([signal[s:s + window_size] for s in starts])


def preprocess_windows(windows, sample_rate=SAMPLE_RATE):
    """Baseline-correct and standardise each window: (n, L) -> (n, L, 1) float32.

    Subtracting a short moving average removes wander and DC offset; dividing
    by the window's own standard deviation removes the gain difference between
    the sensor's ADC counts and MIT-BIH millivolts.
    """
    x = np.asarray(windows, dtype=np.float32)
    baseline = uniform_filter1d(x, size=int(BASELINE_SECONDS * sample_rate), axis=-1, mode="nearest")
    x = x - baseline
    x = x / (x.std(axis=-1, keepdims=True) + 1e-6)
    return x[..., None]


class ArrhythmiaDetector:
    def __init__(self, model_path=None):
        """model_path: a .keras file (default ecg_model.keras), or a directory
        holding config.json + model.weights.h5 (the legacy base model)."""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import keras

        model_path = model_path or DEFAULT_MODEL_FILE
        if os.path.isdir(model_path):
            with open(os.path.join(model_path, "config.json")) as f:
                self.model = keras.models.model_from_json(f.read())
            self.model.load_weights(os.path.join(model_path, "model.weights.h5"))
            self.normalise = False          # notebook model expects raw mV
        else:
            self.model = keras.models.load_model(model_path, compile=False)
            self.normalise = True
        self.window_size = int(self.model.input_shape[1])
        self.model_path = model_path

    def score_windows(self, windows):
        """P(arrhythmia) for an (n, window_size) array."""
        x = preprocess_windows(windows) if self.normalise else np.asarray(windows, np.float32)[..., None]
        return self.model.predict(x, verbose=0).ravel()

    def score(self, signal, stride=None):
        """Per-window P(arrhythmia) for a 1-D signal, as a float array."""
        return self.score_windows(make_windows(np.asarray(signal, dtype=np.float32), self.window_size, stride))
