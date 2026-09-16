"""Load the ECG arrhythmia classifier and score recordings outside the notebook.

    from ecg_model import ArrhythmiaDetector, load_ecg_csv
    det = ArrhythmiaDetector()                 # bundled base model (MIT-BIH)
    probs = det.score(load_ecg_csv("raw_data/collected/normal/x.csv"))

The bundled `base_model/` is the stage-1 network trained on MIT-BIH: it takes
10 s windows (3600 samples at 360 Hz) of raw millivolt ECG with no
normalisation. The fine-tuned stage-2 weights that adapt it to the AD8232
sensor (see training_results/training_summary.json) are not checked in; pass
their path as `model_path` when you have them. On raw ADC counts from the
sensor the base model alone is not expected to be accurate.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_DIR = os.path.join(HERE, "base_model")

SAMPLE_RATE = 360

# ESP32 12-bit ADC (3.3 V / 4095 counts) behind the AD8232's ~1000x gain gives
# ~0.8 uV per count at the electrodes; MIT-BIH is stored in mV.
ADC_COUNTS_TO_MV = 3.3 / 4095 / 1000 * 1000


def load_ecg_csv(path):
    """Read one recording as a float32 vector.

    Accepts the two formats the collector produced: a bare column of ints, or
    `timestamp,value` with a header. A truncated final sample (NaN) is common
    in the collected files and is filled from its neighbour.
    """
    with open(path) as f:
        first = f.readline()
    header = 0 if any(c.isalpha() for c in first) else None
    df = pd.read_csv(path, header=header)
    values = pd.to_numeric(df.iloc[:, -1], errors="coerce").ffill().bfill()
    return values.to_numpy(dtype=np.float32)


def make_windows(signal, window_size, stride=None):
    """Slice a 1-D signal into (n, window_size); a short signal is zero-padded to one window."""
    stride = stride or window_size
    if len(signal) < window_size:
        padded = np.zeros(window_size, dtype=np.float32)
        padded[:len(signal)] = signal
        return padded[None, :]
    starts = range(0, len(signal) - window_size + 1, stride)
    return np.stack([signal[s:s + window_size] for s in starts])


class ArrhythmiaDetector:
    def __init__(self, model_path=None, scale=ADC_COUNTS_TO_MV):
        """model_path: a .keras file, or a directory holding config.json +
        model.weights.h5 (the bundled base model). scale multiplies the input
        before inference (ADC counts -> mV by default; use 1.0 for mV input)."""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import keras

        model_path = model_path or BASE_MODEL_DIR
        if os.path.isdir(model_path):
            with open(os.path.join(model_path, "config.json")) as f:
                self.model = keras.models.model_from_json(f.read())
            self.model.load_weights(os.path.join(model_path, "model.weights.h5"))
        else:
            self.model = keras.models.load_model(model_path)
        self.window_size = int(self.model.input_shape[1])
        self.scale = scale
        self.model_path = model_path

    def score(self, signal, stride=None):
        """Per-window P(arrhythmia) for a 1-D signal, as a float array."""
        windows = make_windows(np.asarray(signal, dtype=np.float32) * self.scale, self.window_size, stride)
        return self.model.predict(windows[..., None], verbose=0).ravel()
