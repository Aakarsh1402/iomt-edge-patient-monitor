"""The scripted ward scenario shared by the demo monitor and the simulated
sensor node, plus real ECG excerpts for it.

Each step: (caption, IMU class id, camera verdict or None, ECG source).
ECG source is the name of a real recording excerpt ("normal", "arrhythmia")
scored by the arrhythmia model at run time, or None when no ECG evidence is
attached. `ECG_FALLBACK_PROB` is used when TensorFlow is not available.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "arrhythmia"))

SCENARIO = [
    ("resting quietly", 0, "Normal", "normal"),
    ("sitting up in bed", 1, "Normal", "normal"),
    ("walking to the bathroom", 2, "Normal", None),
    ("restless, shifting about", 6, "Normal", "normal"),
    ("restless, camera sees distress", 6, "Distress", "normal"),
    ("slumping sideways", 5, None, "normal"),
    ("breathing hard, ECG irregular", 10, "Distress", "arrhythmia"),
    ("FALL", 3, "Danger", "arrhythmia"),
    ("motionless on the floor", 0, "Danger", "arrhythmia"),
]

ECG_FALLBACK_PROB = {"normal": 0.05, "arrhythmia": 0.88}

# Excerpt sources: a held-out volunteer's AD8232 recording (ADC counts) and
# a window of MIT-BIH record 233 (a DS2 test patient) containing PVCs.
NORMAL_FILE = os.path.join(ROOT, "arrhythmia", "raw_data", "collected", "normal", "2025-11-25_18-43-33-Saharsh.csv")
ARRHYTHMIA_RECORD, ARRHYTHMIA_START = "233", 10800     # 7 PVCs in this 10 s


def ecg_excerpt(kind, samples=3600):
    """A `samples`-long 1-D float32 array for the named source."""
    from ecg_model import load_ecg_csv

    if kind == "normal":
        return load_ecg_csv(NORMAL_FILE)[:samples]
    if kind == "arrhythmia":
        import mitbih
        sig, _ = mitbih.load_record(ARRHYTHMIA_RECORD)
        mv = sig[ARRHYTHMIA_START:ARRHYTHMIA_START + samples]
        # Express in ESP32-style ADC counts so it looks like what the node sends.
        return (mv * 400 + 2048).astype(np.float32)
    raise KeyError(kind)
