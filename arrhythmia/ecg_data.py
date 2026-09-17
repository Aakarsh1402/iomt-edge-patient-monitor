"""Window-level datasets for training and evaluating the arrhythmia model.

Two domains:
  * MIT-BIH (mV, annotated) - inter-patient split: DS1 records train, DS2
    records test, so no patient appears on both sides.
  * AD8232 sensor recordings (ADC counts, 10 s each, all labelled "normal")
    - split by subject: two volunteers train, two are held out.

A window is positive when any part of it lies in an abnormal rhythm (AFIB,
VT, bigeminy, ...) or it holds at least MIN_ECTOPIC ectopic beats (V, E, A,
J, S, F, a, e, j); negative when the rhythm is normal and it holds none; and
dropped as ambiguous otherwise. Windows are returned raw; ecg_model.preprocess_windows
does the normalisation so training and inference cannot drift apart.
"""
import glob
import os

import numpy as np

import mitbih
from ecg_model import WINDOW_SIZE, load_ecg_csv

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTED_DIR = os.path.join(HERE, "raw_data", "collected")

MIN_ECTOPIC = 2
VAL_RECORDS = ["106", "115", "208", "220"]              # carved out of DS1 for early stopping
TRAIN_RECORDS = [r for r in mitbih.DS1 if r not in VAL_RECORDS]
TEST_RECORDS = list(mitbih.DS2)
SENSOR_TEST_SUBJECTS = ("Ronith", "Saharsh")


def mitbih_windows(records, stride, window_size=WINDOW_SIZE):
    """(X (n, window_size) float32 mV, y int8, record ids) for the given records."""
    X, y, ids = [], [], []
    for rec in records:
        sig, ann = mitbih.load_record(rec)
        for start, label in mitbih.label_windows(ann, len(sig), window_size, stride, MIN_ECTOPIC):
            if label is None:
                continue
            X.append(sig[start:start + window_size])
            y.append(label)
            ids.append(rec)
    return np.stack(X), np.array(y, dtype=np.int8), np.array(ids)


def sensor_subject(path):
    """'2025-11-25_18-43-33-Saharsh.csv' -> 'Saharsh'."""
    return os.path.basename(path).rsplit(".", 1)[0].rsplit("-", 1)[-1]


def sensor_files(label="normal"):
    return sorted(glob.glob(os.path.join(COLLECTED_DIR, label, "*.csv")))


def sensor_windows(split, window_size=WINDOW_SIZE):
    """Normal AD8232 recordings as (X, y=0, subjects); split is 'train' or 'test'."""
    X, subjects = [], []
    for path in sensor_files("normal"):
        subject = sensor_subject(path)
        if (subject in SENSOR_TEST_SUBJECTS) != (split == "test"):
            continue
        sig = load_ecg_csv(path)
        if len(sig) >= window_size:
            X.append(sig[:window_size])
            subjects.append(subject)
    return np.stack(X), np.zeros(len(X), dtype=np.int8), np.array(subjects)
