"""Loads the trained EHR risk model and scores patient records.

    from ehr_engine import EHRRiskEngine
    engine = EHRRiskEngine()
    result = engine.score(record)
    result.risks["risk_fall_risk_24h"], result.latency_ms, result.warnings

How scoring works
-----------------
The training set (ehr_preprocess.py) has one row per FHIR *observation*: each
row carries demographics + history flags + the single vital/lab that was
recorded at that time, everything else 0. A record with a full panel of
vitals therefore looks nothing like a training row (inputs land 10-2000 SDs
from the training mean and every sigmoid head saturates to 0). So we score a
patient the way the model was trained: one row per observation, then take the
maximum probability per risk across rows ("does any observation trigger this
risk?"), which is also how the per-row training labels were defined.

Features whose value never varied in training (e.g. history_copd) carry no
learned signal, and because the first layer is a BatchNorm with zero moving
variance for them, a non-zero value is amplified ~30x and corrupts every head.
They are pinned to their training value and reported in result.warnings.
"""
import json
import os
import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from ehr_vectorize import build_feature_vector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Headline risks surfaced by the demo and consumed by the fusion engine.
KEY_RISKS = {
    "fall": "risk_fall_risk_24h",
    "seizure": "risk_seizure_risk_24h",
    "cardiac": "risk_mi_24h",
    "sepsis": "risk_sepsis_24h",
    "stroke": "risk_stroke_24h",
}

# |z-score| beyond which a numeric input is almost certainly in the wrong units
# (e.g. d-dimer in ng/mL when training used mg/L).
OUT_OF_RANGE_Z = 25.0


@dataclass
class ScoreResult:
    risks: dict                      # target name -> probability (max over observation rows)
    latency_ms: float
    n_rows: int                      # observation rows scored
    active_features: dict            # feature -> value actually fed to the model
    warnings: list = field(default_factory=list)

    def key_risks(self):
        return {short: self.risks.get(target, 0.0) for short, target in KEY_RISKS.items()}


class EHRRiskEngine:
    def __init__(self, model_dir=HERE):
        import tensorflow as tf  # imported lazily so ehr_vectorize stays TF-free

        with open(os.path.join(model_dir, "ehr_features.json")) as f:
            self.features = json.load(f)
        with open(os.path.join(model_dir, "ehr_targets.json")) as f:
            self.targets = json.load(f)
        self.scaler = joblib.load(os.path.join(model_dir, "ehr_scaler.joblib"))
        self.model = tf.keras.models.load_model(os.path.join(model_dir, "ehr_model.keras"))

        n_in = getattr(self.scaler, "n_features_in_", len(self.features))
        if n_in != len(self.features):
            raise ValueError(
                f"Scaler expects {n_in} features but ehr_features.json lists {len(self.features)}; "
                "the artifacts come from different training runs."
            )
        # Features that were constant across the whole training set.
        self.unsupported = [f for f, v in zip(self.features, self.scaler.var_) if v == 0]
        self._unsupported_idx = [self.features.index(f) for f in self.unsupported]

        # One warm-up call so the first real prediction is not dominated by graph tracing.
        self._predict(np.zeros((1, len(self.features)), dtype=np.float32))

    # ------------------------------------------------------------------ internals
    def _predict(self, x):
        """x: scaled (n, n_features) -> probabilities (n, n_targets)."""
        out = self.model.predict(x, verbose=0)
        # The multi-task model returns one (n, 1) array per head.
        if isinstance(out, (list, tuple)):
            out = np.concatenate([np.asarray(o).reshape(len(x), -1) for o in out], axis=1)
        return np.asarray(out).reshape(len(x), -1)

    @staticmethod
    def _observation_rows(record):
        """Split a record into training-style rows: base (demographics + history)
        plus one row per vital/lab. Yields (label, record_dict)."""
        base = {k: v for k, v in record.items() if k not in ("vitals", "labs")}
        yield "demographics+history", base
        for group in ("vitals", "labs"):
            for name, value in (record.get(group) or {}).items():
                row = dict(base)
                row[group] = {name: value}
                yield name, row

    # ------------------------------------------------------------------ public
    def score(self, record):
        """Score one patient record dict (see EHR_patient_records.json)."""
        rows, labels, active, warns = [], [], {}, []
        for label, row in self._observation_rows(record):
            vec, used = build_feature_vector(row, self.features)
            if label != "demographics+history" and label not in self.features:
                warns.append(f"'{label}' is not a model input; ignored")
                continue
            rows.append(vec)
            labels.append(label)
            active.update(used)

        x = self.scaler.transform(np.concatenate(rows)).astype(np.float32)

        # Pin never-varying features to their training value (z = 0).
        bad = [f for f in self.unsupported if active.get(f)]
        if bad:
            warns.append("not learned by this model (constant in training data), ignored: " + ", ".join(bad))
            x[:, self._unsupported_idx] = 0.0

        # Flag numeric inputs that are wildly outside the training range.
        for i, label in enumerate(labels):
            if label in self.features:
                z = abs(x[i, self.features.index(label)])
                if z > OUT_OF_RANGE_Z:
                    warns.append(f"'{label}'={active.get(label):g} is {z:.0f} SD from the training mean; check units")

        t0 = time.perf_counter()
        probs = self._predict(x)
        latency_ms = (time.perf_counter() - t0) * 1000
        if probs.shape[1] != len(self.targets):
            raise ValueError(f"Model produced {probs.shape[1]} outputs but {len(self.targets)} targets are listed")

        risks = dict(zip(self.targets, map(float, probs.max(axis=0))))
        return ScoreResult(risks, latency_ms, len(rows), active, warns)
