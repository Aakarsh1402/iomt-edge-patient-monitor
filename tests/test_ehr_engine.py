"""EHR risk engine end to end (needs TensorFlow) and its TFLite export."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ehr_risk"))

pytest.importorskip("tensorflow")

from ehr_engine import KEY_RISKS, EHRRiskEngine  # noqa: E402

RECORDS = json.load(open(os.path.join(ROOT, "ehr_risk", "EHR_patient_records.json")))


@pytest.fixture(scope="module")
def engine():
    return EHRRiskEngine()


def test_high_risk_patient_scores_high(engine):
    """Patient 101 (John Doe: CHF, prior falls, seizures) must not come out at 0 %
    - that was the bug the scoring rewrite fixed."""
    r = engine.score(RECORDS["101"])
    assert r.risks[KEY_RISKS["fall"]] > 0.5
    assert r.risks[KEY_RISKS["seizure"]] > 0.5
    assert r.n_rows > 1
    assert set(r.key_risks()) == set(KEY_RISKS)


def test_healthy_patient_scores_low(engine):
    r = engine.score(RECORDS["102"])
    assert max(r.risks[k] for k in KEY_RISKS.values()) < 0.2


def test_missing_fields_are_tolerated(engine):
    r = engine.score({"name": "x", "age": 40, "gender": "F"})
    assert 0 <= max(r.risks.values()) <= 1


def test_tflite_export_agrees_with_keras(engine):
    from ehr_tflite_map import MAP_FILE, load_mapping, make_interpreter
    from ehr_vectorize import build_feature_vector
    assert os.path.exists(MAP_FILE), "run ehr_tflite_map.py"
    mapping = load_mapping()
    assert len(mapping) == len(engine.targets)
    assert sorted(mapping.values()) == sorted(engine.targets)

    interpreter = make_interpreter(os.path.join(ROOT, "ehr_risk", "ehr_model.tflite"))
    inp = interpreter.get_input_details()[0]
    record = RECORDS["101"]
    keras = engine.score(record).risks
    best = {}
    for label, row in engine._observation_rows(record):
        if not (label == "demographics+history" or label in engine.features):
            continue
        x = engine.scaler.transform(build_feature_vector(row, engine.features)[0]).astype(np.float32)
        x[:, engine._unsupported_idx] = 0.0
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        for d in interpreter.get_output_details():
            t = mapping[d["index"]]
            best[t] = max(best.get(t, 0.0), float(interpreter.get_tensor(d["index"]).reshape(-1)[0]))
    top = lambda d: {k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:5]}  # noqa: E731
    assert top(best) == top(keras)
