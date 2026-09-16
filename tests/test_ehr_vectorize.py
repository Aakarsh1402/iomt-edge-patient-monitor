"""Tests for the EHR record -> feature vector encoding.

These run without TensorFlow: ehr_vectorize.py is deliberately numpy-only.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ehr_risk"))

from ehr_vectorize import build_feature_vector, encode_gender, map_history  # noqa: E402

FEATURES = json.load(open(os.path.join(ROOT, "ehr_risk", "ehr_features.json")))
RECORDS = json.load(open(os.path.join(ROOT, "ehr_risk", "EHR_patient_records.json")))


@pytest.mark.parametrize("value,expected", [
    ("M", 1), ("m", 1), ("Male", 1), ("male", 1),
    ("F", 0), ("f", 0), ("Female", 0), ("female", 0),
])
def test_gender_accepts_both_spellings(value, expected):
    """The sample records use 'M'/'F'; training used 'male'/'female'."""
    assert encode_gender(value) == expected


def test_unknown_gender_is_rejected_loudly():
    with pytest.raises(ValueError):
        encode_gender("unknown")


def test_vector_has_one_entry_per_feature():
    vec, _ = build_feature_vector(RECORDS["101"], FEATURES)
    assert vec.shape == (1, len(FEATURES))


def test_vitals_and_labs_land_in_the_right_slots():
    vec, active = build_feature_vector(RECORDS["101"], FEATURES)
    assert vec[0, FEATURES.index("sbp")] == 155
    assert vec[0, FEATURES.index("creatinine")] == pytest.approx(1.8)
    assert active["age"] == 78


def test_history_abbreviations_are_recognised():
    """'CKD', 'CAD' and 'Heart Failure' must set the same flags as the long names."""
    record = {"age": 70, "gender": "M",
              "history": ["Chronic Kidney Disease (CKD)", "Coronary Artery Disease (CAD)",
                          "Heart Failure", "Epilepsy", "Type 2 Diabetes"]}
    _, active = build_feature_vector(record, FEATURES)
    for feature in ["history_ckd", "history_cad", "history_chf", "history_seizure", "history_diabetes"]:
        assert active.get(feature) == 1, feature


def test_short_abbreviations_match_whole_words_only():
    """'cad' must not fire on 'cascade'; 'uti' must not fire on 'nutrition'."""
    record = {"age": 50, "gender": "F", "history": ["Cascade screening", "Poor nutrition"]}
    _, active = build_feature_vector(record, FEATURES)
    assert "history_cad" not in active
    assert "history_uti" not in active


def test_healthy_patient_sets_no_history_flags():
    _, active = build_feature_vector(RECORDS["102"], FEATURES)
    assert not [k for k in active if k.startswith("history_")]


def test_unmapped_history_is_reported():
    """'History of Stroke' has no feature in this 61-input model; say so."""
    mapping = map_history(RECORDS["101"]["history"], FEATURES)
    assert mapping["History of Stroke"] is None
    assert mapping["Epilepsy"] == "history_seizure"


def test_none_history_entry_is_skipped():
    assert map_history(["None"], FEATURES) == {}


def test_missing_sections_do_not_crash():
    """Records arrive half-filled; the vectoriser must degrade, not raise."""
    vec, active = build_feature_vector({"age": 60, "gender": "F"}, FEATURES)
    assert vec.shape == (1, len(FEATURES))
    assert active == {"age": 60}
