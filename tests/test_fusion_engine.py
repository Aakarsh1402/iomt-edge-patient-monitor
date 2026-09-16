"""Tests for the multimodal alert fusion rules."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fusion"))

from fusion_engine import (  # noqa: E402
    ARRHYTHMIA_THRESHOLD,
    UNCORROBORATED_CAP,
    AlertLevel,
    Observation,
    fuse,
)


def test_quiet_patient_produces_no_alert():
    result = fuse(Observation(motion_class="Lying", motion_confidence=0.99,
                              vision_class="Normal", vision_confidence=0.95))
    assert result.level is AlertLevel.NONE
    assert not result.should_page


def test_walking_alone_does_not_page():
    """An ambulatory patient walking is the single most common false alarm."""
    result = fuse(Observation(motion_class="Walking", motion_confidence=0.99))
    assert not result.should_page


def test_lone_sensor_is_capped_below_paging():
    """One sensor with a non-critical event must not page, however confident."""
    result = fuse(Observation(motion_class="Resp.Distress", motion_confidence=1.0))
    assert result.suppressed
    assert result.score == pytest.approx(UNCORROBORATED_CAP)
    assert not result.should_page


def test_agitation_alone_does_not_page():
    """Even with a matching clinical risk, restlessness alone is only a WATCH."""
    result = fuse(Observation(motion_class="Agitation", motion_confidence=1.0,
                              clinical_risks={"risk_seizure_risk_24h": 1.0}))
    assert result.level <= AlertLevel.WATCH
    assert not result.should_page


def test_agitation_plus_camera_distress_does_page():
    """The same event corroborated by the camera escalates."""
    result = fuse(Observation(motion_class="Agitation", motion_confidence=0.9,
                              vision_class="Distress", vision_confidence=0.9,
                              clinical_risks={"risk_seizure_risk_24h": 0.9}))
    assert not result.suppressed
    assert result.should_page
    assert set(result.corroborating_modalities) == {"physical", "visual"}


def test_confirmed_fall_pages_without_corroboration():
    """A fall is critical on its own -- the camera may not even see the patient."""
    result = fuse(Observation(motion_class="FALL", motion_confidence=0.95))
    assert result.should_page
    assert not result.suppressed


def test_fall_with_visual_confirmation_is_critical():
    result = fuse(Observation(motion_class="FALL", motion_confidence=0.98,
                              vision_class="Danger", vision_confidence=0.9,
                              clinical_risks={"risk_fall_risk_24h": 0.93}))
    assert result.level is AlertLevel.CRITICAL


def test_clinical_risk_amplifies_matching_event():
    """Same motion evidence scores higher for a patient with a known fall risk."""
    low = fuse(Observation(motion_class="Slump", motion_confidence=0.9,
                           vision_class="Distress", vision_confidence=0.8,
                           clinical_risks={"risk_fall_risk_24h": 0.0}))
    high = fuse(Observation(motion_class="Slump", motion_confidence=0.9,
                            vision_class="Distress", vision_confidence=0.8,
                            clinical_risks={"risk_fall_risk_24h": 0.95}))
    assert high.score > low.score
    assert "clinical" not in high.corroborating_modalities  # history is not a sensor


def test_clinical_risk_alone_never_pages():
    """A high 24 h risk is context, not an event."""
    result = fuse(Observation(clinical_risks={"risk_fall_risk_24h": 1.0,
                                              "risk_seizure_risk_24h": 1.0}))
    assert result.level is AlertLevel.NONE


def test_low_confidence_classification_is_ignored():
    result = fuse(Observation(motion_class="FALL", motion_confidence=0.2))
    assert result.level is AlertLevel.NONE
    assert result.reasons == []


def test_arrhythmia_below_threshold_contributes_nothing():
    quiet = fuse(Observation(arrhythmia_probability=ARRHYTHMIA_THRESHOLD - 0.01))
    assert quiet.level is AlertLevel.NONE


def test_arrhythmia_corroborates_a_physical_event():
    """Respiratory distress plus an arrhythmic ECG is two independent sensors."""
    result = fuse(Observation(motion_class="Resp.Distress", motion_confidence=0.9,
                              arrhythmia_probability=0.95))
    assert set(result.corroborating_modalities) == {"physical", "cardiac"}
    assert result.should_page
    assert not result.suppressed


def test_missing_modalities_are_tolerated():
    """Sensors drop out constantly in a real ward; fusion must still run."""
    result = fuse(Observation())
    assert result.level is AlertLevel.NONE
    assert result.score == 0.0


def test_unknown_motion_class_gets_a_middling_severity():
    """A future model class must not silently score zero."""
    result = fuse(Observation(motion_class="Wandering", motion_confidence=0.9))
    assert result.score > 0


def test_summary_is_human_readable():
    result = fuse(Observation(motion_class="FALL", motion_confidence=0.97,
                              vision_class="Danger", vision_confidence=0.88))
    text = result.summary()
    assert "CRITICAL" in text and "FALL" in text and "Danger" in text
