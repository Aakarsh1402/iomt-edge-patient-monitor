"""Multimodal alert fusion for the IoMT ward monitor.

The README describes the fusion rule that the whole project is built around --
"alert only when an anomaly is corroborated by more than one signal" -- but
until now it lived only in prose. This module implements it.

Design
------
Each modality contributes evidence, not a decision:

  physical  IMU motion classifier    -> event class + confidence
  visual    camera classifier        -> Normal / Distress / Danger + confidence
  cardiac   ECG arrhythmia CNN       -> P(arrhythmia) for the last window
  clinical  EHR risk model           -> per-condition 24 h risk probabilities

The score is `severity x confidence` for the physical event, plus the visual
and cardiac contributions, all modulated by the patient's clinical risk (a
frail patient with a 90 % fall risk is escalated sooner than a healthy one).

Crucially, a *single* uncorroborated modality is capped below the paging
threshold unless it is a critical event on its own (a confirmed fall, an
in-progress seizure). This is the false-alarm suppression that makes nurses
trust the system: a patient rolling over in bed scores, but does not page.

Everything here is pure Python so it can be unit-tested without TensorFlow,
PyTorch or hardware (see tests/test_fusion_engine.py).
"""
from dataclasses import dataclass, field
from enum import IntEnum

# ---------------------------------------------------------------- modalities

# How alarming each IMU class is on its own, 0..1.
MOTION_SEVERITY = {
    "Lying": 0.0,
    "Sitting": 0.0,
    "Walking": 0.05,     # ambulatory patients are expected to walk
    "Transport": 0.0,    # being wheeled around is normal ward traffic
    "FALL": 1.0,
    "SEIZURE": 1.0,
    "Choking": 1.0,
    "CPR": 1.0,          # someone is already resuscitating: escalate anyway
    "Resp.Distress": 0.7,
    "Vomiting": 0.6,
    "Slump": 0.5,
    "Agitation": 0.35,
}

# Events severe enough to page on their own, without corroboration...
CRITICAL_MOTION = {"FALL", "SEIZURE", "Choking", "CPR"}
# ...but only when the classifier is sure. A 55 % "FALL" on a patient rolling
# over is exactly the false alarm this system exists to suppress.
CRITICAL_ALONE_MIN_CONFIDENCE = 0.75

VISION_SEVERITY = {"Normal": 0.0, "Distress": 0.5, "Danger": 1.0}

# Clinical risks that make a given physical event more plausible. A motion
# event that matches the patient's known risk profile is escalated faster.
MOTION_TO_CLINICAL_RISK = {
    "FALL": "risk_fall_risk_24h",
    "Slump": "risk_fall_risk_24h",
    "SEIZURE": "risk_seizure_risk_24h",
    "Agitation": "risk_seizure_risk_24h",
    "Resp.Distress": "risk_copd_exacerbation_24h",
    "Choking": "risk_copd_exacerbation_24h",
}

# Weights of each evidence stream in the raw score.
W_PHYSICAL = 1.0
W_VISUAL = 0.6
W_CARDIAC = 0.8
# How much a matching clinical risk can amplify the physical evidence (x1..x1.5).
CLINICAL_GAIN = 0.5

# Below this, a lone modality cannot page on its own.
UNCORROBORATED_CAP = 0.55
# P(arrhythmia) above which the ECG counts as corroborating evidence.
ARRHYTHMIA_THRESHOLD = 0.7
# Confidence below which a classifier output is treated as "no opinion".
MIN_CONFIDENCE = 0.5
# Clinical risks below this are not worth mentioning as amplifiers.
MIN_CLINICAL_RISK = 0.05


class AlertLevel(IntEnum):
    NONE = 0      # nothing to show
    INFO = 1      # logged on the ward dashboard
    WATCH = 2     # highlighted; check on the next round
    ALERT = 3     # page the nurse
    CRITICAL = 4  # page the nurse and the rapid-response team

    @property
    def label(self):
        return {0: "OK", 1: "INFO", 2: "WATCH", 3: "ALERT", 4: "CRITICAL"}[int(self)]


# Score thresholds, checked from the top down.
LEVEL_THRESHOLDS = [
    (1.30, AlertLevel.CRITICAL),
    (0.70, AlertLevel.ALERT),
    (0.40, AlertLevel.WATCH),
    (0.10, AlertLevel.INFO),
]


@dataclass
class Observation:
    """One fused snapshot of a patient. Any modality may be None (sensor down)."""
    motion_class: str = None
    motion_confidence: float = 0.0
    vision_class: str = None
    vision_confidence: float = 0.0
    arrhythmia_probability: float = None
    clinical_risks: dict = field(default_factory=dict)
    heart_rate: float = None


@dataclass
class FusionResult:
    level: AlertLevel
    score: float
    reasons: list                      # human-readable evidence, strongest first
    corroborating_modalities: list     # which streams agreed
    suppressed: bool = False           # True if a lone modality was capped

    @property
    def should_page(self):
        return self.level >= AlertLevel.ALERT

    def summary(self):
        why = "; ".join(self.reasons) if self.reasons else "no abnormal signals"
        note = " [uncorroborated - capped]" if self.suppressed else ""
        return f"{self.level.label} ({self.score:.2f}){note}: {why}"


def _confident(confidence):
    return confidence is not None and confidence >= MIN_CONFIDENCE


def fuse(obs, thresholds=LEVEL_THRESHOLDS):
    """Combine the modalities of one Observation into a FusionResult."""
    score = 0.0
    reasons = []
    modalities = []
    critical_alone = False

    # --- physical (IMU) ---
    motion_severity = 0.0
    if obs.motion_class and _confident(obs.motion_confidence):
        motion_severity = MOTION_SEVERITY.get(obs.motion_class, 0.3)
        if motion_severity > 0:
            contribution = W_PHYSICAL * motion_severity * obs.motion_confidence

            # A matching known clinical risk amplifies the physical evidence.
            risk_key = MOTION_TO_CLINICAL_RISK.get(obs.motion_class)
            clinical = obs.clinical_risks.get(risk_key, 0.0) if risk_key else 0.0
            if clinical >= MIN_CLINICAL_RISK:
                contribution *= 1.0 + CLINICAL_GAIN * clinical
                reasons.append(
                    f"known {risk_key.replace('risk_', '').replace('_24h', '')} "
                    f"{clinical * 100:.0f}% amplifies the motion evidence"
                )
                modalities.append("clinical")

            score += contribution
            reasons.insert(0, f"motion '{obs.motion_class}' at {obs.motion_confidence * 100:.0f}% confidence")
            modalities.insert(0, "physical")
            critical_alone = (obs.motion_class in CRITICAL_MOTION
                              and obs.motion_confidence >= CRITICAL_ALONE_MIN_CONFIDENCE)

    # --- visual (camera) ---
    if obs.vision_class and _confident(obs.vision_confidence):
        vision_severity = VISION_SEVERITY.get(obs.vision_class, 0.0)
        if vision_severity > 0:
            score += W_VISUAL * vision_severity * obs.vision_confidence
            reasons.append(f"camera sees '{obs.vision_class}' at {obs.vision_confidence * 100:.0f}% confidence")
            modalities.append("visual")

    # --- cardiac (ECG) ---
    if obs.arrhythmia_probability is not None and obs.arrhythmia_probability >= ARRHYTHMIA_THRESHOLD:
        score += W_CARDIAC * obs.arrhythmia_probability
        reasons.append(f"ECG arrhythmia probability {obs.arrhythmia_probability * 100:.0f}%")
        modalities.append("cardiac")

    # --- corroboration rule ---
    sensor_modalities = [m for m in modalities if m != "clinical"]
    suppressed = False
    if len(sensor_modalities) < 2 and not critical_alone and score > UNCORROBORATED_CAP:
        score = UNCORROBORATED_CAP
        suppressed = True
        reasons.append("only one sensor agrees - held below paging threshold")

    level = AlertLevel.NONE
    for threshold, candidate in thresholds:
        if score >= threshold:
            level = candidate
            break

    return FusionResult(level, round(score, 3), reasons, sensor_modalities, suppressed)
