"""Turn a human-readable patient record (EHR_patient_records.json) into the
feature vector the EHR risk model expects, using the same encoding rules as
the training pipeline (ehr_preprocess.py / ehr_utils.py).

Pure numpy + stdlib so it can be unit-tested without TensorFlow.
"""
import re

import numpy as np

# Keyword lists mirror the check() calls in ehr_preprocess.py, extended with
# the abbreviations / synonyms clinicians actually write in a history list.
HISTORY_KEYWORDS = {
    "history_hypertension": ["hypertension", "high blood pressure"],
    "history_diabetes": ["diabetes", "diabetic"],
    "history_chf": ["congestive heart failure", "heart failure", "chf"],
    "history_copd": ["copd", "chronic obstructive pulmonary"],
    "history_anemia": ["anemia", "anaemia"],
    "history_hyperlipidemia": ["hyperlipidemia", "hyperlipidaemia", "high cholesterol"],
    "history_cad": ["coronary artery disease", "cad", "coronary heart disease"],
    "history_ckd": ["chronic kidney disease", "ckd", "renal failure"],
    "history_obesity": ["obesity", "obese"],
    "history_asthma": ["asthma"],
    "history_pneumonia": ["pneumonia"],
    "history_flu": ["influenza", "flu"],
    "history_afib": ["atrial fibrillation", "afib", "a-fib"],
    "history_dvt_pe": ["dvt", "deep vein thrombosis", "pulmonary embolism"],
    "history_thyroid": ["hypothyroidism", "hyperthyroidism", "thyroid"],
    "history_gout": ["gout"],
    "history_arthritis": ["osteoarthritis", "rheumatoid arthritis", "arthritis"],
    "history_pancreatitis": ["pancreatitis"],
    "history_gi_bleed": ["gi bleed", "gastrointestinal bleed", "varices"],
    "history_ibd": ["crohn", "ulcerative colitis", "ibd", "inflammatory bowel"],
    "history_cellulitis": ["cellulitis"],
    "history_uti": ["urinary tract infection", "uti"],
    "history_seizure": ["epilepsy", "seizure"],
    "history_dementia": ["dementia", "alzheimer"],
}

MALE_VALUES = {"m", "male", "man"}
FEMALE_VALUES = {"f", "female", "woman"}


def encode_gender(value):
    """Training encoded 'male' -> 1, 'female' -> 0 (ehr_utils.encode_data)."""
    s = str(value).strip().lower()
    if s in MALE_VALUES:
        return 1
    if s in FEMALE_VALUES:
        return 0
    raise ValueError(f"Unrecognised gender value {value!r}; use 'M'/'F' or 'male'/'female'")


def _history_flag(feature, history_entries):
    """1 if any keyword for this history feature appears in the record's history list."""
    keywords = HISTORY_KEYWORDS.get(feature, [feature.replace("history_", "").replace("_", " ")])
    text = " ".join(h.lower() for h in history_entries)
    tokens = set(re.findall(r"[a-z0-9\-]+", text))
    for kw in keywords:
        # Short abbreviations must match a whole word ("cad" must not hit "cascade").
        if (kw in tokens) if len(kw) <= 4 else (kw in text):
            return 1
    return 0


def map_history(history_entries, feature_list):
    """Map each free-text history entry to the model feature it sets (or None).

    Lets the demo say exactly which parts of a patient's history the model can
    actually see, instead of guessing from feature-name substrings.
    """
    mapping = {}
    for entry in history_entries:
        if entry.strip().lower() in ("", "none"):
            continue
        hit = next((f for f in feature_list
                    if f.startswith("history_") and _history_flag(f, [entry])), None)
        mapping[entry] = hit
    return mapping


def build_feature_vector(record, feature_list):
    """Return (vector[1, n_features], active) where active maps each non-zero
    feature name to the value that was used. Unknown features default to 0."""
    vitals = record.get("vitals", {}) or {}
    labs = record.get("labs", {}) or {}
    history = record.get("history", []) or []

    vec = np.zeros((1, len(feature_list)), dtype=np.float32)
    active = {}
    for i, feature in enumerate(feature_list):
        if feature == "age":
            val = float(record.get("age", 0) or 0)
        elif feature == "gender":
            val = encode_gender(record["gender"]) if record.get("gender") else 0
        elif feature in vitals:
            val = float(vitals[feature])
        elif feature in labs:
            val = float(labs[feature])
        elif feature.startswith("history_"):
            val = _history_flag(feature, history)
        else:
            val = 0.0
        vec[0, i] = val
        if val:
            active[feature] = val
    return vec, active


def top_risks(probabilities, target_list, n=5):
    """Sort (target, probability) pairs descending."""
    pairs = sorted(zip(target_list, probabilities), key=lambda p: p[1], reverse=True)
    return pairs[:n]


def pretty_target(name):
    return name.replace("risk_", "").replace("_24h", "").replace("_", " ").title()
