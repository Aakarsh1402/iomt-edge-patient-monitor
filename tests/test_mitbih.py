"""MIT-BIH annotation parsing and window labelling (numpy only)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "arrhythmia"))

import mitbih  # noqa: E402


@pytest.mark.parametrize("record", ["100", "208", "233"])
def test_parsed_beat_counts_match_physionet_metadata(record):
    """The metadata JSON was produced by wfdb; our parser must agree with it."""
    beats = mitbih.read_annotations(os.path.join(mitbih.ANNOTATION_DIR, f"{record}.atr"))
    with open(os.path.join(mitbih.SIGNAL_DIR, f"{record}_metadata.json")) as f:
        expected = json.load(f)["beat_counts"]
    counts = {}
    for _, sym, _aux in beats:
        counts[sym] = counts.get(sym, 0) + 1
    for sym, n in expected.items():
        assert counts.get(sym, 0) == n, sym
    assert 0 < beats[-1][0] < 650000
    assert all(b[0] <= a[0] for b, a in zip(beats, beats[1:]))   # monotonic sample indices


def test_every_record_has_signal_and_annotations():
    recs = mitbih.records()
    assert len(recs) == 48
    for r in recs:
        assert os.path.exists(os.path.join(mitbih.ANNOTATION_DIR, f"{r}.atr"))
    assert not set(mitbih.DS1) & set(mitbih.DS2)
    assert set(mitbih.DS1) | set(mitbih.DS2) <= set(recs)


def test_window_labels_from_ectopic_beats():
    ann = [(0, "+", "(N"), (100, "N", ""), (400, "V", ""), (700, "N", ""), (1100, "V", ""), (1400, "A", ""),
           (2100, "N", "")]
    labels = dict(mitbih.label_windows(ann, n_samples=3000, window_size=1000, stride=1000, min_ectopic=2))
    assert labels == {0: None, 1000: 1, 2000: 0}     # one PVC = ambiguous, two = positive, none = negative


def test_window_labels_from_rhythm_marks():
    """Atrial fibrillation is all 'N' beats; the rhythm mark must still make it positive."""
    ann = [(0, "+", "(N"), (500, "N", ""), (1500, "+", "(AFIB"), (1600, "N", ""), (2500, "+", "(N"), (2700, "N", "")]
    labels = dict(mitbih.label_windows(ann, n_samples=3000, window_size=1000, stride=1000))
    assert labels == {0: 0, 1000: 1, 2000: 1}
    assert mitbih.rhythm_segments(ann, 3000) == [(0, 1500, "(N"), (1500, 2500, "(AFIB"), (2500, 3000, "(N")]


def test_rhythm_aux_text_is_parsed():
    ann = mitbih.read_annotations(os.path.join(mitbih.ANNOTATION_DIR, "201.atr"))
    rhythms = {aux for _, sym, aux in ann if sym == "+"}
    assert "(AFIB" in rhythms and "(N" in rhythms
