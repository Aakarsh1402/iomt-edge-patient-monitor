"""Read the MIT-BIH Arrhythmia Database copy under raw_data/mitbih.

The signals were exported to CSV (channel 0 / MLII, millivolts, 360 Hz) and
the beat annotations are the original PhysioNet `.atr` files, which are tiny.
This module parses them without depending on the wfdb package.

    from mitbih import records, load_record
    sig, beats = load_record("208")       # beats: list of (sample_index, symbol)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_DIR = os.path.join(HERE, "raw_data", "mitbih", "arrhythmia")
ANNOTATION_DIR = os.path.join(HERE, "raw_data", "mitbih", "annotations")

SAMPLE_RATE = 360

# Beat symbols that count as arrhythmic, as in the training notebook.
ARRHYTHMIA_SYMBOLS = frozenset("VEAJSFaej")
# Symbols that mark beats at all (the rest are rhythm/quality/comment marks).
BEAT_SYMBOLS = frozenset("NLRaVFJASEj/Qfe!")

# Inter-patient split of de Chazal et al. (2004): train on DS1, test on DS2.
# Records 102, 104, 107 and 217 are paced and conventionally excluded.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

# wfdb annotation code -> symbol (ann_label_table).
_CODE_TO_SYMBOL = {
    1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E",
    11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s", 19: "T", 20: "*", 21: "D",
    22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t", 28: "+", 29: "u", 30: "?",
    31: "!", 32: "[", 33: "]", 34: "e", 35: "n", 36: "@", 37: "x", 38: "f", 39: "(",
    40: ")", 41: "r",
}
_SKIP, _NUM, _SUB, _CHN, _AUX = 59, 60, 61, 62, 63


def records(signal_dir=SIGNAL_DIR):
    return sorted(f[:-4] for f in os.listdir(signal_dir) if f.endswith(".csv"))


def read_annotations(path):
    """Parse a PhysioNet .atr file into a list of (sample_index, symbol).

    Each entry is a little-endian 16-bit word: the top 6 bits are the
    annotation code, the low 10 bits the interval since the previous
    annotation. Codes 59-63 are escapes (long skip, modifiers, aux text).
    """
    words = np.fromfile(path, dtype="<u2")
    out, t, pending_skip, i = [], 0, 0, 0
    while i < len(words):
        code, interval = int(words[i] >> 10), int(words[i] & 0x3FF)
        i += 1
        if code == 0 and interval == 0:
            break
        if code == _SKIP:
            pending_skip = (int(words[i]) << 16) | int(words[i + 1])
            i += 2
        elif code in (_NUM, _SUB, _CHN):
            continue
        elif code == _AUX:
            i += (interval + 1) // 2
        else:
            t += interval + pending_skip
            pending_skip = 0
            out.append((t, _CODE_TO_SYMBOL.get(code, "?")))
    return out


def load_record(name, signal_dir=SIGNAL_DIR, annotation_dir=ANNOTATION_DIR):
    """(signal mV float32, [(sample, symbol), ...]) for one record."""
    sig = np.loadtxt(os.path.join(signal_dir, f"{name}.csv"), dtype=np.float32)
    beats = read_annotations(os.path.join(annotation_dir, f"{name}.atr"))
    return sig, beats


def label_windows(beats, n_samples, window_size, stride, min_ectopic=2):
    """Yield (start, label) for every window; label is 1 (>= min_ectopic
    arrhythmic beats), 0 (none) or None (ambiguous: fewer than min_ectopic)."""
    samples = np.array([s for s, _ in beats])
    ectopic = np.array([sym in ARRHYTHMIA_SYMBOLS for _, sym in beats])
    for start in range(0, n_samples - window_size + 1, stride):
        mask = (samples >= start) & (samples < start + window_size)
        n = int(ectopic[mask].sum())
        yield start, (1 if n >= min_ectopic else 0 if n == 0 else None)
