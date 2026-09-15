"""Probe the IMU classifier with fresh physics-inspired synthetic signals.

Each class is probed with several newly generated windows (see
synthetic_signals.py); a class passes if >= 80 % are classified correctly.
Exit status is 0 when every class passes, so this doubles as a smoke test.

Usage:
    python sanity_check.py                      # uses ward_model_strict.pth
    python sanity_check.py --model ward_model.pth
"""
import argparse
import sys

import numpy as np

from synthetic_signals import make_window
from ward_model import CLASSES, DEFAULT_MODEL_FILE, load_model, predict_window

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default=DEFAULT_MODEL_FILE)
parser.add_argument("--seed", type=int, default=0, help="RNG seed for the synthetic noise (repeatable results)")
args = parser.parse_args()
np.random.seed(args.seed)

print("Loading Model for Physics Check...")
model = load_model(args.model)
print("Model Loaded.\n")

def test_signal(class_id, n_trials=10):
    """Generate n fresh windows for one class and report how many the model gets right."""
    expected = CLASSES[class_id]
    hits, confs = 0, []
    for _ in range(n_trials):
        res, conf, _ = predict_window(model, make_window(class_id))
        hits += res == expected
        confs.append(conf)
    ok = hits >= 0.8 * n_trials
    print(f"{expected:<14} -> {hits:>2}/{n_trials} correct  (mean conf {100*np.mean(confs):5.1f}%)  {'PASS' if ok else 'FAIL'}")
    return ok


print("-" * 60)
results = [test_signal(i) for i in range(len(CLASSES))]
print("-" * 60)
print(f"{sum(results)}/{len(results)} classes passed")
sys.exit(0 if all(results) else 1)
