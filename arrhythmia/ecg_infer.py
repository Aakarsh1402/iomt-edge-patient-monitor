"""Score ECG recordings with the arrhythmia classifier from the command line.

    python ecg_infer.py raw_data/collected/normal/2025-11-25_20-04-09-Aakarsh.csv
    python ecg_infer.py raw_data/collected/            # every CSV underneath, grouped by folder
    python ecg_infer.py raw_data/mitbih/arrhythmia/233.csv   # MIT-BIH record (mV)
    python ecg_infer.py recording.csv --model base_model      # the old notebook model

Sensor ADC counts and MIT-BIH millivolts are both fine: every window is
normalised before inference (see ecg_model.preprocess_windows).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecg_model import ArrhythmiaDetector, load_ecg_csv  # noqa: E402


def collect_csvs(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in sorted(os.walk(p)):
                for f in sorted(files):
                    if f.lower().endswith(".csv"):
                        yield os.path.join(root, f)
        else:
            yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="CSV files or directories")
    parser.add_argument("--model", help=".keras file or model directory (default: ecg_model.keras)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    detector = ArrhythmiaDetector(args.model)
    print(f"Model: {detector.model_path}  (window {detector.window_size} samples)\n")

    by_group = {}
    print(f"{'file':<58}{'windows':>8}{'mean P':>8}{'max P':>8}  verdict")
    for path in collect_csvs(args.paths):
        try:
            probs = detector.score(load_ecg_csv(path))
        except Exception as e:  # unreadable file, wrong shape, ...
            print(f"{path:<58} skipped: {e}")
            continue
        verdict = "ARRHYTHMIA" if probs.max() >= args.threshold else "normal"
        shown = os.path.relpath(path)
        shown = shown if len(shown) <= 56 else "..." + shown[-53:]
        print(f"{shown:<58}{len(probs):>8}{probs.mean():>8.3f}{probs.max():>8.3f}  {verdict}")
        by_group.setdefault(os.path.basename(os.path.dirname(path)), []).append(probs.max())

    if len(by_group) > 1:
        print("\nPer-folder summary (fraction of files flagged):")
        for group, maxes in by_group.items():
            flagged = np.mean(np.array(maxes) >= args.threshold)
            print(f"  {group:<14} n={len(maxes):<4} flagged={flagged:.0%}  mean max-P={np.mean(maxes):.3f}")


if __name__ == "__main__":
    main()
