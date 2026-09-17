"""Re-run the held-out evaluation of ecg_model.keras.

    python ecg_evaluate.py                # DS2 patients + held-out sensor volunteers
    python ecg_evaluate.py --model base_model   # compare the notebook model

Prints AUC / sensitivity / specificity on MIT-BIH DS2 (22 patients never seen
in training), the false-positive rate on sensor recordings from the two
held-out volunteers, and what the model makes of the three post-exercise
sensor files (~140 bpm; never labelled by a clinician, so shown for transparency only).
Exit status is 1 if DS2 AUC falls below --min-auc.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from ecg_data import TEST_RECORDS, mitbih_windows, sensor_files, sensor_windows  # noqa: E402
from ecg_model import WINDOW_SIZE, ArrhythmiaDetector, load_ecg_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-auc", type=float, default=0.85)
    args = parser.parse_args()

    from sklearn.metrics import confusion_matrix, roc_auc_score

    det = ArrhythmiaDetector(args.model)
    print(f"Model: {det.model_path}\n")

    X, y, ids = mitbih_windows(TEST_RECORDS, WINDOW_SIZE)
    p = det.score_windows(X)
    auc = roc_auc_score(y, p)
    tn, fp, fn, tp = confusion_matrix(y, p >= args.threshold).ravel()
    print(f"MIT-BIH DS2 ({len(TEST_RECORDS)} unseen patients, {len(y)} windows, {y.sum()} arrhythmic)")
    print(f"  AUC {auc:.3f}   sensitivity {tp / (tp + fn):.3f}   specificity {tn / (tn + fp):.3f}   "
          f"accuracy {(tp + tn) / len(y):.3f}")
    print(f"  {'record':<8}{'windows':>8}{'arrhythmic':>11}{'mean P':>8}{'flagged':>9}")
    for rec in TEST_RECORDS:
        m = ids == rec
        print(f"  {rec:<8}{m.sum():>8}{int(y[m].sum()):>11}{p[m].mean():>8.2f}{(p[m] >= args.threshold).mean():>9.0%}")

    Xs, _, subjects = sensor_windows("test")
    ps = det.score_windows(Xs)
    print(f"\nAD8232 normal recordings, held-out volunteers ({', '.join(sorted(set(subjects)))}): "
          f"n={len(ps)}  false-positive rate {(ps >= args.threshold).mean():.3f}  mean P {ps.mean():.3f}")

    exercise = sensor_files("arrhythmia")
    if exercise:
        pe = np.array([det.score(load_ecg_csv(f)).max() for f in exercise])
        print(f"AD8232 post-exercise recordings (~140 bpm, n={len(pe)}): P = {', '.join(f'{v:.2f}' for v in pe)}"
              "\n  (unlabelled by a clinician - fast rate plus motion artefact; shown for transparency, not as a result)")

    if auc < args.min_auc:
        sys.exit(f"\nDS2 AUC {auc:.3f} is below {args.min_auc}")


if __name__ == "__main__":
    main()
