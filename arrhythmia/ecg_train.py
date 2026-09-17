"""Train the arrhythmia classifier on a laptop CPU and save ecg_model.keras.

    python ecg_train.py                    # ~5-10 min on a CPU
    python ecg_train.py --epochs 3 --stride 3600 --out /tmp/quick.keras   # smoke run

Protocol
  train  MIT-BIH DS1 (18 records, 10 s windows, 50 % overlap) + sensor normals from 2 volunteers
  val    4 DS1 records held out for early stopping
  test   MIT-BIH DS2 (22 unseen patients) + sensor normals from the 2 other volunteers
The test numbers are written to training_results/ecg_training.json and
reproduced by ecg_evaluate.py.

Why retrain rather than ship the notebook model: the notebook split
overlapping windows at random (a window shares 50 % of its samples with its
neighbours, so train and test were the same beats), fitted a scaler it never
saved, and fed raw millivolts - which is why base_model/ scores every sensor
recording, in ADC counts, as normal.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from ecg_data import (  # noqa: E402
    TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS, mitbih_windows, sensor_windows,
)
from ecg_model import DEFAULT_MODEL_FILE, WINDOW_SIZE, preprocess_windows  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "training_results")


def build_model(window_size=WINDOW_SIZE):
    """Small 1-D CNN. Early pooling keeps it fast on a CPU; the dilated block
    widens the receptive field to ~2.5 s so it can see rhythm, not just beat shape."""
    import keras
    from keras import layers

    inp = layers.Input(shape=(window_size, 1))
    x = inp
    for filters, kernel, pool, dilation in [(32, 15, 4, 1), (64, 9, 4, 1), (128, 7, 2, 1), (128, 7, 1, 4)]:
        x = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool > 1:
            x = layers.MaxPooling1D(pool)(x)
        x = layers.Dropout(0.2)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="ecg_arrhythmia_cnn")


def augment(x, y):
    """Random polarity flip (electrode orientation varies) and a random
    amount of white noise, up to the level the AD8232 produces."""
    import tensorflow as tf

    flip = tf.where(tf.random.uniform(()) < 0.5, -1.0, 1.0)
    x = x * flip + tf.random.normal(tf.shape(x)) * tf.random.uniform((), 0.0, 0.3)
    return x, y


def evaluate(model, X, y, threshold=0.5):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    p = model.predict(preprocess_windows(X), verbose=0, batch_size=256).ravel()
    out = {"n": int(len(y)), "positives": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
        tn, fp, fn, tp = confusion_matrix(y, p >= threshold).ravel()
        out.update(sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                   accuracy=float((tp + tn) / len(y)), confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])
    else:
        out["false_positive_rate"] = float((p >= threshold).mean())
    out["mean_prob"] = float(p.mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stride", type=int, default=WINDOW_SIZE // 2, help="training window stride")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensor-repeat", type=int, default=8,
                        help="oversample the (few) sensor recordings this many times")
    args = parser.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(args.seed)

    print("Building windows...")
    Xtr, ytr, _ = mitbih_windows(TRAIN_RECORDS, args.stride)
    Xs, ys, _ = sensor_windows("train")
    Xtr = np.concatenate([Xtr] + [Xs] * args.sensor_repeat)
    ytr = np.concatenate([ytr] + [ys] * args.sensor_repeat)
    Xva, yva, _ = mitbih_windows(VAL_RECORDS, WINDOW_SIZE)
    print(f"  train {len(ytr)} windows ({ytr.mean():.1%} arrhythmic, {len(ys)} sensor normals x{args.sensor_repeat})")
    print(f"  val   {len(yva)} windows ({yva.mean():.1%} arrhythmic)")

    pos = ytr.mean()
    class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
    train_ds = (tf.data.Dataset.from_tensor_slices((preprocess_windows(Xtr), ytr.astype(np.float32)))
                .shuffle(len(ytr), seed=args.seed).map(augment).batch(args.batch).prefetch(2))
    val_ds = tf.data.Dataset.from_tensor_slices((preprocess_windows(Xva), yva.astype(np.float32))).batch(256)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    model.summary(line_length=90)

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
    minutes = (time.time() - t0) / 60
    model.save(args.out)
    print(f"\nSaved {args.out} after {len(history.history['loss'])} epochs ({minutes:.1f} min)")

    print("\nEvaluating on unseen patients...")
    Xte, yte, _ = mitbih_windows(TEST_RECORDS, WINDOW_SIZE)
    Xst, yst, _ = sensor_windows("test")
    results = {
        "model": os.path.basename(args.out),
        "epochs_run": len(history.history["loss"]),
        "train_minutes": round(minutes, 1),
        "train": {"n": int(len(ytr)), "positives": int(ytr.sum()), "records": TRAIN_RECORDS},
        "val": {**evaluate(model, Xva, yva), "records": VAL_RECORDS},
        "test_mitbih_ds2": {**evaluate(model, Xte, yte), "records": TEST_RECORDS},
        "test_sensor_normal": evaluate(model, Xst, yst),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    for name in ("val", "test_mitbih_ds2", "test_sensor_normal"):
        r = results[name]
        summary = (f"AUC {r['auc']:.3f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}"
                   if "auc" in r else f"false-positive rate {r['false_positive_rate']:.3f}")
        print(f"  {name:<20} n={r['n']:<5} {summary}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ecg_training.json"), "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("loss", "auc")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history["val_" + key], label="val")
        ax.set_title(key); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ecg_training.png"), dpi=110)
    print(f"Wrote {RESULTS_DIR}/ecg_training.json and ecg_training.png")


if __name__ == "__main__":
    main()
