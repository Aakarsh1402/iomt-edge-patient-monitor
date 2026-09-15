"""Evaluate the IMU motion classifier on the held-out (unseen-subject) test set.

Usage:
    python evaluate_model.py                       # ward_model_strict.pth on ward_data_test.npz
    python evaluate_model.py --model ward_model.pth
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from ward_model import CLASSES, DEFAULT_MODEL_FILE, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(HERE, "ward_data_test.npz")
BATCH_SIZE = 64

CLASS_NAMES = [f"{i}: {name}" for i, name in enumerate(CLASSES)]


def evaluate(model_file, data_file, device="cpu"):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run process_ward_data.py first.")

    data = np.load(data_file)
    X = torch.FloatTensor(data["X"])
    y = torch.LongTensor(data["Y"])
    print(f"Evaluating on {len(X)} windows from unseen subjects ({os.path.basename(data_file)})")

    model = load_model(model_file, device)
    print(f"Model loaded: {os.path.basename(model_file)}")

    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_FILE, help="checkpoint to evaluate")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="test-set .npz (X, Y)")
    args = parser.parse_args()

    labels, preds, probs = evaluate(args.model, args.data)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)
    present = sorted(set(labels) | set(preds))
    print(classification_report(labels, preds, labels=present,
                                target_names=[CLASS_NAMES[i] for i in present], digits=4))

    print("-" * 60)
    print("AUC SCORES (one-vs-rest):")
    try:
        auc_scores = roc_auc_score(labels, probs, multi_class="ovr", average=None)
        for i, score in enumerate(auc_scores):
            print(f"  {CLASS_NAMES[i]:<20} : {score:.4f}")
    except ValueError as e:
        print(f"  Could not compute AUC: {e}")

    print("-" * 60)
    print("CONFUSION MATRIX (row = true, col = predicted):")
    cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
    print(pd.DataFrame(cm, index=range(len(CLASSES)), columns=range(len(CLASSES))))

    print("=" * 60)
    print(f"Overall accuracy: {100 * (labels == preds).mean():.2f}%")
    print(" - Precision: when the model predicts 'FALL', how often is it right?")
    print(" - Recall:    when a fall actually happens, how often is it caught?")
    print(" - AUC:       1.0 is perfect, 0.5 is random guessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
