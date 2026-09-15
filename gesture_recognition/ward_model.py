"""Shared definition of the IMU motion classifier (WardGuardianCNN).

Every script in this folder used to carry its own copy of the network; they
drifted apart (different dropout rates, different checkpoint names, different
class labels). This module is now the single source of truth, so a checkpoint
trained by train_ward_model.py is guaranteed to load in the evaluation, sanity
check and fusion demo code.
"""
import os

import numpy as np
import torch
import torch.nn as nn

# 1.5 s windows of 3-axis accelerometer data at 50 Hz.
WINDOW_SIZE = 75
NUM_CHANNELS = 3

# Checkpoint produced by train_ward_model.py (subject-aware "strict" split).
DEFAULT_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ward_model_strict.pth")

CLASSES = [
    "Lying", "Sitting", "Walking", "FALL", "SEIZURE", "Slump",
    "Agitation", "Choking", "Vomiting", "CPR", "Resp.Distress", "Transport",
]
NUM_CLASSES = len(CLASSES)


class WardGuardianCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.cnn(x))


def load_model(model_file=DEFAULT_MODEL_FILE, device="cpu"):
    """Load a trained checkpoint in eval mode. Raises FileNotFoundError if missing."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Checkpoint '{model_file}' not found. Run train_ward_model.py or pass --model."
        )
    model = WardGuardianCNN().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model


def window_to_tensor(window):
    """(75, 3) or (3, 75) array -> (1, 3, 75) float tensor."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.shape == (WINDOW_SIZE, NUM_CHANNELS):
        arr = arr.T
    if arr.shape != (NUM_CHANNELS, WINDOW_SIZE):
        raise ValueError(f"Expected a ({WINDOW_SIZE}, {NUM_CHANNELS}) window, got {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


@torch.no_grad()
def predict_window(model, window):
    """Classify one window. Returns (class_name, confidence, probabilities[12])."""
    probs = torch.softmax(model(window_to_tensor(window)), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs
