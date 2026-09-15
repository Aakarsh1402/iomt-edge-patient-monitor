import os

import matplotlib
matplotlib.use("Agg")  # headless-safe: we only write PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from ward_model import CLASSES as CLASS_LABELS, DEFAULT_MODEL_FILE, load_model

# --- CONFIGURATION ---
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = DEFAULT_MODEL_FILE
DATA_FILE = os.path.join(HERE, "ward_data_test.npz")  # We evaluate ONLY on the unseen test subjects
BATCH_SIZE = 64
device = torch.device("cpu")

CLASSES = [f"{i}: {name}" for i, name in enumerate(CLASS_LABELS)]

# --- LOAD DATA ---
print(f"Loading Test Data from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Did you run 'process_ward_data.py'?")
    exit()

data = np.load(DATA_FILE)
X = torch.FloatTensor(data['X'])
y = torch.LongTensor(data['Y'])

print(f"Evaluating on {len(X)} samples from UNSEEN subjects.")

# No random split needed - this entire file is the test set
test_loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)

# --- LOAD MODEL ---
model = load_model(MODEL_FILE, device)
print("Model Loaded.")

# --- RUN INFERENCE ---
all_preds = []
all_labels = []

print("Running Inference...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- VISUALIZATION 1: CONFUSION MATRIX ---
print("Generating Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha='right')
plt.yticks(range(len(CLASSES)), CLASSES)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xlabel('Predicted By AI')
plt.ylabel('Actual Reality (Test Subjects)')
plt.title('Strict Evaluation: Unseen Patients')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'viz_confusion_matrix_strict.png'))
print("Saved 'viz_confusion_matrix_strict.png'")

# --- VISUALIZATION 2: PER-CLASS ACCURACY ---
print("Generating Accuracy Chart...")
# Calculate accuracy per class (diagonal / row sum)
# Handle division by zero if a class is missing in test set
with np.errstate(divide='ignore', invalid='ignore'):
    class_acc = cm.diagonal() / cm.sum(axis=1)
    class_acc = np.nan_to_num(class_acc) # Replace NaN with 0

plt.figure(figsize=(12, 6))
colors = ['green' if x > 0.90 else 'orange' if x > 0.8 else 'red' for x in class_acc]
bars = plt.bar(CLASSES, class_acc * 100, color=colors)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 110)
plt.ylabel('Accuracy (%)')
plt.title('Reliability on New Patients')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'viz_accuracy_chart_strict.png'))
print("Saved 'viz_accuracy_chart_strict.png'")

print("\nDONE. Check the PNG files.")