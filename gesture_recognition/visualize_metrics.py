import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

# --- CONFIGURATION ---
MODEL_FILE = "ward_model_strict.pth"
DATA_FILE = "ward_data_test.npz" # We evaluate ONLY on the unseen test subjects
BATCH_SIZE = 64
device = torch.device("cpu")

CLASSES = [
    "0: Lying", "1: Sitting", "2: Walking", 
    "3: FALL", "4: SEIZURE", "5: Slump", 
    "6: Agitation", "7: Choking", "8: Vomit", 
    "9: CPR", "10: Resp.Distress", "11: Transport"
]

# --- MODEL DEFINITION (Must match Strict Training Script) ---
class WardGuardianCNN(nn.Module):
    def __init__(self):
        super(WardGuardianCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 12)
        )
    def forward(self, x): return self.classifier(self.cnn(x))

# --- LOAD DATA ---
print(f"Loading Test Data from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Did you run 'process_ward_data_strict.py'?")
    exit()

data = np.load(DATA_FILE)
X = torch.FloatTensor(data['X'])
y = torch.LongTensor(data['Y'])

print(f"Evaluating on {len(X)} samples from UNSEEN subjects.")

# No random split needed - this entire file is the test set
test_loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)

# --- LOAD MODEL ---
model = WardGuardianCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print("Model Loaded.")
except:
    print(f"Error: Could not find '{MODEL_FILE}'")
    exit()

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted By AI')
plt.ylabel('Actual Reality (Test Subjects)')
plt.title('Strict Evaluation: Unseen Patients')
plt.tight_layout()
plt.savefig('viz_confusion_matrix_strict.png')
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
plt.savefig('viz_accuracy_chart_strict.png')
print("Saved 'viz_accuracy_chart_strict.png'")

print("\nDONE. Check the PNG files.")