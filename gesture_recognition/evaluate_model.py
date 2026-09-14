import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

# --- CONFIGURATION ---
BATCH_SIZE = 64
MODEL_FILE = "ward_model.pth"
DATA_FILE = "ward_data.npz"
device = torch.device("cpu")

# Class Names for readable reports
CLASS_NAMES = [
    "0: Lying Down",
    "1: Sitting Up",
    "2: Walking",
    "3: FALL / IMPACT",
    "4: SEIZURE",
    "5: Slump/Pain",
    "6: Agitation",
    "7: CHOKING",
    "8: VOMITING",
    "9: CPR IN PROGRESS",
    "10: RESP. DISTRESS",
    "11: Transport"
]

# --- 1. DEFINE THE MODEL ARCHITECTURE (Must Match Training Exactly) ---
class WardGuardianCNN(nn.Module):
    def __init__(self):
        super(WardGuardianCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 12)
        )

    def forward(self, x):
        return self.classifier(self.cnn(x))

# --- 2. LOAD DATA & MODEL ---
print("Loading Data & Model...")
data = np.load(DATA_FILE)
X = torch.FloatTensor(data['X'])
y = torch.LongTensor(data['Y'])

# Re-create the same split used in training to ensure we test on UNSEEN data
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
# We use a fixed seed here to try and get the same split logic if possible, 
# but for a quick test, using the random_split again gives a statistically valid evaluation.
# ideally we would have saved the test indices, but this works for validation.
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = WardGuardianCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. RUN INFERENCE ---
print(f"Evaluating on {len(test_dataset)} test samples...")

all_preds = []
all_labels = []
all_probs = [] # For AUC

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Get Probabilities (Softmax) for AUC
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get Predictions (Hard Class)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# --- 4. CALCULATE METRICS ---
print("\n" + "="*60)
print("FINAL EVALUATION REPORT")
print("="*60)

# A. Classification Report (Precision, Recall, F1)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
print(report)

# B. AUC Scores (One-vs-Rest)
print("-" * 60)
print("AUC SCORES (Ability to distinguish this class from others):")
try:
    auc_scores = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
    for i, score in enumerate(auc_scores):
        print(f"  {CLASS_NAMES[i]:<20} : {score:.4f}")
except ValueError:
    print("Error calculating AUC. (Might require more samples per class in test set)")

# C. Confusion Matrix
print("-" * 60)
print("CONFUSION MATRIX (Row = True, Col = Predicted):")
cm = confusion_matrix(all_labels, all_preds)
# Print using Pandas for better readability
df_cm = pd.DataFrame(cm, index=[i for i in range(12)], columns=[i for i in range(12)])
print(df_cm)

print("="*60)
print("INTERPRETATION GUIDE:")
print(" - Precision: When model predicts 'Fall', how often is it right?")
print(" - Recall:    When a 'Fall' actually happens, how often does model catch it?")
print(" - AUC:       1.0 is perfect, 0.5 is random guessing.")
print("="*60)