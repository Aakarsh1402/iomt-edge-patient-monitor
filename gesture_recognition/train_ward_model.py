import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_FILE = "ward_model_strict.pth"
DATA_TRAIN = "ward_data_train.npz"
DATA_TEST = "ward_data_test.npz"

device = torch.device("cpu")

# --- MODEL ---
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
print("Loading Data...")
d_train = np.load(DATA_TRAIN)
d_test = np.load(DATA_TEST)

X_train = torch.FloatTensor(d_train['X']).to(device)
y_train = torch.LongTensor(d_train['Y']).to(device)
X_test = torch.FloatTensor(d_test['X']).to(device)
y_test = torch.LongTensor(d_test['Y']).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# --- SETUP ---
model = WardGuardianCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# History Storage
history = {
    'train_acc': [],
    'val_acc': [],
    'val_auc': []
}

print(f"Starting Training ({EPOCHS} Epochs)...")

for epoch in range(EPOCHS):
    # 1. TRAINING LOOP
    model.train()
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    history['train_acc'].append(train_acc)
    
    # 2. VALIDATION LOOP (With AUC)
    model.eval()
    val_correct = 0
    val_total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # Get Probabilities for AUC
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get Accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    history['val_acc'].append(val_acc)
    
    # Calculate AUC (One-vs-Rest)
    try:
        # We must handle cases where a batch might miss a class, though rare with big datasets
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.5 # Fallback if calculation fails (e.g. only 1 class in batch)
        
    history['val_auc'].append(auc)
    
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Val AUC: {auc:.4f}")

# --- SAVE MODEL ---
torch.save(model.state_dict(), MODEL_FILE)
print("Model Saved.")

# --- PLOTTING ---
print("Generating Plots...")
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_acc'], label='Train Acc')
plt.plot(epochs_range, history['val_acc'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot 2: AUC Score
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['val_auc'], label='Validation AUC', color='green')
plt.title('Model AUC Score (Macro-Average)')
plt.xlabel('Epochs')
plt.ylabel('AUC Score (0-1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
print("Done! Open 'training_metrics.png' to see the visualization.")