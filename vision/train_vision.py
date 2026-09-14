import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

# --- CONFIGURATION ---
DATA_DIR = "Ward_Vision_Data"
MODEL_SAVE_PATH = "vision_model_improved.pth"
PLOT_SAVE_PATH = "vision_metrics_improved.png"
BATCH_SIZE = 16 
EPOCHS = 25        # Increased epochs for Fine-Tuning
LEARNING_RATE = 0.001
NUM_CLASSES = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. AGGRESSIVE DATA AUGMENTATION ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), # Increased rotation
        # Add Shear/Scale to simulate weird CCTV angles
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Loading Images...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# --- 2. SETUP MODEL ---
print("Downloading MobileNetV3-Small...")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# STRATEGY: Unfreeze the last block!
# 1. Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# 2. Unfreeze the Classifier Head
for param in model.classifier.parameters():
    param.requires_grad = True

# 3. Unfreeze the LAST Feature Block (The part that sees high-level shapes)
# MobileNetV3 features is a list. Let's unfreeze the last 3 layers.
for param in model.features[-3:].parameters():
    param.requires_grad = True

# Modify Head
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# --- 3. OPTIMIZER & SCHEDULER ---
criterion = nn.CrossEntropyLoss()

# Only optimize parameters that require gradients (The ones we unfroze)
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

# Decay LR by a factor of 0.1 every 7 epochs
# This "Cooling Down" helps the model settle into high accuracy
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- 4. TRAINING LOOP ---
print("\nSTARTING IMPROVED TRAINING...")
start_time = time.time()
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print("-" * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            exp_lr_scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        history[f'{phase}_loss'].append(epoch_loss)
        history[f'{phase}_acc'].append(epoch_acc.item())

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# --- 5. SAVE ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ SUCCESS: Improved Model saved as '{MODEL_SAVE_PATH}'")

# Plotting
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_acc'], label='Train Acc')
plt.plot(epochs_range, history['val_acc'], label='Val Acc')
plt.title('Accuracy (Improved)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_loss'], label='Train Loss')
plt.plot(epochs_range, history['val_loss'], label='Val Loss')
plt.title('Loss (Improved)')
plt.legend()
plt.savefig(PLOT_SAVE_PATH)