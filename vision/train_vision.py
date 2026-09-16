import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless-safe: we only write PNGs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_model import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, build_model, save_class_names

# --- CONFIGURATION ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Ward_Vision_Data")
# run_vision.py loads this name by default, so training and inference agree.
MODEL_SAVE_PATH = os.path.join(HERE, "vision_model.pth")
PLOT_SAVE_PATH = os.path.join(HERE, "vision_metrics_improved.png")
BATCH_SIZE = 16 
EPOCHS = 25        # Increased epochs for Fine-Tuning
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. AGGRESSIVE DATA AUGMENTATION ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), # Increased rotation
        # Add Shear/Scale to simulate weird CCTV angles
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
}

if not os.path.isdir(DATA_DIR):
    sys.exit(f"Dataset not found at {DATA_DIR}.\n"
             "Expected Ward_Vision_Data/{train,val}/{Normal,Distress,Danger}/ with images.")

print("Loading Images...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
NUM_CLASSES = len(class_names)
print(f"Classes: {class_names}  ({dataset_sizes['train']} train / {dataset_sizes['val']} val images)")

# --- 2. SETUP MODEL ---
print("Downloading MobileNetV3-Small...")
model = build_model(NUM_CLASSES, pretrained=True)

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
# Record the class order so inference never has to guess it.
save_class_names(MODEL_SAVE_PATH, class_names)
print(f"✅ SUCCESS: Model saved as '{MODEL_SAVE_PATH}' (classes: {class_names})")

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