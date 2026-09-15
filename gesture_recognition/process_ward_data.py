import os
import numpy as np
import pandas as pd

from synthetic_signals import TRAINING_GENERATORS

# CONFIG
DATASET_PATH = "SisFall_dataset/"
OUTPUT_TRAIN = "ward_data_train.npz"
OUTPUT_TEST = "ward_data_test.npz"
TARGET_HZ = 50; ORIGINAL_HZ = 200; SKIP = ORIGINAL_HZ // TARGET_HZ
WINDOW_SIZE = 75 

# LIST OF TEST SUBJECTS (The "New Patients")
# SisFall has subjects SA01 to SA38 (approx). We reserve the last few.
TEST_SUBJECTS = ['SA23', 'SA24', 'SA25', 'SA26', 'SA27', 'SA28', 'SA29', 'SA30']

train_x, train_y = [], []
test_x, test_y = [], []

print("PHASE 1: Processing Real Data (Subject-Aware Split)...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".txt") and (file.startswith("F") or file.startswith("D")):
            try:
                parts = file.split("_") # ['D01', 'SA01', 'R01.txt']
                code = parts[0]
                subject = parts[1] # 'SA01'
                
                label = -1
                if code in ["D01","D02","D03","D04"]: label = 2 # Walking
                elif code.startswith("F"): label = 3 # Fall
                
                if label != -1:
                    df = pd.read_csv(os.path.join(root, file), header=None, usecols=[0,1,2])
                    data = (df.values / 254.0)[::SKIP]
                    
                    # Decide: Is this a Training Subject or Test Subject?
                    is_test_subject = subject in TEST_SUBJECTS
                    
                    for i in range(0, len(data) - WINDOW_SIZE, 35):
                        window = data[i : i + WINDOW_SIZE]
                        if len(window) == WINDOW_SIZE:
                            if is_test_subject:
                                test_x.append(window)
                                test_y.append(label)
                            else:
                                train_x.append(window)
                                train_y.append(label)
            except (ValueError, IndexError, OSError) as e:
                print(f"  Skipping {file}: {e}")
                continue

print(f"  Real Data Split -> Train: {len(train_x)} | Test: {len(test_x)}")

print("PHASE 2: Generating Harder Synthetic Data...")
# We add more variety to synthetic data to prevent memorization
NUM_TRAIN = 600
NUM_TEST = 150 # Smaller test set for synthetic

def generate(class_id, fn, count, dest_x, dest_y):
    for _ in range(count):
        base = np.zeros((WINDOW_SIZE, 3))
        fn(base)
        dest_x.append(base)
        dest_y.append(class_id)

funcs = TRAINING_GENERATORS  # physics generators live in synthetic_signals.py

print("  Generating Train & Test sets...")
for i, fn in enumerate(funcs):
    if fn: # Skip Real classes (2 and 3)
        generate(i, fn, NUM_TRAIN, train_x, train_y)
        generate(i, fn, NUM_TEST, test_x, test_y)

# Save Separate Files
def save(name, x, y):
    X = np.array(x).transpose(0, 2, 1)
    Y = np.array(y)
    np.savez(name, X=X, Y=Y)

save(OUTPUT_TRAIN, train_x, train_y)
save(OUTPUT_TEST, test_x, test_y)
print(f"SUCCESS: Saved '{OUTPUT_TRAIN}' and '{OUTPUT_TEST}'")