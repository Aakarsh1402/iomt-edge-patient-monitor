import os
import numpy as np
import pandas as pd

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
            except: continue

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

# --- IMPROVED PHYSICS (More Randomness) ---
def f_lying(a): 
    a[:,2]=1.0
    a+=np.random.normal(0, np.random.uniform(0.01, 0.05), a.shape) # Variable noise
def f_sit(a): 
    a[:,1]=1.0
    a+=np.random.normal(0, np.random.uniform(0.02, 0.06), a.shape)
def f_seiz(a): 
    a[:,2]=1.0
    # Random frequency noise (some seizures are faster/slower)
    a+=np.random.normal(0, np.random.uniform(0.3, 0.6), a.shape)
def f_slump(a): 
    angle = np.random.uniform(0.6, 0.8) # Variable slump angle
    a[:,1]=angle; a[:,2]=angle
    a+=np.random.normal(0, 0.02, a.shape)
def f_agit(a):
    freq = np.random.uniform(0.5, 2.0) # Variable rolling speed
    t=np.linspace(0, freq*np.pi, 75)
    roll=np.sin(t)*np.random.uniform(0.4, 0.8)
    a[:,2]=1.0-np.abs(roll); a[:,0]=roll
    a+=np.random.normal(0, 0.15, a.shape)
def f_choke(a):
    a[:,1]=0.9
    interval = np.random.randint(15, 30) # Random cough spacing
    for k in range(5, 70, interval): 
        force = np.random.uniform(1.2, 1.8)
        a[k:k+3,2]+=force; a[k:k+3,1]-=0.5
    a+=np.random.normal(0, 0.05, a.shape)
def f_vomit(a):
    a[:,1]=0.7; a[:,2]=0.7
    freq = np.random.uniform(2.5, 3.5) # Variable heave speed
    t=np.linspace(0, freq*np.pi, 75)
    h=np.sin(t)*np.random.uniform(0.6, 0.9)
    a[:,2]+=h; a[:,1]-=h*0.3
    a+=np.random.normal(0, 0.1, a.shape)
def f_cpr(a):
    a[:,2]=1.0
    freq = np.random.uniform(1.5, 2.2) # Variable CPR speed (90-130 BPM)
    t=np.linspace(0, 1.5, 75)
    c=np.abs(np.sin(2*np.pi*freq*t))*np.random.uniform(0.7, 1.0)
    a[:,2]+=c
    a+=np.random.normal(0, 0.05, a.shape)
def f_resp(a):
    a[:,1]=0.8
    freq = np.random.uniform(0.5, 1.2) # Variable breathing rate
    t=np.linspace(0, 1.5, 75)
    b=np.sin(2*np.pi*freq*t)*np.random.uniform(0.1, 0.2)
    a[:,2]+=b
    a+=np.random.normal(0, 0.02, a.shape)
def f_trans(a): 
    a[:,2]=1.0
    # Variable floor texture (rumble frequency)
    a+=np.random.normal(0, np.random.uniform(0.05, 0.12), a.shape)

funcs = [f_lying, f_sit, None, None, f_seiz, f_slump, f_agit, f_choke, f_vomit, f_cpr, f_resp, f_trans]

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