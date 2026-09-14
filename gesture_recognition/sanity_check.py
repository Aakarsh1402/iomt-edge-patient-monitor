import torch
import numpy as np
import torch.nn as nn

# --- CONFIGURATION ---
MODEL_FILE = "ward_model.pth"
CLASSES = [
    "Lying", "Sitting", "Walking", "FALL", "SEIZURE", "Slump",
    "Agitation", "Choking", "Vomiting", "CPR", "Resp.Distress", "Transport"
]

# --- MODEL DEFINITION (Must match training) ---
class WardGuardianCNN(nn.Module):
    def __init__(self):
        super(WardGuardianCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 12)
        )
    def forward(self, x): return self.classifier(self.cnn(x))

# --- LOAD MODEL ---
print("Loading Model for Physics Check...")
device = torch.device("cpu")
model = WardGuardianCNN()
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print("Model Loaded.\n")
except Exception as e:
    print(f"Error: {e}")
    exit()

def test_signal(name, signal_fn):
    # Generate 1.5s of data (75 samples)
    arr = np.zeros((75, 3))
    signal_fn(arr)
    
    # Convert to Tensor (1, 3, 75)
    inp = torch.FloatTensor(arr).unsqueeze(0).transpose(1, 2).to(device)
    
    # Inference
    with torch.no_grad():
        out = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    
    res = CLASSES[pred.item()]
    # Visual Output
    print(f"Input: {name:<20} -> AI Guess: {res:<15} (Conf: {conf.item()*100:.1f}%)")
    if res.upper() in name.upper() or (name=="Simulated IMPACT" and res=="FALL"):
        print("   ✅ PASS")
    else:
        print(f"   ❌ FAIL (Expected {name})")
    print("-" * 50)

# --- IMPROVED PHYSICS GENERATORS ---

def phys_walking(arr):
    # Walking is NOT a sine wave. It is a series of sharp heel strikes.
    # We add a sharp spike every ~0.5 seconds (Step)
    arr[:, 1] = 0.9 # Mostly upright
    for i in range(0, 75, 25): # 2 steps per second approx
        arr[i:i+5, 1] += 0.4  # Vertical impact
        arr[i:i+5, 2] += 0.2  # Forward momentum
    # Add sensor noise (walking is messy)
    arr += np.random.normal(0, 0.15, arr.shape)

def phys_cpr(arr):
    # CPR is purely DOWNWARD compression, not up-and-down oscillation.
    # Frequency ~1.8 Hz (110 BPM)
    arr[:, 2] = 1.0 # Lying on back
    t = np.linspace(0, 1.5, 75)
    # Use ABS(SIN) to simulate downward force only
    compressions = np.abs(np.sin(2 * np.pi * 1.8 * t)) * 0.8
    arr[:, 2] += compressions 
    arr += np.random.normal(0, 0.05, arr.shape)

def phys_seizure(arr):
    # High frequency chaos (>5Hz)
    arr[:, 2] = 1.0 # Lying
    # Violent random shaking
    arr += np.random.normal(0, 0.6, arr.shape) 

def phys_fall(arr):
    # One massive G-force spike, then silence
    arr[35:40, 1] = 3.5 # 3.5G Impact
    arr[35:40, 2] = 2.0
    # Post-fall noise is low (lying still)
    arr += np.random.normal(0, 0.05, arr.shape)

def phys_vomit(arr):
    # Slow, deep heaving (~0.5 Hz)
    arr[:, 1] = 0.7 # Leaning forward
    arr[:, 2] = 0.7 
    t = np.linspace(0, 1.5, 75)
    # Slow wave
    heave = np.sin(2 * np.pi * 0.5 * t) * 0.6
    arr[:, 2] += heave
    arr += np.random.normal(0, 0.1, arr.shape)

# --- RUN TESTS ---
print("-" * 50)
test_signal("Simulated WALKING", phys_walking)
test_signal("Simulated CPR", phys_cpr)
test_signal("Simulated SEIZURE", phys_seizure)
test_signal("Simulated IMPACT", phys_fall)
test_signal("Simulated VOMITING", phys_vomit)