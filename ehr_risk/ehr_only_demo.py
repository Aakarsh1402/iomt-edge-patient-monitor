import os
import json
import numpy as np
import tensorflow as tf
import joblib
import time
import warnings

# Suppress warnings for a clean demo
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# MODEL FILES
EHR_MODEL_PATH = "ehr_model.keras"
EHR_SCALER = "ehr_scaler.joblib"
EHR_FEATURES = "ehr_features.json"
EHR_TARGETS = "ehr_targets.json"

# DATA FILE (UPDATED NAME)
PATIENT_DB = "EHR_patient_records.json"

# STATE
current_patient_id = "101"
patient_ids = ["101", "102"]
raw_patient_data = {}
last_preds = None
RISK_SCORES = {}

print("\n" + "="*60)
print("🏥 EHR RISK ENGINE: STANDALONE DEMO")
print("="*60)

# ==============================================================================
# 2. LOAD AI BRAIN
# ==============================================================================
print("[INIT] Loading Keras Model & Scalers...")

try:
    # Load the Neural Network and Preprocessing tools
    ehr_model = tf.keras.models.load_model(EHR_MODEL_PATH)
    scaler = joblib.load(EHR_SCALER)
    
    # Load Feature Definitions (Inputs) and Target Definitions (Outputs)
    with open(EHR_FEATURES) as f: feature_list = json.load(f)
    with open(EHR_TARGETS) as f: target_list = json.load(f)
    
    # Load Patient Database
    if not os.path.exists(PATIENT_DB):
        raise FileNotFoundError(f"Could not find '{PATIENT_DB}'. Please create this JSON file.")
        
    with open(PATIENT_DB) as f: patient_db = json.load(f)
    
    print(f"      ✅ Model Loaded ({len(feature_list)} Inputs -> {len(target_list)} Outputs)")

except Exception as e:
    print(f"❌ ERROR: Missing files. {e}")
    print(f"Ensure ehr_model.keras, ehr_scaler.joblib, and {PATIENT_DB} are in this folder.")
    exit()

# ==============================================================================
# 3. CORE LOGIC: PATIENT ANALYSIS
# ==============================================================================
def analyze_patient(pid):
    global raw_patient_data, last_preds, RISK_SCORES
    
    print(f"\n🔄 LOADING PATIENT ID: {pid}...")
    try:
        raw_patient_data = patient_db[pid]
    except KeyError:
        print(f"❌ Error: Patient ID {pid} not found in {PATIENT_DB}")
        return
    
    # STEP A: VECTORIZATION (Convert JSON -> Numpy Array)
    input_vector = np.zeros((1, len(feature_list)))
    
    for i, feature in enumerate(feature_list):
        val = 0
        
        # 1. Demographics
        if feature == 'age': val = raw_patient_data['age']
        elif feature == 'gender': val = 1 if raw_patient_data['gender'] == 'Male' else 0
        
        # 2. Vitals & Labs (Direct Value Mapping)
        elif feature in raw_patient_data.get('vitals', {}): 
            val = raw_patient_data['vitals'][feature]
        elif feature in raw_patient_data.get('labs', {}): 
            val = raw_patient_data['labs'][feature]
            
        # 3. Medical History (Keyword Search)
        elif "history_" in feature:
            cond = feature.replace("history_", "").lower()
            if any(cond in h.lower() for h in raw_patient_data['history']): val = 1
            
        # 4. Medications (Keyword Search)
        elif "med_" in feature:
            clean_med = feature.replace("med_", "").lower()
            if any(clean_med in m.lower() for m in raw_patient_data['medication']): val = 1

        input_vector[0, i] = val

    # STEP B: NORMALIZATION (Scale values to 0-1 range)
    try: 
        input_vector = scaler.transform(input_vector)
    except: 
        pass # Skip if scaler shape mismatch (demo safety)

    # STEP C: INFERENCE (Run the Brain)
    start_t = time.time()
    last_preds = ehr_model.predict(input_vector, verbose=0)[0]
    latency = (time.time() - start_t) * 1000
    
    # STEP D: EXTRACT KEY RISKS
    def get_risk(name):
        try: return float(last_preds[target_list.index(name)])
        except: return 0.0

    RISK_SCORES['fall'] = get_risk("risk_fall_risk_24h")
    RISK_SCORES['seizure'] = get_risk("risk_seizure_risk_24h")
    RISK_SCORES['heart'] = get_risk("risk_mi_24h") # Myocardial Infarction
    RISK_SCORES['sepsis'] = get_risk("risk_sepsis_24h")

    # --- DISPLAY SUMMARY ---
    print(f"   👤 NAME: {raw_patient_data['name']}")
    print(f"   ⏱️ INFERENCE TIME: {latency:.1f} ms")
    print("-" * 40)
    print(f"   📉 FALL RISK:    {RISK_SCORES['fall']*100:.1f}%")
    print(f"   🧠 SEIZURE RISK: {RISK_SCORES['seizure']*100:.1f}%")
    print(f"   ❤️ CARDIAC RISK: {RISK_SCORES['heart']*100:.1f}%")
    print("-" * 40)

def show_deep_dive():
    print(f"\n==========================================")
    print(f"🧬 NEURAL NETWORK INTERNALS")
    print(f"==========================================")
    
    # 1. Show Inputs
    print(f"INPUT LAYER (Active Features):")
    print(f"   • Age: {raw_patient_data['age']}")
    print(f"   • History: {raw_patient_data['history']}")
    print(f"   • Meds: {raw_patient_data['medication']}")
    if 'vitals' in raw_patient_data:
        print(f"   • Vitals: BP {raw_patient_data['vitals'].get('sbp')}/{raw_patient_data['vitals'].get('dbp')}, HR {raw_patient_data['vitals'].get('heart_rate')}")

    # 2. Show Outputs
    print(f"\nOUTPUT LAYER (Top 5 Predicted Risks):")
    # Zip predictions with names, sort by probability
    risk_pairs = list(zip(target_list, last_preds))
    risk_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(5):
        name, prob = risk_pairs[i]
        clean_name = name.replace("risk_", "").replace("_24h", "").upper()
        bar = "█" * int(prob * 20)
        print(f"   {i+1}. {clean_name:<20} |{bar:<20}| {prob*100:.2f}%")
    print("==========================================\n")

# ==============================================================================
# 4. INTERFACE LOOP
# ==============================================================================
# Initial Load
analyze_patient(current_patient_id)

while True:
    print("\n[S]witch Patient | [D]eep Dive Details | [Q]uit")
    cmd = input("Command > ").lower()
    
    if cmd == 's':
        # Toggle ID
        new_idx = (patient_ids.index(current_patient_id) + 1) % len(patient_ids)
        current_patient_id = patient_ids[new_idx]
        analyze_patient(current_patient_id)
        
    elif cmd == 'd':
        show_deep_dive()
        
    elif cmd == 'q':
        break