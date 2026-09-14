import time
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# --- FILES TO LOAD ---
MODEL_PATH = 'ehr_model.tflite'
SCALER_PATH = 'ehr_scaler.joblib'
FEATURES_PATH = 'ehr_features.json'
TARGETS_PATH = 'ehr_targets.json'

def load_artifacts():
    print("--- Loading Artifacts ---")
    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load Column Maps
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    with open(TARGETS_PATH, 'r') as f:
        targets = json.load(f)
        
    # Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Model Loaded. Input Shape: {input_details[0]['shape']}")
    return scaler, features, targets, interpreter, input_details, output_details

def encode_data(df):
    """Must match the training logic exactly."""
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'male': 1, 'female': 0}).fillna(0).astype('int8')
    return df

def generate_dummy_patient(features):
    """Creates a random patient dataframe matching the training columns."""
    data = {}
    for col in features:
        # Create random realistic values based on column name hints
        if 'age' in col: val = np.random.randint(20, 90)
        elif 'heart_rate' in col: val = np.random.randint(50, 120)
        elif 'sbp' in col: val = np.random.randint(90, 180)
        elif 'gender' in col: val = 'male' # Will be encoded
        elif 'history_' in col or 'risk_' in col: val = np.random.choice([0, 1])
        else: val = np.random.random() # Generic labs
        data[col] = [val]
    return pd.DataFrame(data)

def run_inference(interpreter, input_data, input_details, output_details, targets):
    # 1. Set Input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 2. Run Inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    # 3. Get Outputs
    results = {}
    # TFLite outputs might not be in the same order as your targets list
    # We map them by index if the model has multiple outputs
    for i, detail in enumerate(output_details):
        # The model outputs a list of arrays. 
        # We assume the order matches the training target list if compiled correctly.
        # Ideally, we verify tensor names, but TFLite cleans names.
        # For this test, we map strictly by index order.
        if i < len(targets):
            pred = interpreter.get_tensor(detail['index'])[0][0]
            results[targets[i]] = pred
            
    return results, (end_time - start_time) * 1000 # ms

if __name__ == "__main__":
    # 1. Setup
    scaler, features, targets, interpreter, input_details, output_details = load_artifacts()
    
    # 2. Create Dummy Data
    print("\n--- Generating Test Patient ---")
    raw_patient = generate_dummy_patient(features)
    print("Raw Patient Data (First 5 cols):")
    print(raw_patient.iloc[:, :5])
    
    # 3. Preprocess
    # Align columns -> Encode -> FillNA -> Scale -> Float32
    processed_df = raw_patient.reindex(columns=features).fillna(0)
    processed_df = encode_data(processed_df)
    input_data = scaler.transform(processed_df).astype('float32')
    
    # 4. Warmup Run (The first run is always slow due to initialization)
    print("\n--- Warming up Engine ---")
    _, _ = run_inference(interpreter, input_data, input_details, output_details, targets)
    
    # 5. Performance Test Loop
    print("\n--- Starting Performance Test (100 runs) ---")
    latencies = []
    predictions = {}
    
    for _ in range(100):
        preds, lat = run_inference(interpreter, input_data, input_details, output_details, targets)
        latencies.append(lat)
        predictions = preds # Save last result
        
    avg_lat = np.mean(latencies)
    print(f"✅ Average Inference Speed: {avg_lat:.2f} ms per patient")
    
    # 6. Show Prediction Results
    print("\n--- Top 5 Predicted Risks for Dummy Patient ---")
    # Sort by probability
    sorted_risks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    for name, prob in sorted_risks[:5]:
        print(f"{name}: {prob:.2%}")