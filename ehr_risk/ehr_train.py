import os
# --- FORCE CPU MODE (Fixes "Searching for GPU" crashes) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import joblib
import json
import math
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import tools
# Note: Ensure ehr_utils.py still has BATCH_SIZE = 4 or 16. 
# We override it here for safety anyway.
from ehr_utils import data_generator, encode_data, EPOCHS
from ehr_model import create_multi_task_model

# --- CPU SAFETY SETTINGS ---
SAFE_BATCH_SIZE = 16  # CPU can handle slightly larger batches than GPU
print(f"--- Running in CPU Mode (Safe & Stable) ---")

# 1. Load Data
DATA_DIR = 'final_parquet_chunks'
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
files.sort()

# 2. Inspect Columns
try:
    sample = pd.read_parquet(files[0])
    sample = encode_data(sample) 
    targets = [c for c in sample.columns if c.startswith('risk_')]
    features = [c for c in sample.columns if c not in targets and c not in ['patient_id', 'obs_time', 'obs_time_utc']]
    print(f"Features: {len(features)}, Targets: {len(targets)}")
    del sample
except Exception as e:
    print(f"Error reading sample file: {e}")
    exit()

# 3. Fit Scaler
print("--- Fitting Scaler ---")
scaler = StandardScaler()
for f in files:
    df = pd.read_parquet(f, columns=features)
    df = encode_data(df)
    scaler.partial_fit(df)
joblib.dump(scaler, 'ehr_scaler.joblib')

with open('ehr_features.json', 'w') as f: json.dump(features, f)
with open('ehr_targets.json', 'w') as f: json.dump(targets, f)

# 4. Train/Val Split
train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

# 5. Build Model
model = create_multi_task_model(len(features), targets)

# 6. Training
train_rows = sum([pq.ParquetFile(f).metadata.num_rows for f in train_files])
val_rows = sum([pq.ParquetFile(f).metadata.num_rows for f in val_files])
steps_per_epoch = math.ceil(train_rows / SAFE_BATCH_SIZE)
val_steps = math.ceil(val_rows / SAFE_BATCH_SIZE)

train_gen = data_generator(train_files, scaler, features, targets, SAFE_BATCH_SIZE)
val_gen = data_generator(val_files, scaler, features, targets, SAFE_BATCH_SIZE)

print(f"--- Starting Training (Epochs: {EPOCHS}) ---")
print("Note: This will utilize your CPU (10-30% usage). This is normal.")
print("The screen will update once per epoch.")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,  # 2 = One line per epoch (Prevents console freezing)
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# 7. Save and Convert
model.save('ehr_model.keras')
print("✅ Keras model saved.")

print("--- Converting to TFLite ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model = converter.convert()

with open('ehr_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model saved. Ready for Jetson Nano/RPi.")