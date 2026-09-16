import pandas as pd
import numpy as np
import gc
import tensorflow as tf

# --- CONFIGURATION ---
BATCH_SIZE = 4  # <--- CRITICAL: Lowered from 16 to 4 to prevent freezing
EPOCHS = 10
CHUNK_SIZE = 1000

def optimize_dtypes(df):
    """Aggressively reduce memory footprint."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0 and df[col].max() <= 1:
            df[col] = df[col].astype('int8')
        elif df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype('int8')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
        else:
            df[col] = df[col].astype('int32')
    return df

def encode_data(df):
    """Converts string columns to numbers."""
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'male': 1, 'female': 0}).fillna(0).astype('int8')
    return df

def data_generator(file_list, scaler, feature_cols, target_cols, batch_size):
    """
    Yields batches of data. Now forces float32 and handles Keras 3 tuple requirements.
    """
    while True: # Infinite loop for Keras
        shuffled_files = np.random.permutation(file_list)

        for f in shuffled_files:
            try:
                # Read specific columns only to save RAM
                needed_cols = feature_cols + target_cols
                df = pd.read_parquet(f, columns=needed_cols)
                df = encode_data(df)
                df = df.sample(frac=1)
            except Exception as e:
                print(f"Warning: Error reading {f}: {e}. Skipping.")
                continue

            # Process in mini-batches
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]

                # Features
                X_pre_scale = batch_df.reindex(columns=feature_cols).fillna(0)
                X_scaled = scaler.transform(X_pre_scale).astype('float32')

                # Targets
                y_list = [batch_df[col].values.astype('float32') for col in target_cols]

                # Yield Tuple for Keras 3
                if len(y_list) == 1:
                    yield X_scaled, y_list[0]
                else:
                    yield X_scaled, tuple(y_list)

            # Aggressive Cleanup. (Do NOT call keras.backend.clear_session()
            # here: it resets Keras global state from inside model.fit().)
            del df
            del batch_df
            gc.collect()

def set_gpu_memory_growth():
    """Sets up Mixed Precision and Memory Growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU Memory Growth Enabled")
            
            # Enable Mixed Precision (Uses 50% less VRAM on RTX 3060)
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Mixed Precision Enabled (Float16)")
            
        except RuntimeError as e:
            print(e)