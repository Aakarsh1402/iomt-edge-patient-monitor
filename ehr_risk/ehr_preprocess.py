import os
import json
import pandas as pd
import numpy as np
import zipfile
import shutil
import gc
from datetime import datetime
from ehr_utils import optimize_dtypes, CHUNK_SIZE

# --- PATHS ---
ZIP_FILE = 'fhir.zip'
RAW_DIR = 'fhir_raw'
TEMP_PARQUET = 'temp_parquet'
FINAL_PARQUET = 'final_parquet_chunks'

# --- CLEANUP & SETUP ---
def setup_directories():
    print("--- Cleaning directories... ---")
    for d in [RAW_DIR, TEMP_PARQUET, FINAL_PARQUET]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

def extract_zip():
    print(f"--- Extracting {ZIP_FILE}... ---")
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(f"{ZIP_FILE} not found!")
    
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    
    # Consolidate JSONs to root
    json_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith('.json'):
                # Move if nested, or just record path
                full_path = os.path.join(root, file)
                # If it's already in RAW_DIR, just add it, otherwise move it
                if root != RAW_DIR:
                    dest = os.path.join(RAW_DIR, file)
                    shutil.move(full_path, dest)
                    json_files.append(dest)
                else:
                    json_files.append(full_path)
    return json_files

# --- PARSING LOGIC ---
def parse_fhir_bundle_24h(file_path):
    try:
        if os.path.getsize(file_path) == 0: return None
        with open(file_path, 'r') as f: bundle = json.load(f)
        if 'entry' not in bundle: return None

        patient_id = os.path.basename(file_path).split('.')[0]
        patient_features = {'patient_id': patient_id}
        observations = []
        conditions = set()

        for entry in bundle.get('entry', []):
            resource = entry.get('resource', {})
            r_type = resource.get('resourceType')

            if r_type == 'Patient':
                patient_features['gender'] = resource.get('gender')
                birth = resource.get('birthDate')
                if birth:
                    try:
                        patient_features['age'] = (datetime(2025, 1, 1) - datetime.strptime(birth, '%Y-%m-%d')).days // 365
                    except: pass
            elif r_type == 'Condition':
                conditions.add(resource.get('code', {}).get('text', '').lower())
            elif r_type == 'Observation':
                obs_time = resource.get('effectiveDateTime')
                if obs_time:
                    try:
                        observations.append((datetime.fromisoformat(obs_time.replace('Z', '+00:00')), resource))
                    except: pass

        if not observations: return None
        observations.sort(key=lambda x: x[0])

        # MAPPING
        obs_map = {
             '8480-6':'sbp', '8462-4':'dbp', '8867-4':'heart_rate', '9279-1':'respiratory_rate',
             '59408-5':'spo2', '8310-5':'temperature', '6690-2':'wbc_count', '718-7':'hemoglobin',
             '4544-3':'hematocrit', '777-3':'platelet_count', '2093-3':'cholesterol_total',
             '2571-8':'triglycerides', '18262-6':'cholesterol_ldl', '2085-9':'cholesterol_hdl',
             '72514-3':'pain_score', '48643-1':'hba1c', '55758-7':'phq2_score', '70274-6':'gad7_score',
             '8302-2':'height_m', '29463-7':'weight_kg', '3094-0':'bun', '2951-2':'sodium',
             '2823-3':'potassium', '2345-7':'glucose', '1975-2':'bilirubin', '2160-0':'creatinine',
             # New Labs
             '10839-9': 'troponin_i', '1988-5': 'crp', '2524-7': 'lactate', '33959-8': 'procalcitonin', 
             '48065-7': 'd_dimer', '5821-4': 'urine_wbc', '5811-5': 'urine_nitrite', 
             '5799-2': 'urine_leukocyte', '14563-1': 'stool_occult_blood', '1751-7': 'albumin', 
             '1742-6': 'alt', '1920-8': 'ast', '2991-8': 'tsh', '3005-4': 'free_t4', 
             '14338-8': 'lipase', '1798-8': 'amylase', '34714-6': 'inr', '3173-2': 'ptt', 
             '2601-5': 'magnesium', '2777-1': 'phosphate', '11568-8': 'rheumatoid_factor'
        }

        patient_rows = []
        for obs_time, obs in observations:
            row = patient_features.copy()
            row['obs_time'] = obs_time
            
            # Extract Value
            val = None
            code = None
            if 'component' in obs:
                 for comp in obs['component']:
                    code = comp.get('code', {}).get('coding', [{}])[0].get('code')
                    if code in obs_map: row[obs_map[code]] = comp.get('valueQuantity', {}).get('value')
            else:
                code = obs.get('code', {}).get('coding', [{}])[0].get('code')
                if code in obs_map: row[obs_map[code]] = obs.get('valueQuantity', {}).get('value')

            # History Flags (Simplified for speed/safety)
            conditions_text = " ".join(conditions) # Join once for faster search
            
            # Helper to check keywords
            def check(keywords):
                if isinstance(keywords, str): return 1 if keywords in conditions_text else 0
                return 1 if any(k in conditions_text for k in keywords) else 0

            row['history_hypertension'] = check('hypertension')
            row['history_diabetes'] = check('diabetes')
            row['history_chf'] = check('congestive heart failure')
            row['history_copd'] = check('copd')
            row['history_anemia'] = check('anemia')
            row['history_hyperlipidemia'] = check('hyperlipidemia')
            row['history_cad'] = check('coronary artery disease')
            row['history_ckd'] = check('chronic kidney disease')
            row['history_obesity'] = check('obesity')
            row['history_asthma'] = check('asthma')
            row['history_pneumonia'] = check('pneumonia')
            row['history_flu'] = check('influenza')
            # New History
            row['history_afib'] = check('atrial fibrillation')
            row['history_dvt_pe'] = check(['dvt', 'deep vein thrombosis', 'pulmonary embolism'])
            row['history_thyroid'] = check(['hypothyroidism', 'hyperthyroidism', 'thyroid disease'])
            row['history_gout'] = check('gout')
            row['history_arthritis'] = check(['osteoarthritis', 'rheumatoid arthritis'])
            row['history_pancreatitis'] = check('pancreatitis')
            row['history_gi_bleed'] = check(['gi bleed', 'gastrointestinal bleed', 'varices'])
            row['history_ibd'] = check(['crohn', 'ulcerative colitis'])
            row['history_cellulitis'] = check('cellulitis')
            row['history_uti'] = check('urinary tract infection')
            row['history_seizure'] = check(['epilepsy', 'seizure'])
            row['history_dementia'] = check(['dementia', 'alzheimer'])

            # Add Medication checks here if needed, skipping for brevity as main crash source is overhead
            
            patient_rows.append(row)
        return patient_rows
    except Exception:
        return None

def process_single_file(file_path):
    data = parse_fhir_bundle_24h(file_path)
    if data:
        df = pd.DataFrame(data)
        # Pre-optimize before saving to temp
        df = optimize_dtypes(df)
        save_path = os.path.join(TEMP_PARQUET, os.path.basename(file_path).replace('.json', '.parquet'))
        df.to_parquet(save_path, compression='snappy', index=False)
        return True
    return False

def feature_engineering_chunked():
    print("\n--- Starting Feature Engineering & Consolidation ---")
    files = [os.path.join(TEMP_PARQUET, f) for f in os.listdir(TEMP_PARQUET)]
    
    total_files = len(files)
    if total_files == 0:
        print("No files to process!")
        return

    for i in range(0, total_files, CHUNK_SIZE):
        chunk_files = files[i:i+CHUNK_SIZE]
        print(f"Processing final chunk {i//CHUNK_SIZE + 1}...")
        
        df_chunk = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        
        # 1. Feature Engineering (Timestamps & Sort)
        df_chunk['obs_time'] = pd.to_datetime(df_chunk['obs_time'], utc=True)
        df_chunk = df_chunk.sort_values(['patient_id', 'obs_time'])
        
        # 2. Risk Labeling (FULL 51 CONDITIONS)
        conditions = {
            'sepsis': ((df_chunk.get('heart_rate', 0) > 90) & (df_chunk.get('wbc_count', 0) > 12)),
            'mi': ((df_chunk.get('troponin_i', 0) > 0.04) | ((df_chunk.get('history_cad', 0) == 1) & (df_chunk.get('pain_score', 0) > 5))),
            'stroke': ((df_chunk.get('history_afib', 0) == 1) & (df_chunk.get('age', 0) > 70)),
            'heart_failure': ((df_chunk.get('history_chf', 0) == 1) & (df_chunk.get('respiratory_rate', 0) > 20)),
            'pneumonia': ((df_chunk.get('respiratory_rate', 0) > 20) & (df_chunk.get('wbc_count', 0) > 12)),
            'hypertension_crisis': ((df_chunk.get('sbp', 0) > 180) | (df_chunk.get('dbp', 0) > 110)),
            'hypotension': (df_chunk.get('sbp', 999) < 90),
            'tachycardia': (df_chunk.get('heart_rate', 0) > 100),
            'bradycardia': (df_chunk.get('heart_rate', 999) < 60),
            'diabetes_uncontrolled': (df_chunk.get('glucose', 0) > 200),
            'hypoglycemia': (df_chunk.get('glucose', 999) < 70),
            'anemia': (df_chunk.get('hemoglobin', 999) < 10),
            'renal_failure': (df_chunk.get('creatinine', 0) > 2.0),
            'hyperkalemia': (df_chunk.get('potassium', 0) > 5.0),
            'hypokalemia': (df_chunk.get('potassium', 999) < 3.5),
            'hypernatremia': (df_chunk.get('sodium', 0) > 145),
            'hyponatremia': (df_chunk.get('sodium', 999) < 135),
            'liver_disease': ((df_chunk.get('alt', 0) > 40) | (df_chunk.get('ast', 0) > 40)),
            'pancreatitis': ((df_chunk.get('lipase', 0) > 100) | (df_chunk.get('amylase', 0) > 100)),
            'uti': ((df_chunk.get('urine_wbc', 0) > 10) | (df_chunk.get('urine_leukocyte', 0) == 1)),
            'dvt_pe': (df_chunk.get('d_dimer', 0) > 500),
            'gout': (df_chunk.get('history_gout', 0) == 1),
            'thyroid_storm': ((df_chunk.get('tsh', 999) < 0.1) & (df_chunk.get('heart_rate', 0) > 100)),
            'hypothyroidism': (df_chunk.get('tsh', 0) > 4.5),
            'gi_bleed': ((df_chunk.get('stool_occult_blood', 0) == 1) | (df_chunk.get('hemoglobin', 999) < 7)),
            'ibd_flare': ((df_chunk.get('history_ibd', 0) == 1) & (df_chunk.get('crp', 0) > 10)),
            'acidosis': (df_chunk.get('lactate', 0) > 2.0),
            'coagulopathy': ((df_chunk.get('inr', 0) > 1.5) | (df_chunk.get('platelet_count', 999) < 100)),
            'obesity': (df_chunk.get('history_obesity', 0) == 1),
            'depression': (df_chunk.get('phq2_score', 0) > 2),
            'anxiety': (df_chunk.get('gad7_score', 0) > 9),
            'copd_exacerbation': ((df_chunk.get('history_copd', 0) == 1) & (df_chunk.get('spo2', 100) < 90)),
            'asthma_attack': ((df_chunk.get('history_asthma', 0) == 1) & (df_chunk.get('respiratory_rate', 0) > 25)),
            'covid_19': ((df_chunk.get('temperature', 0) > 38) & (df_chunk.get('history_flu', 0) == 1)),
            'dehydration': ((df_chunk.get('bun', 0) > 20) & (df_chunk.get('sodium', 0) > 145)),
            'malnutrition': (df_chunk.get('albumin', 999) < 3.5),
            'rhabdomyolysis': ((df_chunk.get('history_seizure', 0) == 1) & (df_chunk.get('creatinine', 0) > 1.5)),
            'shock': (df_chunk.get('sbp', 999) < 90),
            'seizure_risk': (df_chunk.get('history_seizure', 0) == 1),
            'dementia_risk': (df_chunk.get('history_dementia', 0) == 1),
            'drug_toxicity': (df_chunk.get('alt', 0) > 100),
            'alcohol_withdrawal': ((df_chunk.get('heart_rate', 0) > 100) & (df_chunk.get('history_seizure', 0) == 1)),
            'hyperlipidemia': (df_chunk.get('cholesterol_total', 0) > 200),
            'arthritis_flare': ((df_chunk.get('history_arthritis', 0) == 1) & (df_chunk.get('pain_score', 0) > 5)),
            'cellulitis': ((df_chunk.get('history_cellulitis', 0) == 1) & (df_chunk.get('wbc_count', 0) > 10)),
            'osteoporosis': ((df_chunk.get('age', 0) > 65) & (df_chunk.get('gender', 0) == 0)), 
            'sleep_apnea': ((df_chunk.get('history_obesity', 0) == 1) & (df_chunk.get('history_hypertension', 0) == 1)),
            'post_op_complication': (df_chunk.get('heart_rate', 0) > 110), 
            'medication_nonadherence': ((df_chunk.get('history_hypertension', 0) == 1) & (df_chunk.get('sbp', 0) > 160)),
            'frailty': ((df_chunk.get('age', 0) > 80) & (df_chunk.get('albumin', 999) < 3.0)),
            'fall_risk': ((df_chunk.get('age', 0) > 65) & (df_chunk.get('history_seizure', 0) == 1))
        }

        # Apply rules
        for name, rule in conditions.items():
            # --- CRITICAL FIX: Handle scalar booleans vs Series ---
            if isinstance(rule, bool):
                # If data is missing, rule is just False (scalar)
                df_chunk[f'risk_{name}_24h'] = int(rule)
            else:
                # If data exists, rule is a Series
                df_chunk[f'risk_{name}_24h'] = rule.astype('int8')

        # 3. Final Optimization and Save
        df_chunk = df_chunk.fillna(0)
        df_chunk = optimize_dtypes(df_chunk)
        
        save_path = os.path.join(FINAL_PARQUET, f"chunk_{i}.parquet")
        df_chunk.to_parquet(save_path, index=False)
        
        del df_chunk
        gc.collect()

if __name__ == "__main__":
    setup_directories()
    json_files = extract_zip()
    
    print("--- Parsing JSONs (Sequential Mode to save RAM)... ---")
    
    # --- CHANGED: NO MULTIPROCESSING ---
    count = 0
    total = len(json_files)
    for f in json_files:
        if process_single_file(f):
            count += 1
        if count % 100 == 0:
            print(f"Parsed {count}/{total}...", end='\r')
            gc.collect() # Aggressive Garbage Collection
            
    print(f"\nSuccessfully parsed {count} files.")
        
    feature_engineering_chunked()
    
    # Cleanup raw files to free space
    try:
        shutil.rmtree(RAW_DIR)
        shutil.rmtree(TEMP_PARQUET)
    except: pass
    print("✅ Preprocessing Complete.")