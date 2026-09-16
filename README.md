# IoMT Edge Patient Monitor

**An autonomous, multimodal edge-AI system for continuous patient monitoring and triage.**

Hospital staff cannot watch every patient 24×7, and the automated systems that exist (bed alarms, fall detectors) generate so many false positives that nurses learn to ignore them. This project is an Internet of Medical Things (IoMT) prototype that tackles both problems: a wearable ESP32 edge node streams ECG and motion data over MQTT into a time-series pipeline, and three AI modalities — **ECG arrhythmia detection**, **EHR-based clinical risk prediction**, and **motion/vision-based event recognition** — are fused so that staff are alerted only when an anomaly is corroborated by more than one signal.

> Built by Team AJAS — Aakarsh Mishra, Joshua Koilpillai, Aayush Khatanhar, Saharsh Kallu (with data-collection help from Ronith and Lohith). Full write-up in [`docs/IoMT_Project_Report.pdf`](docs/IoMT_Project_Report.pdf); slides in [`docs/IoMT_Presentation.pptx`](docs/IoMT_Presentation.pptx).

---

## System architecture

```
 ┌──────────────── Edge node (ESP32) ────────────────┐
 │  AD8232 ECG ──┐                                   │
 │               ├─ 360 Hz sampling, DC-offset        │      MQTT (JSON / line protocol)
 │  MPU6050 IMU ─┘  removal, moving-avg filter,      │ ─────────────────────────────────▶  Mosquitto broker
 │                  on-device fall heuristics         │        test/ecg_raw  (360 samples/s)
 └───────────────────────────────────────────────────┘        test/sensors  (1 Hz dashboard)
                                                                          │
 ┌──────────── ESP32-CAM ────────────┐                                    ▼
 │  MJPEG stream over HTTP           │ ──▶ vision model            Telegraf ──▶ InfluxDB ──▶ Grafana dashboard
 └───────────────────────────────────┘                                    │
                                                                          ▼
                     ┌────────────────────────────────────────────────────────────────────┐
                     │  AI layer                                                          │
                     │   • Arrhythmia CNN   (ECG windows)         → medical risk          │
                     │   • EHR multi-task NN (patient history)    → clinical risk         │
                     │   • Motion CNN + Vision CNN (IMU, camera)  → physical event +      │
                     │                                              visual confirmation   │
                     │                                                                    │
                     │   Alert  ⇐  medical risk + physical event + visual confirmation    │
                     │             exceeds threshold                                      │
                     └────────────────────────────────────────────────────────────────────┘
```

**Hardware:** ESP32 dev board, AD8232 single-lead ECG front-end, MPU6050 6-axis IMU, ESP32-CAM (OV2640).
**Transport & storage:** MQTT (PubSubClient) → Telegraf → InfluxDB, visualised in Grafana.

---

## Repository layout

| Path | What it is |
|---|---|
| [`firmware/ecg_imu_node/`](firmware/ecg_imu_node/) | ESP32 sketch: samples the AD8232 at 360 Hz, filters it, batches 1-second windows to `test/ecg_raw`; reads the MPU6050 and publishes a 1 Hz dashboard line to `test/sensors`; contains impact + stillness fall-detection heuristics. |
| [`firmware/camera_webserver/`](firmware/camera_webserver/) | ESP32-CAM MJPEG streaming server (based on the Espressif example) used as the visual input for the vision model. |
| [`arrhythmia/`](arrhythmia/) | ECG arrhythmia classifier — Colab notebook, trained base model weights, MIT-BIH + self-collected datasets, training/validation plots. |
| [`ehr_risk/`](ehr_risk/) | EHR clinical-risk engine — FHIR preprocessing, multi-task Keras model, TFLite export, standalone demo. |
| [`gesture_recognition/`](gesture_recognition/) | IMU motion classifier (1-D CNN in PyTorch) for falls, seizures, agitation, etc. |
| [`vision/`](vision/) | MobileNetV3-Small fine-tuned to classify camera frames as *Normal / Distress / Danger*; runner for webcam, video, images or the ESP32-CAM stream. |
| [`fusion/`](fusion/) | The alert-fusion engine (corroboration rules, alert levels) and `run_monitor.py`, which drives the real IMU + EHR models end to end. |
| [`tests/`](tests/) | `pytest` suite for the fusion rules, EHR encoding and ECG loading — none of it needs TensorFlow or hardware. |
| [`docs/`](docs/) | Project report and presentation. |

---

## The three AI modalities

### 1. Arrhythmia detection (`arrhythmia/`)

A 1-D CNN (Conv 32 → 64 → 128 with BatchNorm, MaxPool and Dropout, then global-average-pooling and a sigmoid head) classifies ECG windows as normal vs. arrhythmic. The bundled base model takes 10-second (3600-sample) windows of raw millivolt ECG; the fine-tuning stage re-windows to 1 second (360 samples).

- **Data.** 48 records from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) (exported to CSV with beat annotations in `raw_data/mitbih/`), plus 120 ten-second recordings collected from team members with our own AD8232 node (`raw_data/collected/normal/`). To obtain "abnormal" rhythms safely, team members recorded after bursts of strenuous exercise (`raw_data/collected/arrhythmia/`).
- **Training.** Two-stage *domain adaptation*: a base model is trained on MIT-BIH, then fine-tuned on the low-cost-sensor data so it generalises to our hardware. Class-weighted loss handles the heavy normal/arrhythmia imbalance; early stopping monitors validation AUC.
- **Results.** Base model on MIT-BIH held-out test: **AUC 0.93**. After fine-tuning on sensor data: **accuracy 99.6 %, AUC 0.999** on the sensor test split (see `training_results/training_summary.json`).
- **What is checked in.** Only the stage-1 base model (`base_model/`, Keras 3 config + weights). The fine-tuned sensor weights behind the 99.6 % figure are not in the repo, and the base model alone scores every AD8232 recording as normal — `ecg_infer.py` runs it anyway so the fine-tuned checkpoint can drop in via `--model`.

| Training curves (base) | Fine-tuning curves |
|---|---|
| ![](arrhythmia/training_results/training_base.png) | ![](arrhythmia/training_results/training_finetune.png) |

### 2. EHR risk prediction (`ehr_risk/`)

A multi-task neural network that reads a patient's recent history and outputs a probability for each of a set of clinical risks (`risk_sepsis_24h`, `risk_mi_24h`, `risk_stroke_24h`, `risk_heart_failure_24h`, …). The full experiment covered 130+ features and 51 conditions × 3 horizons (6 h / 24 h / 48 h) = 153 heads over 10 M+ records; the checkpoint bundled here is the 24 h variant (61 features → 51 heads).

- **Data pipeline (`ehr_preprocess.py`).** Parses FHIR JSON patient bundles (demographics, conditions, observations), builds per-patient time-series of vitals/labs with 3-hour rolling averages and delta features, and streams everything to chunked Parquet so datasets far larger than RAM can be processed.
- **Model (`ehr_model.py`).** Shared body (BatchNorm → Dense 128 → Dense 64) feeding one small sigmoid head per risk, trained jointly with binary cross-entropy and per-head AUC. Training (`ehr_train.py`) uses an out-of-core `tf.data`-style generator over the Parquet chunks.
- **Deployment.** The trained model is exported both as `ehr_model.keras` and as `ehr_model.tflite` for edge inference; `test_ehr_performance.py` benchmarks TFLite latency and `ehr_only_demo.py` scores the sample patients in `EHR_patient_records.json`.
- **Scoring records correctly (`ehr_engine.py`).** Training rows hold *one* observation each (demographics + history + the single vital/lab recorded at that moment), so a patient is scored as one row per observation with the per-risk maximum taken across rows — feeding a full vitals panel at once lands 10–2000 SD out of distribution and every head reads 0 %. Features that never varied in training are pinned (a non-zero value there is amplified ~30× by the first BatchNorm), and inputs in the wrong units (e.g. d-dimer in ng/mL) are flagged.
- **Known limits of the bundled checkpoint.** 31 of the 51 heads never fired during training (their label rules depend on features such as troponin that the 61-input model does not have) and always output 0 — the demo labels those explicitly. TFLite does not preserve head order; `ehr_tflite_map.py` recovers it into `ehr_tflite_outputs.json`.

### 3. Physical event recognition — IMU + vision (`gesture_recognition/`, `vision/`)

This modality exists to kill false alarms. A motion event flagged by the IMU is only escalated if the camera agrees.

- **IMU classifier.** A compact 1-D CNN (`WardGuardianCNN`) over 1.5 s windows (75 samples @ 50 Hz, 3-axis accelerometer) predicting 12 states: lying, sitting, walking, **fall**, **seizure**, slump, agitation, choking, vomiting, CPR-in-progress, respiratory distress, transport. Walking and fall windows come from the [SisFall](https://www.mdpi.com/1424-8220/17/1/198) dataset with a *subject-aware* split (test subjects never seen in training); the remaining classes are generated from physics-inspired synthetic signals. `sanity_check.py` probes the model with hand-crafted signals; `evaluate_model.py` produces the confusion matrix.
- **Vision classifier.** MobileNetV3-Small (ImageNet-pretrained; last three feature blocks + head fine-tuned) with heavy augmentation to cope with odd CCTV-like angles. Classifies frames into *Normal / Distress / Danger*. `run_webcam.py` runs it live on a webcam or the ESP32-CAM stream.

| IMU model — confusion matrix | Vision model — training metrics |
|---|---|
| ![](gesture_recognition/viz_confusion_matrix_strict.png) | ![](vision/vision_metrics_improved.png) |

**Fusion (`fusion/fusion_engine.py`).** Each modality contributes evidence, not a decision. The score is `severity × confidence` for the IMU event, plus weighted camera and ECG contributions, amplified by a *matching* clinical risk from the EHR model (a patient with a 93 % fall risk who slumps is escalated sooner than a healthy one). Two rules keep false alarms down:

- a **single uncorroborated sensor** is capped below the paging threshold — restlessness alone is a WATCH, restlessness plus a camera "Distress" verdict is an ALERT;
- a **critical event** (fall, seizure, choking, CPR) pages on its own, but only when the classifier is ≥ 75 % confident — a 56 % "FALL" on a patient rolling over stays a WATCH until a second sensor agrees.

Alert levels are `OK / INFO / WATCH / ALERT / CRITICAL`; `ALERT` and above page the nurse. `fusion/run_monitor.py` runs the real IMU checkpoint and EHR model through a scripted ward scenario (or live IMU windows over MQTT), and `tests/test_fusion_engine.py` pins the rules down.

---

## Getting started

### Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/              # 40 tests; no TensorFlow, GPU or hardware needed
```

### End-to-end ward monitor

```bash
python fusion/run_monitor.py                  # patient 101: EHR risk profile + IMU classifier + fusion
python fusion/run_monitor.py --patient 102    # healthy patient: same motion, fewer pages
python fusion/run_monitor.py --no-ehr         # PyTorch only, skips TensorFlow
python fusion/run_monitor.py --live --broker 192.168.1.100 --topic test/imu_raw   # real IMU stream
```

`--live` expects `ax,ay,az` messages in g at 50 Hz on the topic. The current `ecg_imu_node` sketch only publishes the 1 Hz dashboard line to `test/sensors` (too slow for a 1.5 s window), so a 50 Hz publish is a small firmware addition still to do.

### Arrhythmia model

`arrhythmia/ecg.ipynb` is a Colab notebook: it downloads MIT-BIH via `wfdb`, trains the base model, and renders an interactive Plotly evaluation dashboard. Outside the notebook:

```bash
cd arrhythmia
python ecg_infer.py raw_data/collected/            # score every recording with the bundled base model
python ecg_infer.py recording.csv --model finetuned.keras --scale 1.0   # a fine-tuned checkpoint, mV input
```

### EHR risk engine

```bash
cd ehr_risk
python ehr_only_demo.py --all      # score the bundled sample patients (interactive without --all)
python test_ehr_performance.py     # TFLite latency benchmark + agreement check against Keras

# To retrain from scratch, place a zip of FHIR JSON bundles at ehr_risk/fhir.zip, then:
python ehr_preprocess.py           # → final_parquet_chunks/
python ehr_train.py                # → ehr_model.keras/.tflite, scaler, feature/target lists, TFLite output map
```

### Gesture (IMU) model

```bash
cd gesture_recognition
python sanity_check.py             # probe ward_model_strict.pth with fresh synthetic signals (exit 1 on failure)
python evaluate_model.py           # precision/recall/AUC + confusion matrix on the unseen-subject test set
python visualize_metrics.py        # regenerate the PNGs below
# To retrain: download SisFall into gesture_recognition/SisFall_dataset/, then
python process_ward_data.py && python train_ward_model.py
```

### Vision model

```bash
cd vision
python run_vision.py                                        # live inference on webcam 0
python run_vision.py --source http://<esp32-cam-ip>:81/stream   # the ESP32-CAM MJPEG stream
python run_vision.py --source clip.mp4 --headless --save annotated.mp4
python run_vision.py --source frames/ --headless            # score a folder of images
# To retrain: put images in Ward_Vision_Data/{train,val}/{Normal,Distress,Danger}/ and run train_vision.py
```

### Firmware

Open either sketch folder in the Arduino IDE with the ESP32 board package installed.

- `firmware/ecg_imu_node/` needs the `PubSubClient` and `ArduinoJson` libraries. Set `ssid`, `password` and `mqtt_server` at the top of the sketch. Wiring: AD8232 `OUTPUT`→GPIO34, `LO+`→GPIO32, `LO-`→GPIO33; MPU6050 over I²C at `0x68`.
- `firmware/camera_webserver/` targets the AI-Thinker ESP32-CAM (select the board model in the sketch). Set the Wi-Fi credentials, flash, and open the printed IP in a browser for the MJPEG stream.

### Backend

Any MQTT broker works (we used Mosquitto). A Telegraf `[[inputs.mqtt_consumer]]` subscribed to `test/sensors` (InfluxDB line protocol) and `test/ecg_raw` (CSV of 360 ints) feeds InfluxDB; Grafana reads from there.

---

## Engineering notes

- **Sensor noise vs. medical-grade data.** The AD8232 on a 12-bit ADC is far noisier than the MIT-BIH recordings. The firmware does slow adaptive baseline tracking for DC-offset removal and a 7-tap moving-average filter before publishing; the model side compensates with the fine-tuning stage on real sensor data.
- **Class imbalance.** Normal beats vastly outnumber arrhythmic ones; class-weighted loss and AUC-based early stopping replaced accuracy as the optimisation target after early models collapsed to the majority class.
- **Out-of-core EHR training.** The FHIR corpus exceeded available RAM, so preprocessing writes chunked Parquet and training streams mini-batches from disk with aggressive dtype down-casting.
- **Honest evaluation for motion data.** SisFall windows are split by *subject*, not randomly, so reported accuracy reflects unseen patients rather than memorised gait signatures. SisFall's waist sensor reads −1 g on Y when upright; synthetic probes must use the same orientation or "walking" is classified as sitting.
- **Score the way you trained.** The EHR model only behaves on training-shaped rows (one observation each). Reconstructing that at inference time, rather than feeding a dense panel, is what turns "0 % for everyone" into a usable risk profile.
- **Multi-output TFLite conversion reorders heads.** Always recover the mapping empirically (`ehr_tflite_map.py`) rather than trusting tensor index or name order.

## Future work

Integration with hospital information systems via HL7/FHIR connectors, federated learning across edge nodes for privacy, a low-power mesh (Zigbee/LoRaWAN) for hospital-scale deployment, sensor self-diagnostics, and a hospital-at-home mode for post-discharge monitoring. See §6 of the report.

## Acknowledgements

MIT-BIH Arrhythmia Database (PhysioNet), SisFall dataset (Universidad de Antioquia), and the Espressif `CameraWebServer` example on which the camera firmware is based.
