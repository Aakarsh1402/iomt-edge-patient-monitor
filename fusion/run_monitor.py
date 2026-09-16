"""Live ward monitor - runs the whole pipeline end to end.

Pulls each patient's clinical risk profile from the EHR model, classifies IMU
windows with the trained motion CNN, and fuses the two (plus optional camera
and ECG evidence) into an alert level per the rules in fusion_engine.py.

    python fusion/run_monitor.py                 # scripted ward scenario
    python fusion/run_monitor.py --no-ehr        # skip TensorFlow, motion only
    python fusion/run_monitor.py --live          # classify IMU windows from MQTT

Without hardware the IMU stream is simulated from the same physics generators
the classifier was trained on, so the demo runs on any machine.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gesture_recognition"))
sys.path.insert(0, os.path.join(ROOT, "ehr_risk"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_engine import AlertLevel, Observation, fuse  # noqa: E402

# Scripted scenario: (seconds, IMU class id, camera verdict, P(arrhythmia)).
# Camera/ECG are None when that sensor has nothing to say.
SCENARIO = [
    ("resting quietly", 0, "Normal", 0.02),
    ("sitting up in bed", 1, "Normal", 0.03),
    ("walking to the bathroom", 2, "Normal", 0.05),
    ("restless, shifting about", 6, "Normal", 0.08),
    ("restless, camera sees distress", 6, "Distress", 0.10),
    ("slumping sideways", 5, None, 0.12),
    ("breathing hard, ECG irregular", 10, "Distress", 0.88),
    ("FALL", 3, "Danger", 0.35),
    ("motionless on the floor", 0, "Danger", 0.30),
]

COLORS = {
    AlertLevel.NONE: "\033[32m", AlertLevel.INFO: "\033[36m", AlertLevel.WATCH: "\033[33m",
    AlertLevel.ALERT: "\033[31m", AlertLevel.CRITICAL: "\033[1;37;41m",
}
RESET = "\033[0m"


def colorize(level, text, enabled):
    return f"{COLORS[level]}{text}{RESET}" if enabled else text


def load_clinical_risks(patient_id, db_path):
    """Score a patient's EHR record once; their 24 h risks are the fusion context."""
    import json

    from ehr_engine import EHRRiskEngine

    with open(db_path) as f:
        db = json.load(f)
    if patient_id not in db:
        raise KeyError(f"patient {patient_id!r} not in {db_path}; known: {', '.join(db)}")
    record = db[patient_id]
    result = EHRRiskEngine().score(record)
    return record, result.risks


def simulated_windows():
    """Yield (caption, imu_window, vision_class, arrhythmia_prob) for the scenario."""
    from synthetic_signals import make_window

    for caption, class_id, vision, ecg in SCENARIO:
        yield caption, make_window(class_id), vision, ecg


def live_windows(broker, topic, port):
    """Yield windows assembled from IMU samples published by the ESP32 node."""
    import paho.mqtt.client as mqtt  # optional dependency

    from ward_model import WINDOW_SIZE

    buffer = []
    client = mqtt.Client()

    def on_message(_client, _userdata, msg):
        try:
            parts = [float(v) for v in msg.payload.decode().split(",")[:3]]
        except ValueError:
            return
        if len(parts) == 3:
            buffer.append(parts)

    client.on_message = on_message
    client.connect(broker, port, 60)
    client.subscribe(topic)
    client.loop_start()
    print(f"Subscribed to {topic} on {broker}:{port}; waiting for IMU samples...")
    try:
        while True:
            if len(buffer) >= WINDOW_SIZE:
                window = np.array(buffer[:WINDOW_SIZE])
                del buffer[:WINDOW_SIZE]
                yield "live IMU window", window, None, None
            else:
                time.sleep(0.2)
    finally:
        client.loop_stop()
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", default="101", help="patient id in EHR_patient_records.json")
    parser.add_argument("--db", default=os.path.join(ROOT, "ehr_risk", "EHR_patient_records.json"))
    parser.add_argument("--model", default=None, help="IMU checkpoint (default: ward_model_strict.pth)")
    parser.add_argument("--no-ehr", action="store_true", help="skip the EHR model (no TensorFlow needed)")
    parser.add_argument("--live", action="store_true", help="read IMU windows from MQTT instead of simulating")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="test/imu_raw")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between scenario steps")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    from ward_model import DEFAULT_MODEL_FILE, load_model, predict_window

    color = not args.no_color and sys.stdout.isatty()

    print("=" * 78)
    print("IoMT WARD MONITOR - multimodal fusion")
    print("=" * 78)

    record, clinical_risks = None, {}
    if not args.no_ehr:
        print("[1/2] Scoring EHR clinical risk...")
        try:
            record, clinical_risks = load_clinical_risks(args.patient, args.db)
        except Exception as e:
            print(f"      EHR model unavailable ({e}); continuing without clinical context.")
    print("[2/2] Loading IMU motion classifier...")
    model = load_model(args.model or DEFAULT_MODEL_FILE)

    if record:
        top = sorted(clinical_risks.items(), key=lambda kv: -kv[1])[:3]
        print(f"\nPatient {args.patient}: {record['name']}, age {record['age']}")
        print("  Clinical risk profile (24 h): " +
              ", ".join(f"{k.replace('risk_', '').replace('_24h', '')} {v * 100:.0f}%" for k, v in top))
    else:
        print(f"\nPatient {args.patient}: no clinical context loaded")

    source = live_windows(args.broker, args.topic, args.port) if args.live else simulated_windows()
    print("\n" + "-" * 78)
    print(f"{'t':>4}  {'situation':<32}{'motion':<20}{'level':<10}score")
    print("-" * 78)

    pages = 0
    for step, (caption, window, vision_class, ecg_prob) in enumerate(source):
        motion_class, motion_conf, _ = predict_window(model, window)
        obs = Observation(
            motion_class=motion_class,
            motion_confidence=motion_conf,
            vision_class=vision_class,
            vision_confidence=0.9 if vision_class else 0.0,
            arrhythmia_probability=ecg_prob,
            clinical_risks=clinical_risks,
        )
        result = fuse(obs)
        pages += result.should_page

        motion_text = f"{motion_class} {motion_conf * 100:.0f}%"
        line = f"{step:>4}  {caption:<32}{motion_text:<20}{result.level.label:<10}{result.score:.2f}"
        print(colorize(result.level, line, color))
        if result.level >= AlertLevel.WATCH:
            for reason in result.reasons:
                print(f"        - {reason}")
        if result.should_page:
            print(colorize(result.level, "        >>> PAGING NURSE", color))

        if not args.live and args.delay:
            time.sleep(args.delay)

    print("-" * 78)
    print(f"{pages} page(s) raised over {step + 1} windows.")


if __name__ == "__main__":
    main()
