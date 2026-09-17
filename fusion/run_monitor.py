"""Live ward monitor - runs the whole pipeline end to end.

Pulls each patient's clinical risk profile from the EHR model, classifies IMU
windows with the trained motion CNN, scores ECG with the arrhythmia CNN, takes
the camera's verdict, and fuses everything into an alert level per the rules
in fusion_engine.py.

    python fusion/run_monitor.py                 # scripted ward scenario, all models
    python fusion/run_monitor.py --no-ehr --no-ecg   # PyTorch only, no TensorFlow
    python fusion/run_monitor.py --live          # consume MQTT (see simulate_node.py)

Without hardware the scenario is fed from the same physics generators the
motion classifier was trained on and from real ECG recordings (a held-out
volunteer's AD8232 trace and a MIT-BIH test patient with PVCs), so the demo
exercises the real models on every machine. `--live` consumes exactly what
the ESP32 node (or fusion/simulate_node.py) publishes.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gesture_recognition"))
sys.path.insert(0, os.path.join(ROOT, "ehr_risk"))
sys.path.insert(0, os.path.join(ROOT, "arrhythmia"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion_engine import AlertLevel, Observation, fuse  # noqa: E402
from scenario import ECG_FALLBACK_PROB, SCENARIO, ecg_excerpt  # noqa: E402

COLORS = {
    AlertLevel.NONE: "\033[32m", AlertLevel.INFO: "\033[36m", AlertLevel.WATCH: "\033[33m",
    AlertLevel.ALERT: "\033[31m", AlertLevel.CRITICAL: "\033[1;37;41m",
}
RESET = "\033[0m"
VISION_MAX_AGE = 5.0          # seconds a camera verdict stays valid
ECG_MAX_AGE = 15.0            # seconds an ECG score stays valid


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


def load_ecg_detector():
    from ecg_model import ArrhythmiaDetector
    return ArrhythmiaDetector()


def simulated_windows(detector):
    """Yield (caption, imu_window, vision_class, vision_conf, arrhythmia_prob) for the scenario."""
    from synthetic_signals import make_window

    cache = {}
    for caption, class_id, vision, ecg in SCENARIO:
        prob = None
        if ecg is not None:
            if detector is None:
                prob = ECG_FALLBACK_PROB[ecg]
            else:
                if ecg not in cache:
                    cache[ecg] = float(detector.score(ecg_excerpt(ecg)).max())
                prob = cache[ecg]
        yield caption, make_window(class_id), vision, 0.9 if vision else 0.0, prob


class LiveFeed:
    """Assembles fusion inputs from the node's MQTT topics.

    IMU samples are grouped into classifier windows; ECG batches fill a
    rolling 10 s buffer that is re-scored whenever it is full; the newest
    camera verdict is attached while it is fresh.
    """

    def __init__(self, broker, port, detector, topics, idle_timeout=None, max_windows=None):
        import paho.mqtt.client as mqtt

        from ward_model import WINDOW_SIZE

        self.window_size = WINDOW_SIZE
        self.detector = detector
        self.topics = topics
        self.idle_timeout, self.max_windows = idle_timeout, max_windows
        self.lock = threading.Lock()
        self.imu, self.ecg = [], []
        self.vision = (None, 0.0, 0.0)          # class, confidence, timestamp
        self.ecg_prob = (None, 0.0)             # probability, timestamp
        self.last_message = time.monotonic()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ward_monitor")
        self.client.on_message = self._on_message
        self.client.connect(broker, port, 60)
        for t in topics.values():
            self.client.subscribe(t)
        self.client.loop_start()

    def _on_message(self, _client, _userdata, msg):
        payload = msg.payload.decode(errors="replace")
        now = time.monotonic()
        self.last_message = now
        try:
            with self.lock:
                if msg.topic == self.topics["imu"]:
                    self.imu.append([float(v) for v in payload.split(",")[:3]])
                elif msg.topic == self.topics["ecg"]:
                    self.ecg.extend(float(v) for v in payload.split(","))
                    if self.detector is not None and len(self.ecg) >= self.detector.window_size:
                        self.ecg = self.ecg[-self.detector.window_size:]
                        prob = float(self.detector.score(np.array(self.ecg, dtype=np.float32)).max())
                        self.ecg_prob = (prob, now)
                elif msg.topic == self.topics["vision"]:
                    name, conf = payload.split(",")
                    self.vision = (name.strip(), float(conf), now)
        except ValueError:
            pass                                    # malformed line: ignore, as the firmware may pad

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()

    def __iter__(self):
        n = 0
        try:
            while self.max_windows is None or n < self.max_windows:
                with self.lock:
                    ready = len(self.imu) >= self.window_size
                    if ready:
                        window = np.array(self.imu[:self.window_size])
                        del self.imu[:self.window_size]
                        now = time.monotonic()
                        vclass, vconf, vt = self.vision
                        if now - vt > VISION_MAX_AGE:
                            vclass, vconf = None, 0.0
                        prob, pt = self.ecg_prob
                        if now - pt > ECG_MAX_AGE:
                            prob = None
                if ready:
                    n += 1
                    yield "live window", window, vclass, vconf, prob
                elif self.idle_timeout and time.monotonic() - self.last_message > self.idle_timeout:
                    print(f"No messages for {self.idle_timeout:g} s; stopping.")
                    return
                else:
                    time.sleep(0.05)
        finally:
            self.close()


def main(argv=None, on_subscribed=None):
    """on_subscribed: optional callback invoked once the live feed is listening (used by live_demo.py)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patient", default="101", help="patient id in EHR_patient_records.json")
    parser.add_argument("--db", default=os.path.join(ROOT, "ehr_risk", "EHR_patient_records.json"))
    parser.add_argument("--model", default=None, help="IMU checkpoint (default: ward_model_strict.pth)")
    parser.add_argument("--no-ehr", action="store_true", help="skip the EHR model")
    parser.add_argument("--no-ecg", action="store_true", help="skip the arrhythmia model (scripted ECG probabilities)")
    parser.add_argument("--live", action="store_true", help="consume sensor topics from MQTT instead of simulating")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--imu-topic", default="test/imu_raw")
    parser.add_argument("--ecg-topic", default="test/ecg_raw")
    parser.add_argument("--vision-topic", default="test/vision")
    parser.add_argument("--idle-timeout", type=float, default=None, help="live: stop after this many silent seconds")
    parser.add_argument("--max-windows", type=int, default=None, help="live: stop after this many IMU windows")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between scenario steps")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the simulated IMU stream")
    args = parser.parse_args(argv)
    np.random.seed(args.seed)

    from ward_model import DEFAULT_MODEL_FILE, load_model, predict_window

    color = not args.no_color and sys.stdout.isatty()

    print("=" * 78)
    print("IoMT WARD MONITOR - multimodal fusion")
    print("=" * 78)

    record, clinical_risks = None, {}
    if not args.no_ehr:
        print("[1/3] Scoring EHR clinical risk...")
        try:
            record, clinical_risks = load_clinical_risks(args.patient, args.db)
        except Exception as e:
            print(f"      EHR model unavailable ({e}); continuing without clinical context.")
    detector = None
    if not args.no_ecg:
        print("[2/3] Loading ECG arrhythmia classifier...")
        try:
            detector = load_ecg_detector()
        except Exception as e:
            print(f"      ECG model unavailable ({e}); using scripted ECG probabilities.")
    print("[3/3] Loading IMU motion classifier...")
    model = load_model(args.model or DEFAULT_MODEL_FILE)

    if record:
        top = sorted(clinical_risks.items(), key=lambda kv: -kv[1])[:3]
        print(f"\nPatient {args.patient}: {record['name']}, age {record['age']}")
        print("  Clinical risk profile (24 h): " +
              ", ".join(f"{k.replace('risk_', '').replace('_24h', '')} {v * 100:.0f}%" for k, v in top))
    else:
        print(f"\nPatient {args.patient}: no clinical context loaded")

    if args.live:
        topics = {"imu": args.imu_topic, "ecg": args.ecg_topic, "vision": args.vision_topic}
        source = LiveFeed(args.broker, args.port, detector, topics, args.idle_timeout, args.max_windows)
        print(f"Subscribed to {', '.join(topics.values())} on {args.broker}:{args.port}; waiting for samples...")
        if on_subscribed:
            on_subscribed()
    else:
        source = simulated_windows(detector)

    print("\n" + "-" * 78)
    print(f"{'t':>4}  {'situation':<30}{'motion':<19}{'ecg':>5}  {'cam':<9}{'level':<9}score")
    print("-" * 78)

    pages, step = 0, -1
    for step, (caption, window, vision_class, vision_conf, ecg_prob) in enumerate(source):
        motion_class, motion_conf, _ = predict_window(model, window)
        obs = Observation(
            motion_class=motion_class,
            motion_confidence=motion_conf,
            vision_class=vision_class,
            vision_confidence=vision_conf,
            arrhythmia_probability=ecg_prob,
            clinical_risks=clinical_risks,
        )
        result = fuse(obs)
        pages += result.should_page

        motion_text = f"{motion_class} {motion_conf * 100:.0f}%"
        ecg_text = f"{ecg_prob:.2f}" if ecg_prob is not None else "-"
        line = (f"{step:>4}  {caption:<30}{motion_text:<19}{ecg_text:>5}  {(vision_class or '-'):<9}"
                f"{result.level.label:<9}{result.score:.2f}")
        print(colorize(result.level, line, color), flush=True)
        if result.level >= AlertLevel.WATCH:
            for reason in result.reasons:
                print(f"        - {reason}")
        if result.should_page:
            print(colorize(result.level, "        >>> PAGING NURSE", color), flush=True)

        if not args.live and args.delay:
            time.sleep(args.delay)

    print("-" * 78)
    print(f"{pages} page(s) raised over {step + 1} windows.")
    return pages


if __name__ == "__main__":
    main()
