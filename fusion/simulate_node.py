"""Stand-in for the ESP32 bedside node: publishes the same MQTT topics the
firmware does, from recorded ECG and the physics-based IMU generators, so the
live pipeline can be run and tested on any computer.

    python fusion/mini_broker.py &                       # or a real Mosquitto
    python fusion/simulate_node.py                       # plays the ward scenario once
    python fusion/simulate_node.py --loop --speed 4      # keep going, 4x real time
    python fusion/run_monitor.py --live                  # ...and watch it in another shell

Topics (see firmware/ecg_imu_node):
  test/ecg_raw   360 comma-separated ADC counts, once per second
  test/imu_raw   "ax,ay,az" in g at 50 Hz (the addition the firmware needs for
                 the motion classifier; the shipped sketch has its IMU disabled)
  test/sensors   1 Hz InfluxDB line for the Grafana dashboard
  test/vision    "Class,confidence" - what run_vision.py --publish sends
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gesture_recognition"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenario import SCENARIO, ecg_excerpt  # noqa: E402

IMU_RATE = 50
ECG_RATE = 360
STEP_SECONDS = 3.0                # each scenario step = two 75-sample IMU windows

TOPIC_ECG, TOPIC_IMU, TOPIC_DASH, TOPIC_VISION = "test/ecg_raw", "test/imu_raw", "test/sensors", "test/vision"


def scenario_events(seed=0):
    """Yield (t_seconds, topic, payload_str) for one pass through the scenario,
    in time order, exactly as the node would emit them."""
    from synthetic_signals import make_window
    from ward_model import WINDOW_SIZE

    rng = np.random.RandomState(seed)
    excerpts = {k: ecg_excerpt(k) for k in ("normal", "arrhythmia")}
    events = []
    t = 0.0
    for caption, class_id, vision, ecg in SCENARIO:
        # IMU: back-to-back windows from the generator, streamed sample by sample.
        n_windows = int(STEP_SECONDS * IMU_RATE / WINDOW_SIZE)
        for w in range(n_windows):
            np.random.seed(rng.randint(1 << 30))          # generators use the global RNG
            window = make_window(class_id)
            for i, (ax, ay, az) in enumerate(window):
                events.append((t + (w * WINDOW_SIZE + i) / IMU_RATE, TOPIC_IMU, f"{ax:.4f},{ay:.4f},{az:.4f}"))
        # ECG: one 360-sample batch per second, cycling through the excerpt.
        if ecg is not None:
            sig = excerpts[ecg]
            for s in range(int(STEP_SECONDS)):
                offset = (int(t) + s) * ECG_RATE % len(sig)
                batch = np.roll(sig, -offset)[:ECG_RATE]
                events.append((t + s + 0.999, TOPIC_ECG, ",".join(str(int(v)) for v in batch)))
        if vision is not None:
            events.append((t + 0.5, TOPIC_VISION, f"{vision},0.90"))
        for s in range(int(STEP_SECONDS)):
            fall = int(class_id == 3)
            events.append((t + s + 0.5, TOPIC_DASH,
                           f"accel accel_x=0.02,accel_y=-0.98,accel_z=0.05,fallStatus={fall},ecgValue=2048"))
        t += STEP_SECONDS
    events.sort(key=lambda e: e[0])
    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="replay the scenario forever")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ESP32_SimNode")
    client.connect(args.broker, args.port, 60)
    client.loop_start()
    if not args.quiet:
        print(f"Simulated node connected to {args.broker}:{args.port}; publishing {TOPIC_ECG}, {TOPIC_IMU}, "
              f"{TOPIC_DASH}, {TOPIC_VISION} at {args.speed:g}x")

    pass_no = 0
    try:
        while True:
            events = scenario_events(args.seed + pass_no)
            start = time.monotonic()
            for t, topic, payload in events:
                delay = t / args.speed - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
                client.publish(topic, payload)
            pass_no += 1
            if not args.quiet:
                print(f"  scenario pass {pass_no} complete ({len(events)} messages)")
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
