"""One-command live demo on a laptop: in-process MQTT broker + simulated
ESP32 node + the ward monitor consuming the stream.

    python fusion/live_demo.py                  # real time (~30 s)
    python fusion/live_demo.py --speed 5        # faster
    python fusion/live_demo.py --no-ehr --no-ecg   # without TensorFlow

This is the same code path as a real deployment - swap the simulated node
for the ESP32 and this broker for Mosquitto and nothing else changes.
"""
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_broker import Broker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=0, help="broker port (0 = any free port)")
    parser.add_argument("--patient", default="101")
    parser.add_argument("--no-ehr", action="store_true")
    parser.add_argument("--no-ecg", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    import run_monitor
    import simulate_node

    with Broker(host="127.0.0.1", port=args.port) as broker:
        print(f"mini MQTT broker on 127.0.0.1:{broker.port}")
        node_args = ["--broker", "127.0.0.1", "--port", str(broker.port), "--speed", str(args.speed), "--quiet"]
        node = threading.Thread(target=simulate_node.main, args=(node_args,), daemon=True)
        monitor_args = ["--live", "--broker", "127.0.0.1", "--port", str(broker.port), "--patient", args.patient,
                        "--idle-timeout", "4"]
        for flag in ("no_ehr", "no_ecg", "no_color"):
            if getattr(args, flag):
                monitor_args.append("--" + flag.replace("_", "-"))
        run_monitor.main(monitor_args, on_subscribed=node.start)   # node starts once the monitor listens


if __name__ == "__main__":
    main()
