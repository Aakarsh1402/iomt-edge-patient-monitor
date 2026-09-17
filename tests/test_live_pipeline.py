"""End-to-end: simulated node -> mini broker -> live monitor, on one machine."""
import os
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fusion"))

pytest.importorskip("torch")
mqtt = pytest.importorskip("paho.mqtt.client")

from mini_broker import Broker  # noqa: E402


def test_scenario_events_cover_every_topic():
    from simulate_node import TOPIC_DASH, TOPIC_ECG, TOPIC_IMU, TOPIC_VISION, scenario_events
    events = scenario_events()
    topics = {t for _, t, _ in events}
    assert topics == {TOPIC_DASH, TOPIC_ECG, TOPIC_IMU, TOPIC_VISION}
    times = [t for t, _, _ in events]
    assert times == sorted(times)
    imu = [p for _, t, p in events if t == TOPIC_IMU]
    assert len(imu) % 75 == 0 and len(imu) >= 9 * 150
    ecg = next(p for _, t, p in events if t == TOPIC_ECG)
    assert len(ecg.split(",")) == 360


def test_live_monitor_pages_on_the_fall(capsys):
    import run_monitor
    from simulate_node import scenario_events

    with Broker(host="127.0.0.1", port=0) as broker:
        def publish():
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="node")
            client.connect("127.0.0.1", broker.port)
            client.loop_start()
            time.sleep(0.5)                        # let the monitor subscribe first
            for _t, topic, payload in scenario_events():
                client.publish(topic, payload)
                time.sleep(0.0005)
            client.loop_stop()
            client.disconnect()

        t = threading.Thread(target=publish, daemon=True)
        t.start()
        pages = run_monitor.main(["--live", "--no-ehr", "--no-ecg", "--no-color", "--broker", "127.0.0.1",
                                  "--port", str(broker.port), "--max-windows", "18", "--idle-timeout", "5"])
        t.join(10)

    out = capsys.readouterr().out
    assert "live window" in out
    assert "FALL" in out
    assert pages >= 1 and "PAGING NURSE" in out
