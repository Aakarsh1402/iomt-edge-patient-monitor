"""The pure-Python MQTT broker that lets the live pipeline run without Mosquitto."""
import os
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fusion"))

from mini_broker import Broker, encode_remaining_length, topic_matches  # noqa: E402


@pytest.mark.parametrize("pattern,topic,expected", [
    ("test/ecg_raw", "test/ecg_raw", True),
    ("test/ecg_raw", "test/imu_raw", False),
    ("test/+", "test/imu_raw", True),
    ("test/+", "test/imu/raw", False),
    ("test/#", "test/imu/raw", True),
    ("#", "anything/at/all", True),
    ("test/+/raw", "test/imu/raw", True),
    ("test", "test/imu", False),
])
def test_topic_matching(pattern, topic, expected):
    assert topic_matches(pattern, topic) is expected


def test_remaining_length_varint():
    assert encode_remaining_length(0) == b"\x00"
    assert encode_remaining_length(127) == b"\x7f"
    assert encode_remaining_length(128) == b"\x80\x01"
    assert encode_remaining_length(16383) == b"\xff\x7f"


def test_publish_subscribe_round_trip():
    mqtt = pytest.importorskip("paho.mqtt.client")
    received, got = [], threading.Event()

    def on_message(_c, _u, msg):
        received.append((msg.topic, msg.payload.decode(), msg.retain))
        if len(received) >= 3:
            got.set()

    with Broker(host="127.0.0.1", port=0) as broker:
        pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pub")
        pub.connect("127.0.0.1", broker.port)
        pub.loop_start()
        pub.publish("ward/status", "retained-hello", retain=True).wait_for_publish(2)

        sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sub")
        sub.on_message = on_message
        sub.connect("127.0.0.1", broker.port)
        sub.subscribe("ward/#")
        sub.loop_start()
        time.sleep(0.2)                       # let the subscription register (and the retained msg arrive)

        pub.publish("ward/imu", "0.1,-0.98,0.02", qos=0)
        pub.publish("ward/ecg", "1,2,3", qos=1).wait_for_publish(2)
        pub.publish("other/topic", "ignored").wait_for_publish(2)
        assert got.wait(3), f"got only {received}"

        sub.loop_stop(); pub.loop_stop()
        sub.disconnect(); pub.disconnect()

    topics = {t for t, _, _ in received}
    assert topics == {"ward/status", "ward/imu", "ward/ecg"}
    assert ("ward/status", "retained-hello", True) in received
    assert broker.messages == 4
