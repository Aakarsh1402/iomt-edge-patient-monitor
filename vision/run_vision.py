"""Run the ward vision classifier on a webcam, a video file, images, or the
ESP32-CAM MJPEG stream.

    python run_vision.py                              # default webcam (index 0)
    python run_vision.py --source 1                   # another webcam
    python run_vision.py --source http://192.168.1.50:81/stream   # ESP32-CAM
    python run_vision.py --source clip.mp4 --headless --save out.mp4
    python run_vision.py --source frames/ --headless  # score a folder of images
    python run_vision.py --source clip.mp4 --headless --publish localhost   # feed the fusion monitor

The original script only supported webcam 0 with a GUI window, which meant the
vision modality could not be demonstrated on a headless machine or against the
ESP32-CAM stream the rest of the project is built around.

With --publish, every verdict is sent to MQTT as "Class,confidence" on
--vision-topic, which is what fusion/run_monitor.py --live consumes.

Press 'q' to quit the live window.
"""
import argparse
import glob
import os
import sys

import cv2
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_model import (  # noqa: E402
    CLASS_COLORS,
    DEFAULT_MODEL_FILE,
    FALLBACK_COLOR,
    load_model,
    predict_image,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Publisher:
    """Optional MQTT sink for verdicts; a no-op when --publish is not given."""

    def __init__(self, broker, port, topic):
        self.client = None
        if broker:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ward_vision")
            self.client.connect(broker, port, 60)
            self.client.loop_start()
            self.topic = topic
            print(f"Publishing verdicts to {topic} on {broker}:{port}")

    def send(self, class_name, confidence):
        if self.client:
            self.client.publish(self.topic, f"{class_name},{confidence:.3f}")

    def close(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def annotate(frame, class_name, confidence):
    color = CLASS_COLORS.get(class_name, FALLBACK_COLOR)
    cv2.putText(frame, f"{class_name} ({confidence * 100:.1f}%)", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return frame


def image_paths(source):
    """Expand a directory or glob into a sorted list of image paths."""
    if os.path.isdir(source):
        files = [p for p in sorted(os.listdir(source)) if p.lower().endswith(IMAGE_SUFFIXES)]
        return [os.path.join(source, p) for p in files]
    matches = sorted(glob.glob(source))
    return [p for p in matches if p.lower().endswith(IMAGE_SUFFIXES)]


def run_images(paths, model, class_names, device, args, publisher):
    counts = {}
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  skipped (unreadable): {path}")
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        name, conf = predict_image(model, class_names, pil, device)
        counts[name] = counts.get(name, 0) + 1
        publisher.send(name, conf)
        print(f"  {os.path.basename(path):<40} {name:<10} {conf * 100:5.1f}%")
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            cv2.imwrite(os.path.join(args.save, os.path.basename(path)), annotate(frame, name, conf))
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing scored"
    print(f"\nSummary: {summary}")
    if args.save:
        print(f"Annotated frames written to {args.save}/")


def run_stream(source, model, class_names, device, args, publisher):
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not capture.isOpened():
        sys.exit(f"Could not open video source {source!r}. "
                 "Check the webcam index, file path, or that the ESP32-CAM stream URL is reachable.")

    writer = None
    if args.save:
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    print("Press 'q' to quit." if not args.headless else "Running headless...")
    frames = 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            name, conf = predict_image(model, class_names, pil, device)
            annotate(frame, name, conf)
            publisher.send(name, conf)

            if writer is not None:
                writer.write(frame)
            if args.headless:
                if frames % args.log_every == 0:
                    print(f"  frame {frames:>6}: {name} ({conf * 100:.1f}%)")
            else:
                cv2.imshow("Ward vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Annotated video written to {args.save}")
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Processed {frames} frame(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="webcam index, video file, image file/folder/glob, or stream URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--headless", action="store_true", help="no GUI window; print predictions instead")
    parser.add_argument("--save", help="write annotated output (video path, or folder for images)")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--log-every", type=int, default=10, help="headless: print every Nth frame")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--publish", metavar="BROKER", help="MQTT broker host to publish verdicts to")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--vision-topic", default="test/vision")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    model, class_names = load_model(args.model, args.device)
    print(f"Loaded {os.path.basename(args.model)} -> classes {class_names}")

    publisher = Publisher(args.publish, args.port, args.vision_topic)
    paths = image_paths(args.source) if not str(args.source).isdigit() else []
    try:
        if paths:
            print(f"Scoring {len(paths)} image(s):")
            run_images(paths, model, class_names, args.device, args, publisher)
        else:
            run_stream(args.source, model, class_names, args.device, args, publisher)
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
