"""Ward vision classifier: builds, loads its checkpoint, and scores an image."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vision"))

pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

from vision_model import (  # noqa: E402
    DEFAULT_CLASS_NAMES, DEFAULT_MODEL_FILE, build_model, load_model, predict_image,
    read_class_names, save_class_names,
)


def test_class_sidecar_round_trip(tmp_path):
    f = str(tmp_path / "m.pth")
    assert read_class_names(f) == DEFAULT_CLASS_NAMES          # no sidecar -> default order
    save_class_names(f, ["b", "a"])
    assert read_class_names(f) == ["b", "a"]


def test_untrained_model_scores_an_image():
    model = build_model(num_classes=3, pretrained=False).eval()
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8))
    name, conf = predict_image(model, ["A", "B", "C"], img)
    assert name in ("A", "B", "C") and 0 <= conf <= 1


def test_checkpoint_loads_and_predicts():
    model, class_names = load_model(DEFAULT_MODEL_FILE)
    assert sorted(class_names) == sorted(DEFAULT_CLASS_NAMES)
    img = Image.new("RGB", (224, 224), (120, 120, 120))
    name, conf = predict_image(model, class_names, img)
    assert name in class_names and 0 <= conf <= 1
