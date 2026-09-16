"""Shared definition of the ward vision classifier (MobileNetV3-Small).

Keeps the architecture, the preprocessing transform and the class-name lookup
in one place so training and inference cannot disagree about any of them.
"""
import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_FILE = os.path.join(HERE, "vision_model.pth")

# ImageFolder sorts class directories alphabetically. Used only when a
# checkpoint has no sidecar .classes.json (older checkpoints).
DEFAULT_CLASS_NAMES = ["Danger", "Distress", "Normal"]

# Colours for the on-screen overlay, in OpenCV's BGR order.
CLASS_COLORS = {
    "Normal": (0, 255, 0),
    "Distress": (0, 255, 255),
    "Danger": (0, 0, 255),
}
FALLBACK_COLOR = (255, 255, 255)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Evaluation transform. train_vision.py adds augmentation on top of this.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def classes_path(model_file):
    """Sidecar file holding the class-name order a checkpoint was trained with."""
    return os.path.splitext(model_file)[0] + ".classes.json"


def save_class_names(model_file, class_names):
    with open(classes_path(model_file), "w") as f:
        json.dump(list(class_names), f, indent=2)


def read_class_names(model_file):
    path = classes_path(model_file)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return list(DEFAULT_CLASS_NAMES)


def build_model(num_classes, pretrained=False):
    """MobileNetV3-Small with the classifier head resized to num_classes."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_model(model_file=DEFAULT_MODEL_FILE, device="cpu"):
    """Load a trained checkpoint in eval mode. Returns (model, class_names)."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Checkpoint '{model_file}' not found. Train one with train_vision.py or pass --model."
        )
    class_names = read_class_names(model_file)
    model = build_model(len(class_names))
    model.load_state_dict(torch.load(model_file, map_location=device))
    return model.to(device).eval(), class_names


@torch.no_grad()
def predict_image(model, class_names, pil_image, device="cpu"):
    """Classify one PIL image. Returns (class_name, confidence)."""
    tensor = eval_transform(pil_image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx])
