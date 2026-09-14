import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "vision_model.pth"
NUM_CLASSES = 3

# Given the states "Danger", "Distress", and "Normal", ImageFolder sorts them alphabetically:
CLASS_NAMES = ["Danger", "Distress", "Normal"] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- SETUP MODEL ---
print("Loading model architecture...")
# Initialize the model structure with NO pre-trained weights since we will load our custom ones
model = models.mobilenet_v3_small(weights=None) 

# Rebuild the exact classifier structure used in training
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)

print(f"Loading weights from {MODEL_PATH}...")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model weights: {e}")
    print("Make sure the MODEL_PATH is correct and the model file exists.")
    exit(1)

model = model.to(device)
model.eval() # Set model to evaluation mode

# --- SETUP TRANSFORMS ---
# Must exactly match the 'val' transforms from train_vision.py
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- WEBCAM LOOP ---
print("\nStarting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0) # 0 is the default webcam index

if not cap.isOpened():
    print("Error: Could not open your webcam. Check your connections or permissions.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # OpenCV captures in BGR, but PyTorch models expect RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy array to PIL Image so we can use torchvision transforms
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transforms and add batch dimension (B, C, H, W)
    input_tensor = data_transforms(pil_image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Get raw confidence probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the top class and its confidence score
        confidence, preds = torch.max(probs, 1)
        
        predicted_class_idx = preds.item()
        confidence_score = confidence.item() * 100
        predicted_class_name = CLASS_NAMES[predicted_class_idx]

    # Determine color (OpenCV uses BGR format)
    if predicted_class_name == "Normal":
        color = (0, 255, 0) # Green
    elif predicted_class_name == "Distress":
        color = (0, 255, 255) # Yellow
    elif predicted_class_name == "Danger":
        color = (0, 0, 255) # Red
    else:
        color = (255, 255, 255) # White fallback
        
    # Overlay prediction on the video frame
    text = f"{predicted_class_name} ({confidence_score:.1f}%)"
    cv2.putText(frame, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Vision Model Inference', frame)

    # Check for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
