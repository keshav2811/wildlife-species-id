import torch
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.cnn import WildlifeClassifier

# Load model
checkpoint = torch.load("checkpoints/epoch_001_acc1.0000.pt", map_location="cpu")

model = WildlifeClassifier(architecture="resnet50", num_classes=6)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# Class names
classes = [
    "AMERICAN GOLDFINCH",
    "BARN OWL",
    "CARMINE BEE-EATER",
    "DOWNY WOODPECKER",
    "EMPEROR PENGUIN",
    "FLAMINGO",
]

# Transform
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Load image
img_path = "test.jpg"  # put your image here
image = cv2.imread(r"C:\Users\kesha\OneDrive\Documents\project\wildlife_species_id\data\raw\AMERICAN GOLDFINCH\001.jpg")
if image is None:
    raise ValueError("Image not found. Check path!")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

aug = transform(image=image)
tensor = aug["image"].unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(tensor)
    _, pred = torch.max(outputs, 1)

print("Prediction:", classes[pred.item()])