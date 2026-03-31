import streamlit as st
import torch
import cv2
import numpy as np
from src.models.cnn import WildlifeClassifier

# Load model
model = WildlifeClassifier(architecture="resnet50", num_classes=6)
checkpoint = torch.load("checkpoints/epoch_001_acc0.9988.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Class names
classes = [
    "AMERICAN GOLDFINCH",
    "BARN OWL",
    "CARMINE BEE-EATER",
    "DOWNY WOODPECKER",
    "EMPEROR PENGUIN",
    "FLAMINGO"
]

st.title("🦜 Wildlife Species Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, 1).item()

    st.success(f"Prediction: {classes[pred]}")