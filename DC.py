import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Dog vs Cat Classifier")

IMG_SIZE = 200

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "dog_cat_model.pt")


def build_model():
    """Must stay identical to train_colab.py's architecture, or load_state_dict() will fail."""
    backbone = models.mobilenet_v2(weights=None)  # weights=None: loading our own trained weights, not ImageNet's
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(backbone.last_channel, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid(),
    )
    return backbone


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        available = os.listdir(APP_DIR)
        st.error(
            f"Model file not found at:\n`{MODEL_PATH}`\n\n"
            f"Files actually present in the app directory:\n{available}\n\n"
            "Check that dog_cat_model.pt is committed to the repo root, "
            "not ignored by .gitignore, and under GitHub's file size limits."
        )
        st.stop()

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# Must match train_colab.py's val_transform exactly, including Normalize —
# pretrained models expect ImageNet-style normalized input, not just 0-1 scaled pixels.
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        val = model(tensor).item()

    # class_to_idx from training: 0=cat, 1=dog
    if val >= 0.5:
        st.success(f"Prediction: Dog 🐶 (confidence {val:.2%})")
    else:
        st.success(f"Prediction: Cat 🐱 (confidence {1-val:.2%})")
