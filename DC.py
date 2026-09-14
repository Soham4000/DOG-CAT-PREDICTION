import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.title("Dog vs Cat Classifier")

IMG_SIZE = 200

# Resolve the model path relative to this script's own location, not whatever
# directory Streamlit happens to be running from — this is a common cause of
# FileNotFoundError even when the file is correctly in the repo.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "dog_cat_model.pt")


class DogCatCNN(nn.Module):
    """Must stay byte-for-byte identical to the architecture in train_colab.py,
    or load_state_dict() will fail with a shape-mismatch error."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # List what's actually in the app directory so the real problem is visible
        # in the Streamlit UI instead of buried in the server logs.
        available = os.listdir(APP_DIR)
        st.error(
            f"Model file not found at:\n`{MODEL_PATH}`\n\n"
            f"Files actually present in the app directory:\n{available}\n\n"
            "Check that dog_cat_model.pt is committed to the repo root, "
            "not ignored by .gitignore, and under GitHub's file size limits."
        )
        st.stop()

    model = DogCatCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW, add batch dim

    with torch.no_grad():
        val = model(tensor).item()

    # class_to_idx from training: 0=cat, 1=dog — confirm this matches the printout
    # from train_colab.py's "Classes:" line before trusting this label order
    if val >= 0.5:
        st.success(f"Prediction: Dog 🐶 (confidence {val:.2%})")
    else:
        st.success(f"Prediction: Cat 🐱 (confidence {1-val:.2%})")
