import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.title("Dog vs Cat Classifier")

IMG_SIZE = 200

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "dog_cat_model.pt")


class DogCatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(23 * 23 * 32, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        available = os.listdir(APP_DIR)
        st.error(f"Model file not found. Files present: {available}")
        st.stop()

    model = DogCatCNN()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = {k: v.float() for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"Model architecture doesn't match the saved weights:\n\n{e}")
        st.stop()

    model.eval()
    return model


model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        val = model(tensor).item()

    if val >= 0.5:
        st.success(f"Prediction: Dog 🐶 (confidence {val:.2%})")
    else:
        st.success(f"Prediction: Cat 🐱 (confidence {1-val:.2%})")
