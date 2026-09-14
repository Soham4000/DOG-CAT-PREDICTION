import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("Dog vs Cat Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_cat_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    img_resized = img.resize((200, 200))
    img_array = np.array(img_resized)
    test_input = img_array.reshape((1, 200, 200, 3)) / 255.0  # match your training preprocessing

    val = model.predict(test_input)

    # val is a probability array, e.g. [[0.87]] — not a plain 0/1, so compare against a threshold
    if val[0][0] >= 0.5:
        st.success("Prediction: Dog 🐶")
    else:
        st.success("Prediction: Cat 🐱")
