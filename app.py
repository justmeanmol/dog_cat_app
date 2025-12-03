import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dog vs Cat Classifier", layout="centered")

# --------------------------
# Load the model
# --------------------------
@st.cache_resource
def load_catdog_model():
    model = load_model("dog_cat.keras")   # ← Your model
    return model

model = load_catdog_model()

st.title("🐶🐱 Dog vs Cat Image Classifier")
st.write("Upload an image or use live camera to classify dog vs cat.")


# --------------------------
# Prediction Function
# --------------------------
def predict(image):
    img = cv2.resize(image, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "🐶 Dog", float(prediction)
    else:
        return "🐱 Cat", float(1 - prediction)


# --------------------------
# 1️⃣ Upload Image Section
# --------------------------
st.header("📤 Upload an Image")

uploaded = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np._
