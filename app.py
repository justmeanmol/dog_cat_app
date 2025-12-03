import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐶")

st.title("🐶🐱 Dog vs Cat Image Classifier")
st.write("Upload an image or use live camera to classify dog vs cat.")

# Cache the model so it loads only once
@st.cache_resource
def load_model_cached():
    model = load_model("dog_cat_model.h5")
    return model

model = load_model_cached()

# Preprocess function
def preprocess(img):
    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Live camera
camera_image = st.camera_input("Take a picture")

img = None

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

elif camera_image:
    img = cv2.imdecode(np.frombuffer(camera_image.getvalue(), np.uint8), 1)

# Prediction
if img is not None:
    st.image(img, caption="Input Image", use_column_width=True)

    processed = preprocess(img)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        st.success("It's a **CAT 😺**!")
    else:
        st.success("It's a **DOG 🐶**!")
