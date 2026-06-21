import streamlit as st
import tensorflow as tf
import tempfile
import numpy as np
from PIL import Image

st.title("CT Scan Classification")

# Upload model
uploaded_model = st.file_uploader(
    "Upload Model (.h5)",
    type=["h5"]
)

if uploaded_model is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name

    model = tf.keras.models.load_model(model_path)

    st.success("Model berhasil dimuat!")

    uploaded_image = st.file_uploader(
        "Upload CT Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(image)

        img = image.resize((224, 224))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        st.write("Prediksi:")
        st.write(prediction)
