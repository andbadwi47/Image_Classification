import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ct_scan_classifier_model.h5")

model = load_model()

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

def predict(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction))

    return predicted_class, confidence, prediction[0]

st.set_page_config(page_title="CT Scan Classifier", page_icon="🩻")

st.title("🩻 CT Scan Classification")

uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Menganalisis gambar..."):
        predicted_class, confidence, probs = predict(image)

    st.success(f"Hasil Prediksi: {predicted_class}")
    st.metric("Confidence", f"{confidence*100:.2f}%")

    prob_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(prob_dict)
