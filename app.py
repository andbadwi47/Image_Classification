import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================================
# KONFIGURASI
# =====================================================

MODEL_PATH = "ct_scan_classifier_model.h5"

# File ID Google Drive Anda
FILE_ID = "1xVpKk127kd9nYrQs9E9q9ECpmd209Ca0"

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

# =====================================================
# DOWNLOAD MODEL DARI GOOGLE DRIVE
# =====================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        return

    st.info("Mengunduh model dari Google Drive...")

    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    response = requests.get(url, stream=True)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    download_model()

    model = tf.keras.models.load_model(MODEL_PATH)

    return model

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="CT Scan Classification",
    page_icon="🩻"
)

st.title("🩻 CT Scan Classification")

st.write(
    "Upload gambar CT Scan untuk dilakukan klasifikasi."
)

# =====================================================
# LOAD MODEL
# =====================================================

try:

    model = load_model()

    st.success("Model berhasil dimuat")

except Exception as e:

    st.error("Gagal memuat model")

    st.exception(e)

    st.stop()

# =====================================================
# AMBIL UKURAN INPUT OTOMATIS
# =====================================================

INPUT_HEIGHT = model.input_shape[1]
INPUT_WIDTH = model.input_shape[2]

st.write(
    f"Ukuran input model: {INPUT_HEIGHT} x {INPUT_WIDTH}"
)

# =====================================================
# UPLOAD GAMBAR
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CT Scan",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDIKSI
# =====================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        img = image.resize(
            (INPUT_WIDTH, INPUT_HEIGHT)
        )

        img = np.array(img)

        img = img.astype(np.float32)

        img = img / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

        with st.spinner(
            "Menganalisis gambar..."
        ):

            prediction = model.predict(
                img,
                verbose=0
            )

        predicted_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            np.max(prediction)
        )

        if predicted_index < len(CLASS_NAMES):

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

        else:

            predicted_class = (
                f"Class {predicted_index}"
            )

        st.success(
            f"Prediksi: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.subheader(
            "Probabilitas Tiap Kelas"
        )

        chart_data = {}

        for i in range(
            prediction.shape[1]
        ):

            if i < len(CLASS_NAMES):

                label = CLASS_NAMES[i]

            else:

                label = f"Class {i}"

            chart_data[label] = float(
                prediction[0][i]
            )

        st.bar_chart(chart_data)

    except Exception as e:

        st.error(
            "Terjadi kesalahan saat prediksi."
        )

        st.exception(e)
