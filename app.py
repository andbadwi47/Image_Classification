import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# =====================================================
# KONFIGURASI
# =====================================================

MODEL_PATH = "ct_scan_classifier_model.h5"

# File ID Google Drive Anda
FILE_ID = "https://drive.google.com/file/d/1xVpKk127kd9nYrQs9E9q9ECpmd209Ca0/view?usp=drive_link"

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

# =====================================================
# DOWNLOAD MODEL
# =====================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        return

    st.warning("⬇️ Model belum tersedia, mengunduh dari Google Drive...")

    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    response = requests.get(
        url,
        stream=True
    )

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=8192
        ):

            if chunk:
                file.write(chunk)

    st.success("✅ Download model selesai")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    download_model()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CT Scan Classification",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 CT Scan Classification System")

st.write(
    "Upload satu atau beberapa gambar CT Scan untuk dilakukan klasifikasi."
)

# =====================================================
# LOAD MODEL
# =====================================================

try:

    model = load_model()

    st.success("✅ Model berhasil dimuat")

    if os.path.exists(MODEL_PATH):

        file_size = (
            os.path.getsize(MODEL_PATH)
            / (1024 * 1024)
        )

        st.info(
            f"📦 Model tersedia ({file_size:.2f} MB)"
        )

except Exception as e:

    st.error("❌ Gagal memuat model")

    st.exception(e)

    st.stop()

# =====================================================
# INFO MODEL
# =====================================================

INPUT_HEIGHT = model.input_shape[1]
INPUT_WIDTH = model.input_shape[2]

st.write(
    f"Input Shape : {model.input_shape}"
)

st.write(
    f"Output Shape : {model.output_shape}"
)

# =====================================================
# UPLOAD FILE
# =====================================================

uploaded_files = st.file_uploader(
    "Upload CT Scan Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =====================================================
# PREDIKSI
# =====================================================

if uploaded_files:

    st.info(
        f"📂 {len(uploaded_files)} gambar dipilih"
    )

    if st.button("🔍 Mulai Analisis"):

        results = []

        progress = st.progress(0)

        for idx, uploaded_file in enumerate(
            uploaded_files
        ):

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                img = image.resize(
                    (
                        INPUT_WIDTH,
                        INPUT_HEIGHT
                    )
                )

                img = np.array(img)

                img = img.astype(
                    np.float32
                )

                img = img / 255.0

                img = np.expand_dims(
                    img,
                    axis=0
                )

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

                if predicted_index < len(
                    CLASS_NAMES
                ):

                    predicted_class = (
                        CLASS_NAMES[
                            predicted_index
                        ]
                    )

                else:

                    predicted_class = (
                        f"Class {predicted_index}"
                    )

                results.append(
                    {
                        "File":
                            uploaded_file.name,
                        "Prediction":
                            predicted_class,
                        "Confidence (%)":
                            round(
                                confidence
                                * 100,
                                2
                            )
                    }
                )

                st.image(
                    image,
                    caption=
                    f"{uploaded_file.name} → {predicted_class}",
                    width=250
                )

            except Exception as e:

                st.error(
                    f"Error pada file: {uploaded_file.name}"
                )

                st.exception(e)

            progress.progress(
                (idx + 1)
                / len(uploaded_files)
            )

        st.success(
            "✅ Analisis selesai"
        )

        st.subheader(
            "📋 Hasil Klasifikasi"
        )

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            label="📥 Download Hasil CSV",
            data=df.to_csv(
                index=False
            ),
            file_name=
            "hasil_klasifikasi.csv",
            mime="text/csv"
        )
