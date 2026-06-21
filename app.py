import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =====================================================
# KONFIGURASI
# =====================================================

MODEL_PATH = "ct_scan_classifier_model.h5"
FILE_ID = "https://drive.google.com/file/d/1xVpKk127kd9nYrQs9E9q9ECpmd209Ca0/view?usp=drive_link"

# SESUAIKAN JIKA URUTAN KELAS ASLI BERBEDA
CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

IMG_SIZE = (227, 227)

# =====================================================
# DOWNLOAD MODEL
# =====================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        return

    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    response = requests.get(url, stream=True)

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    download_model()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model

# =====================================================
# PREDIKSI
# =====================================================

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    img = np.array(image)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

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

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        prediction[0]
    )

# =====================================================
# STREAMLIT
# =====================================================

st.set_page_config(
    page_title="CT Scan Classification",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 CT Scan Classification System")

# =====================================================
# LOAD MODEL
# =====================================================

try:

    model = load_model()

    st.success(
        "✅ Model berhasil dimuat"
    )

except Exception as e:

    st.error(
        "❌ Gagal memuat model"
    )

    st.exception(e)

    st.stop()

# =====================================================
# INFO MODEL
# =====================================================

st.sidebar.success(
    "🟢 Model Ready"
)

st.sidebar.write(
    f"Input Shape: {model.input_shape}"
)

st.sidebar.write(
    f"Output Shape: {model.output_shape}"
)

# =====================================================
# UPLOAD MULTIPLE FILES
# =====================================================

uploaded_files = st.file_uploader(
    "Upload CT Scan Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =====================================================
# ANALISIS
# =====================================================

if uploaded_files:

    st.info(
        f"{len(uploaded_files)} gambar dipilih"
    )

    if st.button(
        "🔍 Mulai Analisis"
    ):

        results = []

        for file in uploaded_files:

            try:

                image = Image.open(file)

                predicted_class, confidence, probs = predict_image(
                    image
                )

                st.image(
                    image,
                    caption=f"{file.name} → {predicted_class}",
                    width=300
                )

                with st.expander(
                    f"Detail {file.name}"
                ):

                    st.write(
                        "Raw Prediction:",
                        probs
                    )

                    st.write(
                        {
                            CLASS_NAMES[i]:
                            float(probs[i])
                            for i in range(
                                len(CLASS_NAMES)
                            )
                        }
                    )

                results.append({
                    "File":
                        file.name,
                    "Prediction":
                        predicted_class,
                    "Confidence (%)":
                        round(
                            confidence * 100,
                            2
                        )
                })

            except Exception as e:

                st.error(
                    f"Error pada {file.name}"
                )

                st.exception(e)

        st.subheader(
            "📋 Hasil Klasifikasi"
        )

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            "📥 Download CSV",
            data=df.to_csv(
                index=False
            ),
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
