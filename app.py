import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# KONFIGURASI
# =====================================

MODEL_PATH = "ct_scan_classifier_model.h5"

# Google Drive File ID
FILE_ID = "https://drive.google.com/file/d/1xVpKk127kd9nYrQs9E9q9ECpmd209Ca0/view?usp=drive_link"

# Nama kelas (ubah jika berbeda)
CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

# =====================================
# DOWNLOAD & LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.info("Mengunduh model dari Google Drive...")

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        gdown.download(
            url=url,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model

# =====================================
# LOAD MODEL
# =====================================

try:

    model = load_model()

except Exception as e:

    st.error("Gagal memuat model.")

    st.exception(e)

    st.stop()

# =====================================
# AMBIL UKURAN INPUT MODEL OTOMATIS
# =====================================

INPUT_HEIGHT = model.input_shape[1]
INPUT_WIDTH = model.input_shape[2]

# =====================================
# STREAMLIT UI
# =====================================

st.set_page_config(
    page_title="CT Scan Classification",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 CT Scan Classification")

st.write(
    "Upload gambar CT Scan untuk melakukan klasifikasi."
)

st.write(
    f"Input model: {INPUT_HEIGHT} x {INPUT_WIDTH}"
)

# =====================================
# UPLOAD GAMBAR
# =====================================

uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDIKSI
# =====================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

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

        prob_dict = {}

        for i in range(
            prediction.shape[1]
        ):

            if i < len(CLASS_NAMES):

                label = CLASS_NAMES[i]

            else:

                label = f"Class {i}"

            prob_dict[label] = float(
                prediction[0][i]
            )

        st.bar_chart(prob_dict)

    except Exception as e:

        st.error(
            "Terjadi kesalahan saat prediksi."
        )

        st.exception(e)
