import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(
    page_title="CT Scan Classification",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 CT Scan Classification")

st.write("""
1. Upload file model (.h5)
2. Upload gambar CT Scan
3. Sistem akan melakukan prediksi otomatis
""")

# ==========================
# Upload Model
# ==========================

uploaded_model = st.file_uploader(
    "Upload Model (.h5)",
    type=["h5"]
)

if uploaded_model is not None:

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".h5"
        ) as tmp:

            tmp.write(uploaded_model.read())
            model_path = tmp.name

        model = tf.keras.models.load_model(model_path)

        st.success("✅ Model berhasil dimuat")

        st.write("Input Shape Model:")

        st.code(str(model.input_shape))

        # ==========================
        # Ambil ukuran input model
        # ==========================

        INPUT_HEIGHT = model.input_shape[1]
        INPUT_WIDTH = model.input_shape[2]

        # ==========================
        # Upload Gambar
        # ==========================

        uploaded_image = st.file_uploader(
            "Upload CT Scan Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:

            image = Image.open(uploaded_image).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            # Resize sesuai model
            img = image.resize(
                (INPUT_WIDTH, INPUT_HEIGHT)
            )

            img = np.array(img)

            img = img.astype(np.float32)

            # Normalisasi
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

            st.subheader("Hasil Prediksi")

            st.write(prediction)

            predicted_class = int(
                np.argmax(prediction)
            )

            confidence = float(
                np.max(prediction)
            )

            st.success(
                f"Class Index: {predicted_class}"
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
                prob_dict[
                    f"Class {i}"
                ] = float(
                    prediction[0][i]
                )

            st.bar_chart(prob_dict)

    except Exception as e:

        st.error("Terjadi Error")

        st.exception(e)
