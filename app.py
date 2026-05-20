from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

from utils.preprocessing import preprocess_image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pneumonia_model.keras"
PREDICTION_THRESHOLD = 0.8


st.set_page_config(
    page_title="Klasifikasi X-Ray Pneumonia",
    layout="centered",
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return load_model(MODEL_PATH)
    except Exception as error:
        st.error(f"Gagal memuat model: {error}")
        return None


def predict(image: Image.Image, threshold: float):
    model = get_model()
    if model is None:
        st.error(
            "Model tidak ditemukan. Jalankan training terlebih dahulu dengan "
            "`python train_model.py` sampai file `pneumonia_model.keras` dibuat."
        )
        return None

    processed_image = preprocess_image(image)
    probability = float(model.predict(processed_image, verbose=0)[0][0])

    if probability >= threshold:
        label = "PNEUMONIA"
        confidence = probability
    else:
        label = "NORMAL"
        confidence = 1.0 - probability

    return label, confidence, probability


st.title("Klasifikasi Citra X-Ray Dada")
st.write(
    "Aplikasi sederhana berbasis Deep Learning untuk mengklasifikasikan citra "
    "X-Ray dada ke dalam kelas NORMAL atau PNEUMONIA menggunakan model "
    "Transfer Learning MobileNetV2."
)

st.warning(
    "Disclaimer: aplikasi ini hanya untuk edukasi dan demonstrasi. Hasil "
    "prediksi tidak boleh digunakan sebagai pengganti diagnosis dokter."
)

uploaded_file = st.file_uploader(
    "Upload gambar X-Ray dada",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG.",
)

prediction_threshold = st.slider(
    "Threshold prediksi PNEUMONIA",
    min_value=0.5,
    max_value=0.95,
    value=PREDICTION_THRESHOLD,
    step=0.05,
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview gambar", use_container_width=True)

        if st.button("Prediksi", type="primary"):
            result = predict(image, prediction_threshold)
            if result is not None:
                label, confidence, raw_probability = result

                if label == "PNEUMONIA":
                    st.error(f"Hasil prediksi: {label}")
                else:
                    st.success(f"Hasil prediksi: {label}")

                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(int(np.clip(confidence * 100, 0, 100)))

                with st.expander("Detail probabilitas model"):
                    st.write(f"Probabilitas PNEUMONIA: {raw_probability * 100:.2f}%")
                    st.write(f"Probabilitas NORMAL: {(1.0 - raw_probability) * 100:.2f}%")

    except (UnidentifiedImageError, OSError):
        st.error("File upload tidak valid. Silakan upload file gambar JPG, JPEG, atau PNG.")
