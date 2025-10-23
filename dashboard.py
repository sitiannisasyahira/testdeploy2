import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="🍎 Dashboard Deteksi & Klasifikasi",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
        .title {text-align: center; color: #2E8B57;}
        .subtitle {text-align: center; font-size:18px; color: gray;}
        .result-box {
            padding: 15px; 
            border-radius: 10px; 
            background-color: #f0f2f6;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🧠 Dashboard Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi objek (Apel/Jeruk) dan Klasifikasi daun (Sehat/Tidak Sehat)</p>", unsafe_allow_html=True)
st.write("---")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/Siti Annisa Syahira_Laporan 4.pt"
    h5_path = "model/Siti Annisa Syahira_Laporan 2.h5"

    # Cek file
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {os.path.abspath(yolo_path)}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model klasifikasi (.h5) tidak ditemukan di: {os.path.abspath(h5_path)}")

    # Load model YOLO dan Keras
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(h5_path)
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
    st.success("✅ Semua model berhasil dimuat!")
    st.write("📏 Input shape model klasifikasi:", classifier.input_shape)
except Exception as e:
    st.error(f"🚨 Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# ==========================
# SIDEBAR
# ==========================
menu = st.sidebar.radio("📂 Pilih Mode Analisis:", ["Deteksi Objek (Apel/Jeruk)", "Klasifikasi Daun"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.info("Gunakan mode yang sesuai dengan data yang ingin kamu analisis 👇")

# ==========================
# FUNGSI KLASIFIKASI DAUN (OTOMATIS SESUAI SHAPE MODEL)
# ==========================
def predict_leaf(image_pil):
    input_shape = classifier.input_shape  # (None, H, W, C)
    target_size = (input_shape[1], input_shape[2])

    # Jika model pakai 1 channel (grayscale)
    if input_shape[3] == 1:
        img = image_pil.convert("L")
    else:
        img = image_pil.convert("RGB")

    # Resize gambar sesuai model
    img_resized = img.resize(target_size)
    img_array = image.img_to_array(img_resized)

    # Kalau model butuh 1 channel tapi input 3, ubah dimensi
    if input_shape[3] == 1 and img_array.ndim == 3:
        img_array = np.expand_dims(img_array[:, :, 0], axis=-1)

    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = classifier.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = "🌿 Daun Sehat" if class_index == 0 else "🍂 Daun Tidak Sehat"
    color = "green" if class_index == 0 else "red"

    return label, confidence, color

# ==========================
# HALAMAN UTAMA
# ==========================
col1, col2 = st.columns(2)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1.image(img, caption="📸 Gambar Asli", use_container_width=True)

    if menu == "Deteksi Objek (Apel/Jeruk)":
        with st.spinner("🔍 Mendeteksi objek dengan YOLO..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            col2.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

            st.subheader("📊 Detail Deteksi:")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls_id]
                st.markdown(f"- **Objek:** {label} | **Akurasi:** `{conf:.2f}`")

            st.success("✅ Deteksi selesai!")

    elif menu == "Klasifikasi Daun":
        with st.spinner("🧬 Menganalisis kondisi daun..."):
            label, confidence, color = predict_leaf(img)
            col2.markdown(f"<div class='result-box'><h3 style='color:{color};'>{label}</h3>"
                          f"<p>Probabilitas: <b>{confidence:.2f}</b></p></div>", unsafe_allow_html=True)
            st.balloons()
else:
    st.info("⬅️ Silakan unggah gambar terlebih dahulu melalui sidebar.")

# ==========================
# FOOTER
# ==========================
st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 | Proyek UAS - Siti Annisa Syahira</p>", unsafe_allow_html=True)
