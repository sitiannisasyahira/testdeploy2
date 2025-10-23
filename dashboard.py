import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Halaman
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
# Load Model
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Siti Annisa Syahira_Laporan 4.pt")  # model deteksi apel & jeruk
    classifier = tf.keras.models.load_model("model/Siti Annisa Syahira_Laporan 2.h5")  # model klasifikasi daun
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar
# ==========================
menu = st.sidebar.radio("📂 Pilih Mode Analisis:", ["Deteksi Objek (Apel/Jeruk)", "Klasifikasi Daun"])
st.sidebar.info("Gunakan mode yang sesuai dengan data yang ingin kamu analisis 👇")
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Fungsi Klasifikasi
# ==========================
def predict_leaf(image_pil):
    img_resized = image_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = classifier.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = "🌿 Daun Sehat" if class_index == 0 else "🍂 Daun Tidak Sehat"
    color = "green" if class_index == 0 else "red"

    return label, confidence, color

# ==========================
# Bagian Utama
# ==========================
col1, col2 = st.columns(2)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1.image(img, caption="📸 Gambar Asli", use_container_width=True)

    if menu == "Deteksi Objek (Apel/Jeruk)":
        with st.spinner("🔍 Mendeteksi objek dengan YOLO..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil deteksi (dengan box)
            col2.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

            # tampilkan label & confidence tiap deteksi
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
# Footer
# ==========================
st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 | Proyek UAS - Siti Annisa Syahira</p>", unsafe_allow_html=True)
