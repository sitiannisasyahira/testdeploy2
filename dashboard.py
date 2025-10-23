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
    page_title="🍎 Dashboard UTS - Deteksi & Klasifikasi",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# CSS STYLING UNTUK NAVBAR
# ==========================
st.markdown("""
    <style>
        /* Global Styles */
        body {
            background-color: #F8FAFC;
        }
        .title {
            text-align: center;
            color: #1E5631;
            font-size: 38px;
            font-weight: bold;
            margin-top: 10px;
        }
        .subtitle {
            text-align: center;
            color: #5f5f5f;
            font-size: 18px;
            margin-bottom: 30px;
        }

        /* Navbar Styling */
        .nav-container {
            background-color: #2E8B57;
            padding: 12px 0;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
        }
        .nav-item {
            display: inline-block;
            margin: 0 30px;
            font-size: 18px;
            font-weight: 500;
            color: white;
            text-decoration: none;
            transition: all 0.3s ease-in-out;
        }
        .nav-item:hover {
            color: #FFD700;
            transform: scale(1.1);
        }
        .active {
            color: #FFD700;
            border-bottom: 3px solid #FFD700;
            padding-bottom: 5px;
        }

        .result-box {
            padding: 20px; 
            border-radius: 15px; 
            background-color: #f0f2f6;
            text-align: center;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
        footer {
            text-align: center;
            color: gray;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🧠 Dashboard UTS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi Apel & Jeruk 🍎🍊 serta Klasifikasi Daun 🌿</p>", unsafe_allow_html=True)

# ==========================
# NAVBAR CUSTOM
# ==========================
pages = ["Beranda", "Deteksi & Klasifikasi", "Tentang Aplikasi"]
page = st.session_state.get("page", "Beranda")

cols = st.columns(len(pages))
for i, p in enumerate(pages):
    if cols[i].button(p):
        st.session_state.page = p
        page = p

# ==========================
# LOAD MODEL DENGAN AMAN
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/Siti Annisa Syahira_Laporan 4.pt"
    h5_path = "model/Siti Annisa Syahira_Laporan 2.h5"

    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"❌ File YOLO tidak ditemukan di: {os.path.abspath(yolo_path)}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"❌ File H5 tidak ditemukan di: {os.path.abspath(h5_path)}")

    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(h5_path)
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================
# FUNGSI PREDIKSI DAUN
# ==========================
def predict_leaf(image_pil):
    input_shape = classifier.input_shape
    target_size = (input_shape[1], input_shape[2])

    if input_shape[3] == 1:
        img = image_pil.convert("L")
    else:
        img = image_pil.convert("RGB")

    img_resized = img.resize(target_size)
    img_array = image.img_to_array(img_resized)

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
# ISI HALAMAN
# ==========================
if page == "Beranda":
    st.markdown("### 👋 Selamat Datang di Dashboard UTS")
    st.write("""
        Aplikasi ini dikembangkan oleh **Siti Annisa Syahira** untuk memenuhi tugas **UTS**.
        Fitur yang tersedia:
        - 🍎 Deteksi **buah Apel dan Jeruk** menggunakan YOLO (.pt)
        - 🌿 Klasifikasi **daun sehat / tidak sehat** menggunakan model TensorFlow (.h5)
        
        Desain ini dibuat agar tampil **menarik, elegan, dan profesional**.
    """)
    st.image("https://cdn.pixabay.com/photo/2016/03/05/22/32/apple-1239429_1280.jpg", use_container_width=True)
    st.info("Klik tombol navigasi di atas untuk mulai menggunakan dashboard ini 🚀")

elif page == "Deteksi & Klasifikasi":
    st.markdown("### 📸 Unggah Gambar untuk Analisis")

    mode = st.radio("Pilih Mode:", ["Deteksi Objek (Apel/Jeruk)", "Klasifikasi Daun"], horizontal=True)
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1.image(img, caption="Gambar Asli", use_container_width=True)

        if mode == "Deteksi Objek (Apel/Jeruk)":
            with st.spinner("🔎 Mendeteksi objek dengan YOLO..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                col2.image(result_img, caption="Hasil Deteksi", use_container_width=True)

                st.subheader("📊 Detail Deteksi:")
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = results[0].names[cls_id]
                    st.markdown(f"- **Objek:** {label} | **Akurasi:** `{conf:.2f}`")
                st.success("✅ Deteksi selesai!")

        elif mode == "Klasifikasi Daun":
            with st.spinner("🧬 Menganalisis kondisi daun..."):
                label, confidence, color = predict_leaf(img)
                col2.markdown(
                    f"<div class='result-box'><h3 style='color:{color};'>{label}</h3>"
                    f"<p>Probabilitas: <b>{confidence:.2f}</b></p></div>",
                    unsafe_allow_html=True,
                )
                st.balloons()
    else:
        st.info("⬆️ Silakan unggah gambar terlebih dahulu untuk melanjutkan.")

elif page == "Tentang Aplikasi":
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.write("""
        Dashboard ini menggunakan teknologi:
        - **Streamlit** untuk antarmuka web interaktif  
        - **YOLOv8** untuk deteksi objek buah  
        - **TensorFlow/Keras** untuk klasifikasi daun  

        Tujuan utama aplikasi ini adalah membantu identifikasi visual sederhana dengan tampilan yang modern dan mudah digunakan.
    """)
    st.success("Dikembangkan oleh **Siti Annisa Syahira (2025)** | Proyek UTS")

# ==========================
# FOOTER
# ==========================
st.write("---")
st.markdown("<footer>© 2025 | Proyek UTS - Siti Annisa Syahira</footer>", unsafe_allow_html=True)
