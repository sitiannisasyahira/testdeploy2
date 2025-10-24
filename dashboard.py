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
    page_title="🍎 Dashboard Klasifikasi dan Deteksi Objek Buah & Daun 🌿",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# CSS STYLING
# ==========================
st.markdown("""
    <style>
        /* BODY */
        body {
            background-color: #F8FAFC;
            font-family: 'Poppins', sans-serif;
        }

        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #E0FBE2, #C8E6C9);
            color: #2E7D32;
        }

        [data-testid="stSidebar"] h2 {
            text-align: center;
            font-weight: bold;
            color: #2E7D32;
        }

        [data-testid="stSidebar"] a {
            text-decoration: none;
            font-size: 17px;
            color: #1B5E20;
            padding: 8px 18px;
            display: block;
            border-radius: 12px;
            transition: all 0.3s ease;
        }

        [data-testid="stSidebar"] a:hover {
            background-color: #A5D6A7;
            transform: scale(1.03);
        }

        .main-title {
            text-align: center;
            color: #2E7D32;
            font-size: 38px;
            font-weight: 800;
            margin-bottom: -10px;
        }

        .sub-title {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-bottom: 30px;
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

# ==========================
# LOAD MODEL
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
# FUNGSI PREDIKSI
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
# NAVIGASI SIDEBAR
# ==========================
st.sidebar.title("🌿 Navigasi Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "🔍 Deteksi", "🌿 Klasifikasi", "ℹ️ Tentang Aplikasi"]
)

# ==========================
# HALAMAN BERANDA
# ==========================
if menu == "🏠 Beranda":
    st.markdown("<h1 class='main-title'> 🍎 Dashboard Klasifikasi dan Deteksi Objek Buah & Daun 🌿</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Deteksi Apel & Jeruk 🍎🍊 dan Klasifikasi Daun 🌿</p>", unsafe_allow_html=True)

    st.markdown("### Selamat Datang di Dashboard Klasifikasi dan Deteksi Objek Buah & Daun 👋")
    st.write("""
        Aplikasi ini dibuat oleh **Siti Annisa Syahira (2208108010085** sebagai bagian dari proyek **UTS Praktikum Pemrograman Big Data**.
        Fungsinya adalah untuk:
        - 🔍 **Mendeteksi buah (Apel dan Jeruk)** menggunakan model YOLO (.pt).  
        - 🌿 **Mengklasifikasi daun** apakah **Sehat** atau **Tidak Sehat** menggunakan model Keras (.h5).  
        
    """)
    st.image("https://cdn.pixabay.com/photo/2017/01/20/00/30/orange-1995056_1280.jpg", use_container_width=True)
    st.success("Klik menu **Deteksi & Klasifikasi** di sidebar untuk mulai 🚀")

# ==========================
# HALAMAN DETEKSI & KLASIFIKASI
# ==========================
elif menu == "🔍 Deteksi":
    st.markdown("<h2 style='color:#2E7D32;'>📸 Unggah Gambar untuk Analisis</h2>", unsafe_allow_html=True)
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
elif menu == "🌿 Klasifikasi":
    st.markdown("<h2 style='color:#2E7D32;'>📸 Unggah Gambar untuk Analisis</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
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

# ==========================
# HALAMAN TENTANG
# ==========================
elif menu == "ℹ️ Tentang Aplikasi":
    st.markdown("<h2 style='color:#2E7D32;'>👩‍💻 Tentang Aplikasi</h2>", unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan menggunakan:
        - **Streamlit** untuk antarmuka web interaktif.
        - **YOLO (You Only Look Once)** untuk deteksi objek buah (Apel dan Jeruk).
        - **TensorFlow / Keras** untuk klasifikasi daun (Sehat / Tidak Sehat).

        Model dilatih secara terpisah menggunakan dataset khusus.
        Tujuan aplikasi ini adalah mempermudah analisis cepat terhadap citra buah dan daun 🌿.
    """)
    st.info("Dikembangkan oleh **Siti Annisa Syahira (2025)** | Proyek UTS")

# ==========================
# FOOTER
# ==========================
st.write("---")
st.markdown("<footer>© 2025 | Proyek UTS - Siti Annisa Syahira</footer>", unsafe_allow_html=True)
