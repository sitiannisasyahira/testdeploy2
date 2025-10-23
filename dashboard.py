import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import base64

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Dashboard UTS - Siti Annisa Syahira", layout="wide")

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Poppins', sans-serif;
    }
    .nav-button {
        display: inline-block;
        margin: 8px;
        padding: 10px 22px;
        background-color: #ffffffcc;
        border-radius: 25px;
        border: 1px solid #ccc;
        color: #333;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background-color: #0078ff;
        color: white;
        transform: scale(1.05);
    }
    .active {
        background-color: #0078ff;
        color: white !important;
    }
    .page-title {
        text-align: center;
        font-weight: 700;
        font-size: 28px;
        color: #0078ff;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .footer {
        text-align:center;
        margin-top:40px;
        color: #666;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Siti Annisa Syahira_Laporan 4.pt")  # model YOLO untuk apel & jeruk
    classifier = tf.keras.models.load_model("model/Siti Annisa Syahira_Laporan 2.h5")  # model daun
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# NAVIGATION SYSTEM
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Beranda"):
        st.session_state.page = "Beranda"
with col2:
    if st.button("🍎 Deteksi Objek"):
        st.session_state.page = "Deteksi"
with col3:
    if st.button("🌿 Klasifikasi"):
        st.session_state.page = "Klasifikasi"
with col4:
    if st.button("ℹ️ Tentang"):
        st.session_state.page = "Tentang"

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================
# PAGE: BERANDA
# ==========================
if st.session_state.page == "Beranda":
    st.markdown("<h1 class='page-title'>🧠 Dashboard UTS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
    st.write("""
    Selamat datang di **Dashboard UTS Siti Annisa Syahira** 🌸  
    Aplikasi ini menggabungkan dua kemampuan:
    1. **Deteksi Objek (YOLO)** untuk mendeteksi **Apel dan Jeruk** berdasarkan model `.pt`.  
    2. **Klasifikasi Gambar (CNN)** untuk menentukan apakah **Daun Sehat atau Tidak Sehat** berdasarkan model `.h5`.

    Silakan pilih menu di atas untuk mencoba fitur-fiturnya 👆
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2909/2909763.png", width=180)
    st.markdown("<div class='footer'>Dibuat dengan ❤️ oleh Siti Annisa Syahira</div>", unsafe_allow_html=True)

# ==========================
# PAGE: DETEKSI OBJEK
# ==========================
elif st.session_state.page == "Deteksi":
    st.markdown("<h1 class='page-title'>🍎 Deteksi Objek Apel & Jeruk</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah gambar buah (apel/jeruk)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        with st.spinner("🔍 Model sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="Hasil Deteksi Objek", use_container_width=True)
        st.success("✅ Deteksi berhasil! Objek teridentifikasi dengan baik.")
        st.balloons()

# ==========================
# PAGE: KLASIFIKASI
# ==========================
elif st.session_state.page == "Klasifikasi":
    st.markdown("<h1 class='page-title'>🌿 Klasifikasi Daun Sehat / Tidak Sehat</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar daun diunggah", use_container_width=True)

            # Ambil ukuran input model otomatis
            try:
                input_shape = classifier.input_shape
                target_size = (input_shape[1], input_shape[2])
            except Exception:
                target_size = (224, 224)

            img_resized = img.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            with st.spinner("🧠 Model sedang menganalisis..."):
                prediction = classifier.predict(img_array)

            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            # Tentukan label
            if class_index == 0:
                label = "🌱 Daun Sehat"
            else:
                label = "🍂 Daun Tidak Sehat"

            st.success(f"**Hasil Prediksi:** {label}")
            st.progress(confidence / 100)
            st.write(f"Tingkat keyakinan: {confidence:.2f}%")

            st.balloons()
            st.snow()
            st.markdown("<div style='text-align:center; font-size:20px;'><b>Selamat!</b> 🎉 Klasifikasi berhasil dilakukan!</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# ==========================
# PAGE: TENTANG
# ==========================
elif st.session_state.page == "Tentang":
    st.markdown("<h1 class='page-title'>ℹ️ Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.write("""
    Aplikasi ini dibuat untuk memenuhi tugas **Ujian Tengah Semester (UTS)**  
    pada mata kuliah **Pemrograman Big Data**.  
    Dibangun menggunakan framework **Streamlit**, dengan dua model utama:
    - **YOLOv8**: Deteksi objek (Apel & Jeruk)  
    - **CNN (Keras/TensorFlow)**: Klasifikasi daun sehat / tidak sehat  

    **Mahasiswi:** Siti Annisa Syahira  
    **NPM:** 2208108010085 
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/4072/4072581.png", width=180)
    st.markdown("<div class='footer'>© 2025 Dashboard UTS - All Rights Reserved</div>", unsafe_allow_html=True)
