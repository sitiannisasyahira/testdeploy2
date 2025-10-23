import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import os

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="🧠 Dashboard UTS", page_icon="🍃", layout="wide")

# ==========================
# CSS - GLASS MORPHISM NAVIGATION
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #b4f1b4, #ffffff);
        font-family: "Poppins", sans-serif;
    }

    /* Navigasi atas */
    .glass-nav {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        display: flex;
        justify-content: center;
        padding: 15px 0;
        z-index: 999;
    }

    .nav-item {
        margin: 0 25px;
        font-size: 18px;
        font-weight: 600;
        color: #1b4332;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .nav-item:hover {
        color: #2d6a4f;
        transform: scale(1.1);
        text-shadow: 0 0 10px #b7e4c7;
    }

    .active {
        color: #40916c;
        border-bottom: 3px solid #2d6a4f;
        padding-bottom: 5px;
    }

    /* Title */
    .page-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        margin-top: 100px;
        color: #1b4332;
        text-shadow: 1px 1px 8px rgba(0,0,0,0.1);
    }

    .upload-box {
        border: 2px dashed #40916c;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        transition: 0.3s;
    }

    .upload-box:hover {
        background-color: rgba(64,145,108,0.05);
    }

    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 40px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# NAVIGATION BAR
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

nav_html = f"""
<div class="glass-nav">
    <div class="nav-item {'active' if st.session_state.page == 'Beranda' else ''}" 
         onclick="window.parent.postMessage({{'page': 'Beranda'}}, '*')">🏠 Beranda</div>
    <div class="nav-item {'active' if st.session_state.page == 'Deteksi' else ''}" 
         onclick="window.parent.postMessage({{'page': 'Deteksi'}}, '*')">🍎 Deteksi Buah</div>
    <div class="nav-item {'active' if st.session_state.page == 'Klasifikasi' else ''}" 
         onclick="window.parent.postMessage({{'page': 'Klasifikasi'}}, '*')">🌿 Klasifikasi Daun</div>
    <div class="nav-item {'active' if st.session_state.page == 'Tentang' else ''}" 
         onclick="window.parent.postMessage({{'page': 'Tentang'}}, '*')">ℹ️ Tentang</div>
</div>

<script>
    window.addEventListener('message', (event) => {{
        if (event.data.page) {{
            window.parent.postMessage({{type: 'streamlit:setPage', page: event.data.page}}, '*');
        }}
    }});
</script>
"""
st.components.v1.html(nav_html, height=80)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Siti Annisa Syahira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Siti Annisa Syahira_Laporan 2.h5")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error("⚠️ Model tidak ditemukan! Pastikan file model sudah diunggah ke folder `model/`.")
    st.stop()

# ==========================
# PAGE CONTENTS
# ==========================
page = st.session_state.page

# BERANDA
if page == "Beranda":
    st.markdown("<h1 class='page-title'>🧠 Dashboard UTS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
    st.write("""
        Selamat datang di **Dashboard UTS** oleh *Siti Annisa Syahira* 🌿  
        Aplikasi ini menggunakan **YOLOv8** untuk deteksi buah (apel & jeruk)  
        dan **TensorFlow/Keras** untuk klasifikasi daun sehat atau tidak sehat.
    """)
    st.image("https://cdn.pixabay.com/photo/2016/03/05/19/02/apple-1239423_1280.jpg", use_container_width=True)
    st.success("Gunakan menu di atas untuk berpindah ke halaman deteksi atau klasifikasi.")

# DETEKSI BUAH
elif page == "Deteksi":
    st.markdown("<h1 class='page-title'>🍎 Deteksi Buah (Apel & Jeruk)</h1>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Unggah gambar buah untuk deteksi", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
            with st.spinner("🔍 Mendeteksi buah..."):
                results = yolo_model(img)
                result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi YOLOv8", use_container_width=True)
            st.success("✅ Deteksi selesai!")
        else:
            st.info("Silakan unggah gambar untuk memulai deteksi.")

# KLASIFIKASI DAUN
elif page == "Klasifikasi":
    st.markdown("<h1 class='page-title'>🌿 Klasifikasi Daun Sehat / Tidak Sehat</h1>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Unggah gambar daun untuk klasifikasi", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar daun diunggah", use_container_width=True)

            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            label = "🌱 Daun Sehat" if np.argmax(prediction) == 0 else "🍂 Daun Tidak Sehat"
            confidence = np.max(prediction) * 100

            st.success(f"**Hasil Prediksi:** {label}")
            st.progress(float(confidence) / 100)
            st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")
        else:
            st.info("Unggah gambar daun untuk memulai klasifikasi.")

# TENTANG
elif page == "Tentang":
    st.markdown("<h1 class='page-title'>ℹ️ Tentang Dashboard Ini</h1>", unsafe_allow_html=True)
    st.write("""
        Dashboard ini merupakan proyek **Ujian Tengah Semester (UTS)**  
        oleh **Siti Annisa Syahira**, menggunakan teknologi:
        - 💡 Streamlit untuk antarmuka interaktif  
        - 🧠 YOLOv8 untuk deteksi buah (apel & jeruk)  
        - 🌿 TensorFlow/Keras untuk klasifikasi daun sehat / tidak sehat  
    """)
    st.info("✨ Desain bergaya Glass Morphism agar tampil elegan, ringan, dan profesional.")

# ==========================
# FOOTER
# ==========================
st.markdown("<p class='footer'>© 2025 | Dashboard UTS - Siti Annisa Syahira 🍃</p>", unsafe_allow_html=True)
