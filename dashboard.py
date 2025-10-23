import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Dashboard UTS - Siti Annisa Syahira", layout="wide")

# ==========================
# CUSTOM STYLE (efek daun)
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f8fff8 0%, #d8f3dc 100%);
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
        background-color: #2d6a4f;
        color: white;
        transform: scale(1.05);
    }
    .page-title {
        text-align: center;
        font-weight: 700;
        font-size: 28px;
        color: #2d6a4f;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .footer {
        text-align:center;
        margin-top:40px;
        color: #666;
        font-size: 14px;
    }

    /* Efek daun jatuh */
    @keyframes fall {
      0% {transform: translateY(-10vh) rotate(0deg);}
      100% {transform: translateY(100vh) rotate(360deg);}
    }
    .leaf {
      position: fixed;
      top: -10vh;
      font-size: 24px;
      animation: fall linear infinite;
      opacity: 0.8;
      z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Siti Annisa Syahira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Siti Annisa Syahira_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# NAVIGASI
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
# BERANDA
# ==========================
if st.session_state.page == "Beranda":
    st.markdown("<h1 class='page-title'>🧠 Dashboard UTS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
    st.write("""
    Selamat datang di **Dashboard UTS Siti Annisa Syahira** 🌸  
    Aplikasi ini memadukan:
    - **Deteksi Objek (YOLOv8)** → Apel 🍎 & Jeruk 🍊  
    - **Klasifikasi Gambar (CNN)** → Daun Sehat 🌱 atau Tidak Sehat 🍂  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=200)
    st.markdown("<div class='footer'>Dibuat dengan ❤️ oleh Siti Annisa Syahira</div>", unsafe_allow_html=True)

# ==========================
# DETEKSI OBJEK
# ==========================
elif st.session_state.page == "Deteksi":
    st.markdown("<h1 class='page-title'>🍎 Deteksi Objek Apel & Jeruk</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah gambar buah (apel/jeruk)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar diunggah", use_container_width=True)

        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        st.success("✅ Deteksi selesai! Objek berhasil ditemukan.")

# ==========================
# KLASIFIKASI
# ==========================
elif st.session_state.page == "Klasifikasi":
    st.markdown("<h1 class='page-title'>🌿 Klasifikasi Daun Sehat / Tidak Sehat</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar daun diunggah", use_container_width=True)

            try:
                input_shape = classifier.input_shape
                target_size = (input_shape[1], input_shape[2])
            except Exception:
                target_size = (224, 224)

            img_resized = img.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            with st.spinner("🧠 Menganalisis kondisi daun..."):
                prediction = classifier.predict(img_array)

            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            if class_index == 0:
                label = "🌱 Daun Sehat"
            else:
                label = "🍂 Daun Tidak Sehat"

            st.success(f"**Hasil Prediksi:** {label}")
            st.progress(confidence / 100)
            st.write(f"Tingkat keyakinan: {confidence:.2f}%")

            # 🌿 Efek daun jatuh
            leaf_animation = "".join([
                f"<div class='leaf' style='left:{np.random.randint(0,100)}%; animation-duration:{np.random.uniform(5,10)}s;'>🍃</div>"
                for _ in range(25)
            ])
            st.markdown(leaf_animation, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# ==========================
# TENTANG
# ==========================
elif st.session_state.page == "Tentang":
    st.markdown("<h1 class='page-title'>ℹ️ Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.write("""
    Aplikasi ini merupakan proyek **UTS Praktikum Pemrograman Big Data** oleh **Siti Annisa Syahira (2208108010085)**.  
    Fitur utama:
    - Deteksi buah Apel & Jeruk dengan model YOLOv8  
    - Klasifikasi daun sehat atau tidak sehat dengan CNN (TensorFlow)  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/765/765613.png", width=160)
    st.markdown("<div class='footer'>© 2025 Dashboard UTS - All Rights Reserved</div>", unsafe_allow_html=True)
