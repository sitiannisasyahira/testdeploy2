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
    page_title="🍎 Dashboard Klasifikasi dan Deteksi Objek Buah & Daun 🍃",
    page_icon="🍃",
    layout="wide"
)

# ==========================
# CSS + EFEK DAUN JATUH
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #F8FAFC;
            overflow-x: hidden;
        }
        .title {
            text-align: center;
            color: #2E8B57;
            font-size: 38px;
            font-weight: bold;
            margin-top: 20px;
        }
        .subtitle {
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
        /* Efek daun jatuh */
        .leaf {
            position: fixed;
            top: -10px;
            animation: fall linear infinite;
            opacity: 0.8;
        }
        @keyframes fall {
            0% { transform: translateY(0) rotate(0deg); opacity: 1; }
            100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
        }
    </style>

    <script>
        const emojis = ['🍃','🍂','🌿','🍁'];
        for(let i=0;i<15;i++){
            let leaf = document.createElement('div');
            leaf.innerHTML = emojis[Math.floor(Math.random()*emojis.length)];
            leaf.classList.add('leaf');
            leaf.style.left = Math.random()*100 + 'vw';
            leaf.style.fontSize = (20 + Math.random()*30) + 'px';
            leaf.style.animationDuration = (5 + Math.random()*5) + 's';
            document.body.appendChild(leaf);
        }
    </script>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("<h1 class='title'>🍎 Dashboard Klasifikasi dan Deteksi Objek Buah & Daun 🍃</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis citra Apel, Jeruk, dan Klasifikasi Daun Sehat / Tidak Sehat</p>", unsafe_allow_html=True)

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
# FUNGSI KLASIFIKASI
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
# NAVIGASI UTAMA (4 MENU)
# ==========================
st.sidebar.title("🌿 Navigasi Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "🔍 Deteksi", "🌿 Klasifikasi", "ℹ️ Tentang Aplikasi"]
)

# ==========================
# BERANDA
# ==========================
if menu == "🏠 Beranda":
    st.markdown("### Selamat Datang di Dashboard 🍎🌿")
    st.write("""
        Aplikasi ini dibuat oleh **Siti Annisa Syahira (2208108010085)** sebagai proyek **UTS Pemrograman Big Data**.  
        Fungsinya adalah untuk:
        - 🔍 **Mendeteksi buah (Apel & Jeruk)** menggunakan model YOLO (.pt)  
        - 🌿 **Mengklasifikasi daun** (Sehat / Tidak Sehat) menggunakan model Keras (.h5)
    """)
    st.image("https://cdn.pixabay.com/photo/2017/01/20/00/30/orange-1995056_1280.jpg", use_container_width=True)
    st.success("Pilih menu di sidebar untuk memulai 🚀")

# ==========================
# DETEKSI BUAH
# ==========================
elif menu == "🔍 Deteksi":
    st.markdown("### 🔍 Deteksi Buah")
    uploaded_file = st.file_uploader("Unggah gambar buah", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Gambar Asli", use_container_width=True)

        with st.spinner("🔎 Mendeteksi buah dengan YOLO..."):
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
    else:
        st.info("⬆️ Silakan unggah gambar buah terlebih dahulu.")

# ==========================
# KLASIFIKASI DAUN
# ==========================
elif menu == "🌿 Klasifikasi":
    st.markdown("### 🌿 Klasifikasi Daun")
    uploaded_leaf = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])

    if uploaded_leaf:
        img_leaf = Image.open(uploaded_leaf)
        col1, col2 = st.columns(2)
        col1.image(img_leaf, caption="Gambar Daun Asli", use_container_width=True)

        with st.spinner("🧬 Menganalisis kondisi daun..."):
            label, confidence, color = predict_leaf(img_leaf)
            col2.markdown(
                f"<div class='result-box'><h3 style='color:{color};'>{label}</h3>"
                f"<p>Probabilitas: <b>{confidence:.2f}</b></p></div>",
                unsafe_allow_html=True,
            )
            st.balloons()
    else:
        st.info("⬆️ Silakan unggah gambar daun terlebih dahulu.")

# ==========================
# TENTANG APLIKASI
# ==========================
elif menu == "ℹ️ Tentang Aplikasi":
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.write("""
        Aplikasi ini dikembangkan menggunakan:
        -  **Streamlit** untuk UI interaktif  
        -  **YOLOv8** untuk deteksi buah  
        -  **TensorFlow/Keras** untuk klasifikasi daun  

        🎓 Proyek UTS Pemrograman Big Data  
        👩‍💻 **Dikembangkan oleh:** Siti Annisa Syahira (2025)
    """)

# ==========================
# FOOTER
# ==========================
st.write("---")
st.markdown("<footer>© 2025 | 🍎 Dashboard Klasifikasi & Deteksi Buah & Daun 🍃 | Siti Annisa Syahira</footer>", unsafe_allow_html=True)
