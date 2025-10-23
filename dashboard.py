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
    page_title="🍎 Dashboard UAS - Deteksi & Klasifikasi",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# CSS STYLING
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #F8FAFC;
        }
        .title {
            text-align: center;
            color: #2E8B57;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: gray;
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

st.markdown("<h1 class='title'>🧠 Dashboard UAS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi Apel & Jeruk 🍎🍊 dan Klasifikasi Daun 🌿</p>", unsafe_allow_html=True)

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
    input_shape = classifier.input_shape  # (None, H, W, C)
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
# NAVIGASI TABS
# ==========================
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "🔍 Deteksi & Klasifikasi", "ℹ️ Tentang Aplikasi"])

# ==========================
# TAB 1 - BERANDA
# ==========================
with tab1:
    st.markdown("### Selamat Datang di Dashboard Proyek UAS 👋")
    st.write("""
        Aplikasi ini dibuat oleh **Siti Annisa Syahira** sebagai bagian dari proyek UAS.
        Fungsinya adalah untuk:
        - 🔍 **Mendeteksi buah (Apel dan Jeruk)** menggunakan model YOLO (.pt).  
        - 🌿 **Mengklasifikasi daun** apakah **Sehat** atau **Tidak Sehat** menggunakan model Keras (.h5).  
        
        Dashboard ini interaktif dan dirancang agar mudah digunakan serta menarik untuk presentasi.
    """)
    st.image("https://cdn.pixabay.com/photo/2017/01/20/00/30/orange-1995056_1280.jpg", use_container_width=True)
    st.success("Klik tab **Deteksi & Klasifikasi** di atas untuk mulai menggunakan aplikasi ini 🚀")

# ==========================
# TAB 2 - DETEKSI & KLASIFIKASI
# ==========================
with tab2:
    st.markdown("### 📸 Unggah Gambar untuk Analisis")

    mode = st.selectbox("Pilih Mode Analisis:", ["Deteksi Objek (Apel/Jeruk)", "Klasifikasi Daun"])
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

# ==========================
# TAB 3 - TENTANG
# ==========================
with tab3:
    st.markdown("### 👩‍💻 Tentang Aplikasi")
    st.write("""
        Aplikasi ini dikembangkan menggunakan:
        - **Streamlit** untuk antarmuka web.
        - **YOLO (You Only Look Once)** untuk deteksi objek buah (Apel dan Jeruk).
        - **TensorFlow / Keras** untuk klasifikasi daun (Sehat / Tidak Sehat).

        Model YOLO dan Keras dilatih secara terpisah menggunakan dataset khusus.
        Tujuan aplikasi ini adalah mempermudah analisis cepat terhadap citra buah dan daun.
    """)
    st.info("Dikembangkan oleh **Siti Annisa Syahira (2025)** | Proyek UAS")

# ==========================
# FOOTER
# ==========================
st.write("---")
st.markdown("<footer>© 2025 | Proyek UAS - Siti Annisa Syahira</footer>", unsafe_allow_html=True)
