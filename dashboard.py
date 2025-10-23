import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🧠 Dashboard UTS", page_icon="🌿", layout="wide")

# ==========================
# CUSTOM CSS (NAVIGASI ELEGAN)
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5fff5, #e8f5e9);
        font-family: "Poppins", sans-serif;
    }

    /* NAVIGATION BAR */
    .navbar {
        position: fixed;
        top: 0;
        width: 100%;
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(15px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 50px;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        z-index: 9999;
    }

    .nav-title {
        font-size: 22px;
        font-weight: 700;
        color: #1b4332;
        letter-spacing: 1px;
    }

    .nav-links {
        display: flex;
        gap: 40px;
    }

    .nav-link {
        font-size: 17px;
        font-weight: 500;
        color: #2d6a4f;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .nav-link:hover {
        color: #40916c;
        transform: translateY(-2px);
    }

    .active-link {
        border-bottom: 3px solid #2d6a4f;
        padding-bottom: 4px;
        color: #1b4332;
        font-weight: 600;
    }

    /* CONTENT */
    .page-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        margin-top: 100px;
        color: #1b4332;
    }

    .subtext {
        text-align: center;
        color: #555;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .upload-box {
        border: 2px dashed #74c69d;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background-color: rgba(255,255,255,0.5);
        transition: all 0.3s;
    }

    .upload-box:hover {
        background-color: rgba(200,255,200,0.3);
    }

    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 60px;
        font-size: 14px;
        padding-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# NAVIGATION
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

selected = st.session_state.page

nav_html = f"""
<div class="navbar">
    <div class="nav-title">🌿 Dashboard UTS</div>
    <div class="nav-links">
        <a class="nav-link {'active-link' if selected == 'Beranda' else ''}" 
           onclick="window.parent.postMessage({{'page':'Beranda'}}, '*')">🏠 Beranda</a>
        <a class="nav-link {'active-link' if selected == 'Deteksi' else ''}" 
           onclick="window.parent.postMessage({{'page':'Deteksi'}}, '*')">🍎 Deteksi Buah</a>
        <a class="nav-link {'active-link' if selected == 'Klasifikasi' else ''}" 
           onclick="window.parent.postMessage({{'page':'Klasifikasi'}}, '*')">🌿 Klasifikasi Daun</a>
        <a class="nav-link {'active-link' if selected == 'Tentang' else ''}" 
           onclick="window.parent.postMessage({{'page':'Tentang'}}, '*')">ℹ️ Tentang</a>
    </div>
</div>

<script>
window.addEventListener('message', (event) => {{
    if (event.data.page) {{
        window.parent.postMessage({{type:'streamlit:setPage', page:event.data.page}}, '*');
    }}
}});
</script>
"""
st.components.v1.html(nav_html, height=70)

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
except:
    st.error("⚠️ Model tidak ditemukan! Pastikan file model ada di folder `model/`.")
    st.stop()

# ==========================
# PAGE CONTENT
# ==========================
page = st.session_state.page

if page == "Beranda":
    st.markdown("<h1 class='page-title'>🧠 Dashboard UTS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtext'>Dikembangkan oleh <b>Siti Annisa Syahira</b> | Menggunakan YOLOv8 dan TensorFlow</p>", unsafe_allow_html=True)
    st.image("https://cdn.pixabay.com/photo/2018/08/06/22/24/apple-3580668_1280.jpg", use_container_width=True)
    st.info("Gunakan menu di atas untuk melakukan deteksi atau klasifikasi gambar 🌿")

elif page == "Deteksi":
    st.markdown("<h1 class='page-title'>🍎 Deteksi Buah (Apel & Jeruk)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtext'>Gunakan model YOLOv8 untuk mendeteksi objek pada gambar buah.</p>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Unggah gambar buah", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar diunggah", use_container_width=True)
            with st.spinner("🔍 Mendeteksi buah..."):
                results = yolo_model(img)
                result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi YOLOv8", use_container_width=True)
            st.success("✅ Deteksi selesai!")

elif page == "Klasifikasi":
    st.markdown("<h1 class='page-title'>🌿 Klasifikasi Daun Sehat / Tidak Sehat</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtext'>Gunakan model TensorFlow untuk menentukan kondisi daun.</p>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar daun diunggah", use_container_width=True)
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            label = "🌱 Daun Sehat" if np.argmax(prediction) == 0 else "🍂 Daun Tidak Sehat"
            conf = np.max(prediction) * 100
            st.success(f"**Hasil Prediksi:** {label}")
            st.progress(float(conf)/100)
            st.write(f"Tingkat keyakinan: {conf:.2f}%")

elif page == "Tentang":
    st.markdown("<h1 class='page-title'>ℹ️ Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dikembangkan sebagai **proyek UTS** oleh *Siti Annisa Syahira*.  
        Menggunakan:
        - YOLOv8 untuk deteksi objek buah 🍎  
        - TensorFlow/Keras untuk klasifikasi daun 🌿  
        - Streamlit sebagai platform dashboard interaktif 🎨
    """)
    st.info("Desain elegan dengan glass-effect, menonjolkan kesederhanaan & profesionalisme.")

st.markdown("<p class='footer'>© 2025 | Dashboard UTS - Siti Annisa Syahira 🌱</p>", unsafe_allow_html=True)
