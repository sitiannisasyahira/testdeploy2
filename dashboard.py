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
    page_title="🧠 Dashboard UTS - Deteksi & Klasifikasi",
    page_icon="🍎",
    layout="wide"
)

# ==========================
# CSS UNTUK NAVIGASI MELAYANG
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #F8FAFC;
        }

        /* Floating Button */
        .fab-container {
            position: fixed;
            bottom: 40px;
            right: 40px;
            z-index: 9999;
        }

        .fab-button {
            width: 65px;
            height: 65px;
            background-color: #2E8B57;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 28px;
            cursor: pointer;
            box-shadow: 2px 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s ease-in-out;
        }

        .fab-button:hover {
            background-color: #3CB371;
            transform: rotate(90deg);
        }

        /* Menu Items */
        .fab-menu {
            display: none;
            flex-direction: column;
            align-items: center;
            margin-bottom: 10px;
        }

        .fab-menu.show {
            display: flex;
            animation: fadeIn 0.4s ease-in-out;
        }

        .fab-item {
            background-color: white;
            color: #2E8B57;
            font-weight: bold;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 6px 0;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }

        .fab-item:hover {
            background-color: #2E8B57;
            color: white;
            transform: scale(1.1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .title {
            text-align: center;
            color: #1E5631;
            font-size: 36px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("<h1 class='title'>🧠 Dashboard UTS - Deteksi & Klasifikasi Citra</h1>", unsafe_allow_html=True)
st.write("---")

# ==========================
# NAVIGATION HANDLER
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

menu_clicked = st.session_state.page

# ==========================
# FLOATING NAVIGATION MENU
# ==========================
fab_html = """
    <div class="fab-container">
        <div class="fab-menu" id="fabMenu">
            <div class="fab-item" onclick="window.parent.postMessage({type: 'streamlit:setPage', page: 'Beranda'}, '*')">🏠 Beranda</div>
            <div class="fab-item" onclick="window.parent.postMessage({type: 'streamlit:setPage', page: 'Deteksi'}, '*')">🔍 Deteksi</div>
            <div class="fab-item" onclick="window.parent.postMessage({type: 'streamlit:setPage', page: 'Klasifikasi'}, '*')">🌿 Klasifikasi</div>
            <div class="fab-item" onclick="window.parent.postMessage({type: 'streamlit:setPage', page: 'Tentang'}, '*')">ℹ️ Tentang</div>
        </div>
        <div class="fab-button" onclick="toggleMenu()">+</div>
    </div>

    <script>
        function toggleMenu() {
            const menu = window.parent.document.getElementById('fabMenu');
            menu.classList.toggle('show');
        }
        window.addEventListener('message', (event) => {
            if (event.data.type === 'streamlit:setPage') {
                window.parent.postMessage(event.data, '*');
            }
        });
    </script>
"""
st.components.v1.html(fab_html, height=300)

# ==========================
# PAGE CONTENTS
# ==========================
if menu_clicked == "Beranda":
    st.markdown("### 👋 Selamat Datang di Dashboard UTS")
    st.write("""
        Aplikasi ini memanfaatkan **YOLOv8** dan **TensorFlow** untuk mendeteksi objek (buah)
        serta mengklasifikasi daun sehat dan tidak sehat 🌿.  
        Gunakan tombol melayang di kanan bawah untuk berpindah halaman!
    """)
    st.image("https://cdn.pixabay.com/photo/2017/01/20/15/06/apples-1995056_1280.jpg", use_container_width=True)

elif menu_clicked == "Deteksi":
    st.markdown("### 🔍 Deteksi Objek (Apel/Jeruk)")
    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar diunggah", use_container_width=True)
        st.info("Model YOLO akan digunakan untuk deteksi buah 🍎🍊")
    else:
        st.warning("Silakan unggah gambar terlebih dahulu.")

elif menu_clicked == "Klasifikasi":
    st.markdown("### 🌿 Klasifikasi Daun Sehat/Tidak Sehat")
    uploaded_file = st.file_uploader("Unggah gambar daun", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar daun", use_container_width=True)
        st.info("Model TensorFlow akan melakukan klasifikasi daun 🌿")
    else:
        st.warning("Silakan unggah gambar terlebih dahulu.")

elif menu_clicked == "Tentang":
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.write("""
        Dashboard ini dikembangkan oleh **Siti Annisa Syahira (2025)**  
        sebagai bagian dari **Ujian Tengah Semester (UTS)**.  
        Dibangun menggunakan:
        - Streamlit 💻  
        - YOLOv8 🧠  
        - TensorFlow/Keras 🌱  
    """)
    st.success("Gunakan tombol melayang di kanan bawah untuk kembali ke halaman lain.")

# ==========================
# FOOTER
# ==========================
st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 | Dashboard UTS - Siti Annisa Syahira</p>", unsafe_allow_html=True)
