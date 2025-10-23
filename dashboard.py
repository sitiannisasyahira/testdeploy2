import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Siti Annisa Syahira_Laporan 4.pt")  # Model deteksi objek (apel & jeruk)
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/Siti Annisa Syahira_Laporan 2.h5")  # Model klasifikasi daun
    except Exception as e:
        st.error(f"❌ Gagal memuat model klasifikasi daun: {e}")
        classifier = None

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Fungsi Prediksi Daun
# ==========================
def predict_leaf(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = classifier.predict(img_array)
    class_names = ["Daun Sehat", "Daun Tidak Sehat"]

    prob = float(np.max(prediction))
    class_index = np.argmax(prediction)
    label = class_names[class_index]

    color = "🟢" if label == "Daun Sehat" else "🔴"
    return label, prob, color

# ==========================
# Fungsi Prediksi Buah (YOLO)
# ==========================
def detect_fruit(img):
    results = yolo_model(img)
    result_img = results[0].plot()
    detected_classes = [r.names[int(c)] for c in results[0].boxes.cls]
    return result_img, detected_classes

# ==========================
# UI
# ==========================
st.title("🍃 Dashboard Klasifikasi & Deteksi Gambar")
st.markdown("### Proyek UAS — Siti Annisa Syahira")
st.write("Gunakan aplikasi ini untuk mendeteksi **buah (apel/jeruk)** dan **klasifikasi kondisi daun** apakah **sehat** atau **tidak sehat**.")

menu = st.sidebar.radio("Pilih Mode Analisis:", ["🧠 Klasifikasi Daun", "🍎 Deteksi Buah (YOLO)"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar Diupload", use_container_width=True)

    if menu == "🧠 Klasifikasi Daun":
        if classifier is not None:
            label, prob, color = predict_leaf(img)
            st.success(f"Hasil Prediksi: {color} **{label}**")
            st.progress(prob)
            st.write(f"🔍 Probabilitas keyakinan model: `{prob:.2f}`")
        else:
            st.error("Model klasifikasi daun belum dimuat.")

    elif menu == "🍎 Deteksi Buah (YOLO)":
        if yolo_model is not None:
            result_img, detected = detect_fruit(img)
            st.image(result_img, caption="📸 Hasil Deteksi Buah", use_container_width=True)

            if detected:
                st.info(f"Buah yang terdeteksi: **{', '.join(detected)}**")
            else:
                st.warning("Tidak ada buah terdeteksi.")
        else:
            st.error("Model YOLO belum dimuat.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Dibuat oleh **Siti Annisa Syahira** | Proyek UAS 2025 🌿")
