# dashboard_uts.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from ultralytics import YOLO
import io
import os
import traceback

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Dashboard UTS - Deteksi & Klasifikasi", layout="wide")

# -----------------------
# Helper: safe load models
# -----------------------
@st.cache_resource
def load_models(yolo_path="model/Siti Annisa Syahira_Laporan 4.pt",
                h5_path="model/Siti Annisa Syahira_Laporan 2.h5"):
    # check files
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO file not found: {os.path.abspath(yolo_path)}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {os.path.abspath(h5_path)}")

    yolo = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(h5_path)
    return yolo, classifier

# -----------------------
# Robust preprocessing using classifier.input_shape
# -----------------------
def preprocess_for_classifier(pil_img, classifier):
    """
    Return numpy array shaped (1, H, W, C) matching classifier.input_shape and scaled [0,1].
    """
    input_shape = classifier.input_shape  # typically (None, H, W, C)
    if not (isinstance(input_shape, tuple) and len(input_shape) >= 4):
        # fallback to 224x224x3
        target_h, target_w, target_c = 224, 224, 3
    else:
        _, target_h, target_w, target_c = input_shape

        # some old models show None or 0 for channels; normalize to 3
        if target_c is None or target_c == 0:
            target_c = 3

    # convert color
    if target_c == 1:
        img = pil_img.convert("L")
    else:
        img = pil_img.convert("RGB")

    img = img.resize((int(target_w), int(target_h)))
    arr = kimage.img_to_array(img)

    # If classifier expects 1 channel but array is HxWx3 -> reduce
    if target_c == 1 and arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, 0:1]

    # If array lacks channel dim, expand
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)

    arr = np.expand_dims(arr, axis=0).astype("float32") / 255.0
    return arr

# -----------------------
# UI state
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

def set_page(p):
    st.session_state.page = p

# -----------------------
# Top navbar (simple buttons)
# -----------------------
cols = st.columns([1,1,1,1])
with cols[0]:
    if st.button("🏠 Beranda"):
        set_page("Beranda")
with cols[1]:
    if st.button("🍎 Deteksi Buah"):
        set_page("Deteksi")
with cols[2]:
    if st.button("🌿 Klasifikasi Daun"):
        set_page("Klasifikasi")
with cols[3]:
    if st.button("ℹ️ Tentang"):
        set_page("Tentang")
st.write("---")

# -----------------------
# Load models (with friendly errors)
# -----------------------
try:
    yolo_model, classifier = load_models()
    st.success("✅ Model berhasil dimuat")
    st.write("Input shape classifier:", classifier.input_shape)
except Exception as e:
    st.error("🚨 Gagal memuat model. Lihat detail di bawah.")
    st.exception(e)
    st.stop()

# -----------------------
# Pages
# -----------------------
page = st.session_state.page

if page == "Beranda":
    st.header("🌿 Dashboard UTS — Deteksi & Klasifikasi Citra")
    st.write("Oleh: Siti Annisa Syahira — gunakan menu di atas untuk berpindah halaman.")
    st.markdown("**Fitur utama:**")
    st.markdown("- Deteksi buah (Apel & Jeruk) menggunakan YOLOv8")
    st.markdown("- Klasifikasi daun sehat / tidak sehat menggunakan model Keras (.h5)")
    # some nice layout
    c1, c2 = st.columns([2,1])
    with c1:
        st.image("https://cdn.pixabay.com/photo/2018/08/06/22/24/apple-3580668_1280.jpg", use_container_width=True)
    with c2:
        st.subheader("Quick Actions")
        if st.button("Mulai Deteksi"):
            set_page("Deteksi")
        if st.button("Mulai Klasifikasi"):
            set_page("Klasifikasi")
    st.markdown("---")
    st.subheader("Informasi model")
    st.write("- YOLO model path: `model/Siti Annisa Syahira_Laporan 4.pt`")
    st.write("- Keras model input shape (ditampilkan di atas).")

elif page == "Deteksi":
    st.header("🔎 Deteksi Buah (Apel & Jeruk)")
    st.write("Unggah gambar dan YOLO akan mendeteksi objek. Opsional: jalankan klasifikasi pada setiap crop yang terdeteksi.")
    uploaded = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg","jpeg","png"])
    classify_crops = st.checkbox("Jalankan klasifikasi pada setiap crop hasil deteksi (opsional)", value=False)
    if uploaded is None:
        st.info("Silakan unggah gambar.")
    else:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar input", use_container_width=True)

            with st.spinner("Menjalankan YOLO..."):
                results = yolo_model(img)  # ultralytics menerima PIL
            # show annotated image
            try:
                annotated = results[0].plot()
                st.image(annotated, caption="Hasil deteksi (annotated)", use_container_width=True)
            except Exception:
                st.info("Gagal menampilkan gambar berannotasi — menampilkan boxes sebagai teks.")

            # iterate boxes and show details
            st.subheader("Detail deteksi")
            boxes = results[0].boxes  # Boxes object
            if boxes is None or len(boxes) == 0:
                st.warning("Tidak ada objek terdeteksi.")
            else:
                # try to obtain coordinates (xyxy)
                coords_list = []
                try:
                    # ultralytics new results: boxes.xyxy, boxes.cls, boxes.conf
                    xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    names = results[0].names
                    for i, (xy, conf, cid) in enumerate(zip(xyxy, confs, cls_ids)):
                        x1, y1, x2, y2 = map(int, xy.tolist())
                        coords_list.append((x1,y1,x2,y2))
                        st.write(f"- {i+1}. {names[cid]} — conf: {conf:.2f} — box: ({x1},{y1})-({x2},{y2})")
                except Exception:
                    # fallback iteration on boxes
                    for i, box in enumerate(boxes):
                        try:
                            cid = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = results[0].names[cid]
                            # some box objects have .xyxy
                            xy = box.xyxy[0] if hasattr(box, "xyxy") else None
                            if xy is not None:
                                x1,y1,x2,y2 = map(int, xy.tolist())
                                coords_list.append((x1,y1,x2,y2))
                                st.write(f"- {i+1}. {name} — conf: {conf:.2f} — box: ({x1},{y1})-({x2},{y2})")
                            else:
                                st.write(f"- {i+1}. {name} — conf: {conf:.2f}")
                        except Exception:
                            st.write(f"- {i+1}. (info deteksi tidak lengkap)")

                # If user asked to classify crops, do so:
                if classify_crops and len(coords_list) > 0:
                    st.subheader("Hasil klasifikasi pada tiap crop")
                    for idx, (x1,y1,x2,y2) in enumerate(coords_list, start=1):
                        try:
                            crop = img.crop((x1,y1,x2,y2))
                            st.image(crop, width=200, caption=f"Crop {idx}")
                            arr = preprocess_for_classifier(crop, classifier)
                            pred = classifier.predict(arr)
                            cls_id = int(np.argmax(pred))
                            conf = float(np.max(pred))
                            label = "Daun Sehat" if cls_id == 0 else "Daun Tidak Sehat"
                            st.write(f"→ Crop {idx}: **{label}** (conf {conf:.2f})")
                        except Exception as e:
                            st.error(f"Gagal mengklasifikasikan crop {idx}: {e}")
                            st.text(traceback.format_exc())

        except Exception as e:
            st.error("Terjadi kesalahan saat proses deteksi:")
            st.exception(e)

elif page == "Klasifikasi":
    st.header("🌿 Klasifikasi Daun (Sehat / Tidak Sehat)")
    st.write("Unggah gambar daun. Preprocessing disesuaikan otomatis dengan input shape model.")
    uploaded = st.file_uploader("Unggah gambar daun", type=["jpg","jpeg","png"])
    if uploaded is None:
        st.info("Silakan unggah gambar.")
    else:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Input gambar", use_container_width=True)

            arr = preprocess_for_classifier(img, classifier)
            st.write("Preview input array shape untuk classifier:", arr.shape)
            with st.spinner("Menjalankan klasifikasi..."):
                pred = classifier.predict(arr)
            cls_id = int(np.argmax(pred))
            conf = float(np.max(pred))
            label = "🌱 Daun Sehat" if cls_id == 0 else "🍂 Daun Tidak Sehat"
            st.success(f"Hasil: **{label}** — confidence: {conf:.4f}")
        except Exception as e:
            st.error("Terjadi kesalahan saat klasifikasi:")
            st.exception(e)
            st.text("Traceback (ringkas):")
            st.text(traceback.format_exc(limit=200))

elif page == "Tentang":
    st.header("ℹ️ Tentang")
    st.write("""
        Dashboard ini dikembangkan oleh **Siti Annisa Syahira** untuk UTS.  
        Teknologi: YOLOv8 (deteksi) dan TensorFlow/Keras (klasifikasi).
    """)
    st.write("Pastikan file model ada di folder `model/` dan memiliki input shape yang benar.")

st.write("---")
st.markdown("© 2025 | Dashboard UTS - Siti Annisa Syahira")
