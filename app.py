import os
import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="YOLOv12 Vehicle Detection & Tracking Demo",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Real-Time Intelligent Traffic Monitoring System (YOLOv12)")
st.caption("Proyek Skripsi S1 Matematika Universitas Brawijaya | Deploy Demo")

# Load Pretrained YOLOv12 Model
@st.cache_resource
def load_model():
    # Mengarahkan path ke folder models/best.pt
    model_path = os.path.join("model", "best.pt")
    
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        # Fallback jika file tidak ditemukan
        return YOLO("yolo12n.pt")

try:
    model = load_model()
    st.sidebar.success("✅ Model YOLOv12 Berhasil Dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")

# Sidebar Options
st.sidebar.header("⚙️ Konfigurasi Model")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
option = st.sidebar.selectbox("Pilih Mode Input Data:", ("Upload Video (.mp4)", "Upload Gambar (.jpg/.png)"))

# ==========================================
# MODE 1: INPUT VIDEO (.mp4)
# ==========================================
if option == "Upload Video (.mp4)":
    st.subheader("📹 Pemrosesan Video Traffic Monitoring")
    video_file = st.file_uploader("Unggah berkas video rekaman CCTV...", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Simpan sementara video yang diunggah
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        btn_start = st.button("▶️ Jalankan Analisis Video YOLOv12")
        
        if btn_start:
            st.info("Memproses frame video secara real-time...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference YOLOv12 pada tiap Frame
                results = model(frame, conf=conf_threshold)[0]
                res_plotted = results.plot()
                
                # Konversi BGR OpenCV ke RGB Streamlit
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
            st.success("🎉 Pemrosesan Video Selesai!")

# ==========================================
# MODE 2: INPUT GAMBAR (.jpg / .png)
# ==========================================
else:
    st.subheader("🖼️ Deteksi Objek pada Sampel Gambar")
    image_file = st.file_uploader("Unggah sampel citra CCTV...", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        image = Image.open(image_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        if st.button("🔍 Jalankan Deteksi YOLOv12"):
            img_array = np.array(image)
            results = model(img_array, conf=conf_threshold)[0]
            res_plotted = results.plot()
            
            with col2:
                st.image(res_plotted, caption="Hasil Deteksi YOLOv12", use_container_width=True)
