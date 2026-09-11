import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI Traffic Monitoring Demo", layout="wide")

st.title("🚗 Intelligent Traffic Monitoring System (YOLOv12)")
st.write("Sistem Pemantauan Lalu Lintas Cerdas - Proyek Skripsi S1 Matematika Universitas Brawijaya")

import os

@st.cache_resource
def load_yolo_model():
    # Mengarahkan path ke folder models/best.pt
    model_path = os.path.join("model", "best.pt")
    
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        # Fallback jika file tidak ditemukan
        return YOLO("yolo12n.pt")

model = load_yolo_model()

uploaded_file = st.file_uploader("Unggah Gambar Sampel CCTV...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Input", use_column_width=True)
    
    if st.button("Jalankan Deteksi YOLOv12"):
        # Inference
        img_array = np.array(image)
        results = model(img_array)[0]
        res_plotted = results.plot()
        
        st.subheader("Hasil Deteksi Objek:")
        st.image(res_plotted, caption="Visualisasi YOLOv12", use_column_width=True)
