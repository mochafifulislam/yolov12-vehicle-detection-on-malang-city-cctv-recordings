# 🚦 Real-Time Motor Vehicle Detection & Counting on Malang City CCTV using YOLOv12 + DeepSORT

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![YOLOv12](https://img.shields.io/badge/YOLOv12-Ultralytics-000000)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)

Repositori ini berisi implementasi sistem pemantauan lalu lintas cerdas (*Intelligent Traffic Monitoring System*) untuk **deteksi, pelacakan, dan penghitungan jumlah kendaraan bermotor** secara *real-time* dari rekaman CCTV persimpangan lalu lintas di Kota Malang.

Proyek ini dikembangkan menggunakan arsitektur state-of-the-art **YOLOv12** untuk *object detection* dan diintegrasikan dengan algoritma **DeepSORT** (*Deep Simple Online and Realtime Tracking*) untuk *multi-object tracking* serta *counting mechanism*.

---

## 📌 Latar Belakang & Tantangan

Kemacetan lalu lintas pada persimpangan utama Kota Malang memerlukan pemantauan volume kendaraan secara presisi dan otomatis. Tantangan utama pada rekaman CCTV meliputi:
* Variasi pencahayaan dan resolusi kamera CCTV.
* Tingkat oklusi (*occlusion*) antar kendaraan yang tinggi, terutama sepeda motor dan mobil.
* Kebutuhan pelacakan kendaraan secara kontinu agar tidak terjadi penghitungan ganda (*double counting*).

---

## 🏗️ Alur Sistem & Metodologi

### 1. Custom Dataset & Annotation
* **Pengumpulan Data**: Rekaman CCTV dari beberapa titik persimpangan utama di Kota Malang.
* **Annotasi**: Pelabelan manual kelas kendaraan (*Motorcycle*, *Car*, *Bus*, *Truck*) dalam format YOLO.
* **Augmentasi Data**: Penyesuaian kecerahan, rotasi ringan, dan *mosaic augmentation*.

### 2. Evaluasi Model
Model diuji menggunakan metrik evaluasi standar *Computer Vision*:

$$mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i$$

---

## 📊 Hasil & Performa Model

| Kelas Kendaraan | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **Motorcycle** | 0.912 | 0.885 | 0.924 | 0.685 |
| **Car** | 0.935 | 0.910 | 0.948 | 0.721 |
| **Bus** | 0.890 | 0.865 | 0.895 | 0.650 |
| **Truck** | 0.882 | 0.850 | 0.880 | 0.638 |
| **Overall (All Classes)** | **0.905** | **0.878** | **0.912** | **0.674** |

---

## 💡 Fitur Utama

- 🔍 **Deteksi Presisi Tinggi**: Berbasis SOTA YOLOv12 dengan *Transfer Learning*.
- 🆔 **Multi-Object Tracking (MOT)**: DeepSORT mencegah penghitungan ulang saat kendaraan terhalang (*occluded*).
- 🚗 **Multi-Class Vehicle Counting**: Menghitung secara terpisah berdasarkan kategori kendaraan.
- 📐 **Virtual Counting Zone/Line**: Garis deteksi kustom yang dapat disesuaikan koordinatnya pada frame video.

---

## 🔒 Catatan Kerahasiaan & Hak Cipta Data (Data Disclaimer)

> ⚠️ **Data Privacy Notice:**
> Rekaman CCTV utuh dan dataset lengkap yang digunakan dalam riset skripsi ini bersifat **terbatas/terproteksi** untuk menjaga privasi publik dan kepemilikan data lokal. Repositori ini hanya menyediakan file sampel video pendek di folder `data/samples/` untuk keperluan pengujian kode (*code demonstration*).

---

## 🚀 Cara Menjalankan Kode

### 1. Clone Repositori
```bash
git clone [https://github.com/mochafifulislam/yolov12-vehicle-detection-on-cctv-recordings-malang-city.git](https://github.com/mochafifulislam/yolov12-vehicle-detection-on-cctv-recordings-malang-city.git)
cd yolov12-vehicle-detection-on-cctv-recordings-malang-city
