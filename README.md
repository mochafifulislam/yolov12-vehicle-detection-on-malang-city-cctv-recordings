# 🚦 YOLOv12 Motor Vehicle Detection & Counting on Malang City CCTV

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)
![YOLOv12](https://img.shields.io/badge/YOLOv12-Ultralytics-000000)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)

An intelligent traffic monitoring system for **motor vehicle detection, multi-object tracking, and vehicle counting** using CCTV recordings from urban roads in **Malang City, Indonesia**.

This project integrates **YOLOv12n** for object detection with **SORT** and **DeepSORT** for multi-object tracking (MOT) and virtual-line-based vehicle counting.

The project was developed as part of research on computer vision-based intelligent transportation systems, focusing on the challenges of heterogeneous urban traffic, vehicle occlusion, and varying illumination conditions.

---

## 📌 Overview

Accurate vehicle detection and counting are important for traffic monitoring, congestion analysis, transportation planning, and traffic management.

Conventional traffic counting methods often require manual observation and significant human effort. This project explores an automated computer vision approach using existing CCTV infrastructure to detect, track, and count vehicles.

The proposed pipeline is:

```text
CCTV Recording
      │
      ▼
Image Preprocessing
      │
      ▼
YOLOv12n Vehicle Detection
      │
      ▼
┌─────────────────────┐
│ Multi-Object Track  │
│                     │
│       SORT          │
│        or           │
│     DeepSORT        │
└─────────────────────┘
      │
      ▼
Vehicle Trajectory
      │
      ▼
Virtual Counting Line
      │
      ▼
Vehicle Count
```

The research evaluates both detection performance and vehicle-counting performance under **bright and low-light conditions**.

---

## 🎯 Research Objectives

The main objectives of this project are:

* Detect motor vehicles from urban CCTV recordings using YOLOv12.
* Track vehicles across consecutive video frames.
* Count vehicles using a virtual counting line.
* Compare the performance of **SORT** and **DeepSORT**.
* Evaluate the robustness of the tracking methods under different illumination conditions.
* Investigate the applicability of YOLOv12-based traffic monitoring in an urban environment.

---

## 🗺️ Dataset

The dataset was collected from public CCTV cameras at five major roads in Malang City:

| Road                      |  Bright | Low-Light |  Total  |
| :------------------------ | :-----: | :-------: | :-----: |
| Jalan MT Haryono          |    80   |     80    |   160   |
| Jalan Ahmad Yani          |    80   |     80    |   160   |
| Jalan Ranu Grati          |    80   |     80    |   160   |
| Jalan Basuki Rahmat       |    80   |     80    |   160   |
| Jalan Brigjen S. Supriadi |    80   |     80    |   160   |
| **Total**                 | **400** |  **400**  | **800** |

The dataset consists of **800 manually collected and annotated images**, equally distributed between bright/daytime and low-light/nighttime conditions.

In addition to the image dataset, **10 CCTV video recordings** were used for vehicle detection, tracking, and counting experiments. Each video has a resolution of **1366 × 768 pixels**, a frame rate of **30 FPS**, and a duration of **30 seconds**.

### 🚗 Vehicle Classes

Vehicles are categorized into three classes based on the Indonesian Highway Capacity Manual (MKJI):

| Class  | Description   |
| :----- | :------------ |
| **MC** | Motorcycle    |
| **LV** | Light Vehicle |
| **HV** | Heavy Vehicle |

The annotations were created manually using **Roboflow** in YOLO-compatible bounding-box format.

---

## 🧹 Data Preprocessing

The preprocessing pipeline consists of:

1. Manual vehicle annotation using Roboflow.
2. Image resizing to **640 × 640 pixels**.
3. RGB-to-grayscale conversion.
4. Replication of the grayscale channel into three channels.
5. Dataset splitting into training and validation subsets.

The grayscale transformation follows:

```text
I_gray = 0.299R + 0.587G + 0.114B
```

The resulting grayscale image is replicated across three channels to maintain compatibility with the pretrained YOLOv12 backbone.

The dataset was divided using a **90:10 training-validation split**. Importantly, **no additional data augmentation was applied** beyond the preprocessing procedures described above.

---

## 🧠 YOLOv12n

The detection model used in this project is **YOLOv12n**, the lightweight variant of YOLOv12.

YOLOv12 uses an attention-centric architecture incorporating:

* **Area Attention**
* **Residual Efficient Layer Aggregation Network (R-ELAN)**
* **FlashAttention**

The YOLOv12n model was selected because of its balance between detection performance and computational cost.

### Training Configuration

| Parameter             | Value                    |
| :-------------------- | :----------------------- |
| Model                 | YOLOv12n                 |
| Framework             | Ultralytics YOLO         |
| Training Strategy     | Transfer Learning        |
| Optimizer             | SGD with Momentum (SGDM) |
| Epochs                | 200                      |
| Batch Size            | 64                       |
| Initial Learning Rate | 0.0013                   |
| Momentum              | 0.949                    |
| Input Size            | 640 × 640                |

The model was trained using pretrained weights and optimized for 200 epochs. The computationally intensive training process was performed using an NVIDIA Tesla T4 GPU through Google Colaboratory.

---

## 📊 YOLOv12 Detection Performance

The best-performing YOLOv12n checkpoint was obtained at **epoch 110**.

The validation results were:

| Class                  | Precision |   Recall  |  mAP@0.5  | mAP@0.5:0.95 |
| :--------------------- | :-------: | :-------: | :-------: | :----------: |
| **Heavy Vehicle (HV)** |   1.000   |   0.982   |   0.995   |     0.837    |
| **Light Vehicle (LV)** |   0.977   |   0.967   |   0.988   |     0.795    |
| **Motorcycle (MC)**    |   0.935   |   0.906   |   0.949   |     0.616    |
| **Overall**            | **0.971** | **0.952** | **0.977** |   **0.749**  |

The overall model achieved:

* **Precision:** 97.1%
* **Recall:** 95.2%
* **mAP@0.5:** 97.7%
* **mAP@0.5:0.95:** 74.9%

These results are reported from the validation performance of the best YOLOv12n checkpoint.

> **Note:** The Heavy Vehicle class contains considerably fewer validation instances than Motorcycle and Light Vehicle. Therefore, its exceptionally high precision and AP should be interpreted with caution.

---

## 🆔 Multi-Object Tracking

After YOLOv12 detects vehicles, the detected objects are passed to a multi-object tracking algorithm.

This project evaluates two tracking approaches:

### SORT

**Simple Online and Realtime Tracking (SORT)** uses:

* Kalman Filter for motion prediction.
* Intersection over Union (IoU) for data association.
* Hungarian algorithm for matching detections with existing tracks.

SORT is computationally efficient but can experience identity switching when vehicles become occluded or move closely together.

### DeepSORT

**DeepSORT** extends SORT by incorporating deep appearance descriptors into the data association process.

The tracker combines:

* Motion information.
* Kalman Filter prediction.
* Appearance features extracted from a pretrained deep neural network.

This improves identity preservation when vehicles experience temporary occlusion or interact closely with other vehicles.

---

## 🚗 Vehicle Counting

Vehicle counting is implemented using a **virtual counting line**.

A vehicle is counted when the centroid of its bounding box crosses the predefined counting line in the specified direction.

```text
             Vehicle Direction
                    ↓
                    ↓
        ┌──────────────────────┐
        │                      │
        │        🚗            │
        │         ↓            │
        │         ↓            │
        ├──────────────────────┤
        │  Virtual Counting    │
        │        Line           │
        ├──────────────────────┤
        │                      │
        │        🚙            │
        │                      │
        └──────────────────────┘
```

Each tracked vehicle maintains a unique identity throughout its trajectory, allowing the system to avoid counting the same vehicle multiple times across consecutive frames.

---

## 📈 Vehicle Counting Performance

SORT and DeepSORT were evaluated using the **same YOLOv12 detector, CCTV recordings, and counting-line configurations**.

### Accuracy by Road

| Road                | SORT Bright | DeepSORT Bright | SORT Low-Light | DeepSORT Low-Light |
| :------------------ | ----------: | --------------: | -------------: | -----------------: |
| MT Haryono          |      50.00% |          79.17% |         44.12% |             67.65% |
| Ahmad Yani          |      73.81% |          90.48% |         29.03% |             54.84% |
| Ranu Grati          |      92.73% |          98.18% |         68.29% |             85.37% |
| Basuki Rahmat       |      52.83% |          77.36% |         34.21% |             68.42% |
| Brigjen S. Supriadi |      90.63% |          96.88% |         53.57% |             57.14% |
| **Average**         |  **72.00%** |      **88.37%** |     **45.84%** |         **66.68%** |

DeepSORT consistently achieved higher counting accuracy than SORT across all observation locations.

### Average Performance

| Lighting Condition |           SORT |           DeepSORT |
| :----------------- | -------------: | -----------------: |
| **Bright**         | 72.00 ± 20.20% |  **88.37 ± 9.73%** |
| **Low-Light**      | 45.84 ± 15.69% | **66.68 ± 12.09%** |

DeepSORT improved the average counting accuracy by approximately **16 percentage points under bright conditions** and **21 percentage points under low-light conditions** compared with SORT.

---

## 🏆 Key Findings

The experimental results demonstrate several important findings:

### 1. YOLOv12n provides strong vehicle detection

The detector achieved **97.1% precision, 95.2% recall, 97.7% mAP@0.5, and 74.9% mAP@0.5:0.95**, providing a strong foundation for subsequent tracking and counting.

### 2. DeepSORT outperforms SORT

DeepSORT achieved higher counting accuracy than SORT under both illumination conditions:

```text
Bright
SORT      : 72.00%
DeepSORT  : 88.37%

Low-Light
SORT      : 45.84%
DeepSORT  : 66.68%
```

The improvement is primarily associated with DeepSORT's use of appearance-based features, which helps preserve object identities during occlusion and vehicle interactions.

### 3. Illumination affects counting performance

Both tracking methods experienced lower accuracy under low-light conditions because reduced illumination decreases image contrast and weakens visual features, leading to missed detections and fragmented trajectories.

However, DeepSORT experienced a smaller performance degradation than SORT, indicating greater robustness under challenging illumination.

---

## 📐 Evaluation Metrics

### Intersection over Union

IoU measures the overlap between predicted and ground-truth bounding boxes:

```text
IoU = |B_pred ∩ B_gt| / |B_pred ∪ B_gt|
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### Mean Average Precision

```text
mAP = (1/N) Σ AP_i
```

where `N` represents the number of object classes.

The model was evaluated using **mAP@0.5** and **mAP@0.5:0.95**.

### Counting Accuracy

Vehicle-counting accuracy is calculated as:

```text
Accuracy = (1 - |C_pred - C_gt| / C_gt) × 100%
```

where:

* `C_pred` = predicted vehicle count
* `C_gt` = manually observed ground-truth count

---

## 💡 Key Features

* 🔍 **YOLOv12n Vehicle Detection**
* 🏍️ **Three Vehicle Classes:** Motorcycle, Light Vehicle, Heavy Vehicle
* 🆔 **Multi-Object Tracking**
* ⚡ **SORT Tracking**
* 🧠 **DeepSORT Appearance-Based Tracking**
* 📏 **Virtual Counting Line**
* 🔄 **Trajectory-Based Vehicle Counting**
* 🌗 **Bright & Low-Light Evaluation**
* 📊 **Detection and Counting Performance Evaluation**
* 📹 **CCTV Video Processing**

---

## ⚙️ System Environment

The research was developed using:

| Component               | Specification                  |
| :---------------------- | :----------------------------- |
| OS                      | Windows 11 Home 64-bit         |
| CPU                     | AMD 3020e with Radeon Graphics |
| RAM                     | 8 GB                           |
| Development Environment | Visual Studio Code             |
| Training GPU            | NVIDIA Tesla T4                |
| Training Platform       | Google Colaboratory            |
| Programming Language    | Python                         |

The personal computer was used for dataset preparation, annotation, and software development, while the computationally intensive model training was accelerated using Google Colaboratory with an NVIDIA Tesla T4 GPU.

---

## ⚠️ Limitations

This research has several limitations:

* Evaluation was conducted at only **five observation locations in Malang City**.
* Only **SORT and DeepSORT** were evaluated.
* Adverse weather conditions such as heavy rain and dense fog were not included.
* The generalizability of the model to other cities and CCTV configurations remains to be investigated.
* The Heavy Vehicle class contains fewer samples than the other vehicle classes.

Future work may investigate:

* Larger multi-city datasets.
* More diverse environmental conditions.
* ByteTrack.
* OC-SORT.
* BoT-SORT.
* Statistical significance analysis.
* Model optimization for edge-computing deployment.

---
