import os
from dataset import download_roboflow_dataset
from model import initialize_model

def run_training():
    # --- Config Hyperparameters ---
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "epv0F946ZB0aFwaP7yL2") # Disarankan via env var
    WORKSPACE = "afifulislam"
    PROJECT = "revised-thesis-dataset"
    VERSION = 4
    
    PRETRAINED_MODEL = "yolo12n.pt"
    EPOCHS = 200
    IMG_SIZE = 640
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0013
    MOMENTUM = 0.949
    PROJECT_NAME = "Deteksi_Kendaraan_Bermotor"

    # 1. Download Dataset
    dataset_yaml_path, ver_num = download_roboflow_dataset(
        api_key=ROBOFLOW_API_KEY,
        workspace=WORKSPACE,
        project_name=PROJECT,
        version_num=VERSION
    )

    if not dataset_yaml_path:
        print("Pelatihan dibatalkan karena masalah pada dataset.")
        return

    # 2. Inisialisasi Model
    model = initialize_model(PRETRAINED_MODEL)
    if not model:
        print("Pelatihan dibatalkan karena masalah inisialisasi model.")
        return

    # 3. Running Training
    RUN_NAME = f"{PRETRAINED_MODEL.split('.')[0]}_DeteksiKendaraan_{ver_num}_{EPOCHS}epochs"

    print("\n🚀 Memulai Pelatihan Model YOLOv12...")
    print(f"   Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Image Size: {IMG_SIZE}")

    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=RUN_NAME,
            patience=0,
            optimizer="SGD",
            lr0=LEARNING_RATE,
            momentum=MOMENTUM,
            exist_ok=True
        )

        print("\n🎉 Pelatihan Selesai!")
        best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
        print(f"📦 Model Terbaik Disimpan di: {best_model_path}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat melatih model: {e}")

if __name__ == "__main__":
    run_training()
