import cv2
import time
import torch
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from tracker import detections_from_results

def run_tracking(
    video_path: str,
    weights_path: str,
    output_path: str = "output_deepsort.mp4",
    conf_thresh: float = 0.5,
    downscale: int = 1,
    fps_output: int = 30
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan Perangkat: {device}")

    # Load Model YOLOv12
    model = YOLO(weights_path)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka file video: {video_path}")

    ret, frame = cap.read()
    orig_h, orig_w = frame.shape[:2]

    out_w = orig_w // downscale if downscale != 1 else orig_w
    out_h = orig_h // downscale if downscale != 1 else orig_h

    # Video Writer Output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps_output, (out_w, out_h))

    # Line Setup (Virtual Counting Line)
    line_pos_ratio = 0.85
    line_y = int(orig_h * line_pos_ratio)
    line_p1, line_p2 = (0, line_y), (orig_w, line_y)

    previous_centroid = {}
    counted_ids = set()
    counter = defaultdict(int)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps_list, total_frames = [], 0
    start_global = time.time()

    print(f"🎬 Memproses video & menyimpan output ke: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_frame = time.time()
        total_frames += 1

        if downscale != 1:
            frame = cv2.resize(frame, (out_w, out_h))

        # 1. Deteksi YOLOv12
        results = model(frame)[0]
        allowed_classes = ["HV", "LV", "MC"]
        dets = detections_from_results(results, model.names, conf_thresh, allowed_classes)

        # 2. Update DeepSORT Tracker
        tracks = tracker.update_tracks(dets, frame=frame)

        # 3. Processing Tracking & Counting Line
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            left, top, right, bottom = [int(x) for x in ltrb]
            w, h = right - left, bottom - top

            cx, cy = int(left + w / 2), int(top + h / 2)

            # Bounding Box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

            # Class Identification
            cls_name = getattr(track, "det_class", "unknown")

            # Labeling
            label_text = f"ID:{track_id} {cls_name}"
            cv2.putText(frame, label_text, (left, max(top - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # Counting Logic via Line Crossing
            prev = previous_centroid.get(track_id)
            previous_centroid[track_id] = (cx, cy)

            if prev and prev[1] < line_y <= cy and track_id not in counted_ids:
                counter[cls_name] += 1
                counted_ids.add(track_id)

        # Draw Counting Line & Overlay Statistics
        cv2.line(frame, line_p1, line_p2, (0, 255, 255), 3)
        
        y0 = 80
        for i, (c_name, cnt) in enumerate(counter.items()):
            cv2.putText(frame, f"{c_name}: {cnt}", (20, y0 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_frame)
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Write to Output Video
        video_writer.write(frame)

    cap.release()
    video_writer.release()

    # Log Evaluation
    print("\n=================================")
    print("      EVALUASI PERFORMA FPS")
    print("=================================")
    print(f"Total Frame Diproses : {total_frames}")
    if fps_list:
        print(f"FPS Rata-rata        : {sum(fps_list)/len(fps_list):.2f}")
    print(f"Total Runtime        : {time.time() - start_global:.2f} detik")
    print("=================================")
    print("Hasil Akhir Perhitungan Kendaraan:")
    for k, v in counter.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    # Masukkan Path lokal kamu saat eksekusi
    VIDEO_INPUT = "input_sample.mp4"
    WEIGHTS = "best.pt"
    run_tracking(VIDEO_INPUT, WEIGHTS)
