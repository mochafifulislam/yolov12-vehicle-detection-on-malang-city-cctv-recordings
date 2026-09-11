import numpy as np

def detections_from_results(results, model_names, conf_thresh=0.5, allowed_classes=None):
    """
    Mengekstrak bounding box dan confidence score dari objek YOLOv12
    ke format bounding box DeepSORT ([x, y, w, h], confidence, class_name)
    """
    dets = []
    if results is None or results.boxes is None:
        return dets

    boxes = results.boxes
    if len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(confs)):
        conf = float(confs[i])
        if conf < conf_thresh:
            continue

        cls_id = int(cls_ids[i])
        cls_name = model_names.get(cls_id, str(cls_id))

        if allowed_classes and cls_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = xyxy[i]
        # Format Bounding Box DeepSORT: [left, top, width, height]
        dets.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf, cls_name))

    return dets
