import time
import numpy as np


class Detection:
    def __init__(self, label, confidence, bbox, class_id=0):
        self.label      = label
        self.confidence = round(confidence, 3)
        self.bbox       = bbox
        self.class_id   = class_id
        self.center     = (
            int((bbox[0] + bbox[2]) / 2),
            int((bbox[1] + bbox[3]) / 2)
        )

    def __repr__(self):
        cx, cy = self.center
        return (
            f"Detection(label={self.label!r}, "
            f"conf={self.confidence:.2f}, "
            f"center=({cx},{cy}))"
        )


class YOLOv5Inference:
    """
    Moteur d'inférence YOLOv8-Nano TFLite.
    Compatible mock (sans modèle) et réel (avec TFLite).
    """

    CLASSES = ["bouteille", "lotus"]

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD        = 0.4
    INPUT_SIZE           = (640, 640)

    def __init__(self, model_path=None):
        self.model_path       = model_path
        self._model           = None
        self._mock_mode       = model_path is None
        self._inference_count = 0
        self._total_latency   = 0.0

        if self._mock_mode:
            print("[YOLOv8Inference] Mode mock — détections synthétiques")
        else:
            self._load_model()

    def detect(self, frame):
        if frame is None:
            return []

        t_start = time.time()

        if self._mock_mode:
            detections = self._mock_detect(frame)
        else:
            detections = self._real_detect(frame)

        latency = (time.time() - t_start) * 1000
        self._inference_count += 1
        self._total_latency   += latency

        return detections

    def _real_detect(self, frame):
        """Inférence réelle via ultralytics YOLOv8."""
        try:
            results    = self._model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                conf     = float(box.conf[0])
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue

                cls_id   = int(box.cls[0])
                label    = self.CLASSES[cls_id] if cls_id < len(self.CLASSES) else str(cls_id)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                detections.append(Detection(
                    label      = label,
                    confidence = conf,
                    bbox       = [x1, y1, x2, y2],
                    class_id   = cls_id
                ))

            return detections

        except Exception as e:
            print(f"[YOLOv8Inference] Erreur inférence : {e}")
            return []

    def _mock_detect(self, frame):
        import random
        time.sleep(random.uniform(0.03, 0.08))
        detections = []
        h, w = frame.shape[:2]
        n = random.randint(0, 2)
        for _ in range(n):
            cx  = random.randint(50, w - 50)
            cy  = random.randint(50, h - 50)
            bw  = random.randint(40, 120)
            bh  = random.randint(40, 120)
            x1  = max(0, cx - bw // 2)
            y1  = max(0, cy - bh // 2)
            x2  = min(w, cx + bw // 2)
            y2  = min(h, cy + bh // 2)
            conf   = random.uniform(0.55, 0.99)
            cls_id = random.randint(0, 1)
            label  = self.CLASSES[cls_id]
            if conf >= self.CONFIDENCE_THRESHOLD:
                detections.append(Detection(
                    label=label, confidence=conf,
                    bbox=[x1, y1, x2, y2], class_id=cls_id
                ))
        return detections

    def preprocess(self, frame):
        import cv2
        resized = cv2.resize(frame, self.INPUT_SIZE)
        tensor  = resized.astype(np.float32) / 255.0
        tensor  = np.transpose(tensor, (2, 0, 1))
        tensor  = np.expand_dims(tensor, axis=0)
        return tensor

    def inference(self, tensor):
        pass

    def postprocess(self, output):
        pass

    def get_stats(self):
        avg = (
            self._total_latency / self._inference_count
            if self._inference_count > 0 else 0.0
        )
        return {
            "inference_count" : self._inference_count,
            "avg_latency_ms"  : round(avg, 2),
            "mode"            : "mock" if self._mock_mode else "yolov8_tflite",
        }

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            print(f"[YOLOv8Inference] Modèle chargé : {self.model_path}")
            print(f"[YOLOv8Inference] Classes : {self.CLASSES}")
        except Exception as e:
            print(f"[YOLOv8Inference] Erreur chargement : {e} — mode mock activé")
            self._mock_mode = True