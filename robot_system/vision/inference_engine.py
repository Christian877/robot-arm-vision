import time
import random
import numpy as np


class Detection:
    """
    Représente un objet détecté dans une frame.

    Attributs :
        label      : nom de la classe (ex: "bottle")
        confidence : score de confiance [0.0 - 1.0]
        bbox       : bounding box [x1, y1, x2, y2] en pixels
        center     : centre de la bbox (cx, cy)
        class_id   : identifiant numérique de la classe
    """

    def __init__(self, label: str, confidence: float,
                 bbox: list, class_id: int = 0):
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
    Moteur d'inférence YOLOv5-Nano.

    En mode mock (model_path=None) : retourne des détections
    synthétiques réalistes pour tester le pipeline sans GPU.

    En mode réel : charge le modèle TFLite et fait la vraie inférence.
    """

    # 10 classes du projet
    CLASSES = [
        "bottle", "cup", "box", "ball", "book",
        "phone", "pen", "scissors", "tape", "key"
    ]

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD        = 0.4
    INPUT_SIZE           = (640, 480)

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._model     = None
        self._mock_mode = model_path is None
        self._inference_count = 0
        self._total_latency   = 0.0

        if self._mock_mode:
            print("[YOLOv5Inference] Mode mock — détections synthétiques")
        else:
            self._load_model()

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list:
        """
        Détecte les objets dans une frame.

        Args:
            frame : image numpy (H, W, 3) uint8

        Returns:
            Liste de Detection, filtrée par confidence et NMS
        """
        if frame is None:
            return []

        t_start = time.time()

        if self._mock_mode:
            detections = self._mock_detect(frame)
        else:
            tensor     = self.preprocess(frame)
            output     = self.inference(tensor)
            detections = self.postprocess(output)

        latency = (time.time() - t_start) * 1000  # ms
        self._inference_count += 1
        self._total_latency   += latency

        return detections

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prépare la frame pour l'inférence.
        Resize + normalisation [0, 1] + reshape en tensor.
        """
        import cv2
        resized = cv2.resize(frame, self.INPUT_SIZE)
        tensor  = resized.astype(np.float32) / 255.0
        tensor  = np.transpose(tensor, (2, 0, 1))   # HWC → CHW
        tensor  = np.expand_dims(tensor, axis=0)     # Batch dim
        return tensor

    def inference(self, tensor: np.ndarray) -> np.ndarray:
        """Lance l'inférence sur le modèle TFLite."""
        if self._model is None:
            raise RuntimeError("Modèle non chargé — utilise detect() en mode mock")
        # Inférence réelle TFLite ici
        pass

    def postprocess(self, output: np.ndarray) -> list:
        """
        Convertit la sortie brute du modèle en liste de Detection.
        Applique NMS (Non-Maximum Suppression).
        """
        pass

    def get_stats(self) -> dict:
        """Retourne les statistiques de performance."""
        avg_latency = (
            self._total_latency / self._inference_count
            if self._inference_count > 0 else 0.0
        )
        return {
            "inference_count" : self._inference_count,
            "avg_latency_ms"  : round(avg_latency, 2),
            "mode"            : "mock" if self._mock_mode else "tflite",
        }

    # ------------------------------------------------------------------
    # Mode mock — détections synthétiques
    # ------------------------------------------------------------------

    def _mock_detect(self, frame: np.ndarray) -> list:
        """
        Génère des détections réalistes basées sur les objets
        visibles dans la frame MockCamera.
        """
        # Simuler la latence d'inférence (~30-80ms)
        time.sleep(random.uniform(0.03, 0.08))

        detections = []
        h, w = frame.shape[:2]

        # Nombre aléatoire d'objets détectés (0 à 3)
        n_objects = random.randint(0, 3)

        for _ in range(n_objects):
            # Position aléatoire dans la frame
            cx = random.randint(50, w - 50)
            cy = random.randint(50, h - 50)
            bw = random.randint(40, 120)
            bh = random.randint(40, 120)

            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)

            confidence = random.uniform(0.55, 0.99)
            label      = random.choice(self.CLASSES)
            class_id   = self.CLASSES.index(label)

            if confidence >= self.CONFIDENCE_THRESHOLD:
                detections.append(Detection(
                    label      = label,
                    confidence = confidence,
                    bbox       = [x1, y1, x2, y2],
                    class_id   = class_id
                ))

        return detections

    # ------------------------------------------------------------------
    # Chargement modèle réel (Phase 2 — Google Colab)
    # ------------------------------------------------------------------

    def _load_model(self):
        """Charge le modèle TFLite depuis model_path."""
        try:
            import tflite_runtime.interpreter as tflite
            self._model = tflite.Interpreter(model_path=self.model_path)
            self._model.allocate_tensors()
            print(f"[YOLOv5Inference] Modèle chargé : {self.model_path}")
        except ImportError:
            print("[YOLOv5Inference] tflite_runtime non disponible — mode mock activé")
            self._mock_mode = True
        except Exception as e:
            print(f"[YOLOv5Inference] Erreur chargement modèle : {e}")
            self._mock_mode = True