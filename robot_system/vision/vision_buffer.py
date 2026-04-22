import threading
import time
from collections import deque

from robot_system.vision.camera_module import CameraCapture
from robot_system.vision.inference_engine import YOLOv5Inference, Detection


class VisionBuffer(threading.Thread):
    """
    Connecte CameraCapture et YOLOv5Inference.

    Tourne dans son propre thread :
    1. Lit les frames depuis CameraCapture
    2. Les envoie à YOLOv5Inference
    3. Stocke les détections dans un buffer circulaire

    La FSM lit les détections via get_latest_detections()
    sans jamais bloquer.
    """

    def __init__(self,
                 camera: CameraCapture,
                 engine: YOLOv5Inference,
                 buffer_size: int = 10,
                 confidence_threshold: float = 0.5):

        super().__init__(daemon=True)
        self._camera               = camera
        self._engine               = engine
        self._buffer_size          = buffer_size
        self._confidence_threshold = confidence_threshold

        # Buffer circulaire des dernières détections
        self._detections_buffer = deque(maxlen=buffer_size)
        # Dernière frame traitée
        self._last_frame        = None
        # Stats
        self._processed_frames  = 0
        self._total_detections  = 0
        self._start_time        = None

        self._running = False
        self._lock    = threading.Lock()

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def start(self):
        self._running    = True
        self._start_time = time.time()
        threading.Thread.start(self)
        print("[VisionBuffer] Démarré")

    def stop(self):
        self._running = False
        self.join(timeout=3)
        print("[VisionBuffer] Arrêté")

    # ------------------------------------------------------------------
    # Interface publique — appelée par la FSM
    # ------------------------------------------------------------------

    def get_latest_detections(self) -> list:
        """
        Retourne les détections les plus récentes.
        Non bloquant — retourne [] si rien de disponible.
        """
        with self._lock:
            if not self._detections_buffer:
                return []
            return list(self._detections_buffer[-1])

    def get_best_detection(self) -> Detection:
        """
        Retourne la détection avec la confiance la plus haute
        parmi les dernières détections. Retourne None si vide.
        """
        detections = self.get_latest_detections()
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)

    def get_detections_by_label(self, label: str) -> list:
        """Retourne toutes les détections d'une classe donnée."""
        return [
            d for d in self.get_latest_detections()
            if d.label == label
        ]

    def get_last_frame(self):
        """Retourne la dernière frame traitée."""
        with self._lock:
            return self._last_frame

    def get_stats(self) -> dict:
        """Retourne les statistiques du buffer."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = round(self._processed_frames / elapsed, 2) if elapsed > 0 else 0

        return {
            "processed_frames" : self._processed_frames,
            "total_detections" : self._total_detections,
            "buffer_size"      : len(self._detections_buffer),
            "fps"              : fps,
            "engine_stats"     : self._engine.get_stats(),
        }

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Boucle interne
    # ------------------------------------------------------------------

    def run(self):
        """
        Boucle principale :
        - Lit une frame
        - Lance l'inférence
        - Stocke les détections
        """
        while self._running:
            frame = self._camera.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # Inférence
            detections = self._engine.detect(frame)

            # Filtrer par confiance
            detections = [
                d for d in detections
                if d.confidence >= self._confidence_threshold
            ]

            # Stocker dans le buffer
            with self._lock:
                self._detections_buffer.append(detections)
                self._last_frame      = frame
                self._processed_frames += 1
                self._total_detections += len(detections)