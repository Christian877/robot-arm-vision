import threading
import time
import numpy as np


class CameraCapture(threading.Thread):
    """
    Capture vidéo en thread dédié — 30 FPS.
    Accepte une vraie caméra OpenCV ou un MockCamera (mode virtuel).

    Usage :
        cam = CameraCapture(source=MockCamera())
        cam.start()
        frame = cam.get_frame()
        cam.stop()
    """

    def __init__(self, source=None, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__(daemon=True)
        self.width       = width
        self.height      = height
        self.fps         = fps
        self._source     = source
        self._frame      = None
        self._running    = False
        self._lock       = threading.Lock()
        self._frame_count = 0
        self._start_time  = None
        self._use_mock    = source is not None

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def start(self):
        """Démarre la capture."""
        self._running    = True
        self._start_time = time.time()

        if self._use_mock:
            self._source.start()
            print(f"[CameraCapture] Mode mock — {self.width}x{self.height} @ {self.fps} FPS")
        else:
            import cv2
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
            print(f"[CameraCapture] Caméra USB — {self.width}x{self.height} @ {self.fps} FPS")

        threading.Thread.start(self)

    def stop(self):
        """Arrête proprement la capture."""
        self._running = False
        self.join(timeout=2)

        if self._use_mock:
            self._source.stop()
        else:
            if hasattr(self, "_cap"):
                self._cap.release()

        print("[CameraCapture] Arrêté")

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """
        Retourne la frame la plus récente (non bloquant).
        Retourne None si aucune frame disponible.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_fps(self) -> float:
        """Retourne le FPS réel mesuré depuis le démarrage."""
        if self._start_time is None or self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return round(self._frame_count / elapsed, 2)

    def get_frame_count(self) -> int:
        return self._frame_count

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Boucle interne (thread)
    # ------------------------------------------------------------------

    def run(self):
        """Boucle principale — tourne dans le thread dédié."""
        interval = 1.0 / self.fps

        while self._running:
            t_start = time.time()

            frame = self._read_frame()

            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1

            elapsed   = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Lecture frame (mock ou vraie caméra)
    # ------------------------------------------------------------------

    def _read_frame(self):
        """Lit une frame depuis la source (mock ou OpenCV)."""
        try:
            if self._use_mock:
                return self._source.get_frame()
            else:
                ret, frame = self._cap.read()
                return frame if ret else None
        except Exception as e:
            print(f"[CameraCapture] Erreur lecture frame : {e}")
            return None