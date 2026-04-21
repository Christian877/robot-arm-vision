import threading
import time
import numpy as np
import cv2


class MockCamera:
    """
    Simule une caméra USB Logitech C270.
    Génère des frames 640x480 à 30 FPS avec des objets
    colorés qui bougent — utilisé pour tester sans hardware.
    """

    WIDTH  = 640
    HEIGHT = 480
    FPS    = 30

    def __init__(self, device_id: int = 0):
        self.device_id   = device_id
        self._frame      = None
        self._running    = False
        self._thread     = None
        self._lock       = threading.Lock()
        self._frame_count = 0
        self._start_time  = None

        # Objets simulés : (label, couleur BGR, position x, position y, rayon)
        self._objects = [
            {"label": "bottle", "color": (0, 255, 0),   "x": 100, "y": 200, "r": 40, "dx": 2, "dy": 1},
            {"label": "cup",    "color": (255, 0, 0),   "x": 400, "y": 300, "r": 35, "dx": -1, "dy": 2},
            {"label": "box",    "color": (0, 165, 255), "x": 250, "y": 150, "r": 50, "dx": 1,  "dy": -1},
        ]

    # ------------------------------------------------------------------
    # Interface publique — même API que la vraie CameraCapture
    # ------------------------------------------------------------------

    def start(self):
        """Démarre le thread de génération de frames."""
        self._running    = True
        self._start_time = time.time()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[MockCamera] Démarré — {self.WIDTH}x{self.HEIGHT} @ {self.FPS} FPS")

    def stop(self):
        """Arrête proprement le thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[MockCamera] Arrêté")

    def get_frame(self) -> np.ndarray:
        """
        Retourne la frame la plus récente.
        Retourne None si aucune frame n'est encore disponible.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_fps(self) -> float:
        """Retourne le FPS réel mesuré."""
        if self._start_time is None or self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return round(self._frame_count / elapsed, 2)

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Boucle interne
    # ------------------------------------------------------------------

    def _run(self):
        interval = 1.0 / self.FPS  # 33ms entre chaque frame

        while self._running:
            t_start = time.time()

            frame = self._generate_frame()

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            # Respect du timing 30 FPS
            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Génération de frame synthétique
    # ------------------------------------------------------------------

    def _generate_frame(self) -> np.ndarray:
        """
        Génère une frame 640x480 avec :
        - Fond gris dégradé
        - Grille de référence
        - Objets colorés qui bougent
        - Timestamp et compteur de frames
        """
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * 40

        # Grille de référence (repères visuels)
        for x in range(0, self.WIDTH, 80):
            cv2.line(frame, (x, 0), (x, self.HEIGHT), (60, 60, 60), 1)
        for y in range(0, self.HEIGHT, 80):
            cv2.line(frame, (0, y), (self.WIDTH, y), (60, 60, 60), 1)

        # Déplacer et dessiner les objets
        for obj in self._objects:
            self._move_object(obj)
            self._draw_object(frame, obj)

        # Timestamp en haut à gauche
        ts = time.strftime("%H:%M:%S")
        cv2.putText(
            frame, f"MockCamera | {ts} | frame {self._frame_count}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
        )

        # FPS en haut à droite
        cv2.putText(
            frame, f"{self.get_fps()} FPS",
            (self.WIDTH - 90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
        )

        return frame

    def _move_object(self, obj: dict):
        """Déplace un objet et rebondit sur les bords."""
        obj["x"] += obj["dx"]
        obj["y"] += obj["dy"]

        if obj["x"] - obj["r"] < 0 or obj["x"] + obj["r"] > self.WIDTH:
            obj["dx"] *= -1
        if obj["y"] - obj["r"] < 0 or obj["y"] + obj["r"] > self.HEIGHT:
            obj["dy"] *= -1

    def _draw_object(self, frame: np.ndarray, obj: dict):
        """Dessine un objet simulé avec son label."""
        cx, cy, r = int(obj["x"]), int(obj["y"]), obj["r"]

        # Cercle rempli
        cv2.circle(frame, (cx, cy), r, obj["color"], -1)
        # Contour blanc
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 1)
        # Label au centre
        cv2.putText(
            frame, obj["label"],
            (cx - 20, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )