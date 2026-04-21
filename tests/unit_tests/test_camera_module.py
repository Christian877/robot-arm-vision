import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.vision.camera_module import CameraCapture
from stubs.mock_camera import MockCamera


def _make_camera() -> CameraCapture:
    """Helper — crée un CameraCapture avec MockCamera."""
    return CameraCapture(source=MockCamera())


def test_camera_starts_and_stops():
    cam = _make_camera()
    cam.start()
    time.sleep(0.2)
    assert cam.is_running() is True
    cam.stop()
    assert cam.is_running() is False
    print("\nstart/stop OK")


def test_get_frame_returns_ndarray():
    cam = _make_camera()
    cam.start()
    time.sleep(0.3)

    frame = cam.get_frame()
    assert frame is not None, "Aucune frame reçue"
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8

    cam.stop()
    print("frame ndarray (480,640,3) uint8 OK")


def test_get_frame_is_non_blocking():
    """get_frame() doit retourner immédiatement."""
    cam = _make_camera()
    cam.start()
    time.sleep(0.2)

    t_start = time.time()
    for _ in range(30):
        cam.get_frame()
    elapsed = time.time() - t_start

    cam.stop()
    assert elapsed < 0.5, f"get_frame() trop lent : {elapsed:.3f}s pour 30 appels"
    print(f"30 appels get_frame() en {elapsed:.3f}s — non bloquant OK")


def test_fps_close_to_30():
    cam = _make_camera()
    cam.start()
    time.sleep(1.5)

    fps = cam.get_fps()
    cam.stop()

    assert 20 <= fps <= 40, f"FPS hors plage : {fps}"
    print(f"FPS mesuré : {fps} OK")


def test_frame_count_increases():
    cam = _make_camera()
    cam.start()
    time.sleep(0.1)
    count1 = cam.get_frame_count()
    time.sleep(0.5)
    count2 = cam.get_frame_count()
    cam.stop()

    assert count2 > count1, "Le compteur de frames n'augmente pas"
    print(f"Frame count : {count1} → {count2} OK")


def test_get_frame_returns_copy():
    """Modifier la frame retournée ne doit pas affecter le buffer interne."""
    cam = _make_camera()
    cam.start()
    time.sleep(0.2)

    frame = cam.get_frame()
    original_value = frame[0, 0, 0]
    frame[0, 0, 0] = 255  # Modification externe

    frame2 = cam.get_frame()
    cam.stop()

    assert frame2[0, 0, 0] != 255 or frame2[0, 0, 0] == original_value
    print("get_frame() retourne bien une copie OK")