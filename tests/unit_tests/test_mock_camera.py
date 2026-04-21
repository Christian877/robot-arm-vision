import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from stubs.mock_camera import MockCamera


def test_camera_starts_and_stops():
    cam = MockCamera(device_id=0)
    cam.start()
    time.sleep(0.2)
    assert cam.is_running() is True
    cam.stop()
    assert cam.is_running() is False
    print("\nstart/stop OK")


def test_frame_shape():
    cam = MockCamera()
    cam.start()
    time.sleep(0.2)

    frame = cam.get_frame()
    assert frame is not None, "Aucune frame reçue"
    assert frame.shape == (480, 640, 3), f"Shape inattendu : {frame.shape}"
    assert frame.dtype == np.uint8

    cam.stop()
    print("frame shape (480, 640, 3) OK")


def test_frame_changes():
    """Vérifie que les frames changent bien (objets qui bougent)."""
    cam = MockCamera()
    cam.start()
    time.sleep(0.1)

    frame1 = cam.get_frame()
    time.sleep(0.2)
    frame2 = cam.get_frame()

    assert frame1 is not None
    assert frame2 is not None
    assert not np.array_equal(frame1, frame2), "Les frames ne changent pas"

    cam.stop()
    print("frames dynamiques OK")


def test_fps_is_close_to_30():
    cam = MockCamera()
    cam.start()
    time.sleep(1.0)  # On attend 1 seconde pour mesurer le FPS

    fps = cam.get_fps()
    cam.stop()

    assert 20 <= fps <= 40, f"FPS hors plage : {fps}"
    print(f"FPS mesuré : {fps} OK")


def test_multiple_get_frame_calls():
    """get_frame() ne doit pas bloquer ni crasher en appelant en boucle."""
    cam = MockCamera()
    cam.start()
    time.sleep(0.1)

    frames = []
    for _ in range(10):
        f = cam.get_frame()
        if f is not None:
            frames.append(f)
        time.sleep(0.05)

    cam.stop()
    assert len(frames) > 0
    print(f"{len(frames)} frames récupérées en boucle OK")