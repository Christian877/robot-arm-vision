import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.vision.camera_module import CameraCapture
from robot_system.vision.inference_engine import YOLOv5Inference
from robot_system.vision.vision_buffer import VisionBuffer
from stubs.mock_camera import MockCamera


def _make_vision_stack() -> VisionBuffer:
    """Helper — crée la stack Vision complète en mode mock."""
    camera = CameraCapture(source=MockCamera())
    engine = YOLOv5Inference(model_path=None)
    buffer = VisionBuffer(camera=camera, engine=engine)
    camera.start()
    buffer.start()
    return buffer


def _teardown(buf: VisionBuffer):
    buf.stop()
    buf._camera.stop()


def test_starts_and_stops():
    buf = _make_vision_stack()
    time.sleep(0.3)
    assert buf.is_running() is True
    _teardown(buf)
    assert buf.is_running() is False
    print("\nstart/stop OK")


def test_get_latest_detections_returns_list():
    buf = _make_vision_stack()
    time.sleep(0.5)

    result = buf.get_latest_detections()
    assert isinstance(result, list)

    _teardown(buf)
    print("get_latest_detections() retourne liste OK")


def test_buffer_fills_up():
    """Après 1s, le buffer doit contenir des entrées."""
    buf = _make_vision_stack()
    time.sleep(1.0)

    stats = buf.get_stats()
    assert stats["processed_frames"] > 0, "Aucune frame traitée"

    _teardown(buf)
    print(f"{stats['processed_frames']} frames traitées OK")


def test_get_best_detection():
    """get_best_detection() retourne None ou un Detection valide."""
    buf = _make_vision_stack()
    time.sleep(1.0)

    best = buf.get_best_detection()

    if best is not None:
        assert hasattr(best, "confidence")
        assert hasattr(best, "label")
        assert hasattr(best, "center")
        assert best.confidence >= 0.5
        print(f"Meilleure détection : {best}")
    else:
        print("Aucune détection (normal, aléatoire) OK")

    _teardown(buf)


def test_confidence_filter():
    """Toutes les détections dans le buffer doivent dépasser le seuil."""
    buf = _make_vision_stack()
    time.sleep(1.0)

    detections = buf.get_latest_detections()
    for d in detections:
        assert d.confidence >= 0.5, (
            f"Détection sous le seuil : {d.confidence}"
        )

    _teardown(buf)
    print("filtre confiance OK")


def test_get_last_frame_not_none():
    buf = _make_vision_stack()
    time.sleep(0.5)

    frame = buf.get_last_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    _teardown(buf)
    print("get_last_frame() OK")


def test_stats_fields():
    buf = _make_vision_stack()
    time.sleep(0.5)

    stats = buf.get_stats()
    assert "processed_frames" in stats
    assert "total_detections" in stats
    assert "fps"              in stats
    assert "engine_stats"     in stats
    assert stats["fps"] > 0

    _teardown(buf)
    print(f"stats OK — {stats['fps']} FPS")


def test_get_detections_by_label():
    """Filtrer par label ne retourne que les bonnes classes."""
    buf = _make_vision_stack()
    time.sleep(1.0)

    results = buf.get_detections_by_label("bottle")
    for d in results:
        assert d.label == "bottle"

    _teardown(buf)
    print("filtre par label OK")