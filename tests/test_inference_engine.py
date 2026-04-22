import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.vision.inference_engine import YOLOv5Inference, Detection


def _make_frame() -> np.ndarray:
    """Crée une frame de test 480x640x3."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_mock_mode_init():
    engine = YOLOv5Inference(model_path=None)
    assert engine._mock_mode is True
    print("\nmode mock initialisé OK")


def test_detect_returns_list():
    engine = YOLOv5Inference()
    frame  = _make_frame()
    result = engine.detect(frame)
    assert isinstance(result, list)
    print("detect() retourne une liste OK")


def test_detect_none_frame():
    engine = YOLOv5Inference()
    result = engine.detect(None)
    assert result == []
    print("detect(None) retourne [] OK")


def test_detection_fields():
    """Vérifie que chaque Detection a les bons champs."""
    engine = YOLOv5Inference()
    frame  = _make_frame()

    # On réessaie jusqu'à obtenir au moins une détection
    detections = []
    for _ in range(20):
        detections = engine.detect(frame)
        if len(detections) > 0:
            break

    if len(detections) == 0:
        print("Aucune détection générée (aléatoire) — test ignoré")
        return

    d = detections[0]
    assert isinstance(d, Detection)
    assert hasattr(d, "label")
    assert hasattr(d, "confidence")
    assert hasattr(d, "bbox")
    assert hasattr(d, "center")
    assert hasattr(d, "class_id")
    assert 0.0 <= d.confidence <= 1.0
    assert len(d.bbox) == 4
    assert len(d.center) == 2
    assert d.label in YOLOv5Inference.CLASSES

    print(f"Detection valide : {d}")


def test_confidence_threshold():
    """Toutes les détections doivent dépasser le seuil."""
    engine = YOLOv5Inference()
    frame  = _make_frame()

    for _ in range(10):
        detections = engine.detect(frame)
        for d in detections:
            assert d.confidence >= YOLOv5Inference.CONFIDENCE_THRESHOLD, (
                f"Confiance trop basse : {d.confidence}"
            )

    print("seuil confidence respecté OK")


def test_preprocess_shape():
    """Le tensor de sortie doit avoir la bonne forme."""
    engine = YOLOv5Inference()
    frame  = _make_frame()
    tensor = engine.preprocess(frame)

    assert tensor.shape == (1, 3, 480, 640), (
        f"Shape inattendu : {tensor.shape}"
    )
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

    print(f"preprocess tensor shape {tensor.shape} OK")


def test_get_stats():
    engine = YOLOv5Inference()
    frame  = _make_frame()

    engine.detect(frame)
    engine.detect(frame)

    stats = engine.get_stats()
    assert stats["inference_count"] == 2
    assert stats["avg_latency_ms"]  >  0
    assert stats["mode"] == "mock"

    print(f"stats : {stats} OK")


def test_latency_under_100ms():
    """Chaque inférence doit rester sous 100ms."""
    engine = YOLOv5Inference()
    frame  = _make_frame()

    for _ in range(5):
        t = time.time()
        engine.detect(frame)
        latency_ms = (time.time() - t) * 1000
        assert latency_ms < 100, f"Latence trop haute : {latency_ms:.1f}ms"

    print("latence < 100ms OK")