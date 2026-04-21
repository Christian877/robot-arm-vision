import sys
import importlib

REQUIRED = [
    ("numpy",   "2."),
    ("cv2",     "4."),
    ("PIL",     "12."),
    ("flask",   "3."),
    ("torch",   "2."),
    ("pytest",  "9."),
]

def test_python_version():
    major, minor = sys.version_info[:2]
    assert major == 3 and minor >= 10, f"Python 3.10+ requis, trouvé {major}.{minor}"
    print(f"\nPython {major}.{minor} OK")

def test_imports():
    for module_name, version_prefix in REQUIRED:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "?")
        assert version.startswith(version_prefix), (
            f"{module_name} version {version} inattendue (attendu {version_prefix}x)"
        )
        print(f"  {module_name:<10} {version}  OK")

def test_numpy_operations():
    import numpy as np
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert np.dot(a, b) == 32
    print("\nnumpy operations OK")

def test_opencv_frame():
    import numpy as np
    import cv2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert frame.shape == (480, 640, 3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (480, 640)
    print("opencv frame 640x480 OK")

def test_torch_tensor():
    import torch
    t = torch.zeros(3, 480, 640)
    assert t.shape == (3, 480, 640)
    print("torch tensor OK")

def test_flask_app():
    from flask import Flask
    app = Flask(__name__)
    assert app is not None
    print("flask app OK")