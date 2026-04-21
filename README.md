# Robot Arm Vision

Bras robotisé 5 DOF avec caméra intelligente YOLOv5 — simulation virtuelle complète.

## Architecture
- **Vision** : YOLOv5-Nano quantifié TFLite INT8 · 30 FPS · <100ms latence
- **Control** : FSM Python · cinématique FK/IK · MotorController S-curve
- **Safety** : SafetyMonitor thread 10 Hz · watchdog · e-stop
- **Simulation** : PyBullet · fichiers URDF
- **Interface** : Flask REST API · WebSocket · Dashboard Vue.js

## Structure
robot_system/
├── vision/          # CameraCapture, YOLOv5Inference
├── control/         # MotorController
├── kinematics/      # FK / IK
├── safety/          # SafetyMonitor
└── communication/   # UART Driver
## Démarrage rapide
```bash
source venv/bin/activate
pytest tests/
```

## Phases
- [x] Phase 0 — Environnement
- [ ] Phase 1 — Vision System
- [ ] Phase 2 — Control System  
- [ ] Phase 3 — Simulation PyBullet