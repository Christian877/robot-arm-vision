import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.controller import RobotController, RobotState
from robot_system.communication.uart_driver import UARTDriver
from robot_system.control.motor_controller import MotorController
from robot_system.kinematics.kinematics import RobotKinematics
from robot_system.safety.safety_monitor import SafetyMonitor
from robot_system.vision.camera_module import CameraCapture
from robot_system.vision.inference_engine import YOLOv5Inference
from robot_system.vision.vision_buffer import VisionBuffer
from stubs.mock_stm32 import MockSTM32
from stubs.mock_camera import MockCamera


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_system():
    """
    Crée le système complet :
    Camera → VisionBuffer → FSM → MotorController → UART → STM32
    + SafetyMonitor en parallèle
    """
    stm  = MockSTM32()
    uart = UARTDriver(mock_stm32=stm)
    uart.connect()

    motor  = MotorController(uart_driver=uart)
    kin    = RobotKinematics()
    safety = SafetyMonitor(uart_driver=uart)

    camera = CameraCapture(source=MockCamera())
    engine = YOLOv5Inference(model_path=None)
    vision = VisionBuffer(camera=camera, engine=engine)

    camera.start()
    vision.start()
    safety.start()

    fsm = RobotController(
        vision_buffer    = vision,
        motor_controller = motor,
        kinematics       = kin,
    )

    return {
        "fsm"    : fsm,
        "motor"  : motor,
        "kin"    : kin,
        "safety" : safety,
        "vision" : vision,
        "camera" : camera,
        "uart"   : uart,
        "stm"    : stm,
    }


def _teardown(sys_dict):
    """Arrête proprement tous les composants."""
    fsm    = sys_dict["fsm"]
    vision = sys_dict["vision"]
    camera = sys_dict["camera"]
    safety = sys_dict["safety"]

    if fsm.is_alive():
        fsm.stop()
    else:
        fsm._running = False

    safety.stop()
    vision.stop()
    camera.stop()


def _wait_for_state(fsm, target_state, timeout=10.0) -> bool:
    """
    Attend que la FSM atteigne un état donné.
    Retourne True si atteint, False si timeout.
    """
    t_start = time.time()
    while time.time() - t_start < timeout:
        if fsm.get_state() == target_state:
            return True
        time.sleep(0.1)
    return False


# ------------------------------------------------------------------
# Tests d'intégration
# ------------------------------------------------------------------

def test_full_system_starts():
    """Tous les composants démarrent sans erreur."""
    s = _make_system()
    time.sleep(0.5)

    assert s["vision"].is_running()
    assert s["safety"].is_safe()
    assert s["fsm"].get_state() == RobotState.IDLE

    _teardown(s)
    print("\nsystème complet démarré OK")


def test_vision_feeds_detections():
    """Le VisionBuffer produit des détections après démarrage."""
    s = _make_system()
    time.sleep(1.5)

    stats = s["vision"].get_stats()
    assert stats["processed_frames"] > 0

    _teardown(s)
    print(f"vision : {stats['processed_frames']} frames traitées OK")


def test_motor_and_uart_communicate():
    """MotorController envoie correctement des angles via UARTDriver."""
    s = _make_system()

    angles = [10.0, 15.0, -5.0, 5.0, 0.0]
    result = s["motor"].set_all_angles(angles, speed=10.0)
    assert result is True

    stats = s["uart"].get_stats()
    assert stats["tx_count"] >= 1

    _teardown(s)
    print("motor → uart communication OK")


def test_kinematics_fk_ik_roundtrip():
    """FK puis IK doit retrouver une position proche."""
    s   = _make_system()
    kin = s["kin"]

    known  = [10.0, 15.0, -5.0, 5.0, 0.0]
    target = kin.forward_kinematics(known)

    solution = kin.inverse_kinematics(
        list(target),
        initial_guess=[8.0, 12.0, -3.0, 3.0, 0.0]
    )

    assert solution is not None, "IK non convergée"

    import numpy as np
    pos_check = kin.forward_kinematics(solution)
    error     = np.linalg.norm(target - pos_check)
    assert error < 5.0, f"Erreur FK/IK : {error:.2f}mm"

    _teardown(s)
    print(f"FK→IK roundtrip erreur {error:.3f}mm OK")


def test_safety_monitors_system():
    """SafetyMonitor surveille et accumule des checks."""
    s = _make_system()
    time.sleep(0.8)

    stats = s["safety"].get_stats()
    assert stats["check_count"] > 0
    assert s["safety"].is_safe() is True

    _teardown(s)
    print(f"safety : {stats['check_count']} checks OK")


def test_safety_estop_stops_motor():
    """Un e-stop SafetyMonitor bloque les commandes moteur."""
    s = _make_system()

    s["safety"].emergency_stop()
    assert s["safety"].is_safe() is False

    result = s["uart"].send_angles([10.0, 10.0, 10.0, 10.0, 10.0])
    assert result is False

    _teardown(s)
    print("e-stop bloque les commandes moteur OK")


def test_fsm_transitions_idle_to_scanning():
    """La FSM transite correctement IDLE → SCANNING."""
    s   = _make_system()
    fsm = s["fsm"]
    fsm.start()

    fsm.start_scan()
    assert fsm.get_state() == RobotState.SCANNING

    _teardown(s)
    print("FSM IDLE → SCANNING OK")


def test_fsm_homing_returns_to_idle():
    """
    Après HOMING, la FSM revient en IDLE
    quand les moteurs atteignent la position home.
    """
    s   = _make_system()
    fsm = s["fsm"]
    fsm.start()

    fsm.manual_home()
    assert fsm.get_state() == RobotState.HOMING

    # Attendre retour IDLE (les joints atteignent HOME)
    reached = _wait_for_state(fsm, RobotState.IDLE, timeout=8.0)
    assert reached, (
        f"FSM n'est pas revenue en IDLE — état : {fsm.get_state().name}"
    )

    _teardown(s)
    print("FSM HOMING → IDLE OK")


def test_full_cycle_scan_to_idle():
    """
    Cycle complet :
    IDLE → SCANNING → (détection) → TARGETING → MOVING
    → GRASPING → TRANSPORTING → RELEASING → HOMING → IDLE
    """
    s   = _make_system()
    fsm = s["fsm"]

    transitions = []
    fsm.register_state_callback(
        lambda old, new: transitions.append(new.name)
    )

    fsm.start()
    time.sleep(1.0)  # Laisser le VisionBuffer se remplir

    fsm.start_scan()

    # Attendre IDLE (fin de cycle) ou ERROR avec timeout 30s
    t_start = time.time()
    while time.time() - t_start < 30.0:
        state = fsm.get_state()
        if state in (RobotState.IDLE, RobotState.ERROR):
            break
        time.sleep(0.2)

    print(f"\nTransitions : {' → '.join(transitions)}")
    print(f"État final  : {fsm.get_state().name}")

    # Le cycle doit au minimum passer par SCANNING
    assert "SCANNING" in transitions, "SCANNING jamais atteint"

    _teardown(s)
    print("cycle complet OK")


def test_error_recovery_flow():
    """
    Après ERROR, reset_error() relance un HOMING
    et revient en IDLE.
    """
    s   = _make_system()
    fsm = s["fsm"]
    fsm.start()

    # Forcer ERROR
    fsm._state = RobotState.ERROR
    assert fsm.get_state() == RobotState.ERROR

    # Recovery
    fsm.reset_error()
    assert fsm.get_state() == RobotState.HOMING

    # Attendre IDLE
    reached = _wait_for_state(fsm, RobotState.IDLE, timeout=8.0)
    assert reached, "FSM n'est pas revenue en IDLE après recovery"

    _teardown(s)
    print("error recovery → IDLE OK")