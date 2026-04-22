import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.controller import RobotController, RobotState
from robot_system.communication.uart_driver import UARTDriver
from robot_system.control.motor_controller import MotorController
from robot_system.kinematics.kinematics import RobotKinematics
from robot_system.vision.camera_module import CameraCapture
from robot_system.vision.inference_engine import YOLOv5Inference
from robot_system.vision.vision_buffer import VisionBuffer
from stubs.mock_stm32 import MockSTM32
from stubs.mock_camera import MockCamera


def _make_full_stack():
    """Crée la stack complète en mode mock sans démarrer la FSM."""
    stm  = MockSTM32()
    uart = UARTDriver(mock_stm32=stm)
    uart.connect()

    motor  = MotorController(uart_driver=uart)
    kin    = RobotKinematics()

    camera = CameraCapture(source=MockCamera())
    engine = YOLOv5Inference(model_path=None)
    vision = VisionBuffer(camera=camera, engine=engine)

    camera.start()
    vision.start()

    fsm = RobotController(
        vision_buffer    = vision,
        motor_controller = motor,
        kinematics       = kin,
    )
    return fsm, vision, camera


def _teardown(fsm, vision, camera):
    """Arrête proprement tous les composants."""
    if fsm.is_alive():
        fsm.stop()
    else:
        fsm._running = False
    vision.stop()
    camera.stop()


def test_initial_state_is_idle():
    fsm, vision, camera = _make_full_stack()
    assert fsm.get_state() == RobotState.IDLE
    _teardown(fsm, vision, camera)
    print("\nétat initial IDLE OK")


def test_start_scan_from_idle():
    fsm, vision, camera = _make_full_stack()
    result = fsm.start_scan()
    assert result is True
    assert fsm.get_state() == RobotState.SCANNING
    _teardown(fsm, vision, camera)
    print("start_scan depuis IDLE OK")


def test_cannot_scan_twice():
    fsm, vision, camera = _make_full_stack()
    fsm.start_scan()
    result = fsm.start_scan()
    assert result is False
    _teardown(fsm, vision, camera)
    print("double scan refusé OK")


def test_manual_home():
    fsm, vision, camera = _make_full_stack()
    fsm.manual_home()
    assert fsm.get_state() == RobotState.HOMING
    _teardown(fsm, vision, camera)
    print("manual_home OK")


def test_state_callback():
    """Le callback doit être appelé à chaque transition."""
    fsm, vision, camera = _make_full_stack()
    transitions = []

    fsm.register_state_callback(
        lambda old, new: transitions.append((old.name, new.name))
    )

    fsm.start_scan()
    fsm.manual_home()

    assert len(transitions) >= 1
    _teardown(fsm, vision, camera)
    print(f"callbacks transitions : {transitions} OK")


def test_reset_error():
    fsm, vision, camera = _make_full_stack()
    fsm._state = RobotState.ERROR
    result = fsm.reset_error()
    assert result is True
    assert fsm.get_state() == RobotState.HOMING
    _teardown(fsm, vision, camera)
    print("reset_error OK")


def test_reset_error_only_from_error():
    fsm, vision, camera = _make_full_stack()
    result = fsm.reset_error()
    assert result is False
    _teardown(fsm, vision, camera)
    print("reset_error refusé hors ERROR OK")


def test_fsm_runs_and_stops():
    fsm, vision, camera = _make_full_stack()
    fsm.start()
    time.sleep(0.5)
    assert fsm.is_alive() is True
    _teardown(fsm, vision, camera)
    print("FSM tourne et s'arrête OK")


def test_stats_fields():
    fsm, vision, camera = _make_full_stack()
    stats = fsm.get_stats()
    assert "state"         in stats
    assert "cycle_count"   in stats
    assert "error_count"   in stats
    assert "success_count" in stats
    _teardown(fsm, vision, camera)
    print(f"stats OK : {stats}")


def test_error_recovery():
    fsm, vision, camera = _make_full_stack()
    fsm._state = RobotState.ERROR
    fsm.error_recovery()
    assert fsm.get_state() == RobotState.HOMING
    _teardown(fsm, vision, camera)
    print("error_recovery OK")