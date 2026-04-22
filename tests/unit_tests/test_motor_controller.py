import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.communication.uart_driver import UARTDriver
from robot_system.control.motor_controller import MotorController
from stubs.mock_stm32 import MockSTM32


def _make_motor() -> MotorController:
    stm  = MockSTM32()
    uart = UARTDriver(mock_stm32=stm)
    uart.connect()
    return MotorController(uart_driver=uart)


def test_set_joint_angle_valid():
    motor  = _make_motor()
    result = motor.set_joint_angle(0, 45.0)
    assert result is True
    print("\nset_joint_angle valide OK")


def test_set_joint_angle_invalid_id():
    motor  = _make_motor()
    result = motor.set_joint_angle(9, 45.0)
    assert result is False
    print("joint_id invalide refusé OK")


def test_set_joint_angle_clamped():
    """Un angle hors limites doit être clampé, pas refusé."""
    motor  = _make_motor()
    result = motor.set_joint_angle(1, 200.0)
    assert result is True
    assert motor._target_angles[1] == 90.0
    print("angle clampé à la limite OK")


def test_set_all_angles():
    motor  = _make_motor()
    angles = [10.0, 20.0, -15.0, 5.0, 30.0]
    result = motor.set_all_angles(angles)
    assert result is True
    print("set_all_angles OK")


def test_set_all_angles_wrong_count():
    motor  = _make_motor()
    result = motor.set_all_angles([10.0, 20.0])
    assert result is False
    print("refus mauvais nombre d'angles OK")


def test_set_gripper():
    motor = _make_motor()
    assert motor.set_gripper(MotorController.GRIPPER_CLOSED) is True
    assert motor.get_gripper_state() == MotorController.GRIPPER_CLOSED
    assert motor.set_gripper(MotorController.GRIPPER_OPEN)   is True
    assert motor.get_gripper_state() == MotorController.GRIPPER_OPEN
    print("gripper open/close OK")


def test_emergency_stop():
    motor  = _make_motor()
    result = motor.emergency_stop()
    assert result is True
    print("emergency_stop OK")


def test_scurve_waypoints():
    """La S-curve doit générer 50 waypoints entre start et end."""
    motor  = _make_motor()
    start  = [0.0,  0.0,  0.0,  0.0,  0.0]
    end    = [45.0, 30.0, -20.0, 10.0, 60.0]

    waypoints = motor._generate_scurve(start, end, duration=2.0)

    assert len(waypoints) == 50

    # Premier waypoint proche du start
    for i in range(5):
        assert abs(waypoints[0][i] - start[i]) < 1.0

    # Dernier waypoint proche du end
    for i in range(5):
        assert abs(waypoints[-1][i] - end[i]) < 1.0

    print(f"S-curve : {len(waypoints)} waypoints OK")


def test_get_joint_feedback():
    motor    = _make_motor()
    feedback = motor.get_joint_feedback()
    assert feedback is not None
    assert len(feedback) == 5
    print(f"feedback encodeurs : {feedback} OK")


def test_stats():
    motor = _make_motor()
    motor.set_all_angles([10.0, 10.0, 10.0, 10.0, 10.0])
    stats = motor.get_stats()
    assert stats["move_count"]    >= 1
    assert len(stats["current_angles"]) == 5
    assert "uart_stats" in stats
    print(f"stats OK — {stats['move_count']} mouvements")