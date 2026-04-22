import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from stubs.mock_stm32 import MockSTM32


def test_start_stop():
    stm = MockSTM32()
    stm.start()
    time.sleep(0.1)
    assert stm._running is True
    stm.stop()
    assert stm._running is False
    print("\nstart/stop OK")


def test_send_valid_angles():
    stm = MockSTM32()
    stm.start()

    result = stm.send_angles([10.0, 20.0, -15.0, 5.0, 30.0])
    assert result is True

    stm.stop()
    print("send_angles valides OK")


def test_send_angles_out_of_limits():
    stm = MockSTM32()
    stm.start()

    # θ₂ limité à [-90, 90] — on envoie 200°
    result = stm.send_angles([0.0, 200.0, 0.0, 0.0, 0.0])
    assert result is False

    stm.stop()
    print("refus angles hors limites OK")


def test_joints_reach_target():
    """Les joints doivent atteindre leur cible en quelques secondes."""
    stm = MockSTM32()
    stm.start()

    target = [30.0, 20.0, -10.0, 5.0, 45.0]
    stm.send_angles(target, speed=10.0)

    # Attendre que les joints arrivent
    timeout = 5.0
    t_start = time.time()
    while stm.is_moving():
        if time.time() - t_start > timeout:
            break
        time.sleep(0.1)

    status = stm.get_status()
    for i, (actual, expected) in enumerate(zip(status["angles"], target)):
        assert abs(actual - expected) < 1.0, (
            f"Joint {i+1} : attendu {expected}°, obtenu {actual:.2f}°"
        )

    stm.stop()
    print("joints atteignent leur cible OK")


def test_gripper_open_close():
    stm = MockSTM32()
    stm.start()

    stm.set_gripper(MockSTM32.GRIPPER_CLOSED)
    status = stm.get_status()
    assert status["gripper"] == MockSTM32.GRIPPER_CLOSED

    stm.set_gripper(MockSTM32.GRIPPER_OPEN)
    status = stm.get_status()
    assert status["gripper"] == MockSTM32.GRIPPER_OPEN

    stm.stop()
    print("gripper open/close OK")


def test_emergency_stop():
    stm = MockSTM32()
    stm.start()

    stm.emergency_stop()
    result = stm.send_angles([10.0, 10.0, 10.0, 10.0, 10.0])
    assert result is False, "La commande doit être refusée après e-stop"

    stm.reset_estop()
    result = stm.send_angles([10.0, 10.0, 10.0, 10.0, 10.0])
    assert result is True, "La commande doit passer après reset"

    stm.stop()
    print("emergency stop + reset OK")


def test_status_fields():
    stm = MockSTM32()
    stm.start()
    time.sleep(0.1)

    status = stm.get_status()

    assert "angles"       in status
    assert "temperature"  in status
    assert "voltage"      in status
    assert "gripper"      in status
    assert "moving"       in status
    assert "estop"        in status

    assert len(status["angles"]) == 5
    assert 30.0 <= status["temperature"] <= 80.0
    assert 11.0 <= status["voltage"]     <= 13.0

    stm.stop()
    print("status fields OK")