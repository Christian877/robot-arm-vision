import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.communication.uart_driver import UARTDriver
from robot_system.safety.safety_monitor import SafetyMonitor
from stubs.mock_stm32 import MockSTM32


def _make_monitor() -> SafetyMonitor:
    stm  = MockSTM32()
    uart = UARTDriver(mock_stm32=stm)
    uart.connect()
    return SafetyMonitor(uart_driver=uart)


def test_starts_and_stops():
    mon = _make_monitor()
    mon.start()
    time.sleep(0.2)
    assert mon._running is True
    mon.stop()
    assert mon._running is False
    print("\nstart/stop OK")


def test_is_safe_initially():
    mon = _make_monitor()
    assert mon.is_safe() is True
    print("is_safe initialement True OK")


def test_check_joint_limits_valid():
    mon    = _make_monitor()
    status = {"angles": [0.0, 0.0, 0.0, 0.0, 0.0]}
    result = mon.check_joint_limits(status)
    assert result is True
    print("check_joint_limits valides OK")


def test_check_joint_limits_violation():
    mon    = _make_monitor()
    status = {"angles": [0.0, 200.0, 0.0, 0.0, 0.0]}
    result = mon.check_joint_limits(status)
    assert result is False
    print("check_joint_limits violation détectée OK")


def test_check_temperature_normal():
    mon    = _make_monitor()
    status = {"temperature": 45.0}
    assert mon.check_temperature(status) is True
    print("check_temperature normale OK")


def test_check_temperature_critical():
    mon    = _make_monitor()
    status = {"temperature": 80.0}
    assert mon.check_temperature(status) is False
    print("check_temperature critique détectée OK")


def test_check_voltage_normal():
    mon    = _make_monitor()
    status = {"voltage": 12.0}
    assert mon.check_voltage(status) is True
    print("check_voltage normale OK")


def test_check_voltage_low():
    mon    = _make_monitor()
    status = {"voltage": 9.0}
    assert mon.check_voltage(status) is False
    print("check_voltage basse détectée OK")


def test_emergency_stop():
    mon = _make_monitor()
    mon.emergency_stop()
    assert mon.is_safe() is False
    print("emergency_stop déclenché OK")


def test_estop_callback():
    """Le callback doit être appelé lors d'un e-stop."""
    mon      = _make_monitor()
    called   = []

    mon.register_estop_callback(lambda: called.append(True))
    mon.emergency_stop()

    assert len(called) == 1
    print("callback e-stop appelé OK")


def test_reset_after_estop():
    mon = _make_monitor()
    mon.emergency_stop()
    assert mon.is_safe() is False
    mon.reset()
    assert mon.is_safe() is True
    print("reset après e-stop OK")


def test_stats_after_run():
    mon = _make_monitor()
    mon.start()
    time.sleep(0.5)
    mon.stop()

    stats = mon.get_stats()
    assert stats["check_count"]  > 0
    assert "violation_count"    in stats
    assert "estop_triggered"    in stats
    print(f"stats OK — {stats['check_count']} checks effectués")