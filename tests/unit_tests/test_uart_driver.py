import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.communication.uart_driver import UARTDriver, UARTMessage
from stubs.mock_stm32 import MockSTM32


def _make_driver() -> UARTDriver:
    stm  = MockSTM32()
    uart = UARTDriver(mock_stm32=stm)
    uart.connect()
    return uart


def test_connect_disconnect():
    uart = _make_driver()
    assert uart.is_connected() is True
    uart.disconnect()
    assert uart.is_connected() is False
    print("\nconnect/disconnect OK")


def test_send_valid_angles():
    uart   = _make_driver()
    result = uart.send_angles([10.0, 20.0, -15.0, 5.0, 30.0])
    assert result is True
    uart.disconnect()
    print("send_angles valides OK")


def test_send_wrong_number_of_angles():
    uart   = _make_driver()
    result = uart.send_angles([10.0, 20.0])
    assert result is False
    uart.disconnect()
    print("refus mauvais nombre d'angles OK")


def test_send_angles_not_connected():
    stm  = MockSTM32()
    uart = UARTDriver(mock_stm32=stm)
    result = uart.send_angles([0.0, 0.0, 0.0, 0.0, 0.0])
    assert result is False
    print("refus si non connecté OK")


def test_set_gripper():
    uart = _make_driver()
    assert uart.set_gripper(1) is True
    assert uart.set_gripper(0) is True
    uart.disconnect()
    print("set_gripper OK")


def test_get_status():
    uart   = _make_driver()
    status = uart.get_status()
    assert isinstance(status, dict)
    assert "angles"      in status
    assert "temperature" in status
    assert "voltage"     in status
    uart.disconnect()
    print("get_status OK")


def test_emergency_stop():
    uart = _make_driver()
    assert uart.emergency_stop() is True
    result = uart.send_angles([10.0, 10.0, 10.0, 10.0, 10.0])
    assert result is False
    uart.disconnect()
    print("emergency_stop OK")


def test_stats():
    uart = _make_driver()
    uart.send_angles([0.0, 0.0, 0.0, 0.0, 0.0])
    uart.get_status()
    stats = uart.get_stats()
    assert stats["tx_count"]  >= 1
    assert stats["rx_count"]  >= 1
    assert stats["mode"]      == "mock"
    assert stats["connected"] is True
    uart.disconnect()
    print(f"stats OK — {stats}")


def test_uart_message_encode():
    """Vérifie l'encodage d'une trame UART."""
    msg   = UARTMessage(UARTMessage.CMD_SET_GRIPPER, b"\x01")
    frame = msg.encode()
    assert frame[0] == 0xAA
    assert frame[1] == UARTMessage.CMD_SET_GRIPPER
    assert frame[2] == 1
    assert frame[3] == 0x01
    print("encodage trame UART OK")