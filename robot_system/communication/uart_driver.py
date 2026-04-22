import threading
import time
import struct


class UARTMessage:
    """
    Représente un message UART structuré.

    Format trame :
        [0xAA] [CMD] [LEN] [DATA...] [CRC]
    """
    HEADER = 0xAA

    # Codes commandes
    CMD_SET_ANGLES    = 0x01
    CMD_SET_GRIPPER   = 0x02
    CMD_GET_STATUS    = 0x03
    CMD_ESTOP         = 0x04
    CMD_RESET_ESTOP   = 0x05
    CMD_HOMING        = 0x06

    def __init__(self, cmd: int, data: bytes = b""):
        self.cmd  = cmd
        self.data = data
        self.len  = len(data)

    def encode(self) -> bytes:
        """Encode le message en trame bytes."""
        payload = bytes([self.HEADER, self.cmd, self.len]) + self.data
        crc     = self._compute_crc(payload)
        return payload + bytes([crc])

    @staticmethod
    def _compute_crc(data: bytes) -> int:
        """CRC simple — XOR de tous les bytes."""
        crc = 0
        for b in data:
            crc ^= b
        return crc & 0xFF


class UARTDriver:
    """
    Driver de communication UART entre Raspberry Pi et STM32.

    En mode virtuel, utilise MockSTM32 directement
    sans passer par un vrai port série.
    """

    def __init__(self, port: str = "/dev/ttyAMA0",
                 baudrate: int = 115200,
                 mock_stm32=None):

        self.port      = port
        self.baudrate  = baudrate
        self._stm32    = mock_stm32
        self._mock     = mock_stm32 is not None
        self._lock     = threading.Lock()
        self._connected = False
        self._tx_count  = 0
        self._rx_count  = 0
        self._error_count = 0

        if self._mock:
            print(f"[UARTDriver] Mode mock — MockSTM32 connecté")
        else:
            print(f"[UARTDriver] Port {port} @ {baudrate} baud")

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Ouvre la connexion UART (ou active le mock)."""
        try:
            if self._mock:
                self._stm32.start()
                self._connected = True
                print("[UARTDriver] Connexion mock établie")
                return True
            else:
                import serial
                self._serial    = serial.Serial(
                    self.port, self.baudrate, timeout=1.0
                )
                self._connected = True
                print(f"[UARTDriver] Port {self.port} ouvert")
                return True
        except Exception as e:
            print(f"[UARTDriver] Erreur connexion : {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Ferme la connexion."""
        if self._mock and self._stm32:
            self._stm32.stop()
        elif hasattr(self, "_serial"):
            self._serial.close()
        self._connected = False
        print("[UARTDriver] Déconnecté")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Commandes
    # ------------------------------------------------------------------

    def send_angles(self, angles: list, speed: float = 5.0) -> bool:
        """
        Envoie les angles cibles θ₁…θ₅ au STM32.

        Args:
            angles : liste de 5 floats (degrés)
            speed  : vitesse de déplacement (°/step)

        Returns:
            True si la commande est acceptée
        """
        if not self._connected:
            print("[UARTDriver] Non connecté")
            return False

        if len(angles) != 5:
            print(f"[UARTDriver] 5 angles requis, reçu {len(angles)}")
            return False

        with self._lock:
            try:
                if self._mock:
                    result = self._stm32.send_angles(angles, speed)
                else:
                    data = struct.pack("6f", *angles, speed)
                    msg  = UARTMessage(UARTMessage.CMD_SET_ANGLES, data)
                    self._serial.write(msg.encode())
                    result = self._read_ack()

                self._tx_count += 1
                return result

            except Exception as e:
                print(f"[UARTDriver] Erreur send_angles : {e}")
                self._error_count += 1
                return False

    def set_gripper(self, state: int) -> bool:
        """
        Contrôle le gripper.

        Args:
            state : 0 = ouvert, 1 = fermé
        """
        if not self._connected:
            return False

        with self._lock:
            try:
                if self._mock:
                    result = self._stm32.set_gripper(state)
                else:
                    data = struct.pack("B", state)
                    msg  = UARTMessage(UARTMessage.CMD_SET_GRIPPER, data)
                    self._serial.write(msg.encode())
                    result = self._read_ack()

                self._tx_count += 1
                return result

            except Exception as e:
                print(f"[UARTDriver] Erreur set_gripper : {e}")
                self._error_count += 1
                return False

    def get_status(self) -> dict:
        """
        Demande le feedback complet au STM32.

        Returns:
            dict avec angles, température, tension, gripper, estop
        """
        if not self._connected:
            return {}

        with self._lock:
            try:
                if self._mock:
                    status = self._stm32.get_status()
                else:
                    msg    = UARTMessage(UARTMessage.CMD_GET_STATUS)
                    self._serial.write(msg.encode())
                    status = self._read_status()

                self._rx_count += 1
                return status

            except Exception as e:
                print(f"[UARTDriver] Erreur get_status : {e}")
                self._error_count += 1
                return {}

    def emergency_stop(self) -> bool:
        """Déclenche l'arrêt d'urgence immédiat."""
        if not self._connected:
            return False

        with self._lock:
            try:
                if self._mock:
                    return self._stm32.emergency_stop()
                else:
                    msg = UARTMessage(UARTMessage.CMD_ESTOP)
                    self._serial.write(msg.encode())
                    return self._read_ack()
            except Exception as e:
                print(f"[UARTDriver] Erreur emergency_stop : {e}")
                return False

    def reset_estop(self) -> bool:
        """Réinitialise l'e-stop."""
        if not self._connected:
            return False
        with self._lock:
            if self._mock:
                return self._stm32.reset_estop()
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "tx_count"    : self._tx_count,
            "rx_count"    : self._rx_count,
            "error_count" : self._error_count,
            "connected"   : self._connected,
            "mode"        : "mock" if self._mock else "uart",
        }

    # ------------------------------------------------------------------
    # Lecture port série réel (non utilisé en mode mock)
    # ------------------------------------------------------------------

    def _read_ack(self) -> bool:
        try:
            response = self._serial.read(1)
            return response == b"\x01"
        except Exception:
            return False

    def _read_status(self) -> dict:
        try:
            data = self._serial.read(64)
            return {"raw": data}
        except Exception:
            return {}