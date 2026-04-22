import threading
import time
import random


class MockSTM32:

    JOINT_LIMITS = [
        (-180, 180),
        (-90,  90),
        (-120, 120),
        (-90,  90),
        (-180, 180),
    ]

    GRIPPER_OPEN   = 0
    GRIPPER_CLOSED = 1

    def __init__(self):
        self._current_angles = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._target_angles  = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._gripper        = self.GRIPPER_OPEN
        self._speed          = 5.0
        self._temperature    = 35.0
        self._voltage        = 12.0
        self._estop          = False
        self._running        = False
        self._thread         = None
        self._lock           = threading.Lock()
        self._command_count  = 0

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._physics_loop, daemon=True)
        self._thread.start()
        print("[MockSTM32] Démarré — simulation physique active")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[MockSTM32] Arrêté")

    def send_angles(self, angles, speed=5.0):
        if self._estop:
            print("[MockSTM32] E-STOP actif — commande refusée")
            return False
        if len(angles) != 5:
            return False
        for i, (angle, (min_a, max_a)) in enumerate(zip(angles, self.JOINT_LIMITS)):
            if not (min_a <= angle <= max_a):
                print(f"[MockSTM32] Joint {i+1} hors limite : {angle}")
                return False
        time.sleep(0.02)
        with self._lock:
            self._target_angles = list(angles)
            self._speed         = max(1.0, min(speed, 50.0))
            self._command_count += 1
        return True

    def set_gripper(self, state):
        if self._estop:
            return False
        time.sleep(0.02)
        with self._lock:
            self._gripper = state
            self._command_count += 1
        action = "fermé" if state == self.GRIPPER_CLOSED else "ouvert"
        print(f"[MockSTM32] Gripper {action}")
        return True

    def get_status(self):
        time.sleep(0.02)
        with self._lock:
            status = {
                "angles"       : list(self._current_angles),
                "target"       : list(self._target_angles),
                "gripper"      : self._gripper,
                "temperature"  : round(self._temperature, 1),
                "voltage"      : round(self._voltage, 2),
                "estop"        : self._estop,
                "moving"       : self._is_moving(),
                "command_count": self._command_count,
            }
        return status

    def emergency_stop(self):
        with self._lock:
            self._estop         = True
            self._target_angles = list(self._current_angles)
        print("[MockSTM32] *** EMERGENCY STOP ***")
        return True

    def reset_estop(self):
        with self._lock:
            self._estop = False
        print("[MockSTM32] E-stop réinitialisé")
        return True

    def is_moving(self):
        with self._lock:
            return self._is_moving()

    def _physics_loop(self):
        interval = 0.02
        while self._running:
            t_start = time.time()
            with self._lock:
                if not self._estop:
                    self._update_joints()
                    self._update_sensors()
            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_joints(self):
        for i in range(5):
            current = self._current_angles[i]
            target  = self._target_angles[i]
            delta   = target - current
            if abs(delta) < 0.1:
                self._current_angles[i] = target
            else:
                step = min(self._speed, abs(delta))
                step = step if delta > 0 else -step
                self._current_angles[i] = current + step

    def _update_sensors(self):
        if self._is_moving():
            self._temperature += random.uniform(0.0, 0.05)
        else:
            self._temperature = max(35.0, self._temperature - 0.02)
        self._temperature = min(self._temperature, 80.0)
        self._voltage = 12.0 + random.uniform(-0.1, 0.1)

    def _is_moving(self):
        for i in range(5):
            if abs(self._current_angles[i] - self._target_angles[i]) > 0.1:
                return True
        return False