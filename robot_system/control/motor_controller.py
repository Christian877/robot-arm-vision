import time
import threading
import numpy as np


class MotorController:
    """
    Contrôle les moteurs du bras via UARTDriver.

    Responsabilités :
    - Valider les limites articulaires
    - Générer les profils S-curve
    - Envoyer les angles au STM32
    - Lire le feedback encodeurs
    - Déclencher l'e-stop si nécessaire
    """

    # Limites mécaniques [min, max] en degrés
    JOINT_LIMITS = [
        (-180, 180),   # θ₁ base
        (-90,  90),    # θ₂ épaule
        (-120, 120),   # θ₃ coude
        (-90,  90),    # θ₄ poignet pitch
        (-180, 180),   # θ₅ poignet roll
    ]

    GRIPPER_OPEN   = 0
    GRIPPER_CLOSED = 1

    # Vitesse par défaut (°/step)
    DEFAULT_SPEED    = 5.0
    MAX_SPEED        = 50.0
    MIN_SPEED        = 1.0

    # Tolérance position atteinte (degrés)
    POSITION_TOLERANCE = 1.0

    def __init__(self, uart_driver):
        self._uart          = uart_driver
        self._lock          = threading.Lock()
        self._current_angles = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._target_angles  = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._gripper_state  = self.GRIPPER_OPEN
        self._move_count     = 0

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def set_joint_angle(self, joint_id: int,
                        angle: float, speed: float = None) -> bool:
        """
        Bouge un seul joint vers un angle cible.

        Args:
            joint_id : 0-4 (θ₁ à θ₅)
            angle    : angle cible en degrés
            speed    : vitesse (°/step), None = DEFAULT_SPEED

        Returns:
            True si la commande est envoyée
        """
        if not (0 <= joint_id <= 4):
            print(f"[MotorController] joint_id invalide : {joint_id}")
            return False

        speed  = self._clamp_speed(speed or self.DEFAULT_SPEED)
        angles = list(self._current_angles)

        clamped = self._clamp_angle(joint_id, angle)
        angles[joint_id] = clamped

        return self._send(angles, speed)

    def set_all_angles(self, angles: list,
                       speed: float = None) -> bool:
        """
        Envoie les 5 angles d'un coup.

        Args:
            angles : liste de 5 floats (degrés)
            speed  : vitesse (°/step)

        Returns:
            True si la commande est acceptée
        """
        if len(angles) != 5:
            print(f"[MotorController] 5 angles requis, reçu {len(angles)}")
            return False

        speed   = self._clamp_speed(speed or self.DEFAULT_SPEED)
        clamped = [
            self._clamp_angle(i, a)
            for i, a in enumerate(angles)
        ]

        return self._send(clamped, speed)

    def set_gripper(self, state: int) -> bool:
        """
        Contrôle le gripper.

        Args:
            state : GRIPPER_OPEN (0) ou GRIPPER_CLOSED (1)

        Returns:
            True si la commande est acceptée
        """
        result = self._uart.set_gripper(state)
        if result:
            with self._lock:
                self._gripper_state = state
        return result

    def apply_motion_profile(self, start_angles: list,
                             end_angles: list,
                             duration: float = 2.0) -> bool:
        """
        Génère et exécute un profil S-curve entre deux positions.

        Args:
            start_angles : angles de départ [θ₁…θ₅]
            end_angles   : angles d'arrivée [θ₁…θ₅]
            duration     : durée totale du mouvement (secondes)

        Returns:
            True si le mouvement s'est terminé correctement
        """
        waypoints = self._generate_scurve(
            start_angles, end_angles, duration
        )

        print(
            f"[MotorController] S-curve : "
            f"{len(waypoints)} waypoints sur {duration:.1f}s"
        )

        dt = duration / len(waypoints)

        for i, wp in enumerate(waypoints):
            result = self._send(wp, speed=self.MAX_SPEED)
            if not result:
                print(f"[MotorController] Échec waypoint {i}")
                return False
            time.sleep(dt)

        return True

    def get_joint_feedback(self) -> list:
        """
        Lit les angles réels depuis les encodeurs STM32.

        Returns:
            Liste de 5 floats, ou None si erreur
        """
        status = self._uart.get_status()
        if not status:
            return None

        angles = status.get("angles", None)
        if angles and len(angles) == 5:
            with self._lock:
                self._current_angles = list(angles)
        return angles

    def emergency_stop(self) -> bool:
        """Coupure immédiate de tous les moteurs."""
        print("[MotorController] *** EMERGENCY STOP ***")
        return self._uart.emergency_stop()

    def is_at_target(self) -> bool:
        """
        Vérifie si tous les joints ont atteint leur cible.
        Compare les angles courants aux angles cibles.
        """
        feedback = self.get_joint_feedback()
        if feedback is None:
            return False

        for current, target in zip(feedback, self._target_angles):
            if abs(current - target) > self.POSITION_TOLERANCE:
                return False
        return True

    def get_current_angles(self) -> list:
        with self._lock:
            return list(self._current_angles)

    def get_gripper_state(self) -> int:
        with self._lock:
            return self._gripper_state

    def get_stats(self) -> dict:
        return {
            "move_count"     : self._move_count,
            "current_angles" : self.get_current_angles(),
            "target_angles"  : list(self._target_angles),
            "gripper"        : self._gripper_state,
            "uart_stats"     : self._uart.get_stats(),
        }

    # ------------------------------------------------------------------
    # S-curve interne
    # ------------------------------------------------------------------

    def _generate_scurve(self, start: list, end: list,
                         duration: float,
                         n_points: int = 50) -> list:
        """
        Génère une trajectoire S-curve entre start et end.

        La S-curve est obtenue via une fonction sigmoïde normalisée :
            s(t) = 1 / (1 + exp(-k*(t - 0.5)))

        Cela produit une accélération douce au départ
        et une décélération douce à l'arrivée.
        """
        t_norm  = np.linspace(0, 1, n_points)
        k       = 10.0  # Raideur de la courbe
        sigmoid = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))

        # Normaliser sigmoid entre 0 et 1
        s_min, s_max = sigmoid[0], sigmoid[-1]
        sigmoid = (sigmoid - s_min) / (s_max - s_min)

        waypoints = []
        for s in sigmoid:
            wp = [
                start[i] + s * (end[i] - start[i])
                for i in range(5)
            ]
            waypoints.append(wp)

        return waypoints

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send(self, angles: list, speed: float) -> bool:
        """Envoie les angles et met à jour l'état interne."""
        result = self._uart.send_angles(angles, speed)
        if result:
            with self._lock:
                self._target_angles = list(angles)
                self._move_count   += 1
        return result

    def _clamp_angle(self, joint_id: int, angle: float) -> float:
        """Limite l'angle dans les bornes mécaniques du joint."""
        min_a, max_a = self.JOINT_LIMITS[joint_id]
        clamped = max(min_a, min(max_a, angle))
        if clamped != angle:
            print(
                f"[MotorController] Joint {joint_id+1} : "
                f"{angle:.1f}° → clampé à {clamped:.1f}°"
            )
        return clamped

    def _clamp_speed(self, speed: float) -> float:
        """Limite la vitesse entre MIN et MAX."""
        return max(self.MIN_SPEED, min(self.MAX_SPEED, speed))