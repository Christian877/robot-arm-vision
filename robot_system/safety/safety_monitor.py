import threading
import time


class SafetyMonitor(threading.Thread):
    """
    Surveille l'état du robot à 10 Hz en thread indépendant.

    Vérifie en continu :
    - Limites articulaires
    - Surcharge moteur (courant)
    - Tension alimentation
    - Température moteurs

    Déclenche emergency_stop() automatiquement si
    une condition est violée.
    """

    # Seuils de sécurité
    MAX_TEMPERATURE  = 70.0   # °C
    MIN_VOLTAGE      = 10.5   # V
    MAX_VOLTAGE      = 13.5   # V
    MAX_JOINT_ERROR  = 10.0   # degrés — écart max courant/cible

    # Limites articulaires (degrés)
    JOINT_LIMITS = [
        (-180, 180),
        (-90,  90),
        (-120, 120),
        (-90,  90),
        (-180, 180),
    ]

    MONITOR_HZ = 10  # Fréquence de surveillance

    def __init__(self, uart_driver, motor_controller=None):
        super().__init__(daemon=True)
        self._uart    = uart_driver
        self._motor   = motor_controller
        self._running = False
        self._lock    = threading.Lock()

        # Compteurs
        self._check_count    = 0
        self._violation_count = 0
        self._estop_triggered = False

        # Callbacks — appelé quand e-stop déclenché
        self._on_estop_callbacks = []

        # Dernier status connu
        self._last_status = {}

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        threading.Thread.start(self)
        print(
            f"[SafetyMonitor] Démarré — "
            f"surveillance à {self.MONITOR_HZ} Hz"
        )

    def stop(self):
        self._running = False
        self.join(timeout=2)
        print("[SafetyMonitor] Arrêté")

    def register_estop_callback(self, callback):
        """Enregistre une fonction appelée lors d'un e-stop."""
        self._on_estop_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self):
        interval = 1.0 / self.MONITOR_HZ  # 100ms

        while self._running:
            t_start = time.time()

            self._run_checks()

            elapsed    = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_checks(self):
        """Lance tous les checks de sécurité."""
        status = self._uart.get_status()
        if not status:
            return

        with self._lock:
            self._last_status  = status
            self._check_count += 1

        violations = []

        if not self.check_joint_limits(status):
            violations.append("joint_limits")

        if not self.check_temperature(status):
            violations.append("temperature")

        if not self.check_voltage(status):
            violations.append("voltage")

        if violations:
            with self._lock:
                self._violation_count += 1

            print(
                f"[SafetyMonitor] Violation détectée : "
                f"{violations} — déclenchement e-stop"
            )
            self.emergency_stop()

    # ------------------------------------------------------------------
    # Checks individuels
    # ------------------------------------------------------------------

    def check_joint_limits(self, status: dict = None) -> bool:
        """
        Vérifie que tous les joints sont dans leurs limites.

        Returns:
            True si tous les joints sont dans les limites
        """
        if status is None:
            status = self._uart.get_status()
        if not status:
            return True

        angles = status.get("angles", [])
        if len(angles) != 5:
            return True

        for i, (angle, (min_a, max_a)) in enumerate(
            zip(angles, self.JOINT_LIMITS)
        ):
            if not (min_a <= angle <= max_a):
                print(
                    f"[SafetyMonitor] Joint {i+1} hors limite : "
                    f"{angle:.1f}° (limite [{min_a}°, {max_a}°])"
                )
                return False
        return True

    def check_temperature(self, status: dict = None) -> bool:
        """
        Vérifie que la température est sous le seuil.

        Returns:
            True si température acceptable
        """
        if status is None:
            status = self._uart.get_status()
        if not status:
            return True

        temp = status.get("temperature", 0.0)
        if temp > self.MAX_TEMPERATURE:
            print(
                f"[SafetyMonitor] Température critique : "
                f"{temp:.1f}°C (max {self.MAX_TEMPERATURE}°C)"
            )
            return False
        return True

    def check_voltage(self, status: dict = None) -> bool:
        """
        Vérifie que la tension est dans la plage acceptable.

        Returns:
            True si tension acceptable
        """
        if status is None:
            status = self._uart.get_status()
        if not status:
            return True

        voltage = status.get("voltage", 12.0)
        if not (self.MIN_VOLTAGE <= voltage <= self.MAX_VOLTAGE):
            print(
                f"[SafetyMonitor] Tension hors plage : "
                f"{voltage:.2f}V "
                f"(plage [{self.MIN_VOLTAGE}V, {self.MAX_VOLTAGE}V])"
            )
            return False
        return True

    def detect_overload(self, status: dict = None) -> bool:
        """
        Détecte une surcharge moteur.
        En mode mock, toujours False (pas de données courant).

        Returns:
            True si surcharge détectée
        """
        return False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def emergency_stop(self) -> None:
        """Déclenche l'arrêt d'urgence et notifie les callbacks."""
        with self._lock:
            if self._estop_triggered:
                return
            self._estop_triggered = True

        print("[SafetyMonitor] *** EMERGENCY STOP DÉCLENCHÉ ***")
        self._uart.emergency_stop()

        for callback in self._on_estop_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[SafetyMonitor] Erreur callback : {e}")

    def watchdog_reset(self) -> None:
        """
        Signal de vie envoyé périodiquement au STM32.
        Si absent, le STM32 coupe les moteurs de lui-même.
        En mode mock, pas implémenté.
        """
        pass

    def reset(self) -> None:
        """Réinitialise l'e-stop après intervention."""
        with self._lock:
            self._estop_triggered = False
        self._uart.reset_estop()
        print("[SafetyMonitor] Réinitialisé")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "check_count"     : self._check_count,
                "violation_count" : self._violation_count,
                "estop_triggered" : self._estop_triggered,
                "last_status"     : dict(self._last_status),
            }

    def is_safe(self) -> bool:
        """Retourne True si aucun e-stop n'est actif."""
        with self._lock:
            return not self._estop_triggered