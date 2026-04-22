import threading
import time
from enum import Enum, auto


class RobotState(Enum):
    IDLE         = auto()
    SCANNING     = auto()
    TARGETING    = auto()
    MOVING       = auto()
    GRASPING     = auto()
    TRANSPORTING = auto()
    RELEASING    = auto()
    HOMING       = auto()
    ERROR        = auto()


class RobotController(threading.Thread):
    """
    Orchestrateur central du bras robotisé — Machine à états (FSM).

    États :
        IDLE         → en attente
        SCANNING     → caméra active, cherche un objet
        TARGETING    → objet détecté, calcul IK
        MOVING       → déplacement vers l'objet
        GRASPING     → fermeture du gripper
        TRANSPORTING → déplacement vers la zone de dépôt
        RELEASING    → ouverture du gripper
        HOMING       → retour position repos
        ERROR        → erreur, attente intervention

    Timeouts par état (secondes) :
        SCANNING     → 10s
        TARGETING    → 5s
        MOVING       → 15s
        GRASPING     → 5s
        TRANSPORTING → 15s
        RELEASING    → 5s
        HOMING       → 15s
    """

    STATE_TIMEOUTS = {
        RobotState.SCANNING     : 10.0,
        RobotState.TARGETING    : 5.0,
        RobotState.MOVING       : 15.0,
        RobotState.GRASPING     : 5.0,
        RobotState.TRANSPORTING : 15.0,
        RobotState.RELEASING    : 5.0,
        RobotState.HOMING       : 15.0,
    }

    # Position de repos (degrés)
    HOME_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Position de dépôt
    DROP_ANGLES = [90.0, 20.0, -10.0, 0.0, 0.0]

    # Fréquence boucle FSM (Hz)
    LOOP_HZ = 10

    def __init__(self, vision_buffer, motor_controller, kinematics):
        super().__init__(daemon=True)
        self._vision   = vision_buffer
        self._motor    = motor_controller
        self._kin      = kinematics

        # État courant
        self._state      = RobotState.IDLE
        self._prev_state = None
        self._lock       = threading.Lock()
        self._running    = False

        # Timing état courant
        self._state_start_time = None

        # Données inter-états
        self._current_detection = None
        self._target_angles     = None

        # Compteurs
        self._cycle_count     = 0
        self._error_count     = 0
        self._success_count   = 0

        # Callbacks état
        self._state_callbacks = []

        print("[RobotController] FSM initialisée")

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        threading.Thread.start(self)
        print("[RobotController] Démarré")

    def stop(self):
        self._running = False
        self.join(timeout=3)
        print("[RobotController] Arrêté")

    def register_state_callback(self, callback):
        """
        Enregistre un callback appelé à chaque changement d'état.
        callback(old_state, new_state)
        """
        self._state_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        with self._lock:
            return self._state

    def start_scan(self) -> bool:
        """Lance un cycle de scan depuis IDLE."""
        with self._lock:
            if self._state != RobotState.IDLE:
                print(
                    f"[RobotController] Impossible de scanner "
                    f"depuis l'état {self._state.name}"
                )
                return False
            self._transition_to(RobotState.SCANNING)
            return True

    def manual_home(self) -> bool:
        """Force le retour à la position home."""
        with self._lock:
            self._transition_to(RobotState.HOMING)
            return True

    def reset_error(self) -> bool:
        """Réinitialise depuis l'état ERROR."""
        with self._lock:
            if self._state != RobotState.ERROR:
                return False
            self._transition_to(RobotState.HOMING)
            return True

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "state"         : self._state.name,
                "cycle_count"   : self._cycle_count,
                "error_count"   : self._error_count,
                "success_count" : self._success_count,
            }

    # ------------------------------------------------------------------
    # Boucle principale FSM
    # ------------------------------------------------------------------

    def run(self):
        interval = 1.0 / self.LOOP_HZ

        while self._running:
            t_start = time.time()

            with self._lock:
                self._step()

            elapsed    = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step(self):
        """Exécute un step de la FSM selon l'état courant."""
        state = self._state

        if self._is_timed_out():
            print(
                f"[RobotController] Timeout état "
                f"{state.name} — passage en ERROR"
            )
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        if state == RobotState.IDLE:
            self._handle_idle()
        elif state == RobotState.SCANNING:
            self._handle_scanning()
        elif state == RobotState.TARGETING:
            self._handle_targeting()
        elif state == RobotState.MOVING:
            self._handle_moving()
        elif state == RobotState.GRASPING:
            self._handle_grasping()
        elif state == RobotState.TRANSPORTING:
            self._handle_transporting()
        elif state == RobotState.RELEASING:
            self._handle_releasing()
        elif state == RobotState.HOMING:
            self._handle_homing()
        elif state == RobotState.ERROR:
            self._handle_error()

    # ------------------------------------------------------------------
    # Handlers d'états
    # ------------------------------------------------------------------

    def _handle_idle(self):
        """IDLE — attend une commande externe."""
        pass

    def _handle_scanning(self):
        """SCANNING — cherche un objet via VisionBuffer."""
        detection = self._vision.get_best_detection()

        if detection is None:
            return  # Pas encore de détection

        if detection.confidence < 0.5:
            return  # Confiance insuffisante

        print(
            f"[RobotController] Objet détecté : "
            f"{detection.label} (conf={detection.confidence:.2f})"
        )
        self._current_detection = detection
        self._transition_to(RobotState.TARGETING)

    def _handle_targeting(self):
        """TARGETING — calcule l'IK pour atteindre l'objet."""
        if self._current_detection is None:
            self._transition_to(RobotState.ERROR)
            return

        cx, cy = self._current_detection.center

        # Convertir pixels → mm (approximation)
        target_x = (cx - 320) * 0.5
        target_y = 200.0
        target_z = (240 - cy) * 0.5

        solution = self._kin.inverse_kinematics(
            [target_x, target_y, target_z],
            initial_guess=self.HOME_ANGLES
        )

        if solution is None:
            print("[RobotController] IK non convergée — ERROR")
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        self._target_angles = solution
        print(
            f"[RobotController] IK résolue : "
            f"{[f'{a:.1f}' for a in solution]}"
        )
        self._transition_to(RobotState.MOVING)

    def _handle_moving(self):
        """MOVING — déplace le bras vers la cible."""
        if self._target_angles is None:
            self._transition_to(RobotState.ERROR)
            return

        result = self._motor.set_all_angles(
            self._target_angles, speed=10.0
        )

        if not result:
            print("[RobotController] Échec commande moteur — ERROR")
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        # Attendre que la position soit atteinte
        if self._motor.is_at_target():
            self._transition_to(RobotState.GRASPING)

    def _handle_grasping(self):
        """GRASPING — ferme le gripper."""
        result = self._motor.set_gripper(1)  # CLOSED

        if not result:
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        self._transition_to(RobotState.TRANSPORTING)

    def _handle_transporting(self):
        """TRANSPORTING — déplace vers la zone de dépôt."""
        result = self._motor.set_all_angles(
            self.DROP_ANGLES, speed=8.0
        )

        if not result:
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        if self._motor.is_at_target():
            self._transition_to(RobotState.RELEASING)

    def _handle_releasing(self):
        """RELEASING — ouvre le gripper."""
        result = self._motor.set_gripper(0)  # OPEN

        if not result:
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        self._success_count += 1
        self._cycle_count   += 1
        print(
            f"[RobotController] Cycle {self._cycle_count} "
            f"terminé avec succès"
        )
        self._transition_to(RobotState.HOMING)

    def _handle_homing(self):
        """HOMING — retour position repos."""
        result = self._motor.set_all_angles(
            self.HOME_ANGLES, speed=8.0
        )

        if not result:
            self._error_count += 1
            self._transition_to(RobotState.ERROR)
            return

        if self._motor.is_at_target():
            self._current_detection = None
            self._target_angles     = None
            self._transition_to(RobotState.IDLE)

    def _handle_error(self):
        """ERROR — attend une intervention manuelle."""
        pass

    # ------------------------------------------------------------------
    # Transitions et helpers
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: RobotState):
        """Effectue une transition d'état."""
        old_state = self._state

        if old_state == new_state:
            return

        print(
            f"[RobotController] "
            f"{old_state.name} → {new_state.name}"
        )

        self._prev_state       = old_state
        self._state            = new_state
        self._state_start_time = time.time()

        for cb in self._state_callbacks:
            try:
                cb(old_state, new_state)
            except Exception as e:
                print(f"[RobotController] Erreur callback : {e}")

    def _is_timed_out(self) -> bool:
        """Vérifie si l'état courant a dépassé son timeout."""
        timeout = self.STATE_TIMEOUTS.get(self._state)
        if timeout is None:
            return False
        if self._state_start_time is None:
            return False
        return (time.time() - self._state_start_time) > timeout

    def error_recovery(self):
        """Tente une récupération depuis ERROR."""
        with self._lock:
            if self._state == RobotState.ERROR:
                print("[RobotController] Tentative recovery → HOMING")
                self._transition_to(RobotState.HOMING)