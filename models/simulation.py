import pybullet as p
import pybullet_data
import numpy as np
import time
import os


class RobotSimulation:
    """
    Environnement de simulation PyBullet du bras robotisé.

    Fonctionnalités :
    - Charge le modèle URDF
    - Applique les angles joints
    - Lit la position de l'effecteur
    - Détecte les collisions
    - Valide les trajectoires
    """

    # Indices des joints motorisés (0-4)
    JOINT_INDICES = [0, 1, 2, 3, 4]

    # Limites joints (radians) — identiques à RobotKinematics
    JOINT_LIMITS_RAD = [
        (-np.pi,          np.pi),
        (-np.pi / 2,      np.pi / 2),
        (-2 * np.pi / 3,  2 * np.pi / 3),
        (-np.pi / 2,      np.pi / 2),
        (-np.pi,          np.pi),
    ]

    # Indice end effector (joint fixe après joint_5)
    EE_JOINT_INDEX = 5

    def __init__(self, urdf_path: str = None, gui: bool = False):
        self._gui      = gui
        self._robot_id = None
        self._plane_id = None
        self._physics  = None
        self._objects  = []

        if urdf_path is None:
            # Chemin par défaut
            base = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(base, "robot_arm.urdf")

        self._urdf_path = urdf_path

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Initialise PyBullet et charge le modèle URDF.

        Returns:
            True si chargement réussi
        """
        try:
            mode = p.GUI if self._gui else p.DIRECT
            self._physics = p.connect(mode)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # Sol
            self._plane_id = p.loadURDF("plane.urdf")

            # Robot
            self._robot_id = p.loadURDF(
                self._urdf_path,
                basePosition   = [0, 0, 0],
                baseOrientation = p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase   = True
            )

            n = p.getNumJoints(self._robot_id)
            print(
                f"[Simulation] Démarrée — "
                f"{n} joints chargés depuis {self._urdf_path}"
            )
            return True

        except Exception as e:
            print(f"[Simulation] Erreur démarrage : {e}")
            return False

    def stop(self):
        """Ferme la simulation PyBullet."""
        try:
            p.disconnect(self._physics)
            print("[Simulation] Arrêtée")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Contrôle joints
    # ------------------------------------------------------------------

    def set_joint_angles(self, angles_deg: list) -> bool:
        """
        Applique les angles aux joints du robot simulé.

        Args:
            angles_deg : liste de 5 angles en degrés

        Returns:
            True si appliqué correctement
        """
        if self._robot_id is None:
            return False

        if len(angles_deg) != 5:
            print(f"[Simulation] 5 angles requis, reçu {len(angles_deg)}")
            return False

        angles_rad = [np.deg2rad(a) for a in angles_deg]

        for i, (joint_idx, angle_rad) in enumerate(
            zip(self.JOINT_INDICES, angles_rad)
        ):
            min_r, max_r = self.JOINT_LIMITS_RAD[i]
            angle_rad    = max(min_r, min(max_r, angle_rad))

            p.resetJointState(
                self._robot_id,
                joint_idx,
                angle_rad
            )

        p.stepSimulation()
        return True

    def get_joint_angles(self) -> list:
        """
        Lit les angles courants des joints.

        Returns:
            Liste de 5 angles en degrés
        """
        if self._robot_id is None:
            return []

        angles = []
        for idx in self.JOINT_INDICES:
            state = p.getJointState(self._robot_id, idx)
            angles.append(np.rad2deg(state[0]))

        return angles

    def get_end_effector_position(self) -> np.ndarray:
        """
        Retourne la position 3D de l'effecteur en mm.

        Returns:
            Array [x, y, z] en mm
        """
        if self._robot_id is None:
            return np.zeros(3)

        state = p.getLinkState(
            self._robot_id,
            self.EE_JOINT_INDEX
        )

        # PyBullet retourne en mètres → convertir en mm
        pos = np.array(state[0]) * 1000.0
        return pos

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def check_collision(self) -> bool:
        """
        Vérifie s'il y a une collision entre les liens du robot.

        Returns:
            True si collision détectée
        """
        if self._robot_id is None:
            return False

        p.stepSimulation()
        contacts = p.getContactPoints(
            bodyA=self._robot_id,
            bodyB=self._robot_id
        )

        return len(contacts) > 0

    def check_collision_with_ground(self) -> bool:
        """
        Vérifie collision entre robot et sol.

        Returns:
            True si collision avec le sol
        """
        if self._robot_id is None or self._plane_id is None:
            return False

        contacts = p.getContactPoints(
            bodyA=self._robot_id,
            bodyB=self._plane_id
        )

        # Ignorer contact base_link avec le sol (normal)
        real_contacts = [
            c for c in contacts
            if c[3] != 0  # Ignorer link 0 (base)
        ]

        return len(real_contacts) > 0

    # ------------------------------------------------------------------
    # Validation trajectoire
    # ------------------------------------------------------------------

    def validate_trajectory(self, waypoints: list) -> dict:
        """
        Valide une trajectoire complète.

        Pour chaque waypoint :
        - Applique les angles
        - Vérifie les collisions
        - Enregistre la position effecteur

        Args:
            waypoints : liste de listes d'angles [θ₁…θ₅] en degrés

        Returns:
            dict avec résultats de validation
        """
        results = {
            "total_waypoints"   : len(waypoints),
            "valid_waypoints"   : 0,
            "collisions"        : 0,
            "ground_collisions" : 0,
            "positions"         : [],
            "success"           : False,
        }

        for i, wp in enumerate(waypoints):
            self.set_joint_angles(wp)

            pos = self.get_end_effector_position()
            results["positions"].append(pos.tolist())

            if self.check_collision():
                results["collisions"] += 1
            elif self.check_collision_with_ground():
                results["ground_collisions"] += 1
            else:
                results["valid_waypoints"] += 1

        results["success"] = (
            results["collisions"] == 0 and
            results["ground_collisions"] == 0
        )

        return results

    def add_target_object(self, position: list,
                          radius: float = 0.03) -> int:
        """
        Ajoute un objet cible dans la scène.

        Args:
            position : [x, y, z] en mètres
            radius   : rayon de la sphère

        Returns:
            ID de l'objet dans PyBullet
        """
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius     = radius,
            rgbaColor  = [1, 0.5, 0, 1]
        )
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius = radius
        )
        obj_id = p.createMultiBody(
            baseMass            = 0.1,
            baseCollisionShapeIndex = collision,
            baseVisualShapeIndex    = visual,
            basePosition            = position
        )
        self._objects.append(obj_id)
        print(f"[Simulation] Objet ajouté — id={obj_id} pos={position}")
        return obj_id

    def get_stats(self) -> dict:
        """Retourne les informations sur la simulation."""
        if self._robot_id is None:
            return {}
        return {
            "robot_id"    : self._robot_id,
            "n_joints"    : p.getNumJoints(self._robot_id),
            "n_objects"   : len(self._objects),
            "ee_position" : self.get_end_effector_position().tolist(),
            "gui_mode"    : self._gui,
        }