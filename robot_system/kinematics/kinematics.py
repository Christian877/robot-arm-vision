import numpy as np


class RobotKinematics:

    DH_PARAMS = [
        (0,   100, np.pi / 2),
        (200,   0, 0),
        (150,   0, 0),
        (0,     0, np.pi / 2),
        (0,    80, 0),
    ]

    JOINT_LIMITS_RAD = [
        (-np.pi,          np.pi),
        (-np.pi / 2,      np.pi / 2),
        (-2 * np.pi / 3,  2 * np.pi / 3),
        (-np.pi / 2,      np.pi / 2),
        (-np.pi,          np.pi),
    ]

    IK_TOLERANCE = 1.0
    IK_MAX_ITER  = 1000
    IK_STEP_SIZE = 0.3

    def __init__(self):
        print("[Kinematics] Bras 5 DOF initialisé")

    def forward_kinematics(self, angles_deg):
        angles_rad = [np.deg2rad(a) for a in angles_deg]
        T = np.eye(4)
        for theta, (a, d, alpha) in zip(angles_rad, self.DH_PARAMS):
            T = T @ self._dh_matrix(theta, a, d, alpha)
        return T[:3, 3]

    def _dh_matrix(self, theta, a, d, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])

    def jacobian(self, angles_deg):
        angles = list(angles_deg)
        J      = np.zeros((3, 5))
        delta  = 0.5
        for j in range(5):
            ap = list(angles)
            am = list(angles)
            ap[j] += delta
            am[j] -= delta
            J[:, j] = (
                self.forward_kinematics(ap) - self.forward_kinematics(am)
            ) / (2.0 * delta)
        return J

    def inverse_kinematics(self, target, initial_guess=None):
        target  = np.array(target, dtype=float)
        angles  = list(initial_guess) if initial_guess else [0.0] * 5
        damping = 0.1

        for iteration in range(self.IK_MAX_ITER):
            pos   = self.forward_kinematics(angles)
            error = target - pos
            dist  = np.linalg.norm(error)

            if dist < self.IK_TOLERANCE:
                print(
                    f"[Kinematics] IK convergée en "
                    f"{iteration} itérations — erreur {dist:.3f}mm"
                )
                if self.validate_solution(angles):
                    return angles
                print("[Kinematics] Solution hors limites")
                return None

            J      = self.jacobian(angles)
            JJt    = J @ J.T
            J_pinv = J.T @ np.linalg.inv(JJt + damping ** 2 * np.eye(3))
            dtheta = self.IK_STEP_SIZE * J_pinv @ error

            angles = [angles[i] + dtheta[i] for i in range(5)]
            angles = self._clamp_angles(angles)

        print(
            f"[Kinematics] IK non convergée après "
            f"{self.IK_MAX_ITER} itérations"
        )
        return None

    def validate_solution(self, angles_deg):
        if len(angles_deg) != 5:
            return False
        for angle, (min_r, max_r) in zip(angles_deg, self.JOINT_LIMITS_RAD):
            if not (min_r <= np.deg2rad(angle) <= max_r):
                return False
        return True

    def _clamp_angles(self, angles_deg):
        clamped = []
        for angle, (min_r, max_r) in zip(angles_deg, self.JOINT_LIMITS_RAD):
            min_d = np.rad2deg(min_r)
            max_d = np.rad2deg(max_r)
            clamped.append(max(min_d, min(max_d, angle)))
        return clamped

    def angles_to_rad(self, angles_deg):
        return [np.deg2rad(a) for a in angles_deg]

    def angles_to_deg(self, angles_rad):
        return [np.rad2deg(a) for a in angles_rad]