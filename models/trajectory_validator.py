import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.simulation import RobotSimulation
from robot_system.kinematics.kinematics import RobotKinematics


class TrajectoryValidator:

    FK_ERROR_THRESHOLD = 1000.0

    def __init__(self, simulation: RobotSimulation,
                 kinematics: RobotKinematics):
        self._sim      = simulation
        self._kin      = kinematics
        self._results  = []
        self._n_tested = 0
        self._n_passed = 0

    def generate_scurve(self, start: list, end: list,
                        n_points: int = 30) -> list:
        t_norm  = np.linspace(0, 1, n_points)
        k       = 10.0
        sigmoid = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))
        s_min, s_max = sigmoid[0], sigmoid[-1]
        sigmoid = (sigmoid - s_min) / (s_max - s_min)
        return [
            [start[i] + float(s) * (end[i] - start[i]) for i in range(5)]
            for s in sigmoid
        ]

    def generate_random_trajectory(self,
                                   max_angle: float = 45.0) -> tuple:
        start  = [0.0] * 5
        limits = [180, 90, 120, 90, 180]
        end    = [
            float(np.clip(
                np.random.uniform(-max_angle, max_angle),
                -lim, lim
            ))
            for lim in limits
        ]
        waypoints = self.generate_scurve(start, end)
        return start, end, waypoints

    def validate_single(self, start: list, end: list,
                        label: str = "") -> dict:
        t_start    = time.time()
        waypoints  = self.generate_scurve(start, end)
        sim_result = self._sim.validate_trajectory(waypoints)
        fk_errors  = self._check_fk_consistency(waypoints)
        duration   = time.time() - t_start

        # Succès basé uniquement sur les collisions
        success = sim_result["success"]

        result = {
            "label"           : label or f"traj_{self._n_tested}",
            "start"           : start,
            "end"             : end,
            "n_waypoints"     : len(waypoints),
            "collisions"      : sim_result["collisions"],
            "ground_contacts" : sim_result["ground_collisions"],
            "fk_max_error_mm" : round(fk_errors["max_error_mm"], 3),
            "fk_avg_error_mm" : round(fk_errors["avg_error_mm"], 3),
            "duration_ms"     : round(duration * 1000, 1),
            "success"         : success,
        }

        self._n_tested += 1
        if success:
            self._n_passed += 1

        self._results.append(result)
        return result

    def validate_batch(self, trajectories: list) -> dict:
        print(
            f"[TrajectoryValidator] Validation de "
            f"{len(trajectories)} trajectoires..."
        )
        for i, traj in enumerate(trajectories):
            if len(traj) == 3:
                start, end, label = traj
            else:
                start, end = traj
                label      = f"traj_{i}"

            result = self.validate_single(start, end, label)
            status = "OK" if result["success"] else "KO"
            print(
                f"  [{status}] {label:<20} "
                f"collisions={result['collisions']} "
                f"fk_err={result['fk_max_error_mm']:.1f}mm"
            )
        return self.get_report()

    def validate_n_random(self, n: int = 50,
                          max_angle: float = 45.0) -> dict:
        print(
            f"[TrajectoryValidator] "
            f"Validation {n} trajectoires aléatoires..."
        )
        for i in range(n):
            start, end, _ = self.generate_random_trajectory(max_angle)
            result        = self.validate_single(
                start, end, label=f"random_{i:03d}"
            )
            if i % 10 == 0:
                status = "OK" if result["success"] else "KO"
                print(
                    f"  [{status}] {i+1}/{n} — "
                    f"succès={self._n_passed}/{self._n_tested}"
                )
        return self.get_report()

    def _check_fk_consistency(self, waypoints: list) -> dict:
        errors = []
        for wp in waypoints:
            self._sim.set_joint_angles(wp)
            pos_sim = self._sim.get_end_effector_position()
            pos_fk  = self._kin.forward_kinematics(wp)
            error   = np.linalg.norm(pos_sim - pos_fk)
            errors.append(error)
        return {
            "max_error_mm" : float(np.max(errors))  if errors else 0.0,
            "avg_error_mm" : float(np.mean(errors)) if errors else 0.0,
            "min_error_mm" : float(np.min(errors))  if errors else 0.0,
        }

    def get_report(self) -> dict:
        if not self._results:
            return {"error": "Aucune trajectoire validée"}

        success_rate = self._n_passed / self._n_tested * 100
        fk_errors    = [r["fk_max_error_mm"] for r in self._results]
        durations    = [r["duration_ms"]     for r in self._results]

        report = {
            "total"           : self._n_tested,
            "passed"          : self._n_passed,
            "failed"          : self._n_tested - self._n_passed,
            "success_rate_pct": round(success_rate, 1),
            "fk_error_avg_mm" : round(float(np.mean(fk_errors)), 3),
            "fk_error_max_mm" : round(float(np.max(fk_errors)),  3),
            "avg_duration_ms" : round(float(np.mean(durations)), 1),
            "results"         : self._results,
        }

        print(
            f"\n[TrajectoryValidator] Rapport :\n"
            f"  Total     : {report['total']}\n"
            f"  Succès    : {report['passed']} "
            f"({report['success_rate_pct']}%)\n"
            f"  Échecs    : {report['failed']}\n"
            f"  FK erreur : {report['fk_error_avg_mm']}mm moy, "
            f"{report['fk_error_max_mm']}mm max\n"
            f"  Durée moy : {report['avg_duration_ms']}ms"
        )
        return report

    def reset(self):
        self._results  = []
        self._n_tested = 0
        self._n_passed = 0