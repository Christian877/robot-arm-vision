import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.simulation import RobotSimulation
from models.trajectory_validator import TrajectoryValidator
from robot_system.kinematics.kinematics import RobotKinematics


URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../models/robot_arm.urdf"
)


def _make_validator():
    sim = RobotSimulation(urdf_path=URDF_PATH, gui=False)
    sim.start()
    kin       = RobotKinematics()
    validator = TrajectoryValidator(simulation=sim, kinematics=kin)
    return validator, sim


def test_generate_scurve():
    validator, sim = _make_validator()
    start     = [0.0]  * 5
    end       = [30.0] * 5
    waypoints = validator.generate_scurve(start, end, n_points=20)

    assert len(waypoints) == 20
    assert len(waypoints[0]) == 5

    for i in range(5):
        assert abs(waypoints[0][i]  - start[i]) < 1.0
        assert abs(waypoints[-1][i] - end[i])   < 1.0

    sim.stop()
    print("\nS-curve 20 waypoints OK")


def test_generate_scurve_points():
    validator, sim = _make_validator()
    start     = [0.0] * 5
    end       = [30.0, 20.0, -10.0, 5.0, 15.0]
    waypoints = validator.generate_scurve(start, end, n_points=30)

    assert len(waypoints) == 30
    assert len(waypoints[0]) == 5

    for i in range(5):
        assert abs(waypoints[0][i]  - start[i]) < 1.0
        assert abs(waypoints[-1][i] - end[i])   < 1.0

    sim.stop()
    print("S-curve start/end corrects OK")


def test_generate_random_trajectory():
    validator, sim = _make_validator()
    start, end, waypoints = validator.generate_random_trajectory(
        max_angle=45.0
    )

    assert len(start)     == 5
    assert len(end)       == 5
    assert len(waypoints) == 30

    limits = [180, 90, 120, 90, 180]
    for angle, lim in zip(end, limits):
        assert -lim <= angle <= lim

    sim.stop()
    print("trajectoire aléatoire dans les limites OK")


def test_validate_single_returns_dict():
    validator, sim = _make_validator()
    result = validator.validate_single(
        start=[0.0]*5, end=[0.0]*5, label="home"
    )

    assert isinstance(result, dict)
    for field in [
        "success", "collisions", "n_waypoints",
        "fk_max_error_mm", "duration_ms"
    ]:
        assert field in result

    assert result["n_waypoints"] == 30

    sim.stop()
    print(f"validate_single retourne dict OK")


def test_validate_single_no_collision_home():
    validator, sim = _make_validator()
    result = validator.validate_single(
        start=[0.0]*5, end=[0.0]*5, label="home"
    )
    assert result["collisions"]      == 0
    assert result["ground_contacts"] == 0
    assert result["success"]         is True

    sim.stop()
    print("home sans collision OK")


def test_fk_consistency_recorded():
    """FK erreur enregistrée même si grande — pas un critère de succès."""
    validator, sim = _make_validator()
    result = validator.validate_single(
        start=[0.0]*5,
        end=[20.0, 15.0, -10.0, 5.0, 0.0],
        label="fk_check"
    )
    assert "fk_max_error_mm" in result
    assert result["fk_max_error_mm"] >= 0.0

    sim.stop()
    print(f"FK erreur enregistrée : {result['fk_max_error_mm']:.1f}mm OK")


def test_validate_batch():
    validator, sim = _make_validator()

    trajectories = [
        ([0.0]*5, [10.0, 10.0,  0.0,  0.0,  0.0], "move_1"),
        ([0.0]*5, [20.0, 15.0, -5.0,  5.0,  0.0], "move_2"),
        ([0.0]*5, [0.0,  20.0, 10.0,  0.0, 10.0], "move_3"),
        ([0.0]*5, [15.0,  0.0,-10.0,  0.0, 20.0], "move_4"),
        ([0.0]*5, [0.0,   0.0,  0.0,  0.0,  0.0], "home"),
    ]

    report = validator.validate_batch(trajectories)

    assert report["total"] == 5
    assert "success_rate_pct" in report

    sim.stop()
    print(
        f"batch {report['total']} trajectoires — "
        f"succès {report['success_rate_pct']}% OK"
    )


def test_50_random_trajectories():
    validator, sim = _make_validator()
    report = validator.validate_n_random(n=50, max_angle=40.0)

    assert report["total"] == 50
    assert report["success_rate_pct"] >= 70.0, (
        f"Taux succès trop bas : {report['success_rate_pct']}%"
    )

    sim.stop()
    print(
        f"50 trajectoires : {report['passed']}/50 "
        f"({report['success_rate_pct']}%) OK"
    )


def test_get_report_fields():
    validator, sim = _make_validator()
    validator.validate_single([0.0]*5, [10.0]*5, "test")
    report = validator.get_report()

    for field in [
        "total", "passed", "failed",
        "success_rate_pct", "fk_error_avg_mm",
        "fk_error_max_mm", "avg_duration_ms"
    ]:
        assert field in report, f"Champ manquant : {field}"

    sim.stop()
    print(f"rapport complet OK")


def test_reset():
    validator, sim = _make_validator()
    validator.validate_single([0.0]*5, [10.0]*5)
    assert validator._n_tested == 1

    validator.reset()
    assert validator._n_tested == 0
    assert validator._n_passed == 0
    assert len(validator._results) == 0

    sim.stop()
    print("reset OK")