import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.simulation import RobotSimulation
from robot_system.kinematics.kinematics import RobotKinematics


URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../models/robot_arm.urdf"
)


def _make_sim() -> RobotSimulation:
    sim = RobotSimulation(urdf_path=URDF_PATH, gui=False)
    assert sim.start() is True
    return sim


def test_simulation_starts():
    sim = _make_sim()
    stats = sim.get_stats()
    assert stats["n_joints"] == 6
    sim.stop()
    print("\nsimulation démarre OK")


def test_set_joint_angles():
    sim    = _make_sim()
    angles = [10.0, 20.0, -15.0, 5.0, 30.0]
    result = sim.set_joint_angles(angles)
    assert result is True
    sim.stop()
    print("set_joint_angles OK")


def test_get_joint_angles():
    sim    = _make_sim()
    angles = [10.0, 20.0, -15.0, 5.0, 30.0]
    sim.set_joint_angles(angles)
    readback = sim.get_joint_angles()
    assert len(readback) == 5
    for sent, read in zip(angles, readback):
        assert abs(sent - read) < 1.0, (
            f"Angle envoyé {sent}° lu {read:.2f}°"
        )
    sim.stop()
    print("get_joint_angles readback OK")


def test_end_effector_position_at_zero():
    sim = _make_sim()
    sim.set_joint_angles([0.0, 0.0, 0.0, 0.0, 0.0])
    pos = sim.get_end_effector_position()
    assert len(pos) == 3
    assert not np.allclose(pos, [0, 0, 0])
    sim.stop()
    print(f"position effecteur à zéro : {pos} OK")


def test_end_effector_changes_with_angles():
    sim  = _make_sim()
    sim.set_joint_angles([0.0, 0.0, 0.0, 0.0, 0.0])
    pos1 = sim.get_end_effector_position()
    sim.set_joint_angles([45.0, 30.0, 0.0, 0.0, 0.0])
    pos2 = sim.get_end_effector_position()
    assert not np.allclose(pos1, pos2)
    sim.stop()
    print("effecteur bouge avec les angles OK")


def test_no_self_collision_at_home():
    sim = _make_sim()
    sim.set_joint_angles([0.0, 0.0, 0.0, 0.0, 0.0])
    collision = sim.check_collision()
    assert collision is False
    sim.stop()
    print("pas de collision à home OK")


def test_validate_trajectory_home():
    """Une trajectoire à home ne doit pas avoir de collisions."""
    sim = _make_sim()
    waypoints = [[0.0, 0.0, 0.0, 0.0, 0.0]] * 10
    result    = sim.validate_trajectory(waypoints)
    assert result["total_waypoints"] == 10
    assert result["collisions"]      == 0
    assert result["success"]         is True
    sim.stop()
    print("trajectoire home validée OK")


def test_validate_scurve_trajectory():
    """
    Valide une trajectoire S-curve entre deux positions.
    Utilise RobotKinematics pour générer les waypoints.
    """
    sim = _make_sim()
    kin = RobotKinematics()

    start = [0.0,  0.0,  0.0,  0.0, 0.0]
    end   = [30.0, 20.0, -10.0, 5.0, 0.0]

    t_norm  = np.linspace(0, 1, 20)
    k       = 10.0
    sigmoid = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))
    s_min, s_max = sigmoid[0], sigmoid[-1]
    sigmoid = (sigmoid - s_min) / (s_max - s_min)

    waypoints = [
        [start[i] + s * (end[i] - start[i]) for i in range(5)]
        for s in sigmoid
    ]

    result = sim.validate_trajectory(waypoints)
    assert result["total_waypoints"] == 20
    assert result["success"] is True, (
        f"Collisions : {result['collisions']}, "
        f"Sol : {result['ground_collisions']}"
    )
    sim.stop()
    print(f"trajectoire S-curve {len(waypoints)} waypoints OK")


def test_add_target_object():
    sim    = _make_sim()
    obj_id = sim.add_target_object([0.2, 0.0, 0.1])
    assert obj_id >= 0
    stats = sim.get_stats()
    assert stats["n_objects"] == 1
    sim.stop()
    print("ajout objet cible OK")


def test_50_random_trajectories():
    """
    Valide 50 trajectoires aléatoires.
    Critère : au moins 80% sans collision.
    """
    sim      = _make_sim()
    success  = 0
    n_tests  = 50

    for _ in range(n_tests):
        angles_start = [0.0] * 5
        angles_end   = [
            float(np.random.uniform(lim[0] * 0.5, lim[1] * 0.5))
            for lim in [
                (-90, 90), (-45, 45), (-60, 60),
                (-45, 45), (-90, 90)
            ]
        ]

        t_norm  = np.linspace(0, 1, 15)
        k       = 10.0
        sigmoid = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))
        s_min, s_max = sigmoid[0], sigmoid[-1]
        sigmoid = (sigmoid - s_min) / (s_max - s_min)

        waypoints = [
            [
                angles_start[i] + s * (angles_end[i] - angles_start[i])
                for i in range(5)
            ]
            for s in sigmoid
        ]

        result = sim.validate_trajectory(waypoints)
        if result["success"]:
            success += 1

    sim.stop()
    rate = success / n_tests * 100
    assert rate >= 80.0, (
        f"Taux de succès trop bas : {rate:.1f}% (min 80%)"
    )
    print(f"50 trajectoires : {success}/50 succès ({rate:.1f}%) OK")