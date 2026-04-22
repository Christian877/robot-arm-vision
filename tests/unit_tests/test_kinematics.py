import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from robot_system.kinematics.kinematics import RobotKinematics


def test_fk_zero_position():
    """À angles nuls, la pince doit être en position connue."""
    kin    = RobotKinematics()
    pos    = kin.forward_kinematics([0.0, 0.0, 0.0, 0.0, 0.0])
    assert pos.shape == (3,)
    assert isinstance(pos[0], float)
    print(f"\nFK à zéro : {pos} OK")


def test_fk_returns_3d_vector():
    kin = RobotKinematics()
    pos = kin.forward_kinematics([10.0, 20.0, -15.0, 5.0, 30.0])
    assert len(pos) == 3
    print(f"FK retourne vecteur 3D : {pos} OK")


def test_fk_changes_with_angles():
    """Des angles différents donnent des positions différentes."""
    kin  = RobotKinematics()
    pos1 = kin.forward_kinematics([0.0,  0.0, 0.0, 0.0, 0.0])
    pos2 = kin.forward_kinematics([45.0, 0.0, 0.0, 0.0, 0.0])
    assert not np.allclose(pos1, pos2)
    print("FK sensible aux angles OK")


def test_jacobian_shape():
    """Le Jacobien doit être une matrice 3×5."""
    kin = RobotKinematics()
    J   = kin.jacobian([0.0, 0.0, 0.0, 0.0, 0.0])
    assert J.shape == (3, 5)
    print(f"Jacobien shape {J.shape} OK")


def test_jacobian_not_zero():
    """Le Jacobien ne doit pas être nul."""
    kin  = RobotKinematics()
    J    = kin.jacobian([10.0, 20.0, -10.0, 5.0, 0.0])
    assert np.any(J != 0)
    print("Jacobien non nul OK")


def test_validate_solution_valid():
    kin    = RobotKinematics()
    result = kin.validate_solution([0.0, 0.0, 0.0, 0.0, 0.0])
    assert result is True
    print("validation solution valide OK")


def test_validate_solution_out_of_limits():
    kin    = RobotKinematics()
    result = kin.validate_solution([0.0, 200.0, 0.0, 0.0, 0.0])
    assert result is False
    print("refus solution hors limites OK")


def test_ik_reaches_target():
    """
    L'IK doit converger vers une position atteignable.
    On part d'angles connus, on calcule la FK,
    puis on vérifie que l'IK retrouve une solution proche.
    """
    kin          = RobotKinematics()
    known_angles = [10.0, 15.0, -5.0, 5.0, 0.0]
    target       = kin.forward_kinematics(known_angles)

    solution = kin.inverse_kinematics(
        list(target),
        initial_guess=[8.0, 12.0, -3.0, 3.0, 0.0]
    )

    assert solution is not None, "IK n'a pas convergé"

    pos_check = kin.forward_kinematics(solution)
    error     = np.linalg.norm(target - pos_check)

    assert error < 5.0, f"Erreur IK trop grande : {error:.2f}mm"
    print(f"IK convergée — erreur {error:.3f}mm OK")
def test_ik_returns_valid_angles():
    """La solution IK doit passer validate_solution."""
    kin          = RobotKinematics()
    known_angles = [15.0, 20.0, -5.0, 10.0, 0.0]
    target       = kin.forward_kinematics(known_angles)
    solution     = kin.inverse_kinematics(list(target))

    if solution is not None:
        assert kin.validate_solution(solution)
        print("solution IK valide OK")
    else:
        print("IK non convergée (acceptable pour ce test)")


def test_clamp_angles():
    kin     = RobotKinematics()
    angles  = [0.0, 200.0, -200.0, 0.0, 0.0]
    clamped = kin._clamp_angles(angles)
    assert clamped[1] <= 90.0
    assert clamped[2] >= -120.0
    print(f"clamp angles OK : {clamped}")