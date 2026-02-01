import pytest
import numpy as np
from var_princ_chpt_4.basis import PlaneWaveBasis
from var_princ_chpt_4.potentials import PotentialWell

@pytest.fixture(scope="function")
def shared_potential():
    basis = PlaneWaveBasis(N=1, a=2) # 3x3 matrix
    shared_potential = PotentialWell(basis=basis, width=1.0, V0= 1)
    return shared_potential


def test_potential_matrix_values(shared_potential):
    # Are diagonal elements negative?
    expected_diag = -( shared_potential.V0 * shared_potential.width ) / shared_potential.a
    actual_diag = shared_potential.potential_matrix()[0,0]

    assert np.isclose(actual_diag, expected_diag), \
    f"Diagonal mismatch! Expected {expected_diag}, got {actual_diag}."

    # Are off-diagonal elements correctly calculated
    # TEST 2: Off-Diagonal (Eq 4.53)
    # Pick indices where you know the q-difference (e.g., k=0 vs k=1)
    q_val = shared_potential.basis.k_vector[1] - shared_potential.basis.k_vector[0]
    expected_off = -( (shared_potential.V0 / shared_potential.a) * np.sin(q_val * shared_potential.width / 2) ) / (q_val / 2)
    print(expected_off)
    print(shared_potential.potential_matrix()[0, 1])

    assert np.isclose(shared_potential.potential_matrix()[0, 1], expected_off), "Off-diagonal formula mismatch"
