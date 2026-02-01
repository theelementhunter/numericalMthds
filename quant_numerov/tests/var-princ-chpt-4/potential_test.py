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

def test_full_box_limit(shared_potential):
    """If width == box_size, potential is constant -V0 everywhere."""
    V0 = 10.0
    # Diagonal should be -V0
    assert np.allclose(np.diag(shared_potential.potential_matrix()), -V0), \
        "Diagonal should be -V0 for full width well"

    # Off-diagonal should be 0
    # Sum of abs vals of off-diags should be near 0
    matrix = shared_potential.potential_matrix()
    np.fill_diagonal(matrix, 0)
    assert np.allclose(matrix, 0,
                       atol=1e-9), "Off-diagonals should be zero for constant potential"

def test_hermicity(shared_potential):
    """Hamiltonian must be Hermition for eigenvalues to be real numbers."""
    matrix = shared_potential.potential_matrix()
    assert np.allclose(matrix, matrix.T), "Potential Matrix is not symmetric. Fix that. Plz. And thxu."