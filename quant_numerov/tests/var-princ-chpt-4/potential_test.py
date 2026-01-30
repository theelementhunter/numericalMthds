import pytest
import numpy as np
from var_princ_chpt_4.basis import PlaneWaveBasis
from var_princ_chpt_4.potentials import PotentialWell

@pytest.fixture(scope="function")
def shared_potential():
    basis = PlaneWaveBasis(N=1, a=1) # 3x3 matrix
    shared_potential = PotentialWell(basis=basis, width=1.0)
    return shared_potential


def test_mask_operations():
    """Testing that the mask operations work correctly.
    If the entire matrix is 0, then it should return a matrix
    filled with ones."""

    assert