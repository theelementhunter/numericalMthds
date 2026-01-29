import pytest
import numpy as np
from var_princ_chpt_4.basis import PlaneWaveBasis

def test_matrix_sizes():
    N = 5
    a = 1
    planewavebasis = PlaneWaveBasis(N,a)

    kinetic_matrix = planewavebasis.build_kinetic_matrix()
    diff_matrix = planewavebasis.build_momentum_difference_matrix()

    assert kinetic_matrix.shape == (2*N+1,2*N+1)
    assert diff_matrix.shape == (2*N+1,2*N+1)