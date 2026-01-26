import pytest
import numpy as np
from var_princ_chpt_4.basis import PlaneWaveBasis

def test_kinetic_matrix():
    N = 5
    a = 1
    planewavebasis = PlaneWaveBasis(N,a)

    kinetic_matrix = planewavebasis.build_kinetic_matrix()

    print(kinetic_matrix)

    assert kinetic_matrix.shape == (2*N+1,2*N+1)