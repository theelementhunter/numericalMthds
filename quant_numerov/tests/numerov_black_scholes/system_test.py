import pytest
import numpy as np
from numerov_black_scholes.system import BlackScholesSystem

def test_dx():
    test_sys = BlackScholesSystem(
        L = 10,
        dt = 0.01,
        N_points= 1000,
        T_final= 1.0,
    )

    assert np.isclose(test_sys.dx, 0.01001001)