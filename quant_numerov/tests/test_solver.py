import pytest
import numpy as np
from quant_numerov.src.numerov_schrod_eqn.potentials import HarmonicOscillator
from quant_numerov.src.numerov_schrod_eqn.solver import NumerovSolver


def test_harmonic_first_excited_state():
    x_grid = np.linspace(0, 5, 1000)
    potential = HarmonicOscillator(k = 1.0)
    solver = NumerovSolver(grid = x_grid, potential = potential)

    #Soln Results
    result = solver.solve(n_target=1)

    # n=1 is Odd parity, Energy = 1.5
    assert result.parity == "odd"
    assert result.energy == pytest.approx(1.5, abs = 1e-3)