import pytest
import numpy as np
import scipy.sparse as sp
from numerov_black_scholes.solver import NumerovBackwardEulerSolver

class MockSystem:
    def __init__(self, N_points=10, L=1.0, dt=0.01):
        self.N_points = N_points
        self.L = L
        self.dt = dt
        self.dx = L / (N_points - 1)

@pytest.fixture
def basic_solver():
    """Creates a standard solver instance for testing."""
    sys = MockSystem()
    return NumerovBackwardEulerSolver(sys, alpha=1.0)

# --- Tests --- #
def test_initialization_attributes(basic_solver):
    """ Are the attributes stored correctly?"""
    assert basic_solver.sys.N_points == 10
    assert basic_solver.alpha == 1.0

    # Check that M and K are sparse matrices
    assert sp.issparse(basic_solver.M)
    assert sp.issparse(basic_solver.K)


def test_matrix_assignment_swap(basic_solver):
    """
    Ensures M and K are not swapped

    Physics Check
    - K (Stiffness) represents 2nd derivative -> Inner diag should be negative (-2)
    - M (Mass) represents averaging -> Inner diag should be positive (10)
    """

    # Extract main diagonals
    diag_M = basic_solver.M.diagonal()
    diag_K = basic_solver.K.diagonal()

    # --- Check Mass Matrix (M) ---
    # Should be positive (10/12) for averaging
    # We check interior points specifically
    assert np.all(diag_M[1:-1] > 0), "Mass matrix M should have positive diagonals (10)"
    assert diag_M[1] == pytest.approx(10.0 / 12.0), "M matrix values are incorrect"

    # Check boundary identity enforcement: M[0,0] and M[-1,-1] must be 1.0
    assert diag_M[0] == 1.0, "M boundary at index 0 should be 1.0 for identity"
    assert diag_M[-1] == 1.0, "M boundary at index -1 should be 1.0 for identity"

    # --- Check Stiffness Matrix (K) ---
    # Interior points should be negative (-2/dx^2)
    interior_K = diag_K[1:-1]
    assert np.all(interior_K < 0), "Inner stiffness matrix K should have negative diagonals (-2)"

    # Check boundary zeroing: K[0,0] and K[-1,-1] must be 0.0
    # This ensures (M - alpha*dt*K) reduces to I at the boundaries
    assert diag_K[0] == 0.0, "Boundary K[0,0] should be 0.0"
    assert diag_K[-1] == 0.0, "Boundary K[-1,-1] should be 0.0"

def test_operator_construction(basic_solver):
    """Does A = M - alpha*dt*K have the right shape and properties?"""
    A = basic_solver.A
    assert A.shape == (10,10)

    # A is a positive operator
    diag_A = A.diagonal()
    assert np.all(diag_A > 0), "Operator A should have positive main diagonal (for stability)"

def test_step_identity_check():
    """
    Physics Check: If dt=0, then u_next should equal u_curr.
    (M-0) * u_next = M * u_curr => M * u_next = M * u_curr
    """

    sys = MockSystem(dt=0.0)
    solver = NumerovBackwardEulerSolver(sys, alpha=1.0)

    # Random initial cond.
    u_curr = np.random.rand(10)

    # Step forward
    u_next = solver.step(u_curr)

    # u_next should be identical to u_curr
    np.testing.assert_array_almost_equal(u_next, u_curr)

def test_step_diffusion():
    """
    Does the tep actually diffuse (Smoothing test)?
    If we've a spike [0,1,0], it should spread our and lower in peak.
    """
    sys = MockSystem(N_points=5)
    solver = NumerovBackwardEulerSolver(sys, alpha=1.0)

    u_curr = np.array([0,0,1.0,0,0])
    u_next = solver.step(u_curr)

    # Peak should decrease (diffusion)
    assert u_next[2] < 1.0

    # Neighbors should increase
    assert u_next[1] > 0.0
    assert u_next[3] > 0.0