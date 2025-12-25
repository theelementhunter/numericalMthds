import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp
from numerov_black_scholes.system import BlackScholesSystem

# Don't use @dataclass for solver. It has complex internal states
# that are cumbersome to align with waht dataclass is doing for us.
class NumerovBackwardEulerSolver():
    def __init__(self, BSS: BlackScholesSystem, alpha: float = 1.0):
        self.sys = BSS
        self.A = alpha
        self.M , self.K = self._build_matrices()

        # Operator 'A'
        # Equation: (M - alpha * dt * K) * u_next = M * u_curr
        self.A = self.M - (self.alpha * self.sys.dt * self.K)

        # (LU Decomposition)
        self.solve_A = sp.linalg.factorized(self.A)

    def _build_matrices(self):
        """
        Constructs Numerov matrices.
        :return: Sparce matrices of shape (N,N).
        """
        N = self.sys.N_points
        dx = self.sys.dx

        # --- Matrix K (Stiffness / Derivative) --- #
        # Pattern: [1,-2, 1] / dx^2
        diag_K = np.ones(N) * -2
        offdiag_K = np.ones(N) * 1
        diagonals_K = [diag_K, offdiag_K, offdiag_K]

        K_matrix = sp.diags(diagonals=diagonals_K, offsets=[0, 1, -1])

        K = K_matrix / (dx**2)

        # --- Matrix M --- #
        # Pattern [1, 10, 1] / 12
        diag_M = np.ones(N) * 10
        offdiag_M = np.ones(N) * 1
        diagonals_M = [diag_M, offdiag_M, offdiag_M]

        M_matrix = sp.diags(diagonals=diagonals_M, offsets=[0, 1, -1])

        M = M_matrix / 12

        return K, M

    def step(self, u_curr):
        # 1 Project current state (RHS)
        b = self.M.dot(u_curr)

        # 2. Solve for next state
        return self.solve_A(b)