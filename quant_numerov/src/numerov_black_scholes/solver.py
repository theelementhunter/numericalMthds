import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp
from numerov_black_scholes.system import BlackScholesSystem

# Don't use @dataclass for solver. It has complex internal states
# that are cumbersome to align with waht dataclass is doing for us.
class NumerovBackwardEulerSolver():
    def __init__(self, BSS: BlackScholesSystem):
        self.sys = BSS
        self.M , self.K = self._build_matrices()

    def _build_matrices(self):
        """
        Constructs Numerov matrices.
        :return: Sparce matrices of shape (N,N).
        """
        N = self.sys.N_points
        dx = self.sys.dx

        # --- Matrix M --- #
        diag_M = np.ones(N) * -2
        offdiag_M = np.ones(N) * 1
        diagonals_M = [diag_M, offdiag_M, offdiag_M]

        M_matrix = sp.diags(diagonals=diagonals_M, offsets=[0, 1, -1])

        M = M_matrix / (dx**2)

        # --- Matrix K --- #
        diag_K = np.ones(N) * 10
        offdiag_K = np.ones(N) * 1
        diagonals_K = [diag_K, offdiag_K, offdiag_K]

        K_matrix = sp.diags(diagonals=diagonals_K, offsets=[0, 1, -1])

        K = K_matrix / 12

        return M, K