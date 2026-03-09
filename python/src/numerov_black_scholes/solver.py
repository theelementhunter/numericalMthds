import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp
from numerov_black_scholes.system import BlackScholesSystem

# Don't use @dataclass for solver. It has complex internal states
# that are cumbersome to align with waht dataclass is doing for us.
class NumerovBackwardEulerSolver():
    def __init__(self, BSS: BlackScholesSystem, alpha: float = 1.0):
        self.sys = BSS
        self.alpha = alpha
        self.M , self.K = self._build_matrices()

        # Operator 'A'
        # Equation: (M - alpha * dt * K) * u_next = M * u_curr
        self.A = self.M - (self.alpha * self.sys.dt * self.K)

        # (LU Decomposition)
        self.solve_A = sp.linalg.factorized(self.A.tocsc())

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

        # --- Clamp Boundaries --- #
        # We convert to LIL format briefly because it's efficient for
        # modifying individual entries.
        K = K.tolil()
        M = M.tolil()

        # Zero out the first and last rows
        K[0, :] = 0;
        K[-1, :] = 0
        M[0, :] = 0;
        M[-1, :] = 0

        # Enforce Identity: 1 * u_next = 1 * u_curr
        # This ensures the boundary equation is just u_new = u_old (static)
        M[0, 0] = 1.0;
        M[-1, -1] = 1.0

        # Note: We leave K as zeros on the boundary rows because
        # the equation is M*u_next - alpha*dt*K*u_next = M*u_curr
        # If K is 0 and M is I, it reduces to I*u_next = I*u_curr.

        return M.tocsr(), K.tocsr()

    def step(self, u_curr, boundary_left=None, boundary_right=None):
        # 1 Project current state (RHS)
        b = self.M.dot(u_curr)

        # 2. Enforce Boundary Values explicitly on the RHS vector 'b'
        # This overrides whatever M.dot(u_curr) calculated for the edges.
        if boundary_left is not None:
            b[0] = boundary_left
        if boundary_right is not None:
            b[-1] = boundary_right

        # 3. Solve for next state
        return self.solve_A(b)