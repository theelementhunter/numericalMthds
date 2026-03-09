import numpy as np
from var_princ_chpt_4.basis import PlaneWaveBasis
from abc import ABC, abstractmethod

class Potential1D(ABC):
    """Abstract Base Class; Defines what a 1D Potential must look like."""
    @abstractmethod
    def potential_matrix(self) -> np.ndarray:
        pass


class PotentialWell(Potential1D):
    def __init__(self, basis: PlaneWaveBasis, width: float = 2.0, V0: float = 1.0):
        self.basis = basis
        self.width = width
        self.V0 = V0
        self.N = self.basis.N
        self.a = self.basis.a

    def potential_matrix(self) -> np.ndarray:
        #--- When k_i = k_j -> - V0*b / a ---#
        diagonal_elements = - ( self.V0 * self.width ) / self.a

        diff_matrix = self.basis.build_momentum_difference_matrix()

        #--- Fixing DivisionByZero ---#
        q_to_0_limit = np.full_like(diff_matrix, 1.0)

        # Boolean Mask -> Where is q NOT zero
        mask = np.abs(diff_matrix) > 1e-9

        # Pull out only values there the mask is True
        safe_q = diff_matrix[mask]

        # Do our math on new safe matrix (No DivisionByZero)
        safe_results = ((-self.V0 / self.a) * np.sin((self.width * safe_q)/ 2.0) ) / (safe_q / 2.0)

        # Paste safe matrix back into
        q_to_0_limit[mask] = safe_results

        # --- Replace Diagonal Elements --- #
        np.fill_diagonal(q_to_0_limit, diagonal_elements)

        return q_to_0_limit

