import numpy as np
from scipy import sparse

class PlaneWaveBasis():
    def __init__(self, N, a):
        self.N = N                                             # Number of Plane Waves
        self.a = a                                             # Length of Unit Cell (Real Space) (1D)
        self.N_vector = np.arange(-self.N, self.N + 1)         # Vector of integers ranging form -N to N
        self.k_vector = (2 * np.pi * self.N_vector ) / self.a  # Vector of k_N


    def build_kinetic_matrix(self):
        """
        Build a kinetic matrix (diagonal) (sparse)
        K = h_bar^2 * k^2 / 2m where h_bar^2/2m == 1
        :return: np.array
        """
        k_squared = self.k_vector ** 2
        kinetic_matrix = sparse.diags(diagonals=k_squared, offsets=0)
        return kinetic_matrix

    def build_momentum_difference_matrix(self):
        """
        Helper for building the potential matrix: compute q = k_i -k_j
        :return:
        """
        return self.k_vector[:, np.newaxis] - self.k_vector[np.newaxis, :]

