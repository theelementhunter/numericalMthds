import numpy as np
from var_princ_chpt_4.basis import PlaneWaveBasis
from var_princ_chpt_4.potentials import PotentialWell

class QuantumSys1D:
    def __init__(self, basis: PlaneWaveBasis, potential: PotentialWell) -> None:
        # System Variables
        self.basis = basis
        self.potential = potential
        self.Hamiltonian = None

        # State Variables -> These could have their own class but figured it was more
        # hassle than it was worth since we aren't able to save the state at more
        # than one point in the solution
        self.eigenvalues = None
        self.eigenvectors = None

    def build_hamiltonian(self) -> np.ndarray:
        """Constructs the Hamiltonian matrix."""
        print("Building Kinetic Energy Matrix...")
        T = self.basis.build_kinetic_matrix()

        print("Building Potential Energy Matrix...")
        V = self.potential.potential_matrix()

        print("Building Full Hamiltonian Matrix...")
        self.Hamiltonian = T + V
        print("Hamiltonian Built.")

    def get_wfn(self, state_index: int, xgrid: np.ndarray) -> np.ndarray:
        # No state lower than the ground state
        if state_index < 0:
            raise ValueError("Invalid state_index. Ground State (0) is the lowest state.")

        # Broadcast grids and coefficients for vectorization
        k_vec_broadcast = self.basis.k_vector[None, :]        # Row of k-points
        x_grid_broadcast = xgrid[:, None]                    # Column of positions
        coeff = self.eigenvectors[:, state_index]

        #
        basis_matrix = np.exp(1j * x_grid_broadcast * k_vec_broadcast)

        # Weighted Sum
        # Matrix multiplication (@) does the sum over the 'k' dimension automatically
        psi = basis_matrix @ coeff

        # Normalization
        psi_normalized = psi * (1 / np.sqrt(self.basis.a))
        return psi_normalized



