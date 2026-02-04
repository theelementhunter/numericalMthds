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
        V = self.potential.build_potential_matrix()

        print("Building Full Hamiltonian Matrix...")
        self.H = T + V
        print("Hamiltonian Built.")
