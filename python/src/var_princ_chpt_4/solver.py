from scipy.linalg import eigh
from system import QuantumSys1D


class Eigensolver:
    @staticmethod #No need for anyone that isn't this program to call on this
    def solve(sys: QuantumSys1D ) -> None:
        """
        Diagonalizes the system's Hamiltonian
        Populates sys.eigenvalues and sys.eigenvectors.
        """
        if sys.Hamiltonian is None:
            raise ValueError("System Hamiltonian is not built. Call sys.build_hamiltonian() first.")

        print("Diagonalizing Hamiltonian...")
        energies, coeff = eigh(sys.Hamiltonian)
        print("Populating system...")
        sys.eigenvalues = energies
        sys.eigenvectors = coeff

        print("Solution Reached.")