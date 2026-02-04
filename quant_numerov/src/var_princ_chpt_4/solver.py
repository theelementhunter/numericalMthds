from scipy.linalg import eig
from system import QuantumSys1D


class Eigensolver:
    @staticmethod #No need for anyone that isn't this program to call on this
    def solve(sys: QuantumSys1D ) -> None:
        """
        Diagonalizes the system's Hamiltonian
        Populates sys.eigenvalues and sys.eigenvectors.
        """
        if sys.H is None:
            raise ValueError("System Hamiltonian is not built. Call sys.build_hamiltonian() first.")

        print("Diagonalizing Hamiltonian...")
        energies, wfns = eig(sys.H)
        print("Populating system...")
        sys.eigenvalues = energies
        sys.eigenvectors = wfns

        print("Solution Reached.")