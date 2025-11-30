import numpy as np
from dataclasses import dataclass
from typing import Tuple



@dataclass
class QuantumState:
    energy: float
    psi: np.ndarray
    nodes: int

"""
@dataclass autogenerates the class below behind the scenes
class quantumState:
    def __init__(self, energy: float, psi: np.ndarray, nodes: int):
        self.energy = energy
        self.psi = psi
        self.nodes = nodes
    def __repr__(self):
        return f"quantumState(energy = {self.energy}, psi=self.psi, nodes = {self.nodes})"
        
    def __eq__(self,other):
        return (self.energy, self.psi, self.nodes) == (other.energy, other.psi, other.nodes)
"""

class NumerovSolver:
    def __init__(self, grid: np.ndarray, potential):
        self.x = grid
        self.dx = grid[1] - grid[0]
        self.N = len(grid)
        self.V = potential.get_potential(grid)

    #private methods (_xyz); for internal use only; keep under hood
    def _g_n(self, epsilon: float) -> np.ndarray:
        return 2 * (epsilon - self.V)

    def _f_n(self, epsilon: float) -> np.ndarray:
        g = self._g_n(epsilon)
        return 1 + (g * (self.dx ** 2) / 12)

    #integration logic
    def _integrate(self, epsilon: float) -> np.ndarray:
        f = self._f_n(epsilon)
        psi = np.zeros(self.N, dtype=float)
        forbidden_indices = np.where(self.V > epsilon)[0]
        icl = forbidden_indices[0] if len(forbidden_indices) > 0 else self.N - 1

        # Forward Integration Initial Conditions
        psi[0] = 0.0
        psi[1] = 1e-5
        for i in range(2, icl + 1):
            psi[i] = ((12 - 10 * f[i-1]) * psi[i-1] - f[i-2] * psi[i-2]) / f[i]

        #Backward Integration Initial Conditions
        psi_R = np.zeros(self.N)
        psi_R[-1] = 1e-5
        psi_R[-2] = 1e-5 * np.exp((self.x[-1]**2 - self.x[-2]**2)/2)
        for i in range(self.N - 3, icl - 1, -1):
            psi_R[i] = ((12 - 10 * f[i+1]) * psi_R[i+1] - f[i+2] * psi_R[i+2]) / f[i+1]

        #Matching psi_L and psi_R after integration
        if psi_R[icl] != 0:
            scale = psi[icl] / psi_R[icl]
            psi[icl:] = psi_R[icl:] * scale

        return psi

    def _count_nodes(self, psi: np.ndarray) -> int:
        return len(np.where(np.diff(np.signbit(psi)))[0])

    #Bisection Method
    def solve(self, n_target: int) -> QuantumState:
        E_min, E_max = 0.0, 10.0

        for _ in range(100):
            E_guess = (E_min + E_max) / 2.0
            psi = self._integrate(E_guess)
            nodes = self._count_nodes(psi)

            if nodes == n_target:
                norm = np.sqrt(np.sum(psi**2) * self.dx)
                return QuantumState(E_guess, psi / norm, nodes)
            elif nodes > n_target:
                E_max = E_guess
            else:
                E_min = E_guess

        raise RuntimeError("Could not converge on energy.")