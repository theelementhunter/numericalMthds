import numpy as np
from dataclasses import dataclass
from typing import Tuple



@dataclass
class QuantumState:
    energy: float
    psi: np.ndarray
    nodes: int
    parity: str

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

    def _set_initial_conditions(self, parity: str, f_n: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Determines psi[0] and psi[1] based on parity. Sets psi[-2] and psi[-1] for backward integration
        Replaces 'set_initial_conditions' function from harmonic1
        :param parity: The symmetry of the wavefunction. Expected values are 'even' or 'odd'
        :param f: A numpy array representing the f_n parameter used in the numerov algorithm
        :return: A four-tuple containing the initial values (psi[0], psi[1], psi[-1], psi[-2]).
        """
        if parity == "odd":
            #Odd Parrity: Node at origin -> psi[0] = 0
            psi_0 = 0.0
            psi_1 = 0.1
        else:
            #Even Parity: Slope is zero at origin;
            #Thus psi[0] can be any arbitrary non-zero number
            psi_0 = 0.1
            #Eqn 1.34; Numerov Expansion for zero slope
            psi_1 = ((12 - 10 * f_n[0]) * psi_0) / (2 * f_n[1])

        #Backward Integration
        psi_2 = 0.1
        psi_3 = psi_2 * np.exp((self.x[-1]**2 - self.x[-2]**2)/2)

        return psi_0, psi_1, psi_2, psi_3

    #integration logic
    def _integrate(self, epsilon: float, parity: str) -> np.ndarray:
        """
        Integrates the psi wavefunction using the numerov method.
        :param epsilon: energy to solve for the solution of the wavefunction
        :return: a 3-tuple containing array of the solved wavefunction, integer
        at which teh wavefunction switches into the classically forbidden reason,
        and the value of the derivative discontinity difference
        """
        f = self._f_n(epsilon)
        psi = np.zeros(self.N, dtype=float)

        #Find Match Point (icl)
        forbidden_indices = np.where(self.V > epsilon)[0]
        icl = forbidden_indices[0] if len(forbidden_indices) > 0 else self.N - 1

        #Wavefunction for Backward Integration
        psi_R = np.zeros(self.N)

        #Initial Conditions
        psi[0], psi[1], psi_R[-1], psi_R[-2]= self._set_initial_conditions(parity = parity, f_n = f)

        #Forward Integration
        for i in range(2, icl + 1):
            var1 = (12 - (10 * f[i - 1])) * psi[i - 1]
            var2 = f[i - 2] * psi[i -2]
            numerator = var1 - var2
            denominator = f[i]
            psi[i] = numerator / denominator

        #Backward Integration
        for i in range(self.N - 3, icl - 1, -1):
            psi_R[i] = ((12 - 10 * f[i+1]) * psi_R[i+1] - f[i+2] * psi_R[i+2]) / f[i+1]

        #Continuity Check
        diff = 0.0
        if psi_R[icl] != 0:
            scale = psi[icl] / psi_R[icl]
            psi_R_scaled = psi_R * scale

            #Stitch wavefunction together
            psi[icl + 1:] = psi_R_scaled[icl + 1:]

            # check discontinuity #eqn 1.38
            num1 = psi[icl - 1] + psi_R_scaled[icl + 1]
            num2 = (14 - (12 * f[icl])) * psi[icl]
            num = num1 - num2
            denom = self.dx

            diff = num / denom

            #Sign standardization
            if psi[icl] < 0:
                diff = -diff

        return psi, icl, diff

    def _count_nodes(self, psi: np.ndarray, icl: int) -> int:
        #Counts nodes in classically allowed region (0 to icl)
        segment = psi[:icl+1]
        return len(np.where(np.diff(np.signbit(segment)))[0]) #a little single line magick

    #Bisection Method
    def solve(self, n_target: int) -> QuantumState:
        #Determine Parity
        parity = "even" if n_target % 2 == 0 else "odd"
        #target_nodes = n_target // 2 #Nodes in half-space

        if parity == "even":
            target_nodes = n_target // 2
        else:
            target_nodes = (n_target - 1) // 2

        #Course Energy Search
        E_min, E_max = 0.0, 20.0 #Wide energy range
        E_guess = 5.0

        for _ in range(100):
            E_guess = (E_min + E_max) / 2.0
            psi, icl, diff = self._integrate(E_guess, parity)
            nodes = self._count_nodes(psi, icl)

            #Coarse Energy Check
            if nodes != target_nodes:
                if nodes > target_nodes:
                    E_max = E_guess  # Too much energy
                else:
                    E_min = E_guess  # Too little energy
                continue  # Skip the Fine Check logic this turn

            #Fine Check

            #Check Convergence
            if abs(E_max - E_min) < 1e-10:
                #Soln Found
                #Finalize Soln
                norm = np.sqrt(np.sum(psi**2) * self.dx)
                return QuantumState(energy=E_guess, psi=psi / norm, nodes=target_nodes, parity=parity)

            #If Soln is not Converged
            if diff < 0:
                E_min = E_guess
            else:
                E_max = E_guess

        raise RuntimeError("Could not converge solution.")