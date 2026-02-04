from basis import PlaneWaveBasis
from potentials import PotentialWell
from solver import Eigensolver
from system import QuantumSys1D
from var_princ_chpt_4 import *
import numpy as np

def main():
    # --- Inputs ---#
    width = 2.0
    potential_depth = 1.0
    n_plane_waves = 50
    box_size = 10.0

    # --- Define Basis --- #
    basis = PlaneWaveBasis(N=n_plane_waves,a=box_size)

    # --- Define Potential --- #
    potential = PotentialWell(basis=basis,width=width,V0=potential_depth)

    # --- Create quantum system; States automatically initialized --- #
    system = QuantumSys1D(basis=basis, potential=potential)

    # --- Build Hamiltonian --- #
    system.build_hamiltonian()

    # --- Solve; States populated --- #
    Eigensolver.solve(system)

    # --- Visualization of 1st Ground State --- #



if __name__ == "__main__":
    main()