from basis import PlaneWaveBasis
from potentials import PotentialWell
from solver import Eigensolver
from system import QuantumSys1D
from var_princ_chpt_4 import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- Inputs ---#
    width = 2.0
    potential_depth = 10.0
    n_plane_waves = 100
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
    # Plot Inputs
    state = 0

    # Create spacial grid
    spacial_grid = np.linspace(start=-box_size/2.0,stop=box_size/2.0,num=100*n_plane_waves)
    psi = system.get_wfn(state_index=state, xgrid = spacial_grid)

    title = f"State {state} Wavefunction"
    # Create figure with two axes sharing the same x-axis
    fig, ax1 = plt.subplots()

    # Plot the Wavefunction (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel(r'$|\Psi|^2$', color=color)
    ax1.plot(spacial_grid, np.abs(psi) ** 2, color=color, label='Probability Density')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second axis for the Potential (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Potential V(x)', color=color)

    # Draw the Potential (Square Well)
    # V is -V0 inside [-width/2, width/2], and 0 outside
    potential_profile = np.where(np.abs(spacial_grid) <= width / 2, -potential_depth, 0)
    ax2.plot(spacial_grid, potential_profile, color=color, linestyle='--', alpha=0.5, label='Potential')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"State {state} with V0={potential_depth}")
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()