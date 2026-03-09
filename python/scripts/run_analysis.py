import numpy as np
import matplotlib.pyplot as plt

# We import our "Engine" parts
# (If these fail, make sure you ran 'pip install -e .' in the terminal)
from python.src.numerov_schrod_eqn.potentials import HarmonicOscillator
from python.src.numerov_schrod_eqn.solver import NumerovSolver


def reconstruct_full_wavefunction(result, grid):
    """
    Post-Processing Helper:
    The solver assumes x > 0. This function uses parity to
    reconstruct the full wavefunction from -x_max to +x_max.
    """
    # 1. Create the Left-side grid (negative x)
    # We slice [1:] to avoid duplicating the x=0 point
    x_left = -grid[1:][::-1]
    x_full = np.concatenate((x_left, grid))

    # 2. Create the Left-side wavefunction based on Parity
    psi_right = result.psi

    if result.parity == "even":
        # Symmetric: psi(-x) = psi(x)
        psi_left = psi_right[1:][::-1]
    else:
        # Anti-Symmetric: psi(-x) = -psi(x)
        psi_left = -1 * psi_right[1:][::-1]

    # 3. Stitch them together
    psi_full = np.concatenate((psi_left, psi_right))

    # 4. Re-Normalize (Since we changed the integration area)
    # Area roughly doubled, so amplitude must drop by 1/sqrt(2)
    # We re-calculate strictly to be safe.
    dx = grid[1] - grid[0]
    norm = np.sqrt(np.sum(psi_full ** 2) * dx)

    return x_full, psi_full / norm


def main():
    # --- 1. CONFIGURATION ---
    x_max = 7.0
    N_points = 1000
    target_states = [0, 1, 2, 3]  # Let's solve for multiple states at once!

    # --- 2. INITIALIZATION ---
    grid = np.linspace(0, x_max, N_points)
    potential = HarmonicOscillator(k=1.0)
    solver = NumerovSolver(grid, potential)

    plt.figure(figsize=(10, 6))

    # --- 3. EXECUTION LOOP ---
    for n in target_states:
        print(f"Solving for n={n}...")

        try:
            # The Engine does the heavy lifting
            result = solver.solve(n_target=n)

            # The Client does the display logic
            x_full, psi_full = reconstruct_full_wavefunction(result, grid)

            # Plotting
            plt.plot(x_full, psi_full, label=f"n={n} (E={result.energy:.4f})")

        except RuntimeError as e:
            print(f"Failed to converge for n={n}: {e}")

    # --- 4. FORMATTING ---
    plt.title("Harmonic Oscillator Wavefunctions (Automated)")
    plt.xlabel("Position (Bohr)")
    plt.ylabel("Psi")
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)  # Center line
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save output
    filename = "harmonic_results.png"
    plt.savefig(filename)
    print(f"\nSuccess! Plot saved to {filename}")
    plt.show()


# Standard Python entry point
if __name__ == "__main__":
    main()