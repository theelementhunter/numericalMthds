import matplotlib.pyplot as plt
import numpy as np

def plot_price_comparison(S_grid, numerical_price, analysical_price, title="Option Pricing"):
    """Plots the Numerical vs Analysical solution side-by-side."""
    plt.figure(figsize=(10,6))

    # Plot Analytical Line
    plt.plot(S_grid, numerical_price, "k-", linewidth=1.5, label="Analytical BS (Exact)")

    # Plot Numerical Dots (every 10th point to show discrete nature)
    plt.plot(S_grid, numerical_price, "r--", marker="o", markevery=10, alpha=0.8, label="Numerov (Calculated)")

    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Value ($)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Focus view on relevant area (At-The-Money)
    # We use the Strike Price (K) as the anchor, not the grid center
    # This ensures the "action" is always in the middle of the plot.

    # Assuming you pass 'K' into this function or derive it from S_grid
    # If K is not available, using the median of S_grid is a safe fallback:
    K_approx = np.median(S_grid)

    plt.xlim(K_approx * 0.5, K_approx * 1.5)  # View +/- 50% around Strike
    plt.ylim(0, K_approx * 0.6)  # Y-axis typically needs less room than X

    plt.tight_layout()
    plt.show()

def plot_error_convergence(N_values, errors):
    """Plots Error vs Grid Size on a log-log scale"""

    # Ensure inputs are numpy arrays for element-wise operations
    N_values = np.array(N_values)
    errors = np.array(errors)

    plt.figure(figsize=(8, 6))

    # 1. Plot the actual data
    plt.loglog(N_values, errors, 'bo-', linewidth=2, label="Numerov Error")

    # 2. Generate Reference Slopes
    # We anchor the reference lines to the first data point (errors[0])
    # so they line up visually.

    # O(N^-2) Reference (Standard 2nd Order FD)
    # Formula: y = error[0] * (N / N[0])^-2
    ref_slope_2 = errors[0] * (N_values / N_values[0])**(-2)

    # O(N^-4) Reference (Ideal Numerov 4th Order)
    # Formula: y = error[0] * (N / N[0])^-4
    ref_slope_4 = errors[0] * (N_values / N_values[0])**(-4)

    # 3. Plot Reference Lines
    plt.loglog(N_values, ref_slope_2, 'r--', alpha=0.6, label=r"Reference $O(N^{-2})$")
    plt.loglog(N_values, ref_slope_4, 'g--', alpha=0.6, label=r"Reference $O(N^{-4})$")

    plt.xlabel("Grid Points (N)")
    plt.ylabel("L2 Error Norm")
    plt.title("Error Convergence: Numerov Scheme")
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()

    plt.show()