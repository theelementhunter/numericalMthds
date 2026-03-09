import numpy as np
from numerov_black_scholes.system import BlackScholesSystem
from numerov_black_scholes.solver import NumerovBackwardEulerSolver
from numerov_black_scholes.state import SimulationRunner
from numerov_black_scholes.analytics import black_scholes_call
from numerov_black_scholes.visualization import plot_price_comparison
from numerov_black_scholes.visualization import plot_error_convergence

def get_put_payoff(x_grid, K):
    """
    Log-space Payoff for Put Option
    x = ln(S/K) -> S = K * exp(x)
    Payoff = max(K - S, 0)
    """
    S = K * np.exp(x_grid)
    return np.maximum(K - S, 0)

if __name__ == '__main__':
    # --- Configuration --- #
    S0 = 100.0             # Spot Price
    K = 100.0              # Strike Price
    sigma = 0.2            # Volatility
    r = 0.5 * sigma**2     # Risk-free Rate # Accounts for Black-Scholes drift (r - 0.5*sigma**2)
    T = 1.0                # Maturity

    # Grid Parameters
    # L=6.0 covers approx +/- 3 sigma in log-space
    L = 14.0
    N = 401
    dt = 0.001

    # --- Setup --- #
    # Conver financial volatility to diffusion coefficient
    # In pure diffusion limit: alpha = 0.5 * sigma^2
    alpha_diff = 0.5 * sigma**2

    sys = BlackScholesSystem(L=L, dt=dt, N_points=N, T_final=T)
    solver = NumerovBackwardEulerSolver(sys, alpha=alpha_diff)

    # --- Initial Conditions --- #
    # We solve for a Put (decaying boundary)
    # We are solving for the TRANSFORMED variable U(x, tau)
    # V(x, tau) = exp(-r * tau) * U(x, tau)
    # At tau=0, V = U.

    u0 = get_put_payoff(sys.x_grid, K)

    # Boundary Conditions for U are STATIC:
    # Left (S=0): The Put is worth K * exp(-r*tau).
    # Since U = V * exp(r*tau), U_left = K.
    u0[0] = K
    u0[-1] = 0.0

    # --- Execution --- #
    runner = SimulationRunner(sys, solver, u0)
    runner.run()


    # --- Post-Processing
    # Extract final raw solution u(x,T)
    _, u_final_transformed = runner.history[-1]

    # Applying discount factor
    # We transform U back to Price V
    put_price_numerical = u_final_transformed * np.exp(-r * T)

    # Recover Stock Prices: S = K * exp(x)
    S_grid = K * np.exp(sys.x_grid)

    # Convert Put to Call using Parity (Call = Put + S - K*e^-rT)
    # Note: u_final is the Put price  (assuming purely diffusive approx. for this stage)
    call_price_numerical = put_price_numerical + S_grid - K * np.exp(-r*T)

    # Benchmark
    call_price_analytical = black_scholes_call(S_grid,K,T,r,sigma)


    # --- Visualization --- #
    plot_price_comparison(
        S_grid=S_grid,
        numerical_price=call_price_numerical,
        analytical_price=call_price_analytical,
        title=f"Numerov Pricing (N={N})"
    )

    # Output specific check
    mid_idx = N // 2
    print(f"Deepest-in-money Check (S={S_grid[mid_idx]:.2f}): {call_price_analytical[mid_idx]:.4f}")