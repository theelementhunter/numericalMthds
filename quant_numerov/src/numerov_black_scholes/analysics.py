import numpy as np
import scipy.stats as si

def black_scholes_call(S, K, T, r, sigma):
    """Calculates the exact Black-Scholes price for a Call Option."""
    # Avoid division by zero if T is 0
    if T == 0:
        return np.maximum(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculates the exact Black-Scholes price for a Put Option."""
    if T == 0:
        return np.maximum(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    put_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return put_price