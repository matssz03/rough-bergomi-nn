"""
Black-Scholes implied volatility inversion.

Uses Newton-Raphson with a Brent-style bisection fallback.  Prices below
the intrinsic-value bound or above the no-arbitrage upper bound return
NaN rather than raising, so that a full surface can be inverted without
special-casing deep-OTM contracts whose Monte Carlo estimate collapses
to zero.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------- #
# Black-Scholes price and vega
# --------------------------------------------------------------------- #

def bs_call_price(S: float, K: float, T: float, sigma: float,
                  r: float = 0.0, q: float = 0.0) -> float:
    """Closed-form Black-Scholes price of a European call."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float,
            r: float = 0.0, q: float = 0.0) -> float:
    """Black-Scholes vega (d Price / d sigma)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


# --------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------- #

def implied_vol(price: float, S: float, K: float, T: float,
                r: float = 0.0, q: float = 0.0,
                sigma_lo: float = 1e-6, sigma_hi: float = 5.0,
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Solve for sigma such that bs_call_price(S, K, T, sigma) == price.

    Hybrid Newton-Raphson + bisection: Newton where safe, bisection as a
    fall-back so the routine converges monotonically over the full
    arbitrage-free price range.
    """
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = S * np.exp(-q * T)
    if not np.isfinite(price) or price < lower - 1e-10 or price > upper + 1e-10:
        return float("nan")

    lo, hi = sigma_lo, sigma_hi
    p_lo = bs_call_price(S, K, T, lo, r, q)
    p_hi = bs_call_price(S, K, T, hi, r, q)
    if p_lo > price:
        return lo
    if p_hi < price:
        return hi

    sigma = np.sqrt(2.0 * np.pi / T) * price / (S * np.exp(-q * T))
    sigma = float(np.clip(sigma, lo, hi))

    for _ in range(max_iter):
        p = bs_call_price(S, K, T, sigma, r, q)
        diff = p - price
        if abs(diff) < tol:
            return sigma
        v = bs_vega(S, K, T, sigma, r, q)
        if v > 1e-10:
            step = diff / v
            sigma_new = sigma - step
            if lo < sigma_new < hi:
                if diff > 0:
                    hi = sigma
                else:
                    lo = sigma
                sigma = sigma_new
                continue
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        sigma = 0.5 * (lo + hi)

    return sigma
