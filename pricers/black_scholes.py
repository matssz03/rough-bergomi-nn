"""
Black-Scholes-Merton closed-form pricer for European vanilla options.

Reference: Black & Scholes (1973), Merton (1973).

Notation:
    S     : spot price of the underlying
    K     : strike price
    T     : time to maturity in years
    r     : risk-free rate (continuously compounded)
    q     : continuous dividend yield (default 0)
    sigma : volatility (annualised)

All functions are vectorised over strikes and maturities.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    """Helper: compute d1 and d2 of the Black-Scholes formula."""
    T = np.maximum(T, 1e-12)  # avoid division by zero at maturity
    sigma_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_T
    d2 = d1 - sigma_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Black-Scholes-Merton price of a European vanilla option.

    Parameters
    ----------
    S, K, T, r, sigma : scalars or arrays (broadcasting supported)
    option_type       : 'call' or 'put'
    q                 : continuous dividend yield (default 0)

    Returns
    -------
    Price of the option (same shape as broadcasting result).
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S, K, T, r, sigma, option_type='call', q=0.0):
    """First-order sensitivity of the option price to the underlying."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Second-order sensitivity (same for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    T = np.maximum(T, 1e-12)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity to the volatility parameter (same for calls and puts).

    Returned in absolute value per unit change in sigma (not per 1%).
    """
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    T = np.maximum(T, 1e-12)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, T, r, sigma, option_type='call', q=0.0):
    """Time decay of the option value (returned per year)."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    T = np.maximum(T, 1e-12)
    common = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == 'call':
        return common - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        return common + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_rho(S, K, T, r, sigma, option_type='call', q=0.0):
    """Sensitivity to the interest rate."""
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_implied_volatility(price, S, K, T, r, option_type='call', q=0.0, tol=1e-8, max_iter=100):
    """
    Invert Black-Scholes to recover the implied volatility from an observed price.

    Uses Newton-Raphson iteration with vega as the derivative.

    Parameters
    ----------
    price : observed option price (scalar or array)
    S, K, T, r : market inputs
    option_type : 'call' or 'put'
    q     : dividend yield
    tol   : convergence tolerance on the absolute price error
    max_iter : maximum number of Newton iterations

    Returns
    -------
    Implied volatility (same shape as `price`).
    """
    sigma = np.full_like(np.asarray(price, dtype=float), 0.20)  # initial guess: 20%
    for _ in range(max_iter):
        model_price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = model_price - price
        if np.all(np.abs(diff) < tol):
            break
        vega = bs_vega(S, K, T, r, sigma, q)
        vega = np.maximum(vega, 1e-8)
        sigma = sigma - diff / vega
        sigma = np.maximum(sigma, 1e-4)
    return sigma


# ============================================================================
# UNIT TESTS — run when the module is executed directly.
# Validated against textbook Hull (2018), Chapter 15 examples.
# ============================================================================

if __name__ == '__main__':
    print("Running unit tests for black_scholes.py")
    print("=" * 60)

    # Test 1: Hull (2018), Example 15.6 — a European call
    # S=42, K=40, r=0.10, sigma=0.20, T=0.5
    # Expected call price: 4.76
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5
    price = bs_price(S, K, T, r, sigma, 'call')
    expected = 4.759
    assert abs(price - expected) < 0.01, f"Test 1 failed: {price} vs {expected}"
    print(f"Test 1 (Hull 2018 call):  price = {price:.4f}  (expected {expected})  OK")

    # Test 2: Put-call parity
    # C - P = S*exp(-qT) - K*exp(-rT)
    C = bs_price(S, K, T, r, sigma, 'call')
    P = bs_price(S, K, T, r, sigma, 'put')
    parity_error = (C - P) - (S - K * np.exp(-r * T))
    assert abs(parity_error) < 1e-10, f"Test 2 failed: parity error = {parity_error}"
    print(f"Test 2 (put-call parity): error = {abs(parity_error):.2e}  OK")

    # Test 3: Delta of ATM call should be around 0.5-0.6 for short maturities
    delta_call = bs_delta(100, 100, 0.5, 0.05, 0.20, 'call')
    assert 0.5 < delta_call < 0.7, f"Test 3 failed: ATM delta = {delta_call}"
    print(f"Test 3 (ATM call delta):  delta = {delta_call:.4f}  OK")

    # Test 4: Implied volatility should invert exactly for a call priced at sigma=0.25
    sigma_true = 0.25
    price = bs_price(100, 100, 1.0, 0.05, sigma_true, 'call')
    sigma_implied = bs_implied_volatility(price, 100, 100, 1.0, 0.05, 'call')
    assert abs(sigma_implied - sigma_true) < 1e-6, f"Test 4 failed: {sigma_implied} vs {sigma_true}"
    print(f"Test 4 (IV round-trip):   sigma_implied = {sigma_implied:.6f}  (expected {sigma_true})  OK")

    # Test 5: Vectorisation — price a whole surface at once
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.array([[0.25], [0.5], [1.0], [2.0]])
    prices = bs_price(100, strikes, maturities, 0.05, 0.20, 'call')
    assert prices.shape == (4, 5), f"Test 5 failed: shape = {prices.shape}"
    print(f"Test 5 (vectorised grid): shape = {prices.shape}  OK")

    print("=" * 60)
    print("All tests passed.")
