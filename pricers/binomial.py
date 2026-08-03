"""
Binomial tree pricer for European vanilla options.

Implements two schemes:
    - Cox, Ross & Rubinstein (1979): u * d = 1, moment-matching on volatility
    - Jarrow & Rudd (1983): risk-neutral drift centring with p = 0.5

Both are vectorised via NumPy backward induction.

Notation follows the standard convention:
    u  : up-move factor
    d  : down-move factor
    p  : risk-neutral probability of the up-move
    dt : time step

Convergence: both schemes converge to Black-Scholes at rate O(1/N) as N -> infinity.
"""

import numpy as np


def _lattice_params(sigma, r, q, T, N, scheme='crr'):
    """
    Compute the u, d, p, dt parameters for the specified binomial scheme.

    Parameters
    ----------
    sigma : volatility
    r     : risk-free rate
    q     : dividend yield
    T     : maturity
    N     : number of time steps
    scheme : 'crr' (Cox-Ross-Rubinstein, default) or 'jr' (Jarrow-Rudd)

    Returns
    -------
    u, d, p, dt : the lattice parameters.
    """
    dt = T / N
    if scheme == 'crr':
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        a = np.exp((r - q) * dt)
        p = (a - d) / (u - d)
    elif scheme == 'jr':
        # Risk-neutral drift centring, p = 0.5
        u = np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt))
        d = np.exp((r - q - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt))
        p = 0.5
    else:
        raise ValueError("scheme must be 'crr' or 'jr'")

    # Numerical safety: clip probabilities to (eps, 1-eps) to avoid NaNs
    # when N is very large or sigma is small
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return u, d, p, dt


def binomial_price(S0, K, T, r, sigma, N=512, option_type='call', scheme='crr', q=0.0):
    """
    Price a European vanilla option using a binomial tree.

    Uses vectorised backward induction: builds the terminal payoff vector
    (N+1 nodes) and rolls back to today.

    Parameters
    ----------
    S0 : spot price
    K  : strike price (scalar or array — vectorised)
    T  : maturity
    r  : risk-free rate
    sigma : volatility
    N  : number of time steps (default 512)
    option_type : 'call' or 'put'
    scheme : 'crr' or 'jr'
    q  : dividend yield (default 0)

    Returns
    -------
    Price (scalar or array, matching K).
    """
    K = np.atleast_1d(K).astype(float)
    is_call = option_type.lower().startswith('c')
    u, d, p, dt = _lattice_params(sigma, r, q, T, N, scheme=scheme)

    # Terminal stock prices: S0 * u^j * d^(N-j) for j = 0, ..., N
    j = np.arange(N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))  # shape (N+1,)

    # Terminal payoffs, broadcast against strikes: shape (len(K), N+1)
    if is_call:
        values = np.maximum(S_T[None, :] - K[:, None], 0.0)
    else:
        values = np.maximum(K[:, None] - S_T[None, :], 0.0)

    # Backward induction, N steps
    disc = np.exp(-r * dt)
    for n in range(N, 0, -1):
        # Combine node j and node j+1 at level n into node j at level n-1
        values = disc * (p * values[:, 1:n+1] + (1 - p) * values[:, 0:n])

    price = values[:, 0]
    return float(price[0]) if price.shape == (1,) else price


# ============================================================================
# UNIT TESTS + convergence to Black-Scholes
# ============================================================================

if __name__ == '__main__':
    from black_scholes import bs_price

    print("Running unit tests for binomial.py")
    
    
    print("=" * 60)

    # Test 1: CRR convergence to Black-Scholes at N=512
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_ref = bs_price(S0, K, T, r, sigma, 'call')
    p_crr_512 = binomial_price(S0, K, T, r, sigma, N=512, option_type='call', scheme='crr')
    err_crr = abs(p_crr_512 - bs_ref)
    assert err_crr < 0.02, f"Test 1 failed: CRR error = {err_crr}"
    print(f"Test 1 (CRR N=512 ATM call):  price = {p_crr_512:.4f}  BS = {bs_ref:.4f}  err = {err_crr:.4f}  OK")

    # Test 2: JR convergence to Black-Scholes at N=512
    p_jr_512 = binomial_price(S0, K, T, r, sigma, N=512, option_type='call', scheme='jr')
    err_jr = abs(p_jr_512 - bs_ref)
    assert err_jr < 0.02, f"Test 2 failed: JR error = {err_jr}"
    print(f"Test 2 (JR N=512 ATM call):   price = {p_jr_512:.4f}  BS = {bs_ref:.4f}  err = {err_jr:.4f}  OK")

    # Test 3: Convergence rate — error should approximately halve when N doubles
    errors = []
    for N in [128, 256, 512, 1024]:
        p = binomial_price(S0, K, T, r, sigma, N=N, option_type='call', scheme='crr')
        errors.append(abs(p - bs_ref))
    print(f"Test 3 (CRR convergence):")
    for i, N in enumerate([128, 256, 512, 1024]):
        print(f"         N = {N:>5d}:  err = {errors[i]:.5f}")
    # Rate is roughly O(1/N), so err(N=1024) < err(N=128) / 4
    assert errors[3] < errors[0] / 3, "Convergence rate insufficient"
    print(f"         Convergence rate: OK")

    # Test 4: Vectorisation over strikes
    strikes = np.array([80, 90, 100, 110, 120])
    prices = binomial_price(S0, strikes, T, r, sigma, N=512, option_type='call', scheme='crr')
    assert prices.shape == (5,), f"Test 4 failed: shape = {prices.shape}"
    # Call prices should be monotonically decreasing in strike
    assert np.all(np.diff(prices) < 0), "Prices not monotonic in strike"
    print(f"Test 4 (vectorised strikes):  prices = {[f'{p:.3f}' for p in prices]}  OK")

    # Test 5: Put-call parity on the tree
    C = binomial_price(S0, K, T, r, sigma, N=512, option_type='call', scheme='crr')
    P = binomial_price(S0, K, T, r, sigma, N=512, option_type='put', scheme='crr')
    parity_err = (C - P) - (S0 - K * np.exp(-r * T))
    assert abs(parity_err) < 0.02, f"Test 5 failed: parity error = {parity_err}"
    print(f"Test 5 (put-call parity):     C - P - (S - K*e^-rT) = {parity_err:.4f}  OK")

    print("=" * 60)
    print("All tests passed.")

