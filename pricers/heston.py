"""
Heston (1993) stochastic volatility pricer for European vanilla options,
implemented via the Fourier COS method of Fang & Oosterlee (2008).

Model dynamics (under Q):
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho dt

The COS method expresses the risk-neutral density of the log-price as a
Fourier cosine series, truncates at N terms, and integrates analytically
each cosine coefficient against the option payoff.

Reference: Fang, F., & Oosterlee, C.W. (2008), SIAM J. Sci. Comput. 31(2).
"""

import numpy as np


def _heston_char_fn(u, T, r, q, v0, kappa, theta, sigma_v, rho):
    """
    Characteristic function of X_T = ln(S_T / S_0) under Heston.

    Uses the "little trap" formulation (Albrecher et al. 2007) for numerical
    stability at long maturities.
    """
    xi = kappa - sigma_v * rho * 1j * u
    d = np.sqrt(xi**2 + sigma_v**2 * (u**2 + 1j * u))
    g2 = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T)

    C = ((r - q) * 1j * u * T
         + kappa * theta / sigma_v**2
         * ((xi - d) * T - 2.0 * np.log((1.0 - g2 * exp_dT) / (1.0 - g2))))
    D = (xi - d) / sigma_v**2 * (1.0 - exp_dT) / (1.0 - g2 * exp_dT)

    return np.exp(C + D * v0)


def _chi_psi(a, b, c, d, k):
    """
    Fang & Oosterlee (2008) auxiliary integrals chi and psi on [c, d].
    Used to build the COS payoff coefficients.
    """
    factor = k * np.pi / (b - a)

    chi = (1.0 / (1.0 + factor**2)) * (
        np.cos(factor * (d - a)) * np.exp(d)
        - np.cos(factor * (c - a)) * np.exp(c)
        + factor * np.sin(factor * (d - a)) * np.exp(d)
        - factor * np.sin(factor * (c - a)) * np.exp(c)
    )

    psi = np.zeros_like(k, dtype=float)
    mask = (k == 0)
    psi[mask] = d - c
    if np.any(~mask):
        f = factor[~mask]
        psi[~mask] = (np.sin(f * (d - a)) - np.sin(f * (c - a))) / f

    return chi, psi


def heston_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                     q=0.0, N_cos=256, L=12.0):
    """
    Price a European call under Heston (1993) via the COS method.

    Parameters
    ----------
    S0       : spot price (scalar)
    K        : strike price (scalar or array — vectorised)
    T        : maturity
    r        : risk-free rate
    v0       : initial variance
    kappa    : mean reversion speed
    theta    : long-run variance
    sigma_v  : vol-of-vol
    rho      : spot-variance correlation
    q        : dividend yield
    N_cos    : number of COS terms
    L        : half-width of the truncation domain in cumulant units

    Returns
    -------
    Call price(s), same shape as K.
    """
    K = np.atleast_1d(K).astype(float)

    # For each strike, we work in log-moneyness x = ln(S0/K)
    # Payoff at maturity: (S_T - K)+ = K * (exp(X_T + x) - 1)+ where X_T = ln(S_T/S_0)
    # Equivalent to (exp(y) - 1)+ with y = X_T + x - x_target where we integrate on the y grid.

    # Cumulants of X_T = ln(S_T / S_0) — scalars, don't depend on K
    c1 = (r - q) * T + (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
    c2 = (1.0 / (8.0 * kappa**3)) * (
        sigma_v * T * kappa * np.exp(-kappa * T) * (v0 - theta)
        * (8.0 * kappa * rho - 4.0 * sigma_v)
        + kappa * rho * sigma_v * (1.0 - np.exp(-kappa * T)) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T
        * (-4.0 * kappa * rho * sigma_v + sigma_v**2 + 4.0 * kappa**2)
        + sigma_v**2 * ((theta - 2.0 * v0) * np.exp(-2.0 * kappa * T)
                        + theta * (6.0 * np.exp(-kappa * T) - 7.0) + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - np.exp(-kappa * T))
    )
    c2 = max(abs(c2), 1e-8)

    # Truncation on X_T (same for all strikes)
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    # COS grid
    k = np.arange(N_cos)
    u = k * np.pi / (b - a)

    # Characteristic function * exp(-i u a) — scalar in K, vector in k
    phi = _heston_char_fn(u, T, r, q, v0, kappa, theta, sigma_v, rho)
    F = np.real(phi * np.exp(-1j * u * a))
    F[0] *= 0.5

    prices = np.empty(len(K))
    for i, Ki in enumerate(K):
        # log-moneyness x_i = ln(S0/K_i)
        # Payoff in y = X_T space: K_i * (exp(y + x_i) - 1)+ on y in [-x_i, b]
        # (i.e., positive when S_T > K_i, i.e., y > -x_i)
        x_i = np.log(S0 / Ki)

        # For the call, integrate the payoff on [max(a, -x_i), b]
        y_lower = max(a, -x_i)
        y_upper = b

        # Payoff = K_i * (exp(y + x_i) - 1) on [y_lower, b]
        # = K_i * exp(x_i) * exp(y) - K_i
        # Cosine coefficients of (exp(x_i) * exp(y) - 1) on [y_lower, b] against cos(k pi (y-a)/(b-a))
        chi, psi = _chi_psi(a, b, y_lower, y_upper, k)
        U = 2.0 / (b - a) * (S0 / Ki * chi - psi)  # exp(x_i) = S0/K_i

        prices[i] = Ki * np.exp(-r * T) * np.sum(F * U)

    return float(prices[0]) if prices.shape == (1,) else prices


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from black_scholes import bs_price, bs_implied_volatility

    print("Running unit tests for heston.py")
    print("=" * 60)

    # Test 1: Degenerate case
    S0, T, r = 100.0, 1.0, 0.05
    sigma_bs = 0.20
    v0 = theta = sigma_bs**2
    kappa = 2.0
    sigma_v = 1e-4
    rho = 0.0

    K = 100.0
    price_h = heston_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho)
    price_bs = bs_price(S0, K, T, r, sigma_bs, 'call')
    err = abs(price_h - price_bs)
    print(f"Test 1 (Heston sigma_v->0 => BS): "
          f"Heston = {price_h:.4f}  BS = {price_bs:.4f}  err = {err:.5f}")
    assert err < 0.01, f"Test 1 failed: err = {err}"
    print(f"   OK")

    # Test 2: Vectorisation + monotonicity
    v0 = 0.04
    theta = 0.04
    sigma_v = 0.30
    rho = -0.70
    strikes = np.array([80, 90, 100, 110, 120])
    prices_v = heston_price_cos(S0, strikes, T, r, v0, kappa, theta, sigma_v, rho)
    print(f"Test 2 (vectorised strikes):     "
          f"prices = {[f'{p:.3f}' for p in prices_v]}")
    assert prices_v.shape == (5,), f"Shape wrong: {prices_v.shape}"
    assert np.all(np.diff(prices_v) < 0), "Call prices should decrease in K"
    print(f"   OK  (monotonically decreasing)")

    # Test 3: Negative skew from negative rho
    iv_smile = bs_implied_volatility(prices_v, S0, strikes, T, r, 'call')
    print(f"Test 3 (negative skew):          IVs = {[f'{iv*100:.2f}%' for iv in iv_smile]}")
    assert iv_smile[0] > iv_smile[-1], "No downward skew"
    print(f"   OK  (IV[K=80] = {iv_smile[0]*100:.2f}% > IV[K=120] = {iv_smile[-1]*100:.2f}%)")

    # Test 4: Match BS across all strikes when sigma_v -> 0
    print("\nTest 4 (BS reconciliation across strikes at sigma_v -> 0):")
    v0 = theta = 0.04
    sigma_v = 1e-4
    rho = 0.0
    all_pass = True
    for K in [80, 90, 100, 110, 120]:
        h = heston_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho)
        b = bs_price(S0, K, T, r, 0.20, 'call')
        err = abs(h - b)
        status = "OK" if err < 0.05 else "FAIL"
        if err >= 0.05:
            all_pass = False
        print(f"   K={K:3.0f}: Heston={h:8.4f}  BS={b:8.4f}  err={err:.5f}  {status}")
    assert all_pass, "Test 4 failed"

    print("=" * 60)
    print("All tests passed.")
