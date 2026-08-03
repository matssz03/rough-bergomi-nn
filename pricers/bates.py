"""
Bates (1996) stochastic-volatility model with log-normal jumps, priced
via the Fourier COS method of Fang & Oosterlee (2008).

Model dynamics (under Q):
    dS_t = (r - q - lam * kappa_bar) S_t dt
           + sqrt(v_t) S_t dW^S_t
           + (Y - 1) S_t dN_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho dt

with N_t a Poisson process of intensity lam, log Y ~ N(mu_J, sigma_J^2)
independent of the diffusion, and the drift compensator
    kappa_bar = E[Y - 1] = exp(mu_J + 0.5 sigma_J^2) - 1
so that S is a martingale after discounting.

The Bates characteristic function factorises as the Heston
characteristic function (with drift adjusted by -lam * kappa_bar) times
the log-normal jump component of Merton (1976):

    phi_Bates(u) = phi_Heston(u; r - q - lam kappa_bar)
                   * exp(lam T (exp(i u mu_J - 0.5 u^2 sigma_J^2) - 1))

Reference: Bates, D.S. (1996), Review of Financial Studies 9(1).
           Fang, F., & Oosterlee, C.W. (2008), SIAM J. Sci. Comput. 31(2).
"""

import numpy as np

from pricers.heston import _heston_char_fn, _chi_psi


def _bates_char_fn(u, T, r, q, v0, kappa, theta, sigma_v, rho,
                   lam, mu_J, sigma_J):
    """
    Characteristic function of X_T = ln(S_T / S_0) under Bates.

    Uses the Heston characteristic function of Section 3.4.1 with an
    adjusted risk-neutral drift, multiplied by the Merton jump factor.
    """
    # Jump compensator: kappa_bar = E[Y - 1] = exp(mu_J + 0.5 sigma_J^2) - 1
    kappa_bar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # Heston part with drift adjusted for the jump compensator
    r_adj = r - lam * kappa_bar
    heston_phi = _heston_char_fn(u, T, r_adj, q, v0, kappa, theta,
                                 sigma_v, rho)

    # Merton jump part: Poisson compound of log-normal jumps
    jump_char = np.exp(1j * u * mu_J - 0.5 * u**2 * sigma_J**2) - 1.0
    jump_phi = np.exp(lam * T * jump_char)

    return heston_phi * jump_phi


def bates_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                    lam, mu_J, sigma_J, q=0.0, N_cos=512, L=15.0):
    """
    Price a European call under Bates (1996) via the COS method.

    Parameters
    ----------
    S0       : spot price (scalar)
    K        : strike price (scalar or array — vectorised)
    T        : maturity
    r        : risk-free rate
    v0       : initial variance
    kappa    : mean reversion speed of variance
    theta    : long-run variance
    sigma_v  : vol-of-vol
    rho      : spot-variance correlation
    lam      : jump intensity (per unit of time)
    mu_J     : mean of log jump size
    sigma_J  : standard deviation of log jump size
    q        : dividend yield
    N_cos    : number of COS terms
    L        : half-width of the truncation domain in cumulant units

    Returns
    -------
    Call price(s), same shape as K.
    """
    K = np.atleast_1d(K).astype(float)

    # Jump compensator
    kappa_bar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # Cumulants of X_T = ln(S_T / S_0) under Bates
    # Diffusion (Heston) part with drift adjusted for jumps
    r_adj = r - lam * kappa_bar
    c1_H = ((r_adj - q) * T
            + (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa)
            - 0.5 * theta * T)
    c2_H = (1.0 / (8.0 * kappa**3)) * (
        sigma_v * T * kappa * np.exp(-kappa * T) * (v0 - theta)
        * (8.0 * kappa * rho - 4.0 * sigma_v)
        + kappa * rho * sigma_v * (1.0 - np.exp(-kappa * T))
        * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T
        * (-4.0 * kappa * rho * sigma_v + sigma_v**2 + 4.0 * kappa**2)
        + sigma_v**2 * ((theta - 2.0 * v0) * np.exp(-2.0 * kappa * T)
                        + theta * (6.0 * np.exp(-kappa * T) - 7.0)
                        + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - np.exp(-kappa * T))
    )

    # Jump part: X_T contribution = sum_{k=1..N_T} log Y_k
    # First cumulant: lam * T * mu_J
    # Second cumulant: lam * T * (mu_J^2 + sigma_J^2)
    # Fourth cumulant: lam * T * (mu_J^4 + 6 mu_J^2 sigma_J^2 + 3 sigma_J^4)
    c1_J = lam * T * mu_J
    c2_J = lam * T * (mu_J**2 + sigma_J**2)
    c4_J = lam * T * (mu_J**4 + 6.0 * mu_J**2 * sigma_J**2 + 3.0 * sigma_J**4)

    c1 = c1_H + c1_J
    c2 = c2_H + c2_J
    c4 = c4_J  # Heston c4 is small, dominated by jump c4
    c2 = max(abs(c2), 1e-8)

    # Truncation on X_T (Fang-Oosterlee 2008 rule for jump-diffusions)
    half_width = L * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))
    a = c1 - half_width
    b = c1 + half_width

    # COS grid
    k = np.arange(N_cos)
    u = k * np.pi / (b - a)

    # Characteristic function * exp(-i u a) — scalar in K, vector in k
    phi = _bates_char_fn(u, T, r, q, v0, kappa, theta, sigma_v, rho,
                         lam, mu_J, sigma_J)
    F = np.real(phi * np.exp(-1j * u * a))
    F[0] *= 0.5

    prices = np.empty(len(K))
    for i, Ki in enumerate(K):
        # log-moneyness x_i = ln(S0/K_i)
        x_i = np.log(S0 / Ki)

        # Call payoff support in y = X_T: [max(a, -x_i), b]
        y_lower = max(a, -x_i)
        y_upper = b

        # Payoff coefficients: K_i * (exp(x_i) exp(y) - 1) on [y_lower, b]
        chi, psi = _chi_psi(a, b, y_lower, y_upper, k)
        U = 2.0 / (b - a) * (S0 / Ki * chi - psi)

        prices[i] = Ki * np.exp(-r * T) * np.sum(F * U)

    return float(prices[0]) if prices.shape == (1,) else prices


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pricers.black_scholes import bs_price, bs_implied_volatility
    from pricers.heston import heston_price_cos

    print("Running unit tests for bates.py")
    print("=" * 60)

    S0, T, r = 100.0, 1.0, 0.05

    # Test 1: Bates -> Heston when lam -> 0
    v0 = theta = 0.04
    kappa = 2.0
    sigma_v = 0.30
    rho = -0.70
    lam = 0.0
    mu_J = 0.0
    sigma_J = 0.10

    K = 100.0
    p_bates = bates_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                              lam, mu_J, sigma_J)
    p_heston = heston_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho)
    err = abs(p_bates - p_heston)
    print(f"Test 1 (Bates lam=0 => Heston):  "
          f"Bates = {p_bates:.4f}  Heston = {p_heston:.4f}  err = {err:.5f}")
    assert err < 1e-6, f"Test 1 failed: err = {err}"
    print(f"   OK")

    # Test 2: Bates -> BS when sigma_v=0 AND lam=0
    v0 = theta = 0.04
    sigma_v = 1e-4
    rho = 0.0
    lam = 0.0
    p_bates = bates_price_cos(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                              lam, mu_J, sigma_J)
    p_bs = bs_price(S0, K, T, r, 0.20, 'call')
    err = abs(p_bates - p_bs)
    print(f"Test 2 (Bates -> BS limit):      "
          f"Bates = {p_bates:.4f}  BS = {p_bs:.4f}  err = {err:.5f}")
    assert err < 0.02, f"Test 2 failed: err = {err}"
    print(f"   OK")

    # Test 3: Jumps make short-maturity smile more pronounced
    v0 = theta = 0.04
    sigma_v = 0.30
    rho = -0.70
    lam_no_jump = 0.0
    lam_jump = 0.5
    mu_J = -0.10   # negative mean jump -> left tail
    sigma_J = 0.15
    T_short = 0.1

    strikes = np.array([90, 95, 100, 105, 110])
    p_no_jump = bates_price_cos(S0, strikes, T_short, r,
                                v0, kappa, theta, sigma_v, rho,
                                lam_no_jump, mu_J, sigma_J)
    p_jump = bates_price_cos(S0, strikes, T_short, r,
                             v0, kappa, theta, sigma_v, rho,
                             lam_jump, mu_J, sigma_J)
    iv_no_jump = bs_implied_volatility(p_no_jump, S0, strikes, T_short, r, 'call')
    iv_jump = bs_implied_volatility(p_jump, S0, strikes, T_short, r, 'call')
    print(f"Test 3 (short-maturity skew from jumps):")
    print(f"   No jumps: IVs = {[f'{iv*100:.2f}%' for iv in iv_no_jump]}")
    print(f"   Jumps:    IVs = {[f'{iv*100:.2f}%' for iv in iv_jump]}")
    skew_no_jump = iv_no_jump[0] - iv_no_jump[-1]
    skew_jump = iv_jump[0] - iv_jump[-1]
    assert skew_jump > skew_no_jump, \
        f"Negative jumps should increase left skew: {skew_no_jump} vs {skew_jump}"
    print(f"   OK  (skew: no-jump = {skew_no_jump*100:.2f} pt, jump = {skew_jump*100:.2f} pt)")

    # Test 4: Vectorised strikes + monotonicity
    v0 = theta = 0.04
    sigma_v = 0.30
    rho = -0.70
    lam = 0.5
    mu_J = -0.10
    sigma_J = 0.15
    strikes = np.array([80, 90, 100, 110, 120])
    prices_v = bates_price_cos(S0, strikes, T, r,
                               v0, kappa, theta, sigma_v, rho,
                               lam, mu_J, sigma_J)
    print(f"Test 4 (vectorised + monotonicity): "
          f"prices = {[f'{p:.3f}' for p in prices_v]}")
    assert prices_v.shape == (5,)
    assert np.all(np.diff(prices_v) < 0), "Call prices should decrease in K"
    print(f"   OK")

    print("=" * 60)
    print("All Bates tests passed.")
