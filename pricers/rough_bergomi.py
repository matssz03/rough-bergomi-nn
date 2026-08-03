"""
Rough Bergomi Monte Carlo pricer (thesis Section 3.4.1).

Implements the hybrid scheme of Bennedsen, Lunde and Pakkanen (2017)
for the rough Bergomi model of Bayer, Friz and Gatheral (2016):

    dS_t = S_t * sqrt(v_t) dW^S_t
    v_t  = xi_0(t) * exp( eta * sqrt(2H) * I_t - 0.5 * eta^2 * t^{2H} )

where I_t = int_0^t (t - s)^{H - 1/2} dW^v_s is the Volterra integral,
and d<W^S, W^v>_t = rho dt.

The scheme provides strong convergence of order H + 1/2 and delivers
accurate paths at a cost that scales linearly with the number of time
steps.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import rfft, irfft
from scipy.stats import norm


# --------------------------------------------------------------------- #
# Hybrid scheme kernel utilities
# --------------------------------------------------------------------- #

def _covariance_matrix(H: float, n: int) -> np.ndarray:
    """
    2x2 covariance matrix of (W_1, Y_1) where
      - W_1 is the standard Brownian increment on [0, 1/n]
      - Y_1 is the exact Volterra integral
                Y_1 = int_0^{1/n} (1/n - s)^{H - 1/2} dW_s
    used at the first time step of the kappa = 1 hybrid scheme.
    """
    alpha = H - 0.5
    cov = np.zeros((2, 2))
    cov[0, 0] = 1.0 / n
    cov[1, 1] = 1.0 / ((2.0 * alpha + 1.0) * n ** (2.0 * alpha + 1.0))
    cov[0, 1] = 1.0 / ((alpha + 1.0) * n ** (alpha + 1.0))
    cov[1, 0] = cov[0, 1]
    return cov


# --------------------------------------------------------------------- #
# rough Bergomi model
# --------------------------------------------------------------------- #

class RoughBergomi:
    """
    Rough Bergomi model with a piecewise-linear forward-variance curve.

    Parameters
    ----------
    xi0 : (M,) array
        Forward-variance term structure sampled at ``xi0_maturities``.
    xi0_maturities : (M,) array
        Maturities (in years) at which ``xi0`` is specified.
    eta : float
        Volatility of volatility.
    H : float
        Hurst parameter of the variance process (H < 1/2 = rough regime).
    rho : float
        Instantaneous correlation between the spot and variance Brownians.
    S0 : float, default 1.0
        Initial spot price.
    r : float, default 0.0
        Risk-free rate.
    q : float, default 0.0
        Continuous dividend yield.
    """

    def __init__(
        self,
        xi0: np.ndarray,
        xi0_maturities: np.ndarray,
        eta: float,
        H: float,
        rho: float,
        S0: float = 1.0,
        r: float = 0.0,
        q: float = 0.0,
    ):
        self.xi0 = np.asarray(xi0, dtype=float)
        self.xi0_maturities = np.asarray(xi0_maturities, dtype=float)
        self.eta = float(eta)
        self.H = float(H)
        self.rho = float(rho)
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)

        if not (0.0 < self.H < 1.0):
            raise ValueError(f"H must lie in (0, 1), got {self.H}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho}")
        if self.eta <= 0.0:
            raise ValueError(f"eta must be positive, got {self.eta}")

    # ---------------- forward variance interpolation ----------------

    def forward_variance(self, t: np.ndarray | float) -> np.ndarray | float:
        """Piecewise-linear interpolation of the forward-variance curve."""
        return np.interp(
            t,
            self.xi0_maturities,
            self.xi0,
            left=float(self.xi0[0]),
            right=float(self.xi0[-1]),
        )

    # ---------------- path simulation ----------------

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
        antithetic: bool = True,
        return_variance: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate ``n_paths`` paths of the spot on [0, T].

        Parameters
        ----------
        T : float
            Horizon in years.
        n_steps : int
            Number of time steps per year (so the total number of steps
            is ``round(n_steps * T)``).
        n_paths : int
            Number of Monte Carlo paths.  When ``antithetic`` is True this
            is rounded down to the nearest even integer.
        seed : int, optional
            RNG seed for reproducibility.
        antithetic : bool, default True
            Enable antithetic variates on the driving Brownian motions.
        return_variance : bool, default False
            If True, also return the simulated variance process.

        Returns
        -------
        t_grid : (N + 1,) array
            Time grid in years.
        S : (N + 1, n_paths) array
            Simulated spot paths, with ``S[0] == S0``.
        v : (N + 1, n_paths) array, optional
            Simulated variance paths (only if ``return_variance`` is True).
        """
        rng = np.random.default_rng(seed)
        N = int(round(n_steps * T))
        dt = T / N
        t_grid = np.linspace(0.0, T, N + 1)
        n_grid = n_steps  # 'n' in the hybrid-scheme convention
        alpha = self.H - 0.5

        # Joint (dW^v, Y) with exact 2x2 covariance (kappa = 1)
        cov = _covariance_matrix(self.H, n_grid)
        L = np.linalg.cholesky(cov)

        if antithetic and n_paths % 2 == 1:
            n_paths += 1  # ensure evenness
        if antithetic:
            n_half = n_paths // 2
            z_v = rng.standard_normal((N, n_half, 2))
            z_v = np.concatenate([z_v, -z_v], axis=1)
            z_perp = rng.standard_normal((N, n_half))
            z_perp = np.concatenate([z_perp, -z_perp], axis=1)
        else:
            z_v = rng.standard_normal((N, n_paths, 2))
            z_perp = rng.standard_normal((N, n_paths))

        dW_Y = z_v @ L.T                # (N, n_paths, 2)
        dW_v = dW_Y[..., 0]             # variance-BM increments
        Y1   = dW_Y[..., 1]             # exact integrals for step 1

        # Correlated spot Brownian
        dW_S = self.rho * dW_v + np.sqrt(1.0 - self.rho ** 2) * np.sqrt(dt) * z_perp

        # Volterra process I_{t_i} = Y_i + sum_{k=2..i} G(k) * dW^v_{i-k+1}
        # with the midpoint kernel G(k) = ((k - 0.5) / n)^alpha
        k_idx = np.arange(1, N + 1)
        G = ((k_idx - 0.5) / n_grid) ** alpha

        # Vectorised FFT convolution of dW^v (N x n_paths) with G[1:]
        L_conv = N + N - 1
        fft_len = 1 << (L_conv - 1).bit_length()
        dW_v_pad = np.zeros((fft_len, n_paths))
        dW_v_pad[:N, :] = dW_v
        G_tail = np.zeros(fft_len)
        G_tail[: N - 1] = G[1:]
        conv = irfft(
            rfft(dW_v_pad, axis=0) * rfft(G_tail)[:, None],
            n=fft_len, axis=0,
        )[:N, :]

        I = Y1 + conv  # (N, n_paths)

        # Variance process at t_1, ..., t_N
        t_i = t_grid[1:]
        xi_at_t = self.forward_variance(t_i)  # (N,)
        v = xi_at_t[:, None] * np.exp(
            self.eta * np.sqrt(2.0 * self.H) * I
            - 0.5 * self.eta ** 2 * t_i[:, None] ** (2.0 * self.H)
        )

        # Left-endpoint variance for the spot Euler step
        v0 = float(self.forward_variance(0.0))
        v_left = np.vstack([np.full((1, n_paths), v0), v[:-1, :]])

        # Log-price increments with drift adjusted for r - q
        log_incr = ((self.r - self.q) * dt
                    - 0.5 * v_left * dt
                    + np.sqrt(np.maximum(v_left, 0.0)) * dW_S)
        log_S = np.concatenate([np.zeros((1, n_paths)), np.cumsum(log_incr, axis=0)], axis=0)
        S = self.S0 * np.exp(log_S)

        if return_variance:
            v_full = np.vstack([np.full((1, n_paths), v0), v])  # (N+1, n_paths)
            return t_grid, S, v_full
        return t_grid, S

    # ---------------- pricing ----------------

    def price_european(
        self,
        maturities: np.ndarray,
        strikes: np.ndarray,
        option_type: str = "otm",
        n_steps: int = 100,
        n_paths: int = 15_000,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Price European vanilla contracts on a maturity x strike grid.

        Parameters
        ----------
        maturities : (n_T,) array
            Maturities in years.  All prices are extracted from the same
            simulation run, evaluated at the closest time-grid indices.
        strikes : (n_K,) array
            Strike prices.
        option_type : {"call", "put", "otm"}
            "call" for all-call, "put" for all-put, "otm" for OTM
            selection (put when K < S0, call when K >= S0).  The OTM
            convention is the numerically stable choice used throughout
            the thesis; the returned array is always converted to
            call-equivalent prices via put-call parity.
        n_steps, n_paths, seed, antithetic : simulation controls

        Returns
        -------
        call_prices : (n_T, n_K) array
            Call-equivalent prices (puts converted by parity when
            ``option_type == "otm"``).
        se_prices : (n_T, n_K) array
            Monte Carlo standard errors on those prices.
        """
        maturities = np.asarray(maturities, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        T_max = float(maturities.max())

        t_grid, S = self.simulate(
            T=T_max, n_steps=n_steps, n_paths=n_paths,
            seed=seed, antithetic=antithetic,
        )

        n_T, n_K = len(maturities), len(strikes)
        call_prices = np.full((n_T, n_K), np.nan)
        se_prices   = np.full((n_T, n_K), np.nan)

        for i, T in enumerate(maturities):
            idx = int(round(T / T_max * (len(t_grid) - 1)))
            ST = S[idx, :]
            df = np.exp(-self.r * T)
            for j, K in enumerate(strikes):
                if option_type == "call" or (option_type == "otm" and K >= self.S0):
                    payoff = np.maximum(ST - K, 0.0)
                    price = df * payoff.mean()
                    se = df * payoff.std(ddof=1) / np.sqrt(len(payoff))
                elif option_type == "put" or (option_type == "otm" and K < self.S0):
                    payoff = np.maximum(K - ST, 0.0)
                    price_put = df * payoff.mean()
                    se = df * payoff.std(ddof=1) / np.sqrt(len(payoff))
                    # Convert to call by put-call parity
                    price = price_put + self.S0 * np.exp(-self.q * T) - K * df
                else:
                    raise ValueError(f"unknown option_type {option_type!r}")
                call_prices[i, j] = price
                se_prices[i, j]   = se

        return call_prices, se_prices

    def implied_vol_surface(
        self,
        maturities: np.ndarray,
        strikes: np.ndarray,
        n_steps: int = 100,
        n_paths: int = 15_000,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Black-Scholes implied-volatility surface on the grid.

        Returns
        -------
        iv : (n_T, n_K) array of implied volatilities
        se_iv : (n_T, n_K) array of MC standard errors on those IVs,
                converted from the price SE via the Black-Scholes vega
                evaluated at the point estimate.
        """
        # Prefer the shared inversion in utils.implied_vol when available;
        # otherwise fall back to the local one below.
        try:
            from utils.implied_vol import implied_vol, bs_vega
        except ImportError:
            implied_vol, bs_vega = _local_implied_vol, _local_bs_vega

        call_prices, se_prices = self.price_european(
            maturities=maturities, strikes=strikes,
            option_type="otm",
            n_steps=n_steps, n_paths=n_paths,
            seed=seed, antithetic=antithetic,
        )
        n_T, n_K = call_prices.shape
        iv = np.full((n_T, n_K), np.nan)
        se_iv = np.full((n_T, n_K), np.nan)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                sigma = implied_vol(call_prices[i, j], self.S0, K, T, self.r, self.q)
                iv[i, j] = sigma
                if np.isfinite(sigma) and sigma > 0.0:
                    vega = bs_vega(self.S0, K, T, sigma, self.r, self.q)
                    if vega > 1e-10:
                        se_iv[i, j] = se_prices[i, j] / vega
        return iv, se_iv


# --------------------------------------------------------------------- #
# Local fall-back for implied volatility (used only if utils is missing)
# --------------------------------------------------------------------- #

def _local_bs_call(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _local_bs_vega(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def _local_implied_vol(price, S, K, T, r=0.0, q=0.0, tol=1e-8, max_iter=100):
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = _local_bs_call(S, K, T, mid, r, q)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
