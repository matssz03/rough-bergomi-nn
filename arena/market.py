"""
Market data-generating process (thesis Section 3.4.3).

Every trading episode of the arena is anchored to a single rough Bergomi
"true world":

    theta_true = (xi_0, eta, rho, H)          sampled once per episode

    market IV quotes = rBergomi(theta_true) + idealised bid-ask noise
    true underlying path S_t = rBergomi(theta_true) simulated in-sample

The bid-ask noise follows the setup of Bakshi, Cao & Chen (1997): an
additive Gaussian perturbation on implied volatility, isotropic across
the grid, with standard deviation chosen so that the resulting spread
is comparable to the one observed on liquid index options.
"""
from __future__ import annotations

import numpy as np

from neural_network.dataset import CFG as DATASET_CFG
from pricers.rough_bergomi import RoughBergomi


# --------------------------------------------------------------------- #
# True-world sampling
# --------------------------------------------------------------------- #

def sample_theta_true(rng: np.random.Generator) -> np.ndarray:
    """
    Draw a rough Bergomi parameter vector uniformly from the training
    domain (Section 3.1.3).
    """
    cfg = DATASET_CFG
    xi  = rng.uniform(cfg.xi_lo,  cfg.xi_hi,  size=8)
    eta = float(rng.uniform(cfg.eta_lo, cfg.eta_hi))
    rho = float(rng.uniform(cfg.rho_lo, cfg.rho_hi))
    H   = float(rng.uniform(cfg.H_lo,   cfg.H_hi))
    return np.concatenate([xi, [eta, rho, H]]).astype(np.float32)


def theta_to_model(theta: np.ndarray, S0: float = 1.0) -> RoughBergomi:
    """Turn an 11-vector back into a RoughBergomi model instance."""
    cfg = DATASET_CFG
    xi0 = theta[:8]
    eta = float(theta[8]); rho = float(theta[9]); H = float(theta[10])
    return RoughBergomi(
        xi0=xi0.astype(np.float64),
        xi0_maturities=np.array(cfg.xi0_maturities),
        eta=eta, H=H, rho=rho, S0=S0,
    )


# --------------------------------------------------------------------- #
# Market quotes (true surface + idealised bid-ask noise)
# --------------------------------------------------------------------- #

def generate_market_quotes(
    theta_true: np.ndarray,
    maturities: np.ndarray,
    strikes: np.ndarray,
    bid_ask_bp: float = 50.0,
    n_paths: int = 15_000,
    n_steps: int = 252,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the market implied-volatility surface for one episode.

    Returns
    -------
    iv_true : (n_T, n_K) noiseless rBergomi implied volatilities
    iv_quote: (n_T, n_K) noisy quotes = iv_true + Gaussian(0, bid_ask_bp/1e4)
    """
    model = theta_to_model(theta_true, S0=1.0)
    iv_true, _ = model.implied_vol_surface(
        maturities=maturities, strikes=strikes,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    noise = rng.standard_normal(iv_true.shape) * (bid_ask_bp / 1e4)
    iv_quote = iv_true + noise
    return iv_true.astype(np.float32), iv_quote.astype(np.float32)


# --------------------------------------------------------------------- #
# True underlying path simulation
# --------------------------------------------------------------------- #

def simulate_underlying_path(
    theta_true: np.ndarray,
    T_horizon_days: int = 21,
    n_steps_per_day: int = 4,
    n_paths: int = 1,
    S0: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the "true" underlying path over the trading horizon.

    Parameters
    ----------
    theta_true : the true rBergomi parameter vector
    T_horizon_days : trading horizon in business days (default 21 = 1 month)
    n_steps_per_day : intraday discretisation for the Euler scheme
    n_paths : number of independent underlying paths (default 1 per episode)
    S0 : initial spot
    seed : RNG seed for reproducibility

    Returns
    -------
    day_grid : (T_horizon_days + 1,) grid of day indices [0, 1, ..., T_days]
    S_daily  : (T_horizon_days + 1, n_paths) daily-sampled spot paths
    """
    T_horizon_years = T_horizon_days / 252.0
    total_steps_per_year = 252 * n_steps_per_day
    N = int(round(total_steps_per_year * T_horizon_years))

    model = theta_to_model(theta_true, S0=S0)
    _, S = model.simulate(
        T=T_horizon_years,
        n_steps=total_steps_per_year,
        n_paths=n_paths,
        seed=seed,
        antithetic=(n_paths % 2 == 0 and n_paths >= 2),
    )
    # Sub-sample to daily observations
    day_idx = np.linspace(0, N, T_horizon_days + 1).astype(int)
    S_daily = S[day_idx, :]
    day_grid = np.arange(T_horizon_days + 1)
    return day_grid, S_daily.astype(np.float32)
