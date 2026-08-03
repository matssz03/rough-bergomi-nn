"""
Four trader agents for the arena (thesis Section 3.4.3).

Every agent implements the same three-method interface:

    calibrate(iv_quotes, maturities, strikes)   fit the agent's model
                                                to the market quotes
    price_surface(S, maturities, strikes)       $ price of every contract
                                                given a spot S and the
                                                calibrated parameters
    delta_surface(S, maturities, strikes)       sticky-strike BS delta,
                                                using the frozen implied
                                                vol of the agent's own
                                                calibrated surface

The sticky-strike delta convention (Derman & Kani, 1994) means every
agent's delta is the Black-Scholes delta evaluated at the agent's own
implied volatility, which is a standard, cheap and consistent choice
for hedge-comparison studies.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import norm

from neural_network.dataset import CFG as DATASET_CFG
from neural_network.model import RoughBergomiApproximator
from pricers.bates import bates_price_cos
from pricers.rough_bergomi import RoughBergomi
from utils.implied_vol import bs_call_price, implied_vol


# --------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------- #

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.calibration_time_s: float = 0.0
        self.calibrated: bool = False

    @abstractmethod
    def calibrate(self, iv_quotes: np.ndarray,
                  maturities: np.ndarray, strikes: np.ndarray) -> None: ...

    @abstractmethod
    def price_surface(self, S: float, maturities: np.ndarray,
                      strikes: np.ndarray) -> np.ndarray: ...

    def delta_surface(self, S: float, T_remaining: np.ndarray,
                      strikes: np.ndarray) -> np.ndarray:
        """
        Sticky-strike BS delta using the agent's calibrated implied-vol
        surface (evaluated at the original maturities).  Concrete
        subclasses must define ``self._iv_calibrated`` as an (n_T, n_K)
        array of implied volatilities on the calibration grid, where
        the row index matches the ORIGINAL maturity of each contract.
        """
        assert self.calibrated, f"agent {self.name} not calibrated"
        n_T, n_K = self._iv_calibrated.shape
        delta = np.zeros((n_T, n_K), dtype=np.float32)
        for i in range(n_T):
            T_i = float(T_remaining[i])
            for j in range(n_K):
                K = float(strikes[j])
                sigma = float(self._iv_calibrated[i, j])
                if T_i <= 1e-8 or sigma <= 0:
                    delta[i, j] = 1.0 if S > K else 0.0
                    continue
                d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T_i) / (sigma * np.sqrt(T_i))
                delta[i, j] = norm.cdf(d1)
        return delta


# --------------------------------------------------------------------- #
# Black-Scholes agent
# --------------------------------------------------------------------- #

class BSAgent(Agent):
    """
    Constant-volatility Black-Scholes agent.  Calibrates a single sigma
    by minimising the squared implied-vol error on the ATM column of
    the quotes, which is the standard practitioner shortcut.
    """
    def __init__(self):
        super().__init__("Black-Scholes")
        self.sigma: float = 0.20

    def calibrate(self, iv_quotes, maturities, strikes):
        t0 = time.time()
        # sigma_BS = average ATM implied vol over all maturities
        j_atm = int(np.argmin(np.abs(np.asarray(strikes) - 1.0)))
        self.sigma = float(np.nanmean(iv_quotes[:, j_atm]))
        n_T, n_K = iv_quotes.shape
        self._iv_calibrated = np.full((n_T, n_K), self.sigma, dtype=np.float32)
        self._maturities = np.asarray(maturities, dtype=np.float32)
        self.calibrated = True
        self.calibration_time_s = time.time() - t0

    def price_surface(self, S, maturities, strikes):
        n_T, n_K = len(maturities), len(strikes)
        prices = np.zeros((n_T, n_K), dtype=np.float32)
        for i, T in enumerate(maturities):
            if T <= 1e-8:
                for j, K in enumerate(strikes):
                    prices[i, j] = max(S - K, 0.0)
                continue
            for j, K in enumerate(strikes):
                prices[i, j] = bs_call_price(S, K, T, self.sigma)
        return prices


# --------------------------------------------------------------------- #
# Bates agent
# --------------------------------------------------------------------- #

class BatesAgent(Agent):
    """
    Full Bates (Heston + Merton jumps) calibrated by L-BFGS-B on the
    price surface with numerical gradients.  Eight parameters:
        (v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J).
    """
    # Parameter bounds
    LO = np.array([1e-4, 0.1, 1e-4, 0.05, -0.99, 0.0, -0.5, 1e-4])
    HI = np.array([0.30, 10.0, 0.30, 3.00, -0.05, 5.0,  0.5, 0.5])

    def __init__(self, r: float = 0.0):
        super().__init__("Bates")
        self.r = r
        self.params: np.ndarray = 0.5 * (self.LO + self.HI)

    def _price_surface(self, params, S, maturities, strikes):
        v0, kappa, theta, sigma_v, rho, lam, mu_J, sigma_J = params
        n_T, n_K = len(maturities), len(strikes)
        prices = np.zeros((n_T, n_K))
        for i, T in enumerate(maturities):
            prices[i, :] = bates_price_cos(
                S, np.asarray(strikes), T, self.r,
                v0, kappa, theta, sigma_v, rho,
                lam, mu_J, sigma_J,
            )
        return prices

    def calibrate(self, iv_quotes, maturities, strikes):
        t0 = time.time()
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)

        # Target: convert IV quotes to call prices via BS
        target_prices = np.zeros_like(iv_quotes)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                target_prices[i, j] = bs_call_price(1.0, float(K), float(T),
                                                    float(iv_quotes[i, j]))

        def obj(p):
            try:
                pr = self._price_surface(p, 1.0, maturities, strikes)
            except Exception:
                return 1e6
            return float(np.mean((pr - target_prices) ** 2))

        x0 = 0.5 * (self.LO + self.HI)
        bounds = list(zip(self.LO, self.HI))
        try:
            result = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 100})
            self.params = result.x.astype(np.float32)
        except Exception:
            self.params = x0.astype(np.float32)

        # Store calibrated IV surface for sticky-strike delta
        pr = self._price_surface(self.params, 1.0, maturities, strikes)
        iv = np.zeros_like(iv_quotes)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                iv[i, j] = implied_vol(float(pr[i, j]), 1.0, float(K), float(T))
        self._iv_calibrated = np.nan_to_num(iv, nan=float(np.nanmean(iv))).astype(np.float32)
        self._maturities = maturities.astype(np.float32)
        self.calibrated = True
        self.calibration_time_s = time.time() - t0

    def price_surface(self, S, maturities, strikes):
        return self._price_surface(self.params, S,
                                   np.asarray(maturities), np.asarray(strikes)
                                   ).astype(np.float32)


# --------------------------------------------------------------------- #
# Rough Bergomi agent (Monte Carlo baseline — slow)
# --------------------------------------------------------------------- #

class RoughBergomiAgent(Agent):
    """
    Full rBergomi agent, calibrated by Nelder-Mead on the Monte Carlo
    pricer (same optimiser as Section 4.3 MC baseline).  Slow but
    provides the honest reference against which the NN agent is judged.
    """
    def __init__(self, mc_seed: int = 42,
                 n_paths: int = 15_000, n_steps: int = 252,
                 max_iter: int = 300):
        super().__init__("rough Bergomi")
        self.mc_seed = mc_seed
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.max_iter = max_iter
        self.theta: np.ndarray = np.zeros(11, dtype=np.float32)

    @staticmethod
    def _bounds():
        cfg = DATASET_CFG
        return np.array(
            [[cfg.xi_lo, cfg.xi_hi]] * 8
            + [[cfg.eta_lo, cfg.eta_hi],
               [cfg.rho_lo, cfg.rho_hi],
               [cfg.H_lo,   cfg.H_hi]],
            dtype=np.float64,
        )

    def _price_surface_iv(self, theta, maturities, strikes, seed=None):
        model = RoughBergomi(
            xi0=theta[:8], xi0_maturities=np.array(DATASET_CFG.xi0_maturities),
            eta=float(theta[8]), H=float(theta[10]), rho=float(theta[9]),
            S0=1.0,
        )
        iv, _ = model.implied_vol_surface(
            maturities=maturities, strikes=strikes,
            n_steps=self.n_steps, n_paths=self.n_paths,
            seed=seed if seed is not None else self.mc_seed,
        )
        return iv

    def calibrate(self, iv_quotes, maturities, strikes):
        t0 = time.time()
        bounds = self._bounds()
        lo, hi = bounds[:, 0], bounds[:, 1]

        def obj(theta):
            theta = np.clip(theta, lo + 1e-6, hi - 1e-6)
            try:
                iv = self._price_surface_iv(theta, maturities, strikes,
                                            seed=self.mc_seed)
            except Exception:
                return 1e6
            return float(np.nanmean((iv - iv_quotes) ** 2))

        x0 = 0.5 * (lo + hi)
        result = minimize(obj, x0, method="Nelder-Mead",
                          options={"maxfev": self.max_iter, "adaptive": True,
                                   "xatol": 1e-4, "fatol": 1e-8})
        self.theta = np.clip(result.x, lo + 1e-6, hi - 1e-6).astype(np.float32)
        self._iv_calibrated = self._price_surface_iv(
            self.theta, maturities, strikes, seed=self.mc_seed,
        ).astype(np.float32)
        self._maturities = np.asarray(maturities, dtype=np.float32)
        self.calibrated = True
        self.calibration_time_s = time.time() - t0

    def price_surface(self, S, maturities, strikes):
        iv = self._iv_calibrated if S == 1.0 else self._price_surface_iv(
            self.theta, maturities, strikes, seed=self.mc_seed,
        )
        n_T, n_K = iv.shape
        prices = np.zeros_like(iv)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                prices[i, j] = bs_call_price(float(S), float(K), float(T),
                                             float(iv[i, j]))
        return prices.astype(np.float32)


# --------------------------------------------------------------------- #
# NN agent
# --------------------------------------------------------------------- #

class NNAgent(Agent):
    """
    Neural-network agent that reuses the trained MLP of Section 4.2 as a
    differentiable surrogate for rough Bergomi.  Calibration follows the
    L-BFGS + autograd procedure of Section 4.3.
    """
    def __init__(self, model_path: str, input_bounds: np.ndarray,
                 output_mean: np.ndarray, output_std: np.ndarray,
                 device: torch.device | str = "cpu", max_iter: int = 200):
        super().__init__("Neural network")
        self.device = torch.device(device)
        self.max_iter = max_iter

        self.model = RoughBergomiApproximator().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        self.input_bounds = input_bounds
        self.output_mean = output_mean
        self.output_std  = output_std
        self.theta: np.ndarray = np.zeros(11, dtype=np.float32)

    def calibrate(self, iv_quotes, maturities, strikes):
        t0 = time.time()
        lo = torch.tensor(self.input_bounds[:, 0], device=self.device,
                          dtype=torch.float32)
        hi = torch.tensor(self.input_bounds[:, 1], device=self.device,
                          dtype=torch.float32)
        mu = torch.tensor(self.output_mean, device=self.device,
                          dtype=torch.float32)
        sd = torch.tensor(self.output_std,  device=self.device,
                          dtype=torch.float32)
        tgt = torch.tensor(iv_quotes.reshape(-1), device=self.device,
                           dtype=torch.float32)
        u = torch.zeros(11, device=self.device, dtype=torch.float32,
                        requires_grad=True)
        optim = torch.optim.LBFGS([u], lr=1.0, max_iter=self.max_iter,
                                  tolerance_grad=1e-8, tolerance_change=1e-10,
                                  history_size=20, line_search_fn="strong_wolfe")

        def closure():
            optim.zero_grad()
            theta_scaled = torch.sigmoid(u)
            y_scaled = self.model(theta_scaled.unsqueeze(0)).squeeze(0)
            y = y_scaled * sd + mu
            loss = ((y - tgt) ** 2).mean()
            loss.backward()
            return loss

        optim.step(closure)

        with torch.no_grad():
            theta_scaled = torch.sigmoid(u)
            theta = (lo + (hi - lo) * theta_scaled).cpu().numpy()
            y = (self.model(theta_scaled.unsqueeze(0)).squeeze(0) * sd + mu)
            iv = y.cpu().numpy().reshape(iv_quotes.shape)

        self.theta = theta.astype(np.float32)
        self._iv_calibrated = iv.astype(np.float32)
        self._maturities = np.asarray(maturities, dtype=np.float32)
        self.calibrated = True
        self.calibration_time_s = time.time() - t0

    def price_surface(self, S, maturities, strikes):
        # Sticky-strike: use calibrated IV to price at any S via BS
        iv = self._iv_calibrated
        n_T, n_K = iv.shape
        prices = np.zeros_like(iv)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                prices[i, j] = bs_call_price(float(S), float(K), float(T),
                                             float(iv[i, j]))
        return prices.astype(np.float32)
