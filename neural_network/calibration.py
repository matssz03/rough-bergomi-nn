"""
Calibration accelerator (thesis Section 4.3).

Given a target Black-Scholes implied volatility surface, we find the
rough Bergomi parameter vector

    theta = (xi_0(t_1), ..., xi_0(t_8), eta, rho, H)

that best fits it in the sense of pointwise squared error on the 8 x 11
grid.  Two calibrators are compared:

    NN calibrator:   uses the trained MLP as a differentiable surrogate
                     and PyTorch L-BFGS on the *unconstrained* logit of
                     the training-domain box.  Analytical gradients via
                     autograd, converges in tens of milliseconds.

    MC calibrator:   uses the rough Bergomi Monte Carlo pricer of
                     Section 3.4.1 as an oracle.  Optimises with the
                     Nelder-Mead simplex, which is robust to Monte
                     Carlo sampling noise.  The Monte Carlo seed is
                     fixed for the duration of one calibration so that
                     the objective is deterministic in theta.

USAGE
-----
NN calibration on the full test set (fast, ~3 minutes):

    python -m neural_network.calibration --nn_only --n_targets 4000

Optional MC baseline on a smaller subset (slow, ~1-3 hours for 20):

    python -m neural_network.calibration --n_mc 20

Then produce Figure 7 and Table 7:

    python -m neural_network.calibration --report_only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from neural_network.dataset import (
    CFG as DATASET_CFG,
    RoughBergomiSurfaceDataset,
)
from neural_network.model import RoughBergomiApproximator
from pricers.rough_bergomi import RoughBergomi


# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #

@dataclass
class CalibrationConfig:
    data_dir:   str = "data"
    model_dir:  str = "results/4_2"
    output_dir: str = "results/4_3"

    n_targets:  int = 4_000            # number of NN calibrations
    n_mc:       int = 0                # number of MC baseline calibrations
    nn_max_iter: int = 200
    mc_max_iter: int = 500             # Nelder-Mead function evaluations cap
    mc_seed:    int = 20260718

    seed: int = 20260717
    device: str = "cpu"


# --------------------------------------------------------------------- #
# Parameter box, scaling and reparameterization
# --------------------------------------------------------------------- #

def _bounds() -> np.ndarray:
    cfg = DATASET_CFG
    return np.array(
        [[cfg.xi_lo, cfg.xi_hi]] * 8
        + [[cfg.eta_lo, cfg.eta_hi],
           [cfg.rho_lo, cfg.rho_hi],
           [cfg.H_lo,   cfg.H_hi]],
        dtype=np.float32,
    )


# --------------------------------------------------------------------- #
# NN calibration (PyTorch L-BFGS on logit-reparameterised box)
# --------------------------------------------------------------------- #

def calibrate_nn(
    target_iv: np.ndarray,
    model: torch.nn.Module,
    bounds: np.ndarray,
    output_mean: np.ndarray,
    output_std:  np.ndarray,
    max_iter:    int = 200,
    device:      torch.device = torch.device("cpu"),
    u_init:      np.ndarray | None = None,
) -> dict:
    """
    Return a dict with the calibrated theta, the final loss, and timing.

    The optimisation variable is an unconstrained u in R^11; the box
    constraints are enforced by
        theta = lo + (hi - lo) * sigmoid(u)
    which makes the problem smooth and unbounded.
    """
    lo = torch.tensor(bounds[:, 0], device=device, dtype=torch.float32)
    hi = torch.tensor(bounds[:, 1], device=device, dtype=torch.float32)
    mu = torch.tensor(output_mean,  device=device, dtype=torch.float32)
    sd = torch.tensor(output_std,   device=device, dtype=torch.float32)
    tgt = torch.tensor(target_iv.reshape(-1),
                       device=device, dtype=torch.float32)

    if u_init is None:
        u_init = np.zeros(11, dtype=np.float32)  # start at box centre
    u = torch.tensor(u_init, device=device, dtype=torch.float32,
                     requires_grad=True)

    optim = torch.optim.LBFGS(
        [u], lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-8, tolerance_change=1e-10,
        history_size=20, line_search_fn="strong_wolfe",
    )

    n_calls = [0]

    def closure():
        optim.zero_grad()
        theta = lo + (hi - lo) * torch.sigmoid(u)
        # Rescale to unit hypercube (network's expected input scaling)
        x = (theta - lo) / (hi - lo)
        y_scaled = model(x.unsqueeze(0)).squeeze(0)
        y = y_scaled * sd + mu           # unscale outputs to raw IV
        loss = ((y - tgt) ** 2).mean()
        loss.backward()
        n_calls[0] += 1
        return loss

    t0 = time.time()
    final_loss = optim.step(closure).item()
    elapsed = time.time() - t0

    with torch.no_grad():
        theta_final = (lo + (hi - lo) * torch.sigmoid(u)).cpu().numpy()
        y_final = ((model((torch.sigmoid(u)).unsqueeze(0)).squeeze(0) * sd + mu)
                   .cpu().numpy().reshape(target_iv.shape))

    return {
        "theta_hat": theta_final.astype(np.float32),
        "iv_hat":    y_final.astype(np.float32),
        "final_loss": float(final_loss),
        "time_s":    float(elapsed),
        "n_calls":   int(n_calls[0]),
    }


# --------------------------------------------------------------------- #
# MC calibration (Nelder-Mead, robust to Monte Carlo noise)
# --------------------------------------------------------------------- #

def calibrate_mc(
    target_iv: np.ndarray,
    bounds:    np.ndarray,
    max_iter:  int = 500,
    mc_seed:   int = 20260718,
    n_paths:   int = 15_000,
    n_steps:   int = 252,
    x0:        np.ndarray | None = None,
) -> dict:
    """
    Baseline Monte Carlo calibration.

    Nelder-Mead is chosen over gradient-based methods because it does
    not rely on differentiability and tolerates the residual Monte Carlo
    sampling noise on the objective.  Reproducibility of that noise
    within one calibration is ensured by fixing the pricer seed.
    """
    xi0_mats   = np.array(DATASET_CFG.xi0_maturities)
    maturities = np.array(DATASET_CFG.maturities)
    strikes    = np.array(DATASET_CFG.strikes)
    lo = bounds[:, 0]; hi = bounds[:, 1]

    n_calls = [0]

    def project(theta):
        return np.clip(theta, lo + 1e-6, hi - 1e-6)

    def objective(theta):
        theta = project(theta)
        xi0 = theta[:8].astype(np.float64)
        eta = float(theta[8]); rho = float(theta[9]); H = float(theta[10])
        try:
            rb = RoughBergomi(xi0=xi0, xi0_maturities=xi0_mats,
                              eta=eta, H=H, rho=rho, S0=1.0)
            iv, _ = rb.implied_vol_surface(
                maturities=maturities, strikes=strikes,
                n_steps=n_steps, n_paths=n_paths,
                seed=mc_seed,
            )
        except Exception:
            return 1e6
        n_calls[0] += 1
        return float(np.nanmean((iv - target_iv) ** 2))

    if x0 is None:
        x0 = 0.5 * (lo + hi).astype(np.float64)

    t0 = time.time()
    result = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={
            "xatol": 1e-4, "fatol": 1e-8,
            "maxfev": max_iter, "adaptive": True,
        },
    )
    elapsed = time.time() - t0

    theta_hat = project(result.x)
    # Evaluate final IV surface once more, deterministically
    xi0 = theta_hat[:8]; eta = float(theta_hat[8])
    rho = float(theta_hat[9]); H = float(theta_hat[10])
    rb = RoughBergomi(xi0=xi0, xi0_maturities=xi0_mats,
                      eta=eta, H=H, rho=rho, S0=1.0)
    iv_hat, _ = rb.implied_vol_surface(
        maturities=maturities, strikes=strikes,
        n_steps=n_steps, n_paths=n_paths, seed=mc_seed,
    )

    return {
        "theta_hat":  theta_hat.astype(np.float32),
        "iv_hat":     iv_hat.astype(np.float32),
        "final_loss": float(result.fun),
        "time_s":     float(elapsed),
        "n_calls":    int(n_calls[0]),
        "converged":  bool(result.success),
    }


# --------------------------------------------------------------------- #
# Run experiment
# --------------------------------------------------------------------- #

def run(cfg: CalibrationConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # ---------------- Load data & model ----------------
    train_full = RoughBergomiSurfaceDataset(
        os.path.join(cfg.data_dir, "data_train.npz"),
    )
    test = RoughBergomiSurfaceDataset(
        os.path.join(cfg.data_dir, "data_test.npz"),
        input_bounds=train_full.input_bounds,
        output_mean=train_full.output_mean,
        output_std=train_full.output_std,
    )

    model = RoughBergomiApproximator().to(device)
    model.load_state_dict(torch.load(
        os.path.join(cfg.model_dir, "model.pt"), map_location=device))
    model.eval()

    bounds = _bounds()
    n_test = len(test)
    n_targets_nn = min(cfg.n_targets, n_test)
    print(f"Test set size: {n_test}. NN calibrations: {n_targets_nn}. "
          f"MC calibrations: {cfg.n_mc}.")

    # ---------------- NN calibration ----------------
    if n_targets_nn > 0:
        print(f"Running NN calibration on {n_targets_nn} targets...")
        nn_results = {
            "theta_true":  np.zeros((n_targets_nn, 11), dtype=np.float32),
            "theta_hat":   np.zeros((n_targets_nn, 11), dtype=np.float32),
            "iv_target":   np.zeros((n_targets_nn, 8, 11), dtype=np.float32),
            "iv_hat":      np.zeros((n_targets_nn, 8, 11), dtype=np.float32),
            "final_loss":  np.zeros(n_targets_nn, dtype=np.float32),
            "time_s":      np.zeros(n_targets_nn, dtype=np.float32),
            "n_calls":     np.zeros(n_targets_nn, dtype=np.int32),
        }
        t0 = time.time()
        for i in range(n_targets_nn):
            target = test.iv[i].reshape(8, 11)
            r = calibrate_nn(
                target_iv=target,
                model=model,
                bounds=bounds,
                output_mean=test.output_mean,
                output_std=test.output_std,
                max_iter=cfg.nn_max_iter,
                device=device,
            )
            nn_results["theta_true"][i] = test.theta[i]
            nn_results["theta_hat"][i]  = r["theta_hat"]
            nn_results["iv_target"][i]  = target
            nn_results["iv_hat"][i]     = r["iv_hat"]
            nn_results["final_loss"][i] = r["final_loss"]
            nn_results["time_s"][i]     = r["time_s"]
            nn_results["n_calls"][i]    = r["n_calls"]
            if (i + 1) % 200 == 0 or i == n_targets_nn - 1:
                dt = time.time() - t0
                print(f"  NN {i+1:>5}/{n_targets_nn}   "
                      f"elapsed {dt:6.1f}s   mean_loss {np.mean(nn_results['final_loss'][:i+1]):.3e}",
                      flush=True)
        np.savez_compressed(
            os.path.join(cfg.output_dir, "nn_calibrations.npz"),
            **nn_results,
        )
        print(f"NN calibration done in {(time.time()-t0)/60:.2f} min. "
              f"Mean time per calibration: {nn_results['time_s'].mean()*1000:.2f} ms")

    # ---------------- MC calibration ----------------
    if cfg.n_mc > 0:
        # Use the same first N_mc test surfaces so we can compare pointwise
        n_mc = min(cfg.n_mc, n_test)
        print(f"Running MC baseline calibration on {n_mc} targets "
              f"(this can take hours)...")
        mc_results = {
            "theta_true":  np.zeros((n_mc, 11), dtype=np.float32),
            "theta_hat":   np.zeros((n_mc, 11), dtype=np.float32),
            "iv_target":   np.zeros((n_mc, 8, 11), dtype=np.float32),
            "iv_hat":      np.zeros((n_mc, 8, 11), dtype=np.float32),
            "final_loss":  np.zeros(n_mc, dtype=np.float32),
            "time_s":      np.zeros(n_mc, dtype=np.float32),
            "n_calls":     np.zeros(n_mc, dtype=np.int32),
            "converged":   np.zeros(n_mc, dtype=bool),
        }
        t0 = time.time()
        for i in range(n_mc):
            target = test.iv[i].reshape(8, 11)
            r = calibrate_mc(
                target_iv=target,
                bounds=bounds,
                max_iter=cfg.mc_max_iter,
                mc_seed=cfg.mc_seed + i,
            )
            mc_results["theta_true"][i] = test.theta[i]
            mc_results["theta_hat"][i]  = r["theta_hat"]
            mc_results["iv_target"][i]  = target
            mc_results["iv_hat"][i]     = r["iv_hat"]
            mc_results["final_loss"][i] = r["final_loss"]
            mc_results["time_s"][i]     = r["time_s"]
            mc_results["n_calls"][i]    = r["n_calls"]
            mc_results["converged"][i]  = r["converged"]
            dt = time.time() - t0
            per = dt / (i + 1)
            eta = per * (n_mc - i - 1)
            print(f"  MC {i+1:>3}/{n_mc}   time {r['time_s']:6.1f}s   "
                  f"n_calls {r['n_calls']:4d}   loss {r['final_loss']:.3e}   "
                  f"converged {r['converged']}   ETA {eta/60:.1f} min",
                  flush=True)
            # Save after every MC calibration (they're expensive)
            np.savez_compressed(
                os.path.join(cfg.output_dir, "mc_calibrations.npz"),
                **{k: v[:i+1] for k, v in mc_results.items()},
            )
        print(f"MC calibration done in {(time.time()-t0)/60:.2f} min.")


# --------------------------------------------------------------------- #
# Reporting: Figure 7 + Table 7
# --------------------------------------------------------------------- #

def report(cfg: CalibrationConfig) -> dict:
    """Generate Figure 7 and Table 7 from saved calibration results."""
    nn_path = os.path.join(cfg.output_dir, "nn_calibrations.npz")
    mc_path = os.path.join(cfg.output_dir, "mc_calibrations.npz")
    if not os.path.exists(nn_path):
        raise FileNotFoundError(f"missing {nn_path}, run with --n_targets N first")

    nn = np.load(nn_path)
    mc = np.load(mc_path) if os.path.exists(mc_path) else None

    # ---------------- Errors ----------------
    def _errors(res):
        err = res["iv_hat"] - res["iv_target"]
        abs_err = np.abs(err)
        rel_err = np.abs(err / res["iv_target"])
        return abs_err, rel_err

    nn_abs, nn_rel = _errors(nn)
    if mc is not None:
        mc_abs, mc_rel = _errors(mc)

    # ---------------- Figure 7 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel a: pooled absolute error CDF (in bp)
    ax = axes[0]
    abs_bp = nn_abs.reshape(-1) * 1e4
    xs = np.sort(abs_bp)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, lw=1.5,
            label=f"NN calibration  (N = {len(nn['final_loss'])})")
    if mc is not None:
        xs_mc = np.sort(mc_abs.reshape(-1) * 1e4)
        ys_mc = np.arange(1, len(xs_mc) + 1) / len(xs_mc)
        ax.plot(xs_mc, ys_mc, lw=1.5, linestyle="--",
                label=f"MC calibration  (N = {len(mc['final_loss'])})")
    ax.set_xlabel("Absolute pointwise error  (bp)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(a) Absolute pointwise error")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")

    # Panel b: pooled relative error CDF (%)
    ax = axes[1]
    rel_pct = nn_rel.reshape(-1) * 100
    xs = np.sort(rel_pct)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, lw=1.5,
            label=f"NN calibration  (N = {len(nn['final_loss'])})")
    if mc is not None:
        xs_mc = np.sort(mc_rel.reshape(-1) * 100)
        ys_mc = np.arange(1, len(xs_mc) + 1) / len(xs_mc)
        ax.plot(xs_mc, ys_mc, lw=1.5, linestyle="--",
                label=f"MC calibration  (N = {len(mc['final_loss'])})")
    ax.set_xlabel("Relative pointwise error  (%)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(b) Relative pointwise error")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")

    fig.suptitle(
        "Figure 7 — Calibration quality on the test set: empirical CDFs of "
        "pointwise post-calibration errors",
        y=1.02, fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "figure7_cdfs.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.savefig(os.path.join(cfg.output_dir, "figure7_cdfs.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close()

    # ---------------- Table 7 ----------------
    def _stats(times, calls):
        return {
            "mean_s":   float(np.mean(times)),
            "median_s": float(np.median(times)),
            "std_s":    float(np.std(times, ddof=1)),
            "n":        int(len(times)),
            "mean_calls": float(np.mean(calls)),
        }

    table7 = {
        "nn": _stats(nn["time_s"], nn["n_calls"]),
        "nn_mean_abs_bp": float(nn_abs.mean() * 1e4),
        "nn_mean_rel_pct": float(nn_rel.mean() * 100),
        "nn_p50_abs_bp":  float(np.percentile(nn_abs.reshape(-1), 50) * 1e4),
        "nn_p95_abs_bp":  float(np.percentile(nn_abs.reshape(-1), 95) * 1e4),
    }
    if mc is not None:
        table7["mc"] = _stats(mc["time_s"], mc["n_calls"])
        table7["mc_mean_abs_bp"] = float(mc_abs.mean() * 1e4)
        table7["mc_mean_rel_pct"] = float(mc_rel.mean() * 100)
        table7["mc_p50_abs_bp"]  = float(np.percentile(mc_abs.reshape(-1), 50) * 1e4)
        table7["mc_p95_abs_bp"]  = float(np.percentile(mc_abs.reshape(-1), 95) * 1e4)
        table7["speedup"] = table7["mc"]["mean_s"] / table7["nn"]["mean_s"]

    with open(os.path.join(cfg.output_dir, "table7.json"), "w") as f:
        json.dump(table7, f, indent=2)

    print("=== Table 7 ===")
    print(json.dumps(table7, indent=2))
    print(f"\nFigure 7 saved to {cfg.output_dir}/figure7_cdfs.png")
    return table7


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--model_dir",  type=str, default="results/4_2")
    p.add_argument("--output_dir", type=str, default="results/4_3")
    p.add_argument("--n_targets",  type=int, default=4000,
                   help="number of NN calibrations (up to test-set size)")
    p.add_argument("--n_mc",       type=int, default=0,
                   help="number of MC baseline calibrations (0 to skip)")
    p.add_argument("--nn_max_iter", type=int, default=200)
    p.add_argument("--mc_max_iter", type=int, default=500)
    p.add_argument("--nn_only",    action="store_true",
                   help="alias for --n_mc 0")
    p.add_argument("--report_only", action="store_true",
                   help="skip calibration, only generate Figure 7 and Table 7 from saved data")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=20260717)
    return p


def _main():
    args = _build_parser().parse_args()
    if args.nn_only:
        args.n_mc = 0
    cfg = CalibrationConfig(
        data_dir=args.data_dir, model_dir=args.model_dir,
        output_dir=args.output_dir,
        n_targets=args.n_targets, n_mc=args.n_mc,
        nn_max_iter=args.nn_max_iter, mc_max_iter=args.mc_max_iter,
        device=args.device, seed=args.seed,
    )

    if args.report_only:
        report(cfg)
        return

    run(cfg)
    report(cfg)


if __name__ == "__main__":
    _main()
