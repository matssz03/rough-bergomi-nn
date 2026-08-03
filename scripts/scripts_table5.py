"""
Section 4.1.2 --- Table 5.

Monte Carlo standard error of the rough Bergomi at-the-money one-year
implied volatility, as a function of the number of paths
N in {5,000; 15,000; 30,000}, at the baseline parameter combination.

The internal SE is the vega-adjusted price SE returned by the pricer on
a single run.  The empirical SE is obtained by running R independent
replications and computing the sample standard deviation of the R
implied-vol estimates, which is the ground-truth benchmark against
which the internal SE is validated.
"""
from __future__ import annotations

import json
import time
import os 
import sys 

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pricers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pricers'))
sys.path.insert(0, 'pricers')

from rough_bergomi import RoughBergomi


# --------------------------------------------------------------------- #
# Baseline setting (matches Section 4.4 / Chapter 4 baseline)
# --------------------------------------------------------------------- #

XI0_MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
XI0_FLAT = np.full(8, 0.04)          # sqrt(0.04) = 20% ATM vol level
ETA_BASE = 1.9
H_BASE   = 0.10
RHO_BASE = -0.70
S0_BASE  = 1.0

# ATM 1-year contract
MATS_TABLE5    = np.array([1.0])
STRIKES_TABLE5 = np.array([1.0])

# Path counts and simulation grid
N_STEPS = 100
N_VALUES = [5_000, 15_000, 30_000]
N_REPLICATIONS = 20


# --------------------------------------------------------------------- #
# Table 5 computation
# --------------------------------------------------------------------- #

def compute_table5(
    n_values: list[int] = N_VALUES,
    n_replications: int = N_REPLICATIONS,
    n_steps: int = N_STEPS,
    verbose: bool = True,
) -> dict:
    model = RoughBergomi(
        xi0=XI0_FLAT, xi0_maturities=XI0_MATS,
        eta=ETA_BASE, H=H_BASE, rho=RHO_BASE,
        S0=S0_BASE, r=0.0, q=0.0,
    )

    results = {}
    for N in n_values:
        internal_ses = []
        replicated_ivs = []
        times = []
        for rep in range(n_replications):
            seed = 1_000 * rep + N
            t0 = time.time()
            iv, se_iv = model.implied_vol_surface(
                maturities=MATS_TABLE5, strikes=STRIKES_TABLE5,
                n_steps=n_steps, n_paths=N, seed=seed,
            )
            times.append(time.time() - t0)
            internal_ses.append(float(se_iv[0, 0]))
            replicated_ivs.append(float(iv[0, 0]))

        internal_ses = np.array(internal_ses)
        replicated_ivs = np.array(replicated_ivs)
        empirical_se = float(replicated_ivs.std(ddof=1))
        internal_se_mean = float(internal_ses.mean())
        mean_time = float(np.mean(times))
        mean_iv = float(replicated_ivs.mean())

        results[str(N)] = {
            "N": int(N),
            "mean_iv": mean_iv,
            "internal_se": internal_se_mean,
            "empirical_se": empirical_se,
            "mean_time_s": mean_time,
            "n_replications": int(n_replications),
        }

        if verbose:
            print(
                f"N = {N:>6}   mean IV = {mean_iv:.4f}   "
                f"internal SE = {internal_se_mean*1e4:6.2f} bp   "
                f"empirical SE = {empirical_se*1e4:6.2f} bp   "
                f"time = {mean_time:.2f}s / surface"
            )

    return results


# --------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------- #

def _line(*cells, widths=(8, 10, 14, 14, 14)):
    return " | ".join(f"{str(c):>{w}}" for c, w in zip(cells, widths))


def format_table5_text(results: dict) -> str:
    """Return a monospaced text table ready to drop into the write-up."""
    lines = []
    header = _line("N", "Mean IV", "Internal SE", "Empirical SE", "Time (s)")
    sep = "-" * len(header)
    lines.append("Table 5 --- Monte Carlo standard error of the rough Bergomi")
    lines.append("            at-the-money one-year implied volatility")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for key in sorted(results, key=lambda k: int(k)):
        r = results[key]
        lines.append(_line(
            f"{r['N']:,}".replace(",", " "),
            f"{r['mean_iv']:.4f}",
            f"{r['internal_se']*1e4:.2f} bp",
            f"{r['empirical_se']*1e4:.2f} bp",
            f"{r['mean_time_s']:.2f}",
        ))
    lines.append(sep)
    lines.append(
        f"Baseline: xi0 flat = 0.04, eta = {ETA_BASE}, H = {H_BASE}, "
        f"rho = {RHO_BASE}, S0 = {S0_BASE}"
    )
    lines.append(
        f"{N_REPLICATIONS} independent replications per row; "
        f"{N_STEPS} time steps per year; antithetic variates."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print("Computing Table 5 ...")
    print()
    results = compute_table5()

    print()
    print(format_table5_text(results))

    with open("results_table5.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Saved results_table5.json")
