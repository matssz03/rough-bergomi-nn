"""
Recompute corrected Sharpe ratios from an existing arena_results.npz,
without re-running the trader arena simulation.

Bug fixed: the original _agent_stats() function in trader_arena.py used
    sharpe = mean_pnl / std_pnl * sqrt(252 / N_episodes)
where N_episodes is the SAMPLE SIZE (500 or 100), not the EPISODE HORIZON
in trading days (504 = 2 years, fixed for every agent). The correct
annualisation factor is sqrt(252 / horizon_days), a constant that must be
identical across every agent regardless of how many episodes were run.

This script reloads the raw per-episode P&L arrays already saved in
arena_results.npz (unaffected by the bug — only the Sharpe field was wrong)
and recomputes every statistic with the corrected formula. It also runs a
two-sample significance test (Lo, 2002) on the Sharpe-ratio gap between any
two agents, which is essential here because the rough Bergomi baseline has
a much smaller sample (100 episodes) than the fast agents (500 episodes).

Usage:
    python recompute_sharpe.py --arena_results results/4_4/arena_results.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------- #
# Corrected statistics
# --------------------------------------------------------------------- #

def agent_stats_fixed(pnl: np.ndarray, horizon_days: int = 504,
                      days_per_year: int = 252) -> dict:
    """
    Corrected version of _agent_stats(): the annualisation factor depends
    only on the fixed episode horizon (in trading days), never on the
    number of episodes in the sample.
    """
    mean_pnl = float(np.mean(pnl))
    std_pnl  = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0

    annualisation_factor = np.sqrt(days_per_year / horizon_days)
    sharpe = float(mean_pnl / std_pnl * annualisation_factor) \
             if std_pnl > 1e-12 else float("nan")

    # Per-episode (non-annualised) Sharpe, used for the significance test
    sharpe_raw = mean_pnl / std_pnl if std_pnl > 1e-12 else float("nan")

    cummax = np.maximum.accumulate(np.cumsum(pnl))
    dd = float(np.max(cummax - np.cumsum(pnl)))
    hit_rate = float(np.mean(pnl > 0))

    return {
        "n_episodes": int(len(pnl)),
        "mean_pnl": mean_pnl,
        "median_pnl": float(np.median(pnl)),
        "std_pnl": std_pnl,
        "sharpe_annualised": sharpe,
        "sharpe_raw": sharpe_raw,
        "max_drawdown": dd,
        "hit_rate": hit_rate,
    }


def lo_2002_se(sharpe_raw: float, n: int) -> float:
    """
    Asymptotic standard error of a (non-annualised) sample Sharpe ratio,
    under the i.i.d. returns assumption of Lo (2002):
        SE(SR) ~ sqrt((1 + SR^2/2) / N)
    """
    return float(np.sqrt((1 + sharpe_raw**2 / 2) / n))


def compare_agents(stats_a: dict, name_a: str, stats_b: dict, name_b: str) -> None:
    """Two-sample significance test on the Sharpe-ratio gap between two agents."""
    se_a = lo_2002_se(stats_a["sharpe_raw"], stats_a["n_episodes"])
    se_b = lo_2002_se(stats_b["sharpe_raw"], stats_b["n_episodes"])
    diff = stats_a["sharpe_raw"] - stats_b["sharpe_raw"]
    se_diff = np.sqrt(se_a**2 + se_b**2)
    t_stat = diff / se_diff if se_diff > 1e-12 else float("nan")
    significant = abs(t_stat) > 1.96

    print(f"\n  {name_a} vs {name_b}:")
    print(f"    Sharpe (per-episode, non-annualised): "
          f"{stats_a['sharpe_raw']:+.4f} (N={stats_a['n_episodes']}) vs "
          f"{stats_b['sharpe_raw']:+.4f} (N={stats_b['n_episodes']})")
    print(f"    SE_A={se_a:.4f}  SE_B={se_b:.4f}  "
          f"diff={diff:+.4f}  SE(diff)={se_diff:.4f}")
    print(f"    t-stat = {t_stat:+.3f}   "
          f"{'SIGNIFICANT at 95%' if significant else 'NOT significant at 95%'}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--arena_results", type=str,
                   default="results/4_4/arena_results.npz",
                   help="Path to the existing arena_results.npz "
                        "(already produced by a prior run of trader_arena.py; "
                        "this script does NOT re-run the simulation).")
    p.add_argument("--horizon_days", type=int, default=504,
                   help="Trading days per episode (2 years = 504 by default).")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to save the corrected stats as JSON "
                        "(defaults to table8_fixed.json next to the input file).")
    args = p.parse_args()

    d = np.load(args.arena_results)
    agent_keys = [k for k in d.files if k.startswith("pnl_")]
    agent_names = [k[len("pnl_"):] for k in agent_keys]

    if not agent_names:
        raise SystemExit(f"No 'pnl_<agent>' arrays found in {args.arena_results}")

    print(f"Loaded {args.arena_results}")
    print(f"Agents found: {agent_names}")
    print(f"Using fixed episode horizon = {args.horizon_days} trading days "
          f"(annualisation factor = sqrt(252/{args.horizon_days}) = "
          f"{np.sqrt(252/args.horizon_days):.5f}, identical for every agent)\n")

    all_stats = {}
    print("=== Corrected statistics ===")
    print(f"{'Agent':<16} {'Mean P&L':>10} {'Std P&L':>9} "
          f"{'Sharpe (old bug)':>17} {'Sharpe (fixed)':>15} {'Max DD':>8}")
    for name in agent_names:
        pnl = d[f"pnl_{name}"]
        stats = agent_stats_fixed(pnl, horizon_days=args.horizon_days)
        all_stats[name] = stats

        # Show what the OLD buggy formula would have given, for comparison
        old_buggy_sharpe = (stats["mean_pnl"] / stats["std_pnl"]
                            * np.sqrt(252 / max(len(pnl), 1))) \
                            if stats["std_pnl"] > 1e-12 else float("nan")

        print(f"{name:<16} {stats['mean_pnl']:>+10.4f} {stats['std_pnl']:>9.4f} "
              f"{old_buggy_sharpe:>+17.4f} {stats['sharpe_annualised']:>+15.4f} "
              f"{stats['max_drawdown']:>8.2f}")

    # Significance tests: neural network vs every other agent
    if "Neural network" in all_stats:
        print("\n=== Significance tests (Lo, 2002) — Neural network vs each agent ===")
        for name in agent_names:
            if name != "Neural network":
                compare_agents(all_stats["Neural network"], "Neural network",
                              all_stats[name], name)

    # Save corrected stats to JSON
    out_path = args.output_json or os.path.join(
        os.path.dirname(args.arena_results) or ".", "table8_fixed.json")
    # Drop the internal-only 'sharpe_raw' field from the saved JSON to keep
    # the same schema as the original table8.json
    to_save = {name: {k: v for k, v in stats.items() if k != "sharpe_raw"}
               for name, stats in all_stats.items()}
    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nSaved corrected statistics to {out_path}")


if __name__ == "__main__":
    main()
