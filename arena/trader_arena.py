"""
Trader arena (thesis Section 4.4).

For each of N_EPISODES independent episodes:
    1. Sample theta_true ~ Uniform(training domain)
    2. Generate market quotes: true rBergomi IV surface + bid-ask noise
    3. Each agent calibrates on the noisy quotes
    4. Trade selection at t=0: agent trades every contract on which
       |P_agent - P_market| > edge (long if under-priced, short if over)
    5. Simulate the true underlying path S_t under theta_true for the
       full 2-year horizon (max maturity on the grid)
    6. Daily delta-hedge each contract with the agent's sticky-strike
       delta, from t=0 until that contract's own maturity
    7. Settle each contract at the intrinsic payoff at its maturity
    8. Record per-contract and per-episode P&L

This "hold-to-maturity" protocol exposes model mis-specification through
the hedge P&L: an agent with a poor implied-vol surface will accumulate
hedge losses as the underlying moves, even if it captures the initial
bid-ask spread on quotes.

Deliverables (Section 4.4):
    Figure 8   cumulative P&L per agent + per-batch mean
    Figure 9   representative underlying trajectory + ITM zone
    Figure 10  ECDFs of episode terminal P&L per agent
    Table 8   agent-by-agent statistics (mean, Sharpe, drawdown, hit rate)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arena.agents import BSAgent, BatesAgent, RoughBergomiAgent, NNAgent
from arena.market import (
    sample_theta_true,
    generate_market_quotes,
    simulate_underlying_path,
)
from neural_network.dataset import (
    CFG as DATASET_CFG,
    RoughBergomiSurfaceDataset,
)
from utils.implied_vol import bs_call_price


# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #

@dataclass
class ArenaConfig:
    data_dir:   str = "data"
    model_dir:  str = "results/4_2"
    output_dir: str = "results/4_4"

    n_episodes_fast: int = 500       # BS + Bates + NN
    n_episodes_rb:   int = 100       # rBergomi (slow MC calibration)
    include_rb:      bool = True

    # Market
    bid_ask_bp:    float = 50.0      # idealised bid-ask on IV, in bp
    edge_bp:       float = 50.0      # trade threshold on IV (bp) per contract

    # Simulation grid (max maturity on the grid drives the horizon)
    days_per_year: int = 252
    steps_per_day: int = 2           # intraday resolution for simulation

    # Frictions (0 for Section 4.4 baseline; activated in Section 4.5)
    trade_cost_bp: float = 0.0
    hedge_cost_bp: float = 0.0

    # Reproducibility
    seed: int = 20260719

    # Runtime
    device: str = "cpu"
    verbose: bool = True


# --------------------------------------------------------------------- #
# Trade selection
# --------------------------------------------------------------------- #

def _select_trades(
    price_agent: np.ndarray,
    price_market: np.ndarray,
    edge_bp: float,
) -> np.ndarray:
    """
    q = +1 (long) if agent thinks under-priced by more than the edge,
    -1 (short) if over-priced, 0 otherwise.  Edge in basis points of IV
    is converted to a dollar threshold via a rough vega proxy that
    scales with the market price of each contract.
    """
    edge_frac = edge_bp / 1e4
    dollar_edge = edge_frac * np.abs(price_market) * 5.0
    mispricing = price_agent - price_market
    q = np.zeros_like(price_market, dtype=np.float32)
    q[mispricing >  dollar_edge] = 1.0
    q[mispricing < -dollar_edge] = -1.0
    return q


# --------------------------------------------------------------------- #
# Hold-to-maturity hedge + P&L
# --------------------------------------------------------------------- #

def _hedge_and_pnl(
    q: np.ndarray,
    S_path: np.ndarray,
    price_market_0: np.ndarray,
    delta_fn,
    maturities: np.ndarray,
    strikes: np.ndarray,
    days_per_year: int = 252,
    trade_cost_bp: float = 0.0,
    hedge_cost_bp: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold each contract until its own maturity.  Options settle at
    intrinsic payoff; the hedge is delta-rebalanced daily with the
    agent's sticky-strike delta.  Returns
        pnl_surface : (n_T, n_K) total P&L per contract
        option_pnl  : (n_T, n_K) option leg P&L
        hedge_pnl   : (n_T, n_K) hedge leg P&L (net of hedge frictions)
    """
    n_T, n_K = q.shape
    n_days_total = len(S_path) - 1

    maturities = np.asarray(maturities, dtype=np.float32)
    strikes    = np.asarray(strikes,    dtype=np.float32)

    days_to_expiry = np.minimum(
        np.round(maturities * days_per_year).astype(int),
        n_days_total,
    )  # (n_T,)

    # ---- Option leg: settle at intrinsic value at each contract's expiry ----
    option_pnl = np.zeros((n_T, n_K), dtype=np.float32)
    for i in range(n_T):
        S_at_expiry = float(S_path[days_to_expiry[i]])
        payoff = np.maximum(S_at_expiry - strikes, 0.0)
        option_pnl[i, :] = q[i, :] * (payoff - price_market_0[i, :])

    # ---- Hedge leg: daily rebalance until expiry, unwind at expiry ----
    hedge_pnl = np.zeros((n_T, n_K), dtype=np.float32)
    hedge_cost_total = np.zeros((n_T, n_K), dtype=np.float32)
    h_prev = np.zeros((n_T, n_K), dtype=np.float32)

    for t in range(n_days_total):
        T_remaining = np.maximum(maturities - t / days_per_year, 1e-8)
        delta_t = delta_fn(float(S_path[t]), T_remaining, strikes)  # (n_T, n_K)
        active = (t < days_to_expiry).astype(np.float32)            # (n_T,)
        h_t = -q * delta_t * active[:, None]
        d_h = h_t - h_prev
        hedge_cost_total += (hedge_cost_bp / 1e4) * np.abs(d_h) * float(S_path[t])
        hedge_pnl += h_t * (S_path[t + 1] - S_path[t])
        h_prev = h_t

    hedge_pnl -= hedge_cost_total

    # Initial trade friction on the option leg
    trade_cost = (trade_cost_bp / 1e4) * np.abs(q) * np.abs(price_market_0)
    option_pnl -= trade_cost

    pnl_surface = option_pnl + hedge_pnl
    return pnl_surface, option_pnl, hedge_pnl


# --------------------------------------------------------------------- #
# Single-episode simulation
# --------------------------------------------------------------------- #

def run_episode(
    episode_id: int,
    agents: list,
    cfg: ArenaConfig,
    maturities: np.ndarray,
    strikes: np.ndarray,
) -> dict:
    """
    Simulate one episode.  Underlying is simulated to the maximum grid
    maturity so that every contract can be held to its own expiry.
    """
    seed = cfg.seed + episode_id
    rng = np.random.default_rng(seed)

    theta_true = sample_theta_true(rng)

    iv_true, iv_quote = generate_market_quotes(
        theta_true, maturities, strikes,
        bid_ask_bp=cfg.bid_ask_bp, seed=seed,
    )

    # Convert IV surfaces to $ prices (S0 = 1)
    price_true_0 = np.zeros_like(iv_true)
    price_market_0 = np.zeros_like(iv_quote)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price_true_0[i, j]   = bs_call_price(1.0, float(K), float(T),
                                                 float(iv_true[i, j]))
            price_market_0[i, j] = bs_call_price(1.0, float(K), float(T),
                                                 float(iv_quote[i, j]))

    # Underlying path over the FULL horizon (max maturity on the grid)
    max_horizon_days = int(np.ceil(float(np.max(maturities)) * cfg.days_per_year))
    _, S_path = simulate_underlying_path(
        theta_true,
        T_horizon_days=max_horizon_days,
        n_steps_per_day=cfg.steps_per_day,
        n_paths=1, S0=1.0, seed=seed + 100_000,
    )
    S_path = S_path[:, 0]

    results = {"theta_true": theta_true, "S_path": S_path,
               "iv_true": iv_true, "iv_quote": iv_quote,
               "agents": {}}

    for agent in agents:
        agent.calibrate(iv_quote, maturities, strikes)
        price_agent_0 = agent.price_surface(1.0, maturities, strikes)
        q = _select_trades(price_agent_0, price_market_0, edge_bp=cfg.edge_bp)

        pnl_surface, option_pnl, hedge_pnl = _hedge_and_pnl(
            q=q,
            S_path=S_path,
            price_market_0=price_market_0,
            delta_fn=agent.delta_surface,
            maturities=maturities,
            strikes=strikes,
            days_per_year=cfg.days_per_year,
            trade_cost_bp=cfg.trade_cost_bp,
            hedge_cost_bp=cfg.hedge_cost_bp,
        )

        results["agents"][agent.name] = {
            "q":                   q,
            "price_agent_0":       price_agent_0,
            "price_market_0":      price_market_0,
            "pnl_surface":         pnl_surface,
            "option_pnl_surface":  option_pnl,
            "hedge_pnl_surface":   hedge_pnl,
            "pnl_total":           float(pnl_surface.sum()),
            "option_pnl_total":    float(option_pnl.sum()),
            "hedge_pnl_total":     float(hedge_pnl.sum()),
            "n_trades":            int(np.abs(q).sum()),
            "calibration_time_s":  agent.calibration_time_s,
        }

    return results


# --------------------------------------------------------------------- #
# Arena orchestration
# --------------------------------------------------------------------- #

def build_agents(cfg: ArenaConfig, include_rb: bool = True) -> list:
    train_full = RoughBergomiSurfaceDataset(
        os.path.join(cfg.data_dir, "data_train.npz"),
    )
    agents = [
        BSAgent(),
        BatesAgent(),
        NNAgent(
            model_path=os.path.join(cfg.model_dir, "model.pt"),
            input_bounds=train_full.input_bounds,
            output_mean=train_full.output_mean,
            output_std=train_full.output_std,
            device=cfg.device,
        ),
    ]
    if include_rb:
        agents.append(RoughBergomiAgent(mc_seed=cfg.seed))
    return agents


def run_arena(cfg: ArenaConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    maturities = np.array(DATASET_CFG.maturities)
    strikes    = np.array(DATASET_CFG.strikes)

    # -- Fast agents (BS, Bates, NN) --
    fast_agents = build_agents(cfg, include_rb=False)
    fast_pnl        = {a.name: np.zeros(cfg.n_episodes_fast, dtype=np.float32) for a in fast_agents}
    fast_option_pnl = {a.name: np.zeros(cfg.n_episodes_fast, dtype=np.float32) for a in fast_agents}
    fast_hedge_pnl  = {a.name: np.zeros(cfg.n_episodes_fast, dtype=np.float32) for a in fast_agents}
    fast_pnl_surfaces = {a.name: [] for a in fast_agents}
    fast_cal_times = {a.name: [] for a in fast_agents}
    fast_n_trades  = {a.name: [] for a in fast_agents}
    representative_episode = None

    print(f"Running {cfg.n_episodes_fast} episodes for fast agents "
          f"(hold-to-maturity protocol, sim horizon = max grid maturity)...",
          flush=True)
    t0 = time.time()
    for e in range(cfg.n_episodes_fast):
        res = run_episode(e, fast_agents, cfg, maturities, strikes)
        for name, r in res["agents"].items():
            fast_pnl[name][e]        = r["pnl_total"]
            fast_option_pnl[name][e] = r["option_pnl_total"]
            fast_hedge_pnl[name][e]  = r["hedge_pnl_total"]
            fast_pnl_surfaces[name].append(r["pnl_surface"])
            fast_cal_times[name].append(r["calibration_time_s"])
            fast_n_trades[name].append(r["n_trades"])
        if e == 0:
            representative_episode = res
        if cfg.verbose and ((e + 1) % 25 == 0 or e == cfg.n_episodes_fast - 1):
            dt = time.time() - t0
            rate = (e + 1) / dt
            eta = (cfg.n_episodes_fast - e - 1) / rate
            print(f"  fast episode {e+1:>4}/{cfg.n_episodes_fast}   "
                  f"elapsed {dt/60:.2f} min   ETA {eta/60:.2f} min",
                  flush=True)
    print(f"Fast agents done in {(time.time()-t0)/60:.2f} min.", flush=True)

    # -- Slow rBergomi baseline --
    rb_pnl = rb_option_pnl = rb_hedge_pnl = None
    rb_pnl_surfaces = rb_cal_times = rb_n_trades = None
    if cfg.include_rb and cfg.n_episodes_rb > 0:
        print(f"Running {cfg.n_episodes_rb} episodes for rBergomi baseline "
              "(slow MC calibration)...", flush=True)
        rb_agent = RoughBergomiAgent(mc_seed=cfg.seed)
        rb_pnl        = np.zeros(cfg.n_episodes_rb, dtype=np.float32)
        rb_option_pnl = np.zeros(cfg.n_episodes_rb, dtype=np.float32)
        rb_hedge_pnl  = np.zeros(cfg.n_episodes_rb, dtype=np.float32)
        rb_pnl_surfaces = []
        rb_cal_times = []
        rb_n_trades = []
        t0 = time.time()
        for e in range(cfg.n_episodes_rb):
            res = run_episode(e, [rb_agent], cfg, maturities, strikes)
            r = res["agents"]["rough Bergomi"]
            rb_pnl[e]        = r["pnl_total"]
            rb_option_pnl[e] = r["option_pnl_total"]
            rb_hedge_pnl[e]  = r["hedge_pnl_total"]
            rb_pnl_surfaces.append(r["pnl_surface"])
            rb_cal_times.append(r["calibration_time_s"])
            rb_n_trades.append(r["n_trades"])
            if cfg.verbose:
                dt = time.time() - t0
                rate = (e + 1) / dt
                eta_s = (cfg.n_episodes_rb - e - 1) / rate
                print(f"  rBergomi {e+1:>3}/{cfg.n_episodes_rb}   "
                      f"cal {r['calibration_time_s']:.1f}s   "
                      f"pnl {r['pnl_total']:+.4e}   "
                      f"ETA {eta_s/60:.1f} min", flush=True)
            np.savez_compressed(
                os.path.join(cfg.output_dir, "rb_partial.npz"),
                pnl=rb_pnl[:e+1],
                option_pnl=rb_option_pnl[:e+1],
                hedge_pnl=rb_hedge_pnl[:e+1],
                pnl_surfaces=np.stack(rb_pnl_surfaces, axis=0),
                calibration_time_s=np.array(rb_cal_times),
                n_trades=np.array(rb_n_trades),
            )
        print(f"rBergomi done in {(time.time()-t0)/60:.2f} min.", flush=True)

    # -- Aggregate + save --
    all_pnl        = dict(fast_pnl)
    all_option_pnl = dict(fast_option_pnl)
    all_hedge_pnl  = dict(fast_hedge_pnl)
    all_pnl_surfaces = {n: np.stack(s, axis=0) for n, s in fast_pnl_surfaces.items()}
    all_cal_times = {n: np.array(t, dtype=np.float32) for n, t in fast_cal_times.items()}
    all_n_trades  = {n: np.array(t, dtype=np.int32)   for n, t in fast_n_trades.items()}
    if rb_pnl is not None:
        all_pnl["rough Bergomi"]        = rb_pnl
        all_option_pnl["rough Bergomi"] = rb_option_pnl
        all_hedge_pnl["rough Bergomi"]  = rb_hedge_pnl
        all_pnl_surfaces["rough Bergomi"] = np.stack(rb_pnl_surfaces, axis=0)
        all_cal_times["rough Bergomi"] = np.array(rb_cal_times, dtype=np.float32)
        all_n_trades["rough Bergomi"]  = np.array(rb_n_trades, dtype=np.int32)

    np.savez_compressed(
        os.path.join(cfg.output_dir, "arena_results.npz"),
        **{f"pnl_{k}":         v for k, v in all_pnl.items()},
        **{f"option_pnl_{k}":  v for k, v in all_option_pnl.items()},
        **{f"hedge_pnl_{k}":   v for k, v in all_hedge_pnl.items()},
        **{f"pnl_surface_{k}": v for k, v in all_pnl_surfaces.items()},
        **{f"cal_time_{k}":    v for k, v in all_cal_times.items()},
        **{f"n_trades_{k}":    v for k, v in all_n_trades.items()},
    )
    if representative_episode is not None:
        np.savez_compressed(
            os.path.join(cfg.output_dir, "representative_episode.npz"),
            theta_true=representative_episode["theta_true"],
            S_path=representative_episode["S_path"],
            iv_true=representative_episode["iv_true"],
            iv_quote=representative_episode["iv_quote"],
        )

    stats = report(cfg, all_pnl, all_option_pnl, all_hedge_pnl,
                   all_cal_times, all_n_trades, representative_episode)
    return stats


# --------------------------------------------------------------------- #
# Reporting: Figures 8, 9, 10 + Table 8
# --------------------------------------------------------------------- #

def _agent_stats(pnl: np.ndarray, cal_times: np.ndarray,
                 n_trades: np.ndarray,
                 horizon_days: int = 504) -> dict:
    mean_pnl = float(np.mean(pnl))
    std_pnl  = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    # Le facteur d'annualisation dépend de la durée de l'épisode (504 jours
    # = 2 ans), PAS du nombre d'épisodes simulés. Ce facteur doit être
    # identique pour tous les agents, indépendamment de N.
    annualisation_factor = np.sqrt(252 / horizon_days)
    sharpe = float(mean_pnl / std_pnl * annualisation_factor) \
             if std_pnl > 1e-12 else float("nan")
    cummax   = np.maximum.accumulate(np.cumsum(pnl))
    dd       = float(np.max(cummax - np.cumsum(pnl)))
    hit_rate = float(np.mean(pnl > 0))
    return {
        "n_episodes":       int(len(pnl)),
        "mean_pnl":         mean_pnl,
        "median_pnl":       float(np.median(pnl)),
        "std_pnl":          std_pnl,
        "sharpe_annualised": sharpe,
        "max_drawdown":     dd,
        "hit_rate":         hit_rate,
        "mean_calibration_time_s": float(np.mean(cal_times)),
        "mean_n_trades":    float(np.mean(n_trades)),
    }


def report(
    cfg: ArenaConfig,
    all_pnl: dict,
    all_option_pnl: dict,
    all_hedge_pnl:  dict,
    all_cal_times:  dict,
    all_n_trades:   dict,
    representative_episode: dict | None,
) -> dict:
    # ---- Table 8 ----
    table8 = {name: _agent_stats(all_pnl[name],
                                 all_cal_times[name],
                                 all_n_trades[name])
              for name in all_pnl}
    for name in all_pnl:
        table8[name]["mean_option_pnl"] = float(np.mean(all_option_pnl[name]))
        table8[name]["mean_hedge_pnl"]  = float(np.mean(all_hedge_pnl[name]))

    with open(os.path.join(cfg.output_dir, "table8.json"), "w") as f:
        json.dump(table8, f, indent=2)

    colours = {"Black-Scholes": "tab:red", "Bates": "tab:orange",
               "rough Bergomi": "tab:green",  "Neural network": "tab:blue"}

    # ---- Figure 8: cumulative P&L + batched mean ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for name, pnl in all_pnl.items():
        cum = np.cumsum(pnl)
        ax.plot(np.arange(1, len(cum) + 1), cum, lw=1.5,
                label=name, color=colours.get(name))
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Cumulative P&L (S0 units)")
    ax.set_title("(a) Cumulative P&L")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ref_n = max(len(v) for v in all_pnl.values())
    batch = max(1, min(25, ref_n // 5)) if ref_n >= 5 else 1
    for name, pnl in all_pnl.items():
        n = len(pnl); n_batches = n // batch
        if n_batches == 0:
            continue
        batched = pnl[:n_batches * batch].reshape(n_batches, batch).mean(axis=1)
        ax.plot(np.arange(1, len(batched) + 1) * batch, batched,
                lw=1.5, marker="o", markersize=3,
                label=name, color=colours.get(name))
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Episode index")
    ax.set_ylabel(f"Mean P&L per batch of {batch}")
    ax.set_title(f"(b) Batched mean P&L (batch = {batch})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Figure 8 — Trader arena: cumulative and batched P&L per agent",
                 y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "figure8_cumulative_pnl.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.savefig(os.path.join(cfg.output_dir, "figure8_cumulative_pnl.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close()

    # ---- Figure 9: representative underlying path ----
    if representative_episode is not None:
        S_path = representative_episode["S_path"]
        if S_path.ndim == 2:
            S_path = S_path[:, 0]
        strikes = np.array(DATASET_CFG.strikes)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(np.arange(len(S_path)), S_path, lw=1.5, color="k",
                label=r"Spot $S_t$ under $\theta_{true}$")
        for K in strikes:
            ax.axhline(K, ls=":", color="grey", alpha=0.5, lw=0.7)
        ax.axhspan(1.0, strikes.max() + 0.02, alpha=0.05, color="tab:green",
                   label="ITM for call")
        ax.axhspan(strikes.min() - 0.02, 1.0, alpha=0.05, color="tab:red",
                   label="OTM for call")
        ax.set_xlabel("Day within episode (max maturity)")
        ax.set_ylabel("Spot price")
        ax.set_title("Figure 9 — Representative episode: "
                     "underlying trajectory and ITM zones",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, "figure9_trajectory.png"),
                    dpi=180, bbox_inches="tight", facecolor="white")
        plt.savefig(os.path.join(cfg.output_dir, "figure9_trajectory.pdf"),
                    bbox_inches="tight", facecolor="white")
        plt.close()

    # ---- Figure 10: ECDFs of terminal P&L ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pnl in all_pnl.items():
        xs = np.sort(pnl)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=1.8, label=f"{name}  (N = {len(pnl)})",
                color=colours.get(name))
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel("Episode terminal P&L (S0 units)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Figure 10 — Empirical CDF of terminal P&L per agent",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "figure10_ecdfs.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.savefig(os.path.join(cfg.output_dir, "figure10_ecdfs.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close()

    # ---- Sanity output ----
    print("\n=== Table 8 (with option / hedge leg decomposition) ===")
    print(json.dumps(table8, indent=2))
    print("\n=== Sanity check: agent activity ===")
    for name in all_pnl:
        n_trades_mean = float(np.mean(all_n_trades[name]))
        opt_mean = table8[name]["mean_option_pnl"]
        hdg_mean = table8[name]["mean_hedge_pnl"]
        total   = table8[name]["mean_pnl"]
        print(f"  {name:>15s}   trades/ep={n_trades_mean:5.1f}   "
              f"option={opt_mean:+.4e}   hedge={hdg_mean:+.4e}   "
              f"total={total:+.4e}")
    print(f"\nFigures 8, 9, 10 saved to {cfg.output_dir}/")
    return table8


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir",        type=str, default="data")
    p.add_argument("--model_dir",       type=str, default="results/4_2")
    p.add_argument("--output_dir",      type=str, default="results/4_4")
    p.add_argument("--n_episodes_fast", type=int, default=500)
    p.add_argument("--n_episodes_rb",   type=int, default=100)
    p.add_argument("--no_rb", action="store_true",
                   help="skip the rough Bergomi baseline (much faster)")
    p.add_argument("--bid_ask_bp", type=float, default=50.0)
    p.add_argument("--edge_bp",    type=float, default=50.0)
    p.add_argument("--steps_per_day", type=int, default=2,
                   help="intraday sub-steps for the underlying sim")
    p.add_argument("--trade_cost_bp", type=float, default=0.0)
    p.add_argument("--hedge_cost_bp", type=float, default=0.0)
    p.add_argument("--seed",   type=int, default=20260719)
    p.add_argument("--device", type=str, default="cpu")
    return p


def _main():
    args = _build_parser().parse_args()
    cfg = ArenaConfig(
        data_dir=args.data_dir, model_dir=args.model_dir,
        output_dir=args.output_dir,
        n_episodes_fast=args.n_episodes_fast,
        n_episodes_rb=args.n_episodes_rb,
        include_rb=not args.no_rb,
        bid_ask_bp=args.bid_ask_bp, edge_bp=args.edge_bp,
        steps_per_day=args.steps_per_day,
        trade_cost_bp=args.trade_cost_bp,
        hedge_cost_bp=args.hedge_cost_bp,
        seed=args.seed, device=args.device,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    run_arena(cfg)


if __name__ == "__main__":
    _main()
