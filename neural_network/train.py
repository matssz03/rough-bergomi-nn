"""
Neural network training and evaluation (thesis Section 4.2).

Trains the ``RoughBergomiApproximator`` MLP on the synthetic surfaces
produced by ``neural_network.dataset`` and produces the two artefacts
of Section 4.2:

    Figure 6 -- heatmaps of pointwise approximation error on the test
                set (mean absolute error, standard deviation of absolute
                error, mean relative error, in the format of Horvath,
                Muguruza and Tomas 2021, Figures 6 and 7).
    Table 6  -- speed benchmark of the network against the Monte Carlo
                pricer of ``pricers.rough_bergomi``.

Also stores the stress-test error decomposition used by Section 4.5.

USAGE
-----
Standard training run (assumes data_train.npz, data_test.npz and
data_stress.npz are already in the ``data/`` directory):

    python -m neural_network.train

Change hyperparameters or paths with CLI arguments; see ``--help``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys 


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pricers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pricers'))
sys.path.insert(0, 'pricers')

from neural_network.dataset import (
    CFG as DATASET_CFG,
    RoughBergomiSurfaceDataset,
)
from neural_network.model import RoughBergomiApproximator
from rough_bergomi import RoughBergomi


# --------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "results/4_2"

    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 500
    patience: int = 40
    val_fraction: float = 0.10

    lr_decay_epochs: tuple = (200, 350, 450)
    lr_decay_factor: float = 0.5

    seed: int = 20260717
    device: str = "cpu"       # "cuda" if available; MLP is small so CPU is fine
    n_workers: int = 0        # DataLoader workers


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #

def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---------------- Load datasets ----------------
    print("Loading datasets...")
    train_full = RoughBergomiSurfaceDataset(
        os.path.join(cfg.data_dir, "data_train.npz"),
    )
    test = RoughBergomiSurfaceDataset(
        os.path.join(cfg.data_dir, "data_test.npz"),
        input_bounds=train_full.input_bounds,
        output_mean=train_full.output_mean,
        output_std=train_full.output_std,
    )
    stress_path = os.path.join(cfg.data_dir, "data_stress.npz")
    stress = None
    if os.path.exists(stress_path):
        stress = RoughBergomiSurfaceDataset(
            stress_path,
            input_bounds=train_full.input_bounds,
            output_mean=train_full.output_mean,
            output_std=train_full.output_std,
        )

    print(f"  train_full : {len(train_full):>6}")
    print(f"  test       : {len(test):>6}")
    print(f"  stress     : {len(stress) if stress else 0:>6}")

    # Split off a validation set
    n_val = int(cfg.val_fraction * len(train_full))
    n_tr = len(train_full) - n_val
    gen = torch.Generator().manual_seed(cfg.seed)

    X_all = torch.from_numpy(train_full.X).float()
    Y_all = torch.from_numpy(train_full.Y).float()
    full_ds = TensorDataset(X_all, Y_all)
    train_ds, val_ds = random_split(full_ds, [n_tr, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.n_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.n_workers,
    )

    # ---------------- Model, loss, optimiser ----------------
    model = RoughBergomiApproximator().to(device)
    print(model.summary())
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=list(cfg.lr_decay_epochs),
        gamma=cfg.lr_decay_factor,
    )

    # ---------------- Training ----------------
    print(f"Training for up to {cfg.epochs} epochs, patience {cfg.patience}...")
    best_val = float("inf")
    best_state = None
    stale = 0
    history = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimiser.zero_grad()
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()
            optimiser.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_hat = model(xb)
                loss = criterion(y_hat, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_loss_sum / val_n
        scheduler.step()
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "lr": scheduler.get_last_lr()[0]})

        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:4d}   train {train_loss:.4e}   "
                  f"val {val_loss:.4e}   best {best_val:.4e}   "
                  f"stale {stale}   lr {scheduler.get_last_lr()[0]:.2e}",
                  flush=True)
        if stale >= cfg.patience:
            print(f"  early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model.pt"))
    print(f"Training done in {train_time/60:.2f} min")

    # ---------------- Test-set evaluation ----------------
    print("Evaluating on test set...")
    model.eval()
    X_te = torch.from_numpy(test.X).float().to(device)
    Y_te = torch.from_numpy(test.Y).float().to(device)
    with torch.no_grad():
        Y_hat_scaled = model(X_te).cpu().numpy()

    iv_hat_te = Y_hat_scaled * test.output_std + test.output_mean       # unscaled
    iv_te     = test.iv
    abs_err = iv_hat_te - iv_te
    rel_err = abs_err / iv_te
    abs_err_grid = abs_err.reshape(-1, 8, 11)
    rel_err_grid = rel_err.reshape(-1, 8, 11)

    mean_abs = np.mean(np.abs(abs_err_grid), axis=0)   # (8, 11)
    std_abs  = np.std(abs_err_grid, axis=0, ddof=1)    # (8, 11)
    mean_rel = np.mean(np.abs(rel_err_grid), axis=0)   # (8, 11)
    max_abs  = np.max(np.abs(abs_err_grid), axis=0)    # (8, 11)

    summary = {
        "n_train": n_tr,
        "n_val": n_val,
        "n_test": len(test),
        "global_mean_abs_bp": float(np.mean(np.abs(abs_err)) * 1e4),
        "global_std_abs_bp":  float(np.std(abs_err, ddof=1) * 1e4),
        "global_max_abs_bp":  float(np.max(np.abs(abs_err)) * 1e4),
        "global_mean_rel_pct": float(np.mean(np.abs(rel_err)) * 100),
        "worst_cell_mean_abs_bp": float(mean_abs.max() * 1e4),
        "best_cell_mean_abs_bp":  float(mean_abs.min() * 1e4),
        "train_time_s": train_time,
        "final_train_loss": float(history[-1]["train"]),
        "final_val_loss": float(history[-1]["val"]),
        "best_val_loss": float(best_val),
    }
    print(json.dumps(summary, indent=2))

    np.savez_compressed(
        os.path.join(cfg.output_dir, "test_errors.npz"),
        mean_abs=mean_abs, std_abs=std_abs,
        mean_rel=mean_rel, max_abs=max_abs,
        abs_err=abs_err, rel_err=rel_err,
        maturities=np.array(DATASET_CFG.maturities),
        strikes=np.array(DATASET_CFG.strikes),
    )

    # ---------------- Stress-set evaluation ----------------
    if stress is not None:
        X_st = torch.from_numpy(stress.X).float().to(device)
        with torch.no_grad():
            Y_st_scaled = model(X_st).cpu().numpy()
        iv_hat_st = Y_st_scaled * stress.output_std + stress.output_mean
        iv_st = stress.iv
        abs_err_st = iv_hat_st - iv_st
        rel_err_st = abs_err_st / iv_st
        stress_summary = {
            "n_stress": len(stress),
            "mean_abs_bp": float(np.mean(np.abs(abs_err_st)) * 1e4),
            "max_abs_bp":  float(np.max(np.abs(abs_err_st)) * 1e4),
            "mean_rel_pct": float(np.mean(np.abs(rel_err_st)) * 100),
        }
        summary["stress"] = stress_summary
        print("Stress-set summary:")
        print(json.dumps(stress_summary, indent=2))
        np.savez_compressed(
            os.path.join(cfg.output_dir, "stress_errors.npz"),
            abs_err=abs_err_st, rel_err=rel_err_st,
            theta=stress.theta,
        )

    # ---------------- Figure 6: heatmaps ----------------
    _make_figure6(mean_abs, std_abs, mean_rel, cfg.output_dir, len(test))

    # ---------------- Training curve (annex) ----------------
    _make_training_curve(history, cfg.output_dir)

    # ---------------- Table 6: speed benchmark ----------------
    table6 = _speed_benchmark(model, test, device, cfg.output_dir)
    summary["table6"] = table6
    print("Table 6:")
    print(json.dumps(table6, indent=2))

    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump({"config": asdict(cfg), "summary": summary,
                   "history": history}, f, indent=2)
    print(f"All artefacts saved to {cfg.output_dir}/")

    return summary


# --------------------------------------------------------------------- #
# Speed benchmark (Table 6)
# --------------------------------------------------------------------- #

def _speed_benchmark(model: nn.Module,
                     test: RoughBergomiSurfaceDataset,
                     device: torch.device,
                     output_dir: str) -> dict:
    """
    Time the Monte Carlo pricer against the trained neural network on
    representative test-set surfaces.
    """
    xi0_mats   = np.array(DATASET_CFG.xi0_maturities)
    maturities = np.array(DATASET_CFG.maturities)
    strikes    = np.array(DATASET_CFG.strikes)

    # Monte Carlo -- 30 independent surfaces at 15 000 paths, 252 steps
    n_bench = 30
    mc_times = []
    for k in range(n_bench):
        theta = test.theta[k]
        xi0 = theta[:8]
        eta = float(theta[8])
        rho = float(theta[9])
        H   = float(theta[10])
        rb = RoughBergomi(
            xi0=xi0, xi0_maturities=xi0_mats,
            eta=eta, H=H, rho=rho, S0=1.0,
        )
        t0 = time.time()
        _ = rb.implied_vol_surface(
            maturities=maturities, strikes=strikes,
            n_steps=DATASET_CFG.n_steps, n_paths=DATASET_CFG.n_paths,
            seed=999_000 + k,
        )
        mc_times.append(time.time() - t0)
    mc_mean = float(np.mean(mc_times))
    mc_std  = float(np.std(mc_times, ddof=1))

    # NN single-surface (warm up first, then time 200 calls)
    X_te = torch.from_numpy(test.X).float().to(device)
    with torch.no_grad():
        _ = model(X_te[:1])
    n_single = 200
    nn_single_times = []
    with torch.no_grad():
        for k in range(n_single):
            idx = k % len(X_te)
            t0 = time.time()
            _ = model(X_te[idx:idx + 1])
            nn_single_times.append(time.time() - t0)
    nn_single = float(np.mean(nn_single_times))

    # NN batch of 100
    with torch.no_grad():
        _ = model(X_te[:100])
    nn_batch_times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.time()
            _ = model(X_te[:100])
            nn_batch_times.append(time.time() - t0)
    nn_batch_100 = float(np.mean(nn_batch_times))
    nn_batch_per = nn_batch_100 / 100.0

    table6 = {
        "mc_mean_s": mc_mean,
        "mc_std_s":  mc_std,
        "nn_single_s": nn_single,
        "nn_batch_100_s": nn_batch_100,
        "nn_batch_per_s": nn_batch_per,
        "speedup_single": mc_mean / nn_single,
        "speedup_batch":  mc_mean / nn_batch_per,
    }
    with open(os.path.join(output_dir, "table6.json"), "w") as f:
        json.dump(table6, f, indent=2)
    return table6


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #

def _make_figure6(mean_abs, std_abs, mean_rel, output_dir: str, n_test: int):
    maturities = np.array(DATASET_CFG.maturities)
    strikes    = np.array(DATASET_CFG.strikes)
    extent = [strikes[0] - 0.05, strikes[-1] + 0.05,
              maturities[-1] + 0.1, maturities[0] - 0.1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, data, title, unit in (
        (axes[0], mean_abs * 1e4, r"Mean $|$absolute error$|$", "bp"),
        (axes[1], std_abs  * 1e4, r"Std of absolute error",     "bp"),
        (axes[2], mean_rel * 100, r"Mean $|$relative error$|$", "%"),
    ):
        im = ax.imshow(data, aspect="auto", extent=extent, cmap="viridis",
                       vmin=0, vmax=data.max())
        ax.set_title(f"{title}  ({unit})")
        ax.set_xlabel(r"Moneyness $K/S_0$")
        ax.set_ylabel(r"Maturity $T$ (years)")
        ax.set_xticks(strikes)
        ax.set_yticks(maturities)
        ax.set_xticklabels([f"{s:.1f}" for s in strikes], fontsize=8)
        ax.set_yticklabels([f"{t:.1f}" for t in maturities], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Neural network approximation error on the test set "
                 f"(N_test = {n_test} surfaces)",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure6_heatmaps.png"),
                dpi=140, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure6_heatmaps.pdf"),
                bbox_inches="tight")
    plt.close()


def _make_training_curve(history: list[dict], output_dir: str):
    epochs = [h["epoch"] for h in history]
    train_l = [h["train"] for h in history]
    val_l   = [h["val"]   for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, train_l, label="Train loss", lw=1.5)
    ax.semilogy(epochs, val_l,   label="Validation loss", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised outputs)")
    ax.set_title("MLP training curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_training.png"),
                dpi=140, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--output_dir", type=str, default="results/4_2")
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int, default=40)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=20260717)
    return p


def _main():
    args = _build_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    _main()
