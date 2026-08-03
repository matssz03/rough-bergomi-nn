"""
Neural network dataset generation (thesis Section 3.1.3).

Generates the synthetic rough Bergomi implied-volatility surfaces used
to train and evaluate the neural network approximator of Section 4.2:

    * training set     N_train  = 20 000   in-domain
    * test set         N_test   =  4 000   in-domain
    * stress-test set  N_stress =  2 000   out-of-domain

Grid                8 maturities x 11 strikes = 88 IV points per surface
Simulation          hybrid scheme of Bennedsen, Lunde and Pakkanen (2017)
                    15 000 paths, 252 time steps per year, antithetic

Parameter ranges (Section 3.1.3):
    xi_0(t)  in [0.01, 0.16]^8    (8 dims: initial forward variance TS)
    eta      in [0.5, 4.0]         (vol of vol)
    rho      in [-0.95, -0.10]     (spot-variance correlation)
    H        in [0.025, 0.500]     (Hurst parameter)

Total single-CPU cost at ~3-5 s / surface : 20-30 hours.  The script
parallelises across CPU cores with a multiprocessing.Pool and saves
intermediate chunks so that any interrupted run can be resumed by
re-invoking it without re-computing anything already on disk.

USAGE
-----
Sequential single-core (slow, but the simplest thing to reason about):

    python -m neural_network.dataset --split train --n_workers 1
    python -m neural_network.dataset --split test  --n_workers 1
    python -m neural_network.dataset --split stress --n_workers 1

Parallel across all available cores (recommended):

    python -m neural_network.dataset --split train
    python -m neural_network.dataset --split test
    python -m neural_network.dataset --split stress

Resume an interrupted run (safe to re-run at any moment; skips
chunks already on disk):

    python -m neural_network.dataset --split train

Merge chunks into a single dataset.npz once all three splits are done:

    python -m neural_network.dataset --merge
"""
from __future__ import annotations

import sys
import argparse
import glob
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pricers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pricers'))
sys.path.insert(0, 'pricers')


import numpy as np

from rough_bergomi import RoughBergomi


# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetConfig:
    n_train:  int = 20_000
    n_test:   int =  4_000
    n_stress: int =  2_000
    n_paths:  int = 15_000
    n_steps:  int =    252   # per year (Section 3.1.2)
    chunk_size: int =  100    # surfaces per saved chunk
    output_dir: str = "data"

    # xi0 term structure sampled at these 8 maturities (Section 3.1.3)
    xi0_maturities: tuple = (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0)
    maturities: tuple     = (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0)
    strikes:    tuple     = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                              1.1, 1.2, 1.3, 1.4, 1.5)

    # Training-domain bounds
    xi_lo:  float =  0.01
    xi_hi:  float =  0.16
    eta_lo: float =  0.5
    eta_hi: float =  4.0
    rho_lo: float = -0.95
    rho_hi: float = -0.10
    H_lo:   float =  0.025
    H_hi:   float =  0.500

    # Reproducibility
    seed_train:  int = 1_000_000
    seed_test:   int = 2_000_000
    seed_stress: int = 3_000_000


CFG = DatasetConfig()


# --------------------------------------------------------------------- #
# Parameter sampling
# --------------------------------------------------------------------- #

def sample_in_domain(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Draw n parameter vectors uniformly inside the training domain.

    Returns
    -------
    theta : (n, 11) array with columns
        [xi0_1, ..., xi0_8, eta, rho, H].
    """
    xi  = rng.uniform(CFG.xi_lo,  CFG.xi_hi,  size=(n, 8))
    eta = rng.uniform(CFG.eta_lo, CFG.eta_hi, size=n)
    rho = rng.uniform(CFG.rho_lo, CFG.rho_hi, size=n)
    H   = rng.uniform(CFG.H_lo,   CFG.H_hi,   size=n)
    return np.concatenate([xi, eta[:, None], rho[:, None], H[:, None]], axis=1)


def sample_out_of_domain(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Draw n stress-test parameter vectors that lie outside the training
    box on at least one dimension (Proposition P3, Section 4.5).

    We enlarge each dimension by ~20 % and enforce that every returned
    vector violates at least one training-domain boundary.
    """
    xi  = rng.uniform(CFG.xi_lo * 0.5,  CFG.xi_hi * 1.3,  size=(n, 8))
    eta = rng.uniform(CFG.eta_lo * 0.5, CFG.eta_hi * 1.2, size=n)
    rho = rng.uniform(-0.99, -0.05, size=n)
    H   = rng.uniform(0.010, 0.550, size=n)

    for i in range(n):
        in_box = (
            CFG.xi_lo  <= xi[i].min() and xi[i].max() <= CFG.xi_hi
            and CFG.eta_lo <= eta[i] <= CFG.eta_hi
            and CFG.rho_lo <= rho[i] <= CFG.rho_hi
            and CFG.H_lo   <= H[i]   <= CFG.H_hi
        )
        if in_box:
            which = rng.integers(0, 4)
            if which == 0:
                eta[i] = rng.choice([CFG.eta_lo * 0.7, CFG.eta_hi * 1.15])
            elif which == 1:
                H[i] = rng.choice([0.015, 0.55])
            elif which == 2:
                rho[i] = rng.choice([-0.99, -0.05])
            else:
                xi[i] = xi[i] * rng.choice([0.6, 1.25])

    return np.concatenate([xi, eta[:, None], rho[:, None], H[:, None]], axis=1)


# --------------------------------------------------------------------- #
# Worker: price one surface
# --------------------------------------------------------------------- #

def _price_one_surface(args):
    """
    Price a single implied-volatility surface. Runs inside a worker
    process, so it must be self-contained and picklable.
    """
    theta_row, seed = args
    xi0 = theta_row[:8]
    eta = float(theta_row[8])
    rho = float(theta_row[9])
    H   = float(theta_row[10])

    model = RoughBergomi(
        xi0=xi0, xi0_maturities=np.array(CFG.xi0_maturities),
        eta=eta, H=H, rho=rho, S0=1.0, r=0.0, q=0.0,
    )
    iv, se = model.implied_vol_surface(
        maturities=np.array(CFG.maturities),
        strikes=np.array(CFG.strikes),
        n_steps=CFG.n_steps, n_paths=CFG.n_paths,
        seed=int(seed), antithetic=True,
    )
    return iv.astype(np.float32), se.astype(np.float32)


# --------------------------------------------------------------------- #
# Chunk-based, resumable generation
# --------------------------------------------------------------------- #

def _split_info(split: str):
    if split == "train":
        return CFG.n_train,  CFG.seed_train,  True
    if split == "test":
        return CFG.n_test,   CFG.seed_test,   True
    if split == "stress":
        return CFG.n_stress, CFG.seed_stress, False
    raise ValueError(f"unknown split {split!r}")


def _chunk_path(split: str, chunk_idx: int) -> str:
    return os.path.join(
        CFG.output_dir, f"chunks_{split}", f"chunk_{chunk_idx:05d}.npz"
    )


def _prepare_split(split: str) -> tuple[np.ndarray, list[int], int, int]:
    """
    Draw all parameter vectors for the split (once, deterministically),
    scan the output directory for chunks already on disk, and return the
    list of chunk indices that still need to be computed.
    """
    n_total, seed_base, in_domain = _split_info(split)
    rng = np.random.default_rng(seed_base)
    if in_domain:
        theta = sample_in_domain(rng, n_total)
    else:
        theta = sample_out_of_domain(rng, n_total)

    chunk_dir = os.path.join(CFG.output_dir, f"chunks_{split}")
    os.makedirs(chunk_dir, exist_ok=True)

    n_chunks = (n_total + CFG.chunk_size - 1) // CFG.chunk_size
    todo = []
    for c in range(n_chunks):
        if not os.path.exists(_chunk_path(split, c)):
            todo.append(c)

    return theta, todo, n_total, seed_base


def _worker_seed(seed_base: int, split: str, global_idx: int) -> int:
    """Deterministic per-surface seed."""
    offset = {"train": 0, "test": 100_000_000, "stress": 200_000_000}[split]
    return seed_base + offset + global_idx


def generate_split(split: str, n_workers: int | None = None,
                   verbose: bool = True) -> None:
    """
    Generate (or resume) one dataset split. Safe to interrupt with Ctrl+C
    at any moment: only fully-saved chunks survive, but nothing that is
    already on disk is ever recomputed.
    """
    theta, todo, n_total, seed_base = _prepare_split(split)
    n_done = (n_total + CFG.chunk_size - 1) // CFG.chunk_size - len(todo)
    done_surfaces = n_done * CFG.chunk_size
    if not todo:
        if verbose:
            print(f"[{split}] all chunks already on disk ({n_total} surfaces)")
        return

    n_workers = n_workers or max(1, cpu_count() - 1)
    if verbose:
        print(f"[{split}] {n_total} surfaces total, {len(todo)} chunks to go, "
              f"{n_workers} worker(s), chunk_size = {CFG.chunk_size}")

    t0 = time.time()
    surfaces_done_in_run = 0

    def _run_chunk(chunk_idx: int, pool: Pool | None) -> None:
        nonlocal surfaces_done_in_run
        start = chunk_idx * CFG.chunk_size
        end = min(start + CFG.chunk_size, n_total)
        idxs = list(range(start, end))
        args = [
            (theta[i], _worker_seed(seed_base, split, i)) for i in idxs
        ]

        if pool is None:
            results = [_price_one_surface(a) for a in args]
        else:
            results = list(pool.imap(_price_one_surface, args))

        iv_chunk = np.stack([r[0] for r in results], axis=0)
        se_chunk = np.stack([r[1] for r in results], axis=0)
        theta_chunk = theta[start:end]

        # Save atomically: write to a tmp path, then rename. Note that
        # np.savez_compressed appends ".npz" to the given path if it is
        # not already present, so we work with the actual path returned.
        target = _chunk_path(split, chunk_idx)
        tmp_stem = target + ".writing"     # will become tmp_stem + ".npz"
        np.savez_compressed(
            tmp_stem,
            theta=theta_chunk.astype(np.float32),
            iv=iv_chunk,
            se=se_chunk,
            indices=np.array(idxs, dtype=np.int64),
        )
        os.replace(tmp_stem + ".npz", target)

        surfaces_done_in_run += (end - start)
        if verbose:
            elapsed = time.time() - t0
            rate = surfaces_done_in_run / max(elapsed, 1e-3)
            remaining_surfaces = (
                sum(min((c + 1) * CFG.chunk_size, n_total)
                    - c * CFG.chunk_size for c in todo)
                - surfaces_done_in_run
            )
            eta_s = remaining_surfaces / max(rate, 1e-3)
            n_nan = int(np.isnan(iv_chunk).sum())
            print(
                f"[{split}] chunk {chunk_idx + 1}/{(n_total + CFG.chunk_size - 1) // CFG.chunk_size}"
                f"  surfaces done in run: {surfaces_done_in_run:>6}"
                f"  rate: {rate * 60:.1f}/min"
                f"  ETA: {eta_s/3600:5.2f} h"
                f"  NaN in chunk: {n_nan}",
                flush=True,
            )

    if n_workers == 1:
        for chunk_idx in todo:
            _run_chunk(chunk_idx, pool=None)
    else:
        with Pool(processes=n_workers) as pool:
            for chunk_idx in todo:
                _run_chunk(chunk_idx, pool=pool)

    if verbose:
        total_time = time.time() - t0
        print(f"[{split}] done. {surfaces_done_in_run} new surfaces in "
              f"{total_time/3600:.2f} h "
              f"({surfaces_done_in_run/max(total_time,1e-3):.2f}/s)")


# --------------------------------------------------------------------- #
# Merge chunks into a single .npz
# --------------------------------------------------------------------- #

def merge_split(split: str, output_path: str | None = None) -> str:
    """Concatenate all chunks of a split into a single .npz file."""
    chunk_dir = os.path.join(CFG.output_dir, f"chunks_{split}")
    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunks found in {chunk_dir}")

    theta_all, iv_all, se_all, idx_all = [], [], [], []
    for f in files:
        d = np.load(f)
        theta_all.append(d["theta"])
        iv_all.append(d["iv"])
        se_all.append(d["se"])
        idx_all.append(d["indices"])
    theta = np.concatenate(theta_all, axis=0)
    iv    = np.concatenate(iv_all,    axis=0)
    se    = np.concatenate(se_all,    axis=0)
    indices = np.concatenate(idx_all, axis=0)

    # Reorder to guarantee a canonical row order == parameter draw order
    order = np.argsort(indices)
    theta = theta[order]
    iv    = iv[order]
    se    = se[order]
    indices = indices[order]

    if output_path is None:
        output_path = os.path.join(CFG.output_dir, f"data_{split}.npz")
    np.savez_compressed(
        output_path,
        theta=theta, iv=iv, se=se,
        indices=indices,
        maturities=np.array(CFG.maturities),
        strikes=np.array(CFG.strikes),
        xi0_maturities=np.array(CFG.xi0_maturities),
    )
    n_nan = int(np.isnan(iv).sum())
    print(f"[{split}] merged {len(theta)} surfaces -> {output_path}   "
          f"(NaN cells: {n_nan})")
    return output_path


def merge_all(output_dir: str | None = None) -> None:
    output_dir = output_dir or CFG.output_dir
    for split in ("train", "test", "stress"):
        chunk_dir = os.path.join(output_dir, f"chunks_{split}")
        if os.path.isdir(chunk_dir) and glob.glob(os.path.join(chunk_dir, "chunk_*.npz")):
            merge_split(split, os.path.join(output_dir, f"data_{split}.npz"))
        else:
            print(f"[{split}] no chunks, skipped")


# --------------------------------------------------------------------- #
# PyTorch Dataset wrapper (used by neural_network.train)
# --------------------------------------------------------------------- #

class RoughBergomiSurfaceDataset:
    """
    Thin dataset wrapper over a data_{split}.npz produced by this script.

    Loading is lazy in the sense that the .npz is memory-mapped, so the
    entire training set does not need to fit in RAM.  Input parameters
    are scaled to [0, 1] using the training-domain bounds, and outputs
    are z-scored using the (mu, sigma) statistics of the training set,
    which must be provided by the training script for the test/stress
    splits so that all inputs and outputs are on the same scale.

    Import this into training code as:
        from neural_network.dataset import RoughBergomiSurfaceDataset
    """
    def __init__(
        self,
        path: str,
        input_bounds: np.ndarray | None = None,
        output_mean:  np.ndarray | None = None,
        output_std:   np.ndarray | None = None,
        drop_nan: bool = True,
    ):
        d = np.load(path, mmap_mode="r")
        theta = np.asarray(d["theta"], dtype=np.float32)
        iv    = np.asarray(d["iv"],    dtype=np.float32).reshape(len(theta), -1)

        if drop_nan:
            keep = ~np.isnan(iv).any(axis=1)
            theta = theta[keep]
            iv    = iv[keep]

        if input_bounds is None:
            input_bounds = np.array([
                [CFG.xi_lo, CFG.xi_hi]] * 8
                + [[CFG.eta_lo, CFG.eta_hi],
                   [CFG.rho_lo, CFG.rho_hi],
                   [CFG.H_lo,   CFG.H_hi]],
                dtype=np.float32,
            )
        lo = input_bounds[:, 0]
        hi = input_bounds[:, 1]
        self.X = (theta - lo) / (hi - lo)

        if output_mean is None or output_std is None:
            output_mean = iv.mean(axis=0)
            output_std  = iv.std(axis=0, ddof=1)
            output_std  = np.where(output_std < 1e-6, 1.0, output_std)
        self.Y = (iv - output_mean) / output_std

        self.theta = theta
        self.iv = iv
        self.input_bounds = input_bounds
        self.output_mean = output_mean
        self.output_std = output_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--split", choices=["train", "test", "stress", "all"],
                   default="train",
                   help="which split to generate (default: train)")
    p.add_argument("--n_workers", type=int, default=None,
                   help="number of worker processes "
                        "(default: cpu_count() - 1)")
    p.add_argument("--merge", action="store_true",
                   help="merge existing chunks into data_{split}.npz")
    p.add_argument("--output_dir", type=str, default=None,
                   help="override the default output directory ('data')")
    return p


def _main():
    args = _build_parser().parse_args()
    if args.output_dir is not None:
        # Rebind the module-level config so downstream helpers see it.
        global CFG
        CFG = DatasetConfig(**{**CFG.__dict__, "output_dir": args.output_dir})

    if args.merge:
        merge_all()
        return

    splits = ("train", "test", "stress") if args.split == "all" else (args.split,)
    for split in splits:
        generate_split(split, n_workers=args.n_workers, verbose=True)


if __name__ == "__main__":
    _main()
