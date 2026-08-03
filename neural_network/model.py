"""
Neural network architecture for the rough Bergomi implied-volatility
approximator (thesis Section 3.4.2).

The map to approximate takes an eleven-dimensional parameter vector
    theta = (xi_0(t_1), ..., xi_0(t_8), eta, rho, H)
to the eighty-eight-dimensional vectorised Black-Scholes implied
volatility surface

    IV : [0.1, 0.3, ..., 2.0] years  x  [0.5, 0.6, ..., 1.5] moneyness.

Architecture (Section 3.4.2, matching Horvath, Muguruza and Tomas 2021):

    Input (11) -> [Linear(30) + ELU] x 4 -> Linear(88)

    ~5878 parameters, small enough to fit and evaluate on CPU.

The final layer is linear (no activation), consistent with the fact
that implied volatilities are real-valued and the outputs are z-scored
externally by the training script.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RoughBergomiApproximator(nn.Module):
    """
    Feed-forward multilayer perceptron used as the pricing approximator.

    Parameters
    ----------
    in_dim : int
        Number of input features (default 11 = 8 xi0 + eta + rho + H).
    hidden : int
        Width of the hidden layers (default 30).
    n_hidden : int
        Number of hidden layers (default 4).
    out_dim : int
        Number of output features (default 88 = 8 maturities x 11 strikes).
    activation : str
        Activation to use on hidden layers. Default "elu" reproduces the
        Horvath et al. (2021) architecture; "relu" and "tanh" are also
        accepted for ablation studies.
    """

    def __init__(
        self,
        in_dim: int = 11,
        hidden: int = 30,
        n_hidden: int = 4,
        out_dim: int = 88,
        activation: str = "elu",
    ):
        super().__init__()
        acts = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in acts:
            raise ValueError(f"unsupported activation {activation!r}")
        Act = acts[activation]

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden))
            layers.append(Act())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))  # linear output
        self.net = nn.Sequential(*layers)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.n_hidden = n_hidden
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    def n_parameters(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            f"RoughBergomiApproximator("
            f"in_dim={self.in_dim}, "
            f"hidden={self.hidden}, "
            f"n_hidden={self.n_hidden}, "
            f"out_dim={self.out_dim}, "
            f"activation={self.activation!r})",
            f"  trainable parameters: {self.n_parameters()}",
        ]
        return "\n".join(lines)
