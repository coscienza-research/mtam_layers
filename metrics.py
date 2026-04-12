"""
metrics.py
Metrics & logger utilities for Modern Hetero-Associative Networks (mTAM)
Compatible with the exact mTAM CSA properties (Step 4 integration).

Author: Andrea Ladiana

Dependencies:
  - PyTorch
Shapes (batch-first unless noted):
  - p: (B, K)
  - logits S: (B, K)
  - x_out^{(a)}: (B, d)
  - V^{(a)}: (d, K)
  - z^{(a)}: (B, d)
  - K^{(b)}: (d, K)
  - z_list: list of (B, d); K_list: list of (d, K)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

Tensor = torch.Tensor

# --------- Basic information-theoretic metrics ---------

def entropy(p: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """Shannon entropy of a probability vector p along `dim` (in nats)."""
    p_safe = p.clamp_min(eps)
    return -(p_safe * p_safe.log()).sum(dim=dim)

def topk_mass(p: Tensor, k: int) -> Tensor:
    """Sum of the largest-k probabilities per batch element."""
    k = max(1, min(k, p.size(-1)))
    vals, _ = torch.topk(p, k=k, dim=-1)
    return vals.sum(dim=-1)

def accuracy_top1(p: Tensor, mu_star: Tensor) -> Tensor:
    """Top-1 accuracy: fraction of batch where argmax(p) == mu_star.
    - p: (B, K)
    - mu_star: (B,) or scalar long tensor
    Returns: scalar tensor in [0,1].
    """
    pred = p.argmax(dim=-1)
    if mu_star.dim() == 0:
        mu_star = mu_star.expand_as(pred)
    return (pred == mu_star).float().mean()

def margin_from_logits(S: Tensor, mu_star: Tensor) -> Tensor:
    """Compute margin Δ = S_{mu*} - max_{nu!=mu*} S_{nu} per batch element.
    - S: (B, K) logits
    - mu_star: (B,) or scalar
    Returns: (B,)
    """
    B, K = S.shape
    if mu_star.dim() == 0:
        mu_star = mu_star.expand(B)
    gather_star = S.gather(dim=1, index=mu_star.view(-1, 1)).squeeze(1)  # (B,)
    # mask mu* and take max over others
    mask = torch.ones_like(S, dtype=torch.bool)
    mask.scatter_(1, mu_star.view(-1, 1), False)
    S_masked = S.masked_fill(~mask, float('-inf'))
    max_others, _ = S_masked.max(dim=1)
    return gather_star - max_others

def p_lower_bound(beta: float, delta: Tensor, K: int) -> Tensor:
    """Bound p_{mu*} ≥ 1 / (1 + (K-1) * exp(-beta * Δ)).
    - delta: (B,) margins Δ
    Returns: (B,)
    """
    return 1.0 / (1.0 + (K - 1) * torch.exp(-beta * delta))

# --------- Reconstruction metrics ---------

def reconstruction_overlap(xa: Tensor, Va: Tensor, mu_star: Tensor) -> Tensor:
    """Compute normalized overlap between x^{(a)} and the target column v^{(a)}_{mu*}.
    - xa: (B, d)
    - Va: (d, K)
    - mu_star: (B,) or scalar
    Returns: (B,) overlaps in [-1, 1].
    """
    B, d = xa.shape
    if mu_star.dim() == 0:
        mu_star = mu_star.expand(B)
    v_star = Va.t().gather(dim=0, index=mu_star.view(-1, 1).expand(B, d))  # (B, d)
    # cosine similarity as normalized overlap
    num = (xa * v_star).sum(dim=1)
    den = (xa.norm(dim=1) * v_star.norm(dim=1)).clamp_min(1e-12)
    return num / den

def mse_reconstruction(xa: Tensor, Va: Tensor, mu_star: Tensor) -> Tensor:
    """Mean squared error ||x^{(a)} - v^{(a)}_{mu*}||^2 per batch.
    Returns: (B,)
    """
    B, d = xa.shape
    if mu_star.dim() == 0:
        mu_star = mu_star.expand(B)
    v_star = Va.t().gather(dim=0, index=mu_star.view(-1, 1).expand(B, d))  # (B, d)
    return ((xa - v_star) ** 2).sum(dim=1)

# --------- Energy helpers and diagnostics ---------

def check_energy_monotone(
    energies,  # List[Tensor] of shape (B,)
    atol: float = 1e-6,
    rtol: float = 1e-7,
    per_element: bool = False,
) -> bool:
    """
    True if E^{t+1} <= E^t + atol + rtol*|E^t| for all t.
    With per_element=False the comparison is on the batch mean (more robust).
    """
    if len(energies) < 2:
        return True
    for t in range(len(energies) - 1):
        Et, Et1 = energies[t], energies[t + 1]
        if per_element:
            ok_t = torch.all(Et1 <= Et + atol + rtol * Et.abs())
        else:
            ok_t = Et1.mean() <= Et.mean() + atol + rtol * Et.abs().mean()
        if not bool(ok_t):
            return False
    return True

def convergence_delta_z(z_hist: List[List[Tensor]]) -> Tensor:
    """Compute mean L2 delta between consecutive z across layers and batch.
    - z_hist: list over time t of lists over layers a of tensors (B, d)
    Returns: scalar tensor (mean over t, a, B).
    """
    if len(z_hist) < 2:
        return torch.tensor(0.0)
    deltas = []
    for t in range(len(z_hist) - 1):
        zt, zt1 = z_hist[t], z_hist[t + 1]
        layer_deltas = [ (za1 - za0).norm(dim=1).mean() for za0, za1 in zip(zt, zt1) ]
        deltas.append(torch.stack(layer_deltas).mean())
    return torch.stack(deltas).mean()

def cooperativity_index(
    contribs: Dict[Tuple[int, int], Tensor],
    mu_hat: Optional[Tensor] = None,
) -> Tensor:
    """Compute a cooperativity index: for each batch element, measure how balanced the per-arc
    contributions are for the winning μ (or provided mu_hat).
    Returns scalar in [0,1], where 1 indicates perfectly balanced contributions across arcs.
    """
    # stack per-arc contributions for the chosen mu per batch: shape (B, E)
    arcs = list(contribs.keys())
    B, K = next(iter(contribs.values())).shape
    if mu_hat is None:
        # choose winning μ from total contributions sum
        total = sum(contribs[e] for e in arcs)  # (B, K)
        assert isinstance(total, Tensor)
        mu_hat = total.argmax(dim=-1)          # (B,)
    E = len(arcs)
    C = torch.stack([contribs[e] for e in arcs], dim=2)  # (B, K, E)
    idx = mu_hat.view(B, 1, 1).expand(B, 1, E)
    per_arc = C.gather(dim=1, index=idx).squeeze(1)      # (B, E)
    # normalize per batch
    per_arc = per_arc - per_arc.mean(dim=1, keepdim=True)
    # balance index: 1 - normalized variance
    var = (per_arc ** 2).mean(dim=1) / (per_arc.abs().mean(dim=1).clamp_min(1e-12) ** 2 + 1e-12)
    # map to [0,1] with 1/(1+var) (heuristic)
    return (1.0 / (1.0 + var)).mean()

# --------- Trial logging ---------

@dataclass
class TrialLogs:
    energies: List[Tensor]           # list over t of (B,)
    ps: List[Tensor]                 # list over t of (B, K)
    z_hist: Optional[List[List[Tensor]]] = None  # list over t of list over a of (B, d)

    def is_energy_monotone(self, atol: float = 1e-10) -> bool:
        return check_energy_monotone(self.energies, atol=atol)

    def mean_entropy(self) -> List[float]:
        return [ float(entropy(p).mean().item()) for p in self.ps ]

    def last_p(self) -> Tensor:
        return self.ps[-1] if self.ps else torch.empty(0)

    def last_energy(self) -> Tensor:
        return self.energies[-1] if self.energies else torch.empty(0)

    def convergence_score(self) -> float:
        if self.z_hist is None:
            return float('nan')
        return float(convergence_delta_z(self.z_hist).item())
