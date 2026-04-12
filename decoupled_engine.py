"""
decoupled_engine.py
Modern Hetero-Associative Networks (mTAM) — Decoupled per-edge attention engine
Author: Andrea Ladiana

This engine is used for ablation against CSA:
 - Each directed edge (a<-b) computes its own softmax p^{(a<-b)} over μ.
 - Layer update: z^{(a)} = sum_{(a<-b)} K^{(b)} p^{(a<-b)}.
 - Optional "decoupled" energy: E = sum_a 1/2||z^a||^2
     - (1/β) * sum_{(a<-b)} lse(β z^a·K^b) + C_shift.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _ensure_batch(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"Expected tensor with shape (B, d) or (d,), got {tuple(x.shape)}.")
    return x


class DecoupledEngine(nn.Module):
    def __init__(
        self,
        edges: List[Tuple[int, int]],
        beta: float = 1.0,
        use_shift: bool = True,
        L: int | None = None,
    ):
        super().__init__()
        if len(edges) == 0:
            raise ValueError("edges must be non-empty.")
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a finite positive scalar.")

        self.edges = list(edges)  # list of (a, b): a <- b
        inferred_L = max(max(a, b) for (a, b) in edges) + 1
        if L is None:
            L = inferred_L
        if L <= 0:
            raise ValueError(f"L must be >= 1, got {L}.")
        if inferred_L > L:
            raise ValueError(
                f"L={L} is too small for provided edges (requires at least {inferred_L})."
            )

        self.beta = float(beta)
        self.use_shift = bool(use_shift)
        self.L = int(L)

        for a, b in self.edges:
            if not (0 <= a < self.L and 0 <= b < self.L):
                raise ValueError(f"Edge ({a}, {b}) is out of range for L={self.L}.")

    def _validate_inputs(
        self,
        z_list: List[Tensor],
        K_list: List[Tensor],
    ) -> List[Tensor]:
        if len(z_list) != self.L:
            raise ValueError(f"Expected {self.L} z tensors, got {len(z_list)}.")
        if len(K_list) != self.L:
            raise ValueError(f"Expected {self.L} key banks, got {len(K_list)}.")

        z_batched = [_ensure_batch(z) for z in z_list]
        batch_sizes = [z.size(0) for z in z_batched]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"All z tensors must share batch size, got {batch_sizes}.")

        if any(Kb.dim() != 2 for Kb in K_list):
            bad = [i for i, Kb in enumerate(K_list) if Kb.dim() != 2]
            raise ValueError(f"All key banks must have shape (d, K), invalid indices: {bad}.")

        k_dims = [Kb.size(1) for Kb in K_list]
        if len(set(k_dims)) != 1:
            raise ValueError(f"All key banks must share K dimension, got {k_dims}.")
        if k_dims[0] <= 0:
            raise ValueError("Key banks must have at least one stored pattern (K >= 1).")

        for a, b in self.edges:
            if z_batched[a].size(1) != K_list[b].size(0):
                raise ValueError(
                    "Dimension mismatch on edge "
                    f"({a}, {b}): z[{a}] has d={z_batched[a].size(1)}, "
                    f"K[{b}] expects d={K_list[b].size(0)}."
                )

        return z_batched

    def _validate_p_edge(
        self,
        p_edge: Dict[Tuple[int, int], Tensor],
        batch_size: int,
        k_dim: int,
    ) -> None:
        expected = set(self.edges)
        provided = set(p_edge.keys())
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(f"p_edge keys mismatch. Missing={missing}, extra={extra}.")

        for edge, p in p_edge.items():
            if p.dim() != 2:
                raise ValueError(f"p_edge[{edge}] must have shape (B, K), got {tuple(p.shape)}.")
            if p.size(0) != batch_size or p.size(1) != k_dim:
                raise ValueError(
                    f"p_edge[{edge}] shape {tuple(p.shape)} does not match "
                    f"expected ({batch_size}, {k_dim})."
                )

    def per_edge_logits(
        self,
        z_list: List[Tensor],
        K_list: List[Tensor],
    ) -> Dict[Tuple[int, int], Tensor]:
        # returns dict edge -> (B, K) logits
        z_batched = self._validate_inputs(z_list, K_list)
        out: Dict[Tuple[int, int], Tensor] = {}
        for (a, b) in self.edges:
            za = z_batched[a]  # (B, d)
            Kb = K_list[b]  # (d, K)
            out[(a, b)] = za @ Kb  # (B, K)
        return out

    def per_edge_p(
        self,
        z_list: List[Tensor],
        K_list: List[Tensor],
    ) -> Dict[Tuple[int, int], Tensor]:
        logits = self.per_edge_logits(z_list, K_list)
        return {e: torch.softmax(self.beta * S, dim=-1) for e, S in logits.items()}

    def update_z(
        self,
        z_list: List[Tensor],
        K_list: List[Tensor],
        p_edge: Dict[Tuple[int, int], Tensor],
    ) -> List[Tensor]:
        # z^{(a)} = sum_{(a<-b)} K^{(b)} p^{(a<-b)}
        z_batched = self._validate_inputs(z_list, K_list)
        batch_size = z_batched[0].size(0)
        k_dim = K_list[0].size(1)
        self._validate_p_edge(p_edge, batch_size=batch_size, k_dim=k_dim)

        L = self.L
        z_new: List[Tensor] = []
        for a in range(L):
            acc = None
            for (aa, b), p in p_edge.items():
                if aa != a:
                    continue
                Kb = K_list[b]  # (d, K)
                z_ab = (Kb @ p.transpose(0,1)).transpose(0,1)  # (B,d)
                acc = z_ab if acc is None else (acc + z_ab)
            if acc is None:
                acc = z_batched[a]
            z_new.append(acc)
        return z_new

    def energy(self, z_list: List[Tensor], K_list: List[Tensor]) -> Tensor:
        # E = sum_a 1/2||z_a||^2 - (1/β) * sum_{(a<-b)} lse(β z^a·K^b) + C_shift
        z_batched = self._validate_inputs(z_list, K_list)
        logits = self.per_edge_logits(z_list, K_list)

        # quadratic
        quad: Optional[Tensor] = None
        for a in range(self.L):
            za = z_batched[a]
            t = 0.5 * (za * za).sum(dim=-1)     # (B,)
            quad = t if quad is None else (quad + t)

        # sum of lse
        lse_sum: Optional[Tensor] = None
        for S in logits.values():
            lse = torch.logsumexp(self.beta * S, dim=-1)  # (B,)
            lse_sum = lse if lse_sum is None else (lse_sum + lse)

        assert quad is not None
        assert lse_sum is not None
        E = quad - (1.0 / self.beta) * lse_sum
        if self.use_shift:
            # optional normalization term: (#edges / β) log K
            any_S = next(iter(logits.values()))
            Kdim = any_S.size(-1)
            E = E + (len(self.edges) / self.beta) * math.log(Kdim)
        return E

    def step(
        self,
        z_list: List[Tensor],
        K_list: List[Tensor],
        damping: float = 1.0,
        mode: str = "jacobi",
    ) -> Tuple[List[Tensor], Dict[Tuple[int, int], Tensor]]:
        if not (0.0 < damping <= 1.0):
            raise ValueError(f"damping must be in (0, 1], got {damping}.")
        if mode not in ("jacobi", "gauss-seidel"):
            raise ValueError(f"mode must be 'jacobi' or 'gauss-seidel', got {mode}.")

        z_batched = self._validate_inputs(z_list, K_list)
        if mode == "jacobi":
            p_edge = self.per_edge_p(z_batched, K_list)
            z_new = self.update_z(z_batched, K_list, p_edge)
            z_next = [(1.0 - damping) * z + damping * zn for z, zn in zip(z_batched, z_new)]
            return z_next, p_edge

        # Gauss–Seidel: update in sequence
        z_work = [z.clone() for z in z_batched]
        for a in range(self.L):
            p_edge = self.per_edge_p(z_work, K_list)
            z_new_a = self.update_z(z_work, K_list, p_edge)[a]
            z_work[a] = (1.0 - damping) * z_work[a] + damping * z_new_a
        p_edge = self.per_edge_p(z_work, K_list)
        return z_work, p_edge

    def forward(
        self,
        z_list: List[Tensor],
        K_list: List[Tensor],
        T: int = 1,
        damping: float = 1.0,
        mode: str = "jacobi",
        log: bool = False,
    ) -> Tuple[List[Tensor], Dict[Tuple[int, int], Tensor], Optional[Dict[str, List[object]]]]:
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")

        z_work = [z.clone() for z in self._validate_inputs(z_list, K_list)]
        logs: Optional[Dict[str, List[object]]] = {"E": [], "p_edge": []} if log else None
        p_last: Optional[Dict[Tuple[int, int], Tensor]] = None
        for _ in range(T):
            if log:
                assert logs is not None
                logs["E"].append(self.energy(z_work, K_list).detach().cpu())
            z_work, p_last = self.step(z_work, K_list, damping=damping, mode=mode)
            if log:
                assert logs is not None
                # stack p_edge in a dict of tensors for analysis
                logs["p_edge"].append({e: p.detach().cpu() for e, p in p_last.items()})
        assert p_last is not None
        return z_work, p_last, logs
