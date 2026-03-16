"""
decoupled_engine.py
Modern Hetero-Associative Networks (mTAM) — Decoupled per-edge attention engine
Author: Andrea Ladiana

This engine is used for ablation against CSA:
 - Each directed edge (a<-b) computes its own softmax p^{(a<-b)} over μ.
 - Layer update: z^{(a)} = sum_{(a<-b)} K^{(b)} p^{(a<-b)}.
 - Optional "decoupled" energy: E = sum_a 1/2||z^a||^2 - (1/β) * sum_{(a<-b)} lse(β z^a·K^b) + C_shift.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn

Tensor = torch.Tensor


def _ensure_batch(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


class DecoupledEngine(nn.Module):
    def __init__(self, edges: List[Tuple[int,int]], beta: float = 1.0, use_shift: bool = True):
        super().__init__()
        assert beta > 0.0
        self.edges = list(edges)               # list of (a,b): a <- b
        self.beta = float(beta)
        self.use_shift = bool(use_shift)
        self.L = (max([max(a,b) for (a,b) in edges]) + 1) if len(edges) > 0 else 0

    @torch.no_grad()
    def per_edge_logits(self, z_list: List[Tensor], K_list: List[Tensor]) -> Dict[Tuple[int,int], Tensor]:
        # returns dict edge -> (B, K) logits
        out: Dict[Tuple[int,int], Tensor] = {}
        for (a,b) in self.edges:
            za = _ensure_batch(z_list[a])     # (B, d)
            Kb = K_list[b]                    # (d, K)
            out[(a,b)] = za @ Kb              # (B, K)
        return out

    @torch.no_grad()
    def per_edge_p(self, z_list: List[Tensor], K_list: List[Tensor]) -> Dict[Tuple[int,int], Tensor]:
        logits = self.per_edge_logits(z_list, K_list)
        beta = torch.tensor(self.beta, device=next(iter(logits.values())).device,
                            dtype=next(iter(logits.values())).dtype)
        return {e: torch.softmax(beta * S, dim=-1) for e, S in logits.items()}

    @torch.no_grad()
    def update_z(self, z_list: List[Tensor], K_list: List[Tensor],
                 p_edge: Dict[Tuple[int,int], Tensor]) -> List[Tensor]:
        # z^{(a)} = sum_{(a<-b)} K^{(b)} p^{(a<-b)}
        L = self.L
        z_new: List[Tensor] = []
        for a in range(L):
            acc = None
            for (aa,b), p in p_edge.items():
                if aa != a: continue
                Kb = K_list[b]                                  # (d, K)
                z_ab = (Kb @ p.transpose(0,1)).transpose(0,1)  # (B,d)
                acc = z_ab if acc is None else (acc + z_ab)
            if acc is None:
                za = _ensure_batch(z_list[a])
                acc = za
            z_new.append(acc)
        return z_new

    @torch.no_grad()
    def energy(self, z_list: List[Tensor], K_list: List[Tensor]) -> Tensor:
        # E = sum_a 1/2||z_a||^2 - (1/β) * sum_{(a<-b)} lse(β z^a·K^b) + C_shift
        logits = self.per_edge_logits(z_list, K_list)
        beta = torch.tensor(self.beta, device=next(iter(logits.values())).device,
                            dtype=next(iter(logits.values())).dtype)
        # quadratic
        quad = None
        for a in range(self.L):
            za = _ensure_batch(z_list[a])
            t = 0.5 * (za * za).sum(dim=-1)     # (B,)
            quad = t if quad is None else (quad + t)
        # sum of lse
        lse_sum = None
        for S in logits.values():
            lse = torch.logsumexp(beta * S, dim=-1)  # (B,)
            lse_sum = lse if lse_sum is None else (lse_sum + lse)
        E = quad - (1.0 / beta) * lse_sum
        if self.use_shift:
            # optional normalization term: (#edges / β) log K
            any_S = next(iter(logits.values()))
            Kdim = any_S.size(-1)
            E = E + (len(self.edges) / beta) * torch.log(torch.tensor(Kdim, device=E.device, dtype=E.dtype))
        return E

    @torch.no_grad()
    def step(self, z_list: List[Tensor], K_list: List[Tensor], damping: float = 1.0,
             mode: str = "jacobi") -> Tuple[List[Tensor], Dict[Tuple[int,int], Tensor]]:
        assert 0.0 < damping <= 1.0
        assert mode in ("jacobi", "gauss-seidel")
        if mode == "jacobi":
            p_edge = self.per_edge_p(z_list, K_list)
            z_new = self.update_z(z_list, K_list, p_edge)
            z_next = [(1.0 - damping) * _ensure_batch(z) + damping * zn for z, zn in zip(z_list, z_new)]
            return z_next, p_edge

        # Gauss–Seidel: update in sequence
        z_work = [_ensure_batch(z).clone() for z in z_list]
        for a in range(self.L):
            p_edge = self.per_edge_p(z_work, K_list)
            z_new_a = self.update_z(z_work, K_list, p_edge)[a]
            z_work[a] = (1.0 - damping) * z_work[a] + damping * z_new_a
        p_edge = self.per_edge_p(z_work, K_list)
        return z_work, p_edge

    @torch.no_grad()
    def forward(self, z_list: List[Tensor], K_list: List[Tensor], T: int = 1,
                damping: float = 1.0, mode: str = "jacobi", log: bool = False):
        z_work = [_ensure_batch(z).clone() for z in z_list]
        logs = {"E": [], "p_edge": []} if log else None
        p_last = None
        for _ in range(max(1, T)):
            if log:
                assert logs is not None
                logs["E"].append(self.energy(z_work, K_list).detach().cpu())
            z_work, p_last = self.step(z_work, K_list, damping=damping, mode=mode)
            if log:
                assert logs is not None
                # stack p_edge in a dict of tensors for analysis
                logs["p_edge"].append({e: p.detach().cpu() for e, p in p_last.items()})
        return z_work, p_last, logs
