"""
entropy_gated_consensus.py

Entropy-Gated Consensus (EG-CSA) for mTAM.

This module implements the mechanism introduced in Section 7
("The Cost of Choice: Entropy-Gated Consensus") of the manuscript:

- per-modality logits:           ell^(a)_mu
- local posteriors:              q^(a)
- shared consensus posterior:    p
- dissent entropy:               D(Z) = (1/L) sum_a KL(q^(a) || p)
- logistic gate:                 g = sigma(kappa * (D - tau))
- gated posterior per modality:  p_tilde^(a) = g q^(a) + (1-g) p
- gated update:                  z_new^(a) = sum_b A_bar[a,b] K^(b) p_tilde^(a)

All operations are batch-vectorized and autograd-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .mtam import mTAM_Core


@dataclass
class EGConsensusDiagnostics:
    """Diagnostics for entropy-gated consensus at one iteration."""

    consensus_entropy: Tensor
    dissent_entropy: Tensor
    gate: Tensor
    weighted_modal_variance: Tensor
    dissent_upper_bound: Tensor
    synergy: Tensor


@dataclass
class EGConsensusStep:
    """Outputs of one EG-CSA step."""

    Z_new: Tensor
    S: Tensor
    local_logits: Tensor
    p_consensus: Tensor
    q_local: Tensor
    p_gated: Tensor
    diagnostics: EGConsensusDiagnostics


class EntropyGatedConsensus:
    """Implements the Entropy-Gated Consensus mechanism on top of mTAM core.

    Parameters
    ----------
    core : mTAM_Core
        Base CSA engine providing graph topology and temperature.
    tau : float
        Dissent threshold. Larger values keep the gate more closed.
    kappa : float
        Logistic steepness for the gate.
    eps : float
        Numerical floor used in logarithms.
    """

    def __init__(
        self,
        core: mTAM_Core,
        tau: float,
        kappa: float = 10.0,
        eps: float = 1e-12,
    ) -> None:
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}.")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got {kappa}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.core = core
        self.tau = float(tau)
        self.kappa = float(kappa)
        self.eps = float(eps)

    @property
    def beta_tilde(self) -> float:
        return self.core.beta_tilde

    def _validate_shapes(self, Z: Tensor, K: Tensor) -> None:
        if Z.dim() != 3:
            raise ValueError(f"Z must have shape (B, L, d), got {tuple(Z.shape)}.")
        if K.dim() != 3:
            raise ValueError(f"K must have shape (L, d, M), got {tuple(K.shape)}.")

        if Z.size(1) != self.core.L or Z.size(2) != self.core.d:
            raise ValueError(
                f"Z must have shape (B, {self.core.L}, {self.core.d}), got {tuple(Z.shape)}."
            )
        if K.size(0) != self.core.L or K.size(1) != self.core.d:
            raise ValueError(
                f"K must have shape ({self.core.L}, {self.core.d}, M), got {tuple(K.shape)}."
            )
        if K.size(2) <= 0:
            raise ValueError("K must include at least one stored pattern (M >= 1).")

    def per_modality_logits(
        self,
        Z: Tensor,
        K: Tensor,
        K_eff: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute per-modality logits ell^(a) of shape (B, L, M)."""
        self._validate_shapes(Z, K)
        if K_eff is None:
            K_eff = self.core.effective_keys(K)
        return torch.einsum("nai,aim->nam", Z, K_eff)

    def local_posteriors(self, local_logits: Tensor) -> Tensor:
        """Compute q^(a) = softmax(beta_tilde * ell^(a)) as (B, L, M)."""
        if local_logits.dim() != 3:
            raise ValueError(
                f"local_logits must have shape (B, L, M), got {tuple(local_logits.shape)}."
            )
        return torch.softmax(self.beta_tilde * local_logits, dim=-1)

    def consensus_entropy(self, p: Tensor) -> Tensor:
        """Shannon entropy H(p) for each batch element."""
        p_safe = p.clamp_min(self.eps)
        return -(p_safe * p_safe.log()).sum(dim=-1)

    def dissent_entropy(self, q_local: Tensor, p_consensus: Tensor) -> Tensor:
        """Compute D(Z) = (1/L) sum_a KL(q^(a) || p)."""
        if q_local.dim() != 3:
            raise ValueError(f"q_local must have shape (B, L, M), got {tuple(q_local.shape)}.")
        if p_consensus.dim() != 2:
            raise ValueError(
                f"p_consensus must have shape (B, M), got {tuple(p_consensus.shape)}."
            )

        q_safe = q_local.clamp_min(self.eps)
        p_safe = p_consensus.clamp_min(self.eps).unsqueeze(1)
        kl_per_modality = (q_safe * (q_safe.log() - p_safe.log())).sum(dim=-1)  # (B, L)
        return kl_per_modality.mean(dim=1)

    def weighted_modal_variance(self, local_logits: Tensor, p_consensus: Tensor) -> Tensor:
        """Compute V_mod(Z) = sum_mu p_mu * V_mu(Z)."""
        modal_mean = local_logits.mean(dim=1, keepdim=True)  # (B, 1, M)
        v_mu = (local_logits - modal_mean).pow(2).mean(dim=1)  # (B, M)
        return (p_consensus * v_mu).sum(dim=-1)

    def consensus_synergy(self, S: Tensor, local_logits: Tensor) -> Tensor:
        """Synergy I(Z) from the free-energy decomposition in the manuscript.

        F      = -(1/bt) logsumexp(bt * S)
        F^(a)  = -(1/bt) logsumexp(bt * ell^(a))
        I      = bt * (mean_a F^(a) - F)
        """
        bt = self.beta_tilde
        F_global = -(1.0 / bt) * torch.logsumexp(bt * S, dim=-1)  # (B,)
        F_local = -(1.0 / bt) * torch.logsumexp(bt * local_logits, dim=-1)  # (B, L)
        return bt * (F_local.mean(dim=1) - F_global)

    def gate_value(self, dissent: Tensor) -> Tensor:
        """Logistic gate g = sigma(kappa * (D - tau))."""
        return torch.sigmoid(self.kappa * (dissent - self.tau))

    def gated_distribution(
        self,
        q_local: Tensor,
        p_consensus: Tensor,
        gate: Tensor,
    ) -> Tensor:
        """Compute p_tilde^(a) = g q^(a) + (1-g) p."""
        g = gate.view(-1, 1, 1)
        return g * q_local + (1.0 - g) * p_consensus.unsqueeze(1)

    def gated_update(self, K_eff: Tensor, p_gated: Tensor) -> Tensor:
        """Compute z_new from gated distributions for each modality."""
        return torch.einsum("aim,nam->nai", K_eff, p_gated)

    def step(self, Z: Tensor, K: Tensor, damping: float = 1.0) -> EGConsensusStep:
        """Run one EG-CSA iteration."""
        if not (0.0 < damping <= 1.0):
            raise ValueError(f"damping must be in (0, 1], got {damping}.")

        self._validate_shapes(Z, K)

        K_eff = self.core.effective_keys(K)
        local_logits = self.per_modality_logits(Z, K, K_eff=K_eff)  # (B, L, M)
        S = local_logits.sum(dim=1)  # (B, M)

        p_consensus = self.core.compute_consensus(S)  # (B, M)
        q_local = self.local_posteriors(local_logits)  # (B, L, M)

        dissent = self.dissent_entropy(q_local, p_consensus)  # (B,)
        gate = self.gate_value(dissent)  # (B,)
        p_gated = self.gated_distribution(q_local, p_consensus, gate)  # (B, L, M)

        Z_gate = self.gated_update(K_eff, p_gated)
        Z_new = (1.0 - damping) * Z + damping * Z_gate

        weighted_var = self.weighted_modal_variance(local_logits, p_consensus)
        diagnostics = EGConsensusDiagnostics(
            consensus_entropy=self.consensus_entropy(p_consensus),
            dissent_entropy=dissent,
            gate=gate,
            weighted_modal_variance=weighted_var,
            dissent_upper_bound=(self.beta_tilde ** 2) * weighted_var,
            synergy=self.consensus_synergy(S, local_logits),
        )

        return EGConsensusStep(
            Z_new=Z_new,
            S=S,
            local_logits=local_logits,
            p_consensus=p_consensus,
            q_local=q_local,
            p_gated=p_gated,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        Z: Tensor,
        K: Tensor,
        T: int = 1,
        damping: float = 1.0,
        return_trajectory: bool = False,
    ) -> Tuple[Tensor, EGConsensusStep, Optional[Dict[str, List[Tensor]]]]:
        """Run T EG-CSA iterations.

        Returns
        -------
        Z_out : (B, L, d)
        last_step : EGConsensusStep
        traj : optional dictionary of diagnostic trajectories
        """
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")

        traj: Optional[Dict[str, List[Tensor]]] = None
        if return_trajectory:
            traj = {
                "consensus_entropy": [],
                "dissent_entropy": [],
                "gate": [],
                "weighted_modal_variance": [],
                "dissent_upper_bound": [],
                "synergy": [],
            }

        last_step: Optional[EGConsensusStep] = None
        for _ in range(T):
            last_step = self.step(Z, K, damping=damping)
            Z = last_step.Z_new

            if traj is not None:
                diag = last_step.diagnostics
                traj["consensus_entropy"].append(diag.consensus_entropy.detach())
                traj["dissent_entropy"].append(diag.dissent_entropy.detach())
                traj["gate"].append(diag.gate.detach())
                traj["weighted_modal_variance"].append(diag.weighted_modal_variance.detach())
                traj["dissent_upper_bound"].append(diag.dissent_upper_bound.detach())
                traj["synergy"].append(diag.synergy.detach())
        assert last_step is not None
        return Z, last_step, traj
