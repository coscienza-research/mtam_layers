"""
mtam_block.py — End-to-end Trainable mTAM Modules
===================================================

Drop-in multi-modal Transformer blocks based on Consensus Split-Bank
Attention (CSA). Each module is a standard ``nn.Module`` with a fully
differentiable forward pass (softmax + linear readout), so gradients
flow through the CSA consensus mechanism exactly as they do through
standard multi-head attention.

Architecture Mapping (CSA ↔ Transformer):
─────────────────────────────────────────
  Standard Transformer          mTAM / CSA
  ═══════════════════          ═══════════
  Q = W_Q · x                  z^{(a)} = W_Q^{(a)} · x^{(a)}
  K = W_K · x                  K^{(b)} (learnable key banks)
  V = W_V · x                  V^{(a)} (learnable value banks)
  Attn = softmax(QK^T/√d)      p = softmax(β̃ · S_global)   ← SHARED ∀ a
  Out = Attn · V                x_out^{(a)} = V^{(a)} · p

Key differences:
  1. CSA computes ONE shared consensus p over all modalities (not L
     independent attention distributions).
  2. Multiple CCCP iterations refine p, unlike single-pass attention.
  3. The CSA energy E(z) provides a natural Lyapunov regularizer.

Analogous to Ramsauer et al. (2020) "Hopfield Networks is All You Need"
which showed that MHN retrieval ≡ attention. mTAM extends this to
multi-modal hetero-association: CSA ≡ multi-modal cross-attention with
a global consensus bottleneck.

Author: Andrea Ladiana
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mtam_layers.mtam import mTAM_Core, complete_graph


# ═══════════════════════════════════════════════════
#  mTAM Block — Drop-in Multi-Modal Transformer Block
# ═══════════════════════════════════════════════════

class mTAMBlock(nn.Module):
    r"""Multi-modal Transformer block with Consensus Split-Bank Attention.

    This is the direct multi-modal analogue of a standard Transformer
    encoder block (Vaswani et al., 2017), and of the Hopfield layer
    (Ramsauer et al., 2020) for single-modality settings.

    Architecture:

        For each modality a:
          z^{(a)} = LayerNorm(W_Q^{(a)} · x^{(a)})       ← query projection
          [Run T CCCP iterations of CSA]
          x_out^{(a)} = W_O^{(a)} · V^{(a)} · p           ← output projection
          x^{(a)} ← x^{(a)} + Dropout(x_out^{(a)})        ← residual
          x^{(a)} ← x^{(a)} + Dropout(FFN(LayerNorm(x^{(a)})))  ← feed-forward

    The entire forward pass is differentiable. Gradients flow through
    softmax(β̃ · S) exactly as they flow through standard attention.

    Parameters
    ----------
    in_dims : list of int
        Input dimensions N_a per modality (length L).
    d : int
        Shared CSA embedding dimension. Analogous to ``d_model`` in
        Transformers, or ``head_dim`` in Ramsauer et al.
    num_patterns : int
        Number of prototypes M ≡ K. These are the "stored patterns" in
        the associative memory; analogous to sequence length in attention.
    edges : list of (a, b)
        Directed interaction graph. For the fully connected case
        (all modalities attend to all others), use ``complete_graph(L)``.
    beta : float
        Inverse temperature. β̃ = β/√d controls sharpness of consensus.
        Analogous to the scaling factor 1/√d_k in Transformers.
    T_steps : int
        Number of CCCP iterations per forward pass. T=1 gives standard
        single-pass attention; T>1 allows iterative refinement.
    ffn_ratio : int
        Expansion ratio for the feed-forward sub-layer.
    dropout : float
        Dropout rate for residual connections.
    """

    def __init__(
        self,
        in_dims: List[int],
        d: int,
        num_patterns: int,
        edges: List[Tuple[int, int]],
        beta: float = 1.0,
        T_steps: int = 1,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.L = len(in_dims)
        self.d = d
        self.M = num_patterns
        self.T_steps = T_steps

        # ── Query projections: W_Q^{(a)} : ℝ^{N_a} → ℝ^d ──
        self.Wq = nn.ModuleList([
            nn.Identity() if N == d else nn.Linear(N, d, bias=False)
            for N in in_dims
        ])

        # ── Pre-CSA Layer Normalization ──
        self.norm_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.L)])

        # ── Key and Value banks (learnable "memory") ──
        # K^{(b)} ∈ ℝ^{d × M}: keys for scoring
        # V^{(a)} ∈ ℝ^{d × M}: values for readout
        self.K_banks = nn.Parameter(torch.empty(self.L, d, num_patterns))
        self.V_banks = nn.Parameter(torch.empty(self.L, d, num_patterns))
        nn.init.normal_(self.K_banks, std=d ** -0.5)
        nn.init.normal_(self.V_banks, std=d ** -0.5)

        # ── Output projections: W_O^{(a)} : ℝ^d → ℝ^{N_a} ──
        self.Wo = nn.ModuleList([
            nn.Identity() if N == d else nn.Linear(d, N, bias=False)
            for N in in_dims
        ])

        # ── CSA Core Engine ──
        self.core = mTAM_Core(
            edges=edges, L=self.L, d=d, beta=beta, use_shift=True,
        )

        # ── Feed-Forward Network (per-modality) ──
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(N) for N in in_dims])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(N, N * ffn_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(N * ffn_ratio, N),
            )
            for N in in_dims
        ])

        # ── Dropout ──
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_list: List[Tensor],
        return_consensus: bool = False,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """
        Parameters
        ----------
        x_list : list of (B, N_a) — one per modality
        return_consensus : bool — if True, return consensus p

        Returns
        -------
        x_out : list of (B, N_a) — residual-updated modality embeddings
        p     : (B, M) consensus distribution (if requested, else None)
        """
        if len(x_list) != self.L:
            raise ValueError(f"Expected {self.L} modality inputs, got {len(x_list)}.")

        # ── 1. Project to query space & normalize ──
        z_list = [
            norm(wq(x))
            for wq, norm, x in zip(self.Wq, self.norm_q, x_list)
        ]
        Z = torch.stack(z_list, dim=1)  # (B, L, d)

        # ── 2. Run CSA iterations ──
        #  The forward pass of mTAM_Core is fully differentiable:
        #  softmax(β̃ · S) is differentiable, and the linear readout K̃·p
        #  is differentiable. Gradients propagate through all T steps.
        Z_out, p, _ = self.core(
            Z, self.K_banks, T=self.T_steps, damping=1.0,
        )

        assert p is not None
        # ── 3. Value readout + output projection + residual ──
        X_read = mTAM_Core.value_readout(self.V_banks, p)  # (B, L, d)
        x_attn = [
            self.dropout(wo(X_read[:, a, :]))
            for a, wo in enumerate(self.Wo)
        ]
        x_list = [x + xa for x, xa in zip(x_list, x_attn)]  # residual

        # ── 4. Feed-forward sub-layer + residual ──
        x_list = [
            x + self.dropout(ffn(norm(x)))
            for x, ffn, norm in zip(x_list, self.ffn, self.norm_ffn)
        ]

        return x_list, (p if return_consensus else None)


# ═══════════════════════════════════════════════════
#  mTAM Encoder — Stacked mTAM Blocks
# ═══════════════════════════════════════════════════

class mTAMEncoder(nn.Module):
    """Stack of mTAMBlocks forming a multi-modal encoder.

    Analogous to ``nn.TransformerEncoder`` but for multi-modal inputs
    bound through CSA consensus.

    Parameters
    ----------
    num_layers : int
        Number of stacked mTAMBlocks.
    in_dims, d, num_patterns, edges, beta, T_steps, ffn_ratio, dropout :
        Forwarded to each mTAMBlock (same architecture for all layers).
    """

    def __init__(
        self,
        num_layers: int,
        in_dims: List[int],
        d: int,
        num_patterns: int,
        edges: List[Tuple[int, int]],
        beta: float = 1.0,
        T_steps: int = 1,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            mTAMBlock(
                in_dims=in_dims, d=d, num_patterns=num_patterns,
                edges=edges, beta=beta, T_steps=T_steps,
                ffn_ratio=ffn_ratio, dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x_list: List[Tensor],
        return_all_consensus: bool = False,
    ) -> Tuple[List[Tensor], Optional[List[Tensor]]]:
        """
        Returns
        -------
        x_list : list of (B, N_a) — final modality embeddings
        all_p  : list of (B, M) consensus from each layer (if requested)
        """
        all_p = [] if return_all_consensus else None
        for block in self.blocks:
            x_list, p = block(x_list, return_consensus=return_all_consensus)
            if all_p is not None and p is not None:
                all_p.append(p)
        return x_list, all_p


# ═══════════════════════════════════════════════════
#  Example: mTAM Classifier
# ═══════════════════════════════════════════════════

class mTAMClassifier(nn.Module):
    """Example end-to-end trainable multi-modal classifier using mTAM.

    Architecture:
        [per-modality input embeddings]
            → mTAMEncoder (stacked CSA blocks)
            → concatenate final modality representations
            → classification head

    This demonstrates the complete integration of CSA into a trainable
    pipeline, end-to-end differentiable via standard backpropagation.

    Parameters
    ----------
    in_dims : list of int
        Per-modality input dimensions.
    d : int
        CSA embedding dimension.
    num_patterns : int
        Number of prototypes in CSA memory banks.
    num_classes : int
        Number of output classes.
    num_layers : int
        Number of stacked mTAM blocks.
    edges : list of (a, b), optional
        Interaction graph. Defaults to fully connected (no self-loops).
    beta : float
        Inverse temperature for CSA.
    T_steps : int
        CCCP iterations per block.
    """

    def __init__(
        self,
        in_dims: List[int],
        d: int,
        num_patterns: int,
        num_classes: int,
        num_layers: int = 2,
        edges: Optional[List[Tuple[int, int]]] = None,
        beta: float = 1.0,
        T_steps: int = 1,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        L = len(in_dims)
        if edges is None:
            edges = complete_graph(L, self_loops=False)

        self.encoder = mTAMEncoder(
            num_layers=num_layers,
            in_dims=in_dims, d=d, num_patterns=num_patterns,
            edges=edges, beta=beta, T_steps=T_steps,
            ffn_ratio=ffn_ratio, dropout=dropout,
        )

        # Classification head: concatenate all modality outputs → logits
        total_dim = sum(in_dims)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(
        self,
        x_list: List[Tensor],
    ) -> Tensor:
        """
        Parameters
        ----------
        x_list : list of (B, N_a) per-modality inputs

        Returns
        -------
        logits : (B, num_classes)
        """
        x_list, _ = self.encoder(x_list)
        # Pool: concatenate modality representations
        x_cat = torch.cat(x_list, dim=-1)  # (B, Σ N_a)
        return self.classifier(x_cat)
