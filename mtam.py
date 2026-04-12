"""
mtam.py — Multi-modal Transformer Associative Memory (mTAM) Core Engine
=========================================================================

Definitive implementation of Consensus Split-Bank Attention (CSA) as
derived from the mTAM energy functional.  Every function is annotated
with the corresponding LaTeX equation label.


  Adjacency (row-stochastic):
      Ā_{ab} = A_{ab} / d^{in}_a           [eq:abar_notation]

  Effective inverse temperature:
      β̃ = β / √d                           [eq:tb_def]

  Global Score:
      S_μ = Σ_{a,b} Ā_{ab} ⟨z^(a), k_μ^(b)⟩  [eq:global_score]

  Consensus Probability:
      p = softmax(β̃ · S)                    [eq:consensus_p]

  CCCP Update (Jacobi):
      z_new^(a) = Σ_b Ā_{ab} K^(b) p       [eq:jacobi-update]

  Energy (Lyapunov / DC):
      E = Σ_a ½‖z^(a)‖² − β̃⁻¹ lse(β̃ S) + C_shift
                                             [eq:global-energy]
  Value Readout:
      x^(a) = V^(a) p                       [eq:value-readout]

Tensor layout (batch-first, fully vectorized — zero Python loops
over layers or edges):

  Z      : (B, L, d)     query states per layer
  K      : (L, d, M)     key banks  (M = number of stored patterns)
  V      : (L, d, M)     value banks (optional, for readout)
  A_bar  : (L, L)        row-stochastic adjacency matrix
  S      : (B, M)        global pattern scores
  p      : (B, M)        consensus distribution

All operations are Autograd-compatible.

Author : Andrea Ladiana
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ═══════════════════════════════════════════════════
#  Graph Utilities
# ═══════════════════════════════════════════════════

def build_adjacency(
    edges: List[Tuple[int, int]],
    L: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Build binary adjacency A and row-normalized Ā.

    Convention: edge (a, b) means a ← b (target a receives from source b).
    Row-normalization: Ā_{ab} = A_{ab} / d^{in}_a   [eq:abar_notation]

    Guarantees: each row of Ā sums to 1 (row-stochastic).

    Parameters
    ----------
    edges : list of (target, source) pairs
    L     : number of layers (nodes)

    Returns
    -------
    A     : (L, L) binary adjacency
    A_bar : (L, L) row-stochastic
    """
    if L <= 0:
        raise ValueError(f"L must be >= 1, got {L}.")
    if len(edges) == 0:
        raise ValueError("Edge list is empty.")
    A = torch.zeros(L, L, device=device, dtype=dtype)
    for a, b in edges:
        if not (0 <= a < L and 0 <= b < L):
            raise ValueError(f"Edge ({a}, {b}) out of range [0, {L}).")
        A[a, b] = 1.0
    d_in = A.sum(dim=1, keepdim=True)  # (L, 1) — in-degree per target
    zero_mask = d_in.squeeze(1) == 0
    if zero_mask.any():
        bad = zero_mask.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"Layers {bad} have d_in = 0. Every target must have ≥ 1 incoming edge."
        )
    A_bar = A / d_in  # row-stochastic  [eq:abar_notation]
    return A, A_bar


def complete_graph(L: int, self_loops: bool = False) -> List[Tuple[int, int]]:
    """All directed edges among L layers.

    Set ``self_loops=True`` for L = 1 (auto-associative / standard MHN).
    """
    if L <= 0:
        raise ValueError(f"L must be >= 1, got {L}.")
    if L == 1 and not self_loops:
        raise ValueError("L=1 requires self_loops=True to avoid an empty graph.")
    if self_loops:
        return [(a, b) for a in range(L) for b in range(L)]
    return [(a, b) for a in range(L) for b in range(L) if a != b]


# ═══════════════════════════════════════════════════
#  mTAM Core Engine
# ═══════════════════════════════════════════════════

class mTAM_Core(nn.Module):
    r"""Consensus Split-Bank Attention (CSA) engine for mTAM.

    Encapsulates the graph topology (Ā) and the inverse temperature β,
    and provides **fully-vectorized, Autograd-compatible** methods for
    every stage of the CCCP iteration.

    When L = 1 with a self-loop, the dynamics reduce exactly to the
    standard Modern Hopfield Network (softmax attention).

    Parameters
    ----------
    edges : list of (a, b)
        Directed arcs a ← b.  Layer indices in ``[0, L)``.
    L : int
        Number of layers (modalities).
    d : int
        Shared embedding dimension.
    beta : float
        Raw inverse temperature β > 0.  Effective: β̃ = β / √d.
    use_shift : bool
        Add C_shift = β̃⁻¹ ln M to the energy so that E(0) = 0.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        L: int,
        d: int,
        beta: float = 1.0,
        use_shift: bool = True,
    ) -> None:
        super().__init__()
        if L <= 0:
            raise ValueError(f"L must be >= 1, got {L}.")
        if d <= 0:
            raise ValueError(f"d must be >= 1, got {d}.")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be > 0.")
        self.L = int(L)
        self.d = int(d)
        self.beta = float(beta)
        self.use_shift = bool(use_shift)
        self.edges = list(edges)

        A, A_bar = build_adjacency(edges, L)
        self.register_buffer("A", A)        # (L, L)  binary
        self.register_buffer("A_bar", A_bar)  # (L, L)  row-stochastic

    def _validate_Z(self, Z: Tensor) -> None:
        if Z.dim() != 3:
            raise ValueError(f"Z must have shape (B, L, d), got {tuple(Z.shape)}.")
        if Z.size(1) != self.L or Z.size(2) != self.d:
            raise ValueError(
                f"Z must have shape (B, {self.L}, {self.d}), got {tuple(Z.shape)}."
            )

    def _validate_K(self, K: Tensor) -> None:
        if K.dim() != 3:
            raise ValueError(f"K must have shape (L, d, M), got {tuple(K.shape)}.")
        if K.size(0) != self.L or K.size(1) != self.d:
            raise ValueError(
                f"K must have shape ({self.L}, {self.d}, M), got {tuple(K.shape)}."
            )
        if K.size(2) <= 0:
            raise ValueError("K must contain at least one pattern (M >= 1).")

    def _validate_K_eff(self, K_eff: Tensor, expected_M: int) -> None:
        if K_eff.dim() != 3:
            raise ValueError(
                f"K_eff must have shape (L, d, M), got {tuple(K_eff.shape)}."
            )
        if K_eff.size(0) != self.L or K_eff.size(1) != self.d:
            raise ValueError(
                f"K_eff must have shape ({self.L}, {self.d}, M), got {tuple(K_eff.shape)}."
            )
        if K_eff.size(2) != expected_M:
            raise ValueError(
                f"K_eff has M={K_eff.size(2)} but expected M={expected_M}."
            )

    @staticmethod
    def _validate_S(S: Tensor) -> None:
        if S.dim() != 2:
            raise ValueError(f"S must have shape (B, M), got {tuple(S.shape)}.")
        if S.size(1) <= 0:
            raise ValueError("S must contain at least one pattern score (M >= 1).")

    @staticmethod
    def _validate_p(p: Tensor, expected_M: Optional[int] = None) -> None:
        if p.dim() != 2:
            raise ValueError(f"p must have shape (B, M), got {tuple(p.shape)}.")
        if p.size(1) <= 0:
            raise ValueError("p must contain at least one pattern probability (M >= 1).")
        if expected_M is not None and p.size(1) != expected_M:
            raise ValueError(f"p has M={p.size(1)} but expected M={expected_M}.")

    # ── helpers ──────────────────────────────────────

    @property
    def beta_tilde(self) -> float:
        r"""Effective inverse temperature β̃ = β / √d  [eq:tb_def]."""
        return self.beta / math.sqrt(self.d)

    # ── pre-computation ─────────────────────────────

    def effective_keys(self, K: Tensor) -> Tensor:
        r"""Graph-mixed keys (base-space form of the lifted patterns).

        K̃[a, :, μ] = Σ_b Ā_{ab} K[b, :, μ]

        Cf. Remark after Lemma 5: the mTAM score equals
        S_μ = ⟨Z, K̃_μ⟩  with  K̃ = (Ā ⊗ I_d) K.

        Parameters
        ----------
        K : (L, d, M)

        Returns
        -------
        K_eff : (L, d, M)
        """
        self._validate_K(K)
        return torch.einsum("ab, bim -> aim", self.A_bar, K)

    # ── score ───────────────────────────────────────

    def compute_scores(self, Z: Tensor, K: Tensor,
                       K_eff: Optional[Tensor] = None) -> Tensor:
        r"""Global pattern scores  [eq:global_score].

        S_μ(Z) = Σ_{a,b} Ā_{ab} ⟨z^{(a)}, k_μ^{(b)}⟩
               = Σ_a ⟨z^{(a)}, K̃^{(a)}_μ⟩

        Parameters
        ----------
        Z     : (B, L, d)
        K     : (L, d, M)       — raw keys (used only if K_eff is None)
        K_eff : (L, d, M), opt. — pre-computed effective keys

        Returns
        -------
        S : (B, M)
        """
        self._validate_Z(Z)
        self._validate_K(K)
        if K_eff is None:
            K_eff = self.effective_keys(K)
        else:
            self._validate_K_eff(K_eff, expected_M=K.size(2))
        return torch.einsum("nai, aim -> nm", Z, K_eff)

    # ── consensus ───────────────────────────────────

    def compute_consensus(self, S: Tensor) -> Tensor:
        r"""Consensus distribution  [eq:consensus_p].

        p = softmax(β̃ · S)

        Parameters
        ----------
        S : (B, M)

        Returns
        -------
        p : (B, M)  (probability simplex)
        """
        self._validate_S(S)
        return torch.softmax(self.beta_tilde * S, dim=-1)

    # ── update ──────────────────────────────────────

    def compute_update(self, K: Tensor, p: Tensor,
                       K_eff: Optional[Tensor] = None) -> Tensor:
        r"""Synchronous CCCP update (Jacobi)  [eq:jacobi-update].

        z_new^{(a)} = Σ_b Ā_{ab} K^{(b)} p  =  K̃^{(a)} p

        Derived as ∇_{z^{(a)}} V(z_old)  — Appendix, Gradient Consistency
        proof.

        Parameters
        ----------
        K     : (L, d, M)
        p     : (B, M)
        K_eff : (L, d, M), opt.

        Returns
        -------
        Z_new : (B, L, d)
        """
        self._validate_K(K)
        self._validate_p(p, expected_M=K.size(2))
        if K_eff is None:
            K_eff = self.effective_keys(K)
        else:
            self._validate_K_eff(K_eff, expected_M=K.size(2))
        return torch.einsum("aim, nm -> nai", K_eff, p)

    # ── energy (Lyapunov) ───────────────────────────

    def compute_energy(
        self,
        Z: Tensor,
        S: Optional[Tensor] = None,
        K: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Lyapunov energy  [eq:global-energy].

        E(z) = Σ_a ½‖z^{(a)}‖²  −  β̃⁻¹ log Σ_μ exp(β̃ S_μ)  +  C_shift

        DC decomposition:  E = U(z) − V(z) + C_shift
            U(z) = Σ_a ½‖z‖²             (strictly convex)
            V(z) = lse_{β̃}(S(z))         (convex — lse ∘ linear)

        Supply either pre-computed S **or** K (from which S is derived).

        Parameters
        ----------
        Z : (B, L, d)
        S : (B, M), optional
        K : (L, d, M), optional

        Returns
        -------
        E : (B,)  per-batch energy  (differentiable)
        """
        self._validate_Z(Z)
        if S is None:
            if K is None:
                raise ValueError("Provide either S or K to compute energy.")
            self._validate_K(K)
            S = self.compute_scores(Z, K)
        else:
            self._validate_S(S)

        bt = self.beta_tilde
        M = S.shape[-1]

        # U(z): quadratic confinement — Σ_a ½‖z^{(a)}‖²
        quad = 0.5 * Z.pow(2).sum(dim=(-2, -1))  # (B,)

        # V(z): interaction  —  β̃⁻¹ lse(β̃ S)
        # PyTorch logsumexp is numerically stable (subtracts max internally)
        lse = torch.logsumexp(bt * S, dim=-1)  # (B,)
        interaction = (1.0 / bt) * lse

        E = quad - interaction

        # C_shift = β̃⁻¹ ln M  ⟹  E(0) = 0
        if self.use_shift:
            E = E + (1.0 / bt) * math.log(M)

        return E

    # ── value readout ───────────────────────────────

    @staticmethod
    def value_readout(V: Tensor, p: Tensor) -> Tensor:
        r"""Per-layer readout  [eq:value-readout].

        x^{(a)}_new = V^{(a)} p

        Parameters
        ----------
        V : (L, d, M)
        p : (B, M)

        Returns
        -------
        X : (B, L, d)
        """
        if V.dim() != 3:
            raise ValueError(f"V must have shape (L, d, M), got {tuple(V.shape)}.")
        m = V.size(2)
        mTAM_Core._validate_p(p, expected_M=m)
        return torch.einsum("aim, nm -> nai", V, p)

    # ── single CCCP step ────────────────────────────

    def step(
        self,
        Z: Tensor,
        K: Tensor,
        damping: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""One CCCP iteration (synchronous Jacobi update).

        1.  K̃ = Ā K                         (effective keys — once)
        2.  S_μ = ⟨Z, K̃_μ⟩                  (score)         [eq:global_score]
        3.  p  = softmax(β̃ S)               (consensus)     [eq:consensus_p]
        4.  Z_new = K̃ p                     (update)        [eq:jacobi-update]
        5.  Z_out = (1−γ) Z + γ Z_new       (damping, γ=1 ⟹ strict descent)

        Theorem 4 (Monotonic Descent): with γ = 1 the energy satisfies
        E(Z_{t+1}) ≤ E(Z_t) with equality iff Z_t is a fixed point.

        Parameters
        ----------
        Z       : (B, L, d)
        K       : (L, d, M)
        damping : float in (0, 1].  γ = 1 gives the exact CCCP step.

        Returns
        -------
        Z_new : (B, L, d)
        p     : (B, M)
        S     : (B, M)
        """
        if not (0.0 < damping <= 1.0):
            raise ValueError(f"damping must be in (0, 1], got {damping}.")
        self._validate_Z(Z)
        self._validate_K(K)
        K_eff = self.effective_keys(K)                # pre-compute once
        S = self.compute_scores(Z, K, K_eff=K_eff)   # [eq:global_score]
        p = self.compute_consensus(S)                 # [eq:consensus_p]
        Z_cccp = self.compute_update(K, p, K_eff=K_eff)  # [eq:jacobi-update]

        if damping < 1.0:
            Z_new = (1.0 - damping) * Z + damping * Z_cccp
        else:
            Z_new = Z_cccp

        return Z_new, p, S

    # ── multi-step forward ──────────────────────────

    def forward(
        self,
        Z: Tensor,
        K: Tensor,
        T: int = 1,
        damping: float = 1.0,
        return_trajectory: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, List[Tensor]]]]:
        """Run T CCCP iterations.

        Parameters
        ----------
        Z : (B, L, d) — initial states
        K : (L, d, M) — key banks
        T : int ≥ 1
        damping : float in (0, 1]
        return_trajectory : bool
            If True, collect energy and p at each step.

        Returns
        -------
        Z_out  : (B, L, d) — final states
        p_last : (B, M)    — final consensus
        traj   : dict ({'energy': [...], 'p': [...]}) or None
        """
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")
        if not (0.0 < damping <= 1.0):
            raise ValueError(f"damping must be in (0, 1], got {damping}.")
        self._validate_Z(Z)
        self._validate_K(K)

        traj: Optional[Dict[str, List[Tensor]]] = None
        if return_trajectory:
            traj = {"energy": [], "p": []}

        p_last: Optional[Tensor] = None
        for _ in range(T):
            if traj is not None:
                traj["energy"].append(
                    self.compute_energy(Z, K=K).detach()
                )
            Z, p_last, _ = self.step(Z, K, damping=damping)
            if traj is not None:
                assert p_last is not None
                traj["p"].append(p_last.detach())

        # Final energy after the last update
        if traj is not None:
            traj["energy"].append(
                self.compute_energy(Z, K=K).detach()
            )

        assert p_last is not None
        return Z, p_last, traj


# ═══════════════════════════════════════════════════
#  Learnable Layer (wraps mTAM_Core)
# ═══════════════════════════════════════════════════

class mTAM_Layer(nn.Module):
    r"""High-level mTAM layer with learnable projections and banks.

    Accepts raw inputs x^{(a)} ∈ ℝ^{N_a}, projects them to query space
    z^{(a)} = W_Q^{(a)} x^{(a)}, runs T CSA iterations, and returns
    per-layer readouts  x_out^{(a)} = V^{(a)} p.

    Parameters
    ----------
    in_dims : list of int
        Input dimensions N_a per layer (length L).
    d : int
        Shared embedding dimension.
    num_patterns : int
        Number of prototypes M (≡ K in the LaTeX).
    edges : list of (a, b)
        Directed arcs a ← b.
    beta : float
        Raw inverse temperature.
    use_shift : bool
        Include C_shift in the energy.
    """

    def __init__(
        self,
        in_dims: List[int],
        d: int,
        num_patterns: int,
        edges: List[Tuple[int, int]],
        beta: float = 1.0,
        use_shift: bool = True,
    ) -> None:
        super().__init__()
        self.L = len(in_dims)
        self.d = d
        self.M = num_patterns

        # Per-layer query projections  W_Q^{(a)} : ℝ^{N_a} → ℝ^d
        self.Wq = nn.ModuleList([
            nn.Identity() if N == d else nn.Linear(N, d, bias=False)
            for N in in_dims
        ])

        # Key banks K^{(b)}  and  value banks V^{(a)}
        self.K_banks = nn.Parameter(torch.empty(self.L, d, num_patterns))
        self.V_banks = nn.Parameter(torch.empty(self.L, d, num_patterns))
        nn.init.kaiming_uniform_(self.K_banks, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V_banks, a=math.sqrt(5))

        # Core engine
        self.core = mTAM_Core(
            edges=edges, L=self.L, d=d, beta=beta, use_shift=use_shift,
        )

    def forward(
        self,
        x_list: List[Tensor],
        T: int = 1,
        damping: float = 1.0,
        return_trajectory: bool = False,
    ) -> Tuple[List[Tensor], Tensor, Optional[Dict[str, List[Tensor]]]]:
        """
        Parameters
        ----------
        x_list : list of (B, N_a) — one per layer
        T      : number of CCCP iterations (≥ 1)
        damping : damping factor γ ∈ (0, 1]

        Returns
        -------
        x_out : list of (B, d) — per-layer readouts  V^{(a)} p
        p     : (B, M)
        traj  : trajectory dict (optional)
        """
        if len(x_list) != self.L:
            raise ValueError(f"Expected {self.L} inputs, got {len(x_list)}.")

        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")
        if not (0.0 < damping <= 1.0):
            raise ValueError(f"damping must be in (0, 1], got {damping}.")

        batch_sizes = [x.size(0) for x in x_list]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"All modality inputs must share batch size, got {batch_sizes}.")

        # Project to query space:  z^{(a)} = W_Q^{(a)} x^{(a)}
        Z = torch.stack(
            [wq(x) for wq, x in zip(self.Wq, x_list)], dim=1
        )  # (B, L, d)

        # Run T CCCP iterations
        Z_out, p, traj = self.core(
            Z, self.K_banks, T=T, damping=damping,
            return_trajectory=return_trajectory,
        )

        # Value readout:  x^{(a)} = V^{(a)} p   [eq:value-readout]
        X_out = mTAM_Core.value_readout(self.V_banks, p)  # (B, L, d)
        x_out = [X_out[:, a, :] for a in range(self.L)]

        return x_out, p, traj
