"""
test_mtam.py — Comprehensive Verification Suite for mTAM / CSA core engine
===========================================================================

Run:  python test_mtam.py
  or: pytest test_mtam.py -v

This suite tests the mathematical properties, boundary conditions, edge cases,
and numerical stability of the mTAM engine, including the integration of 
end-to-end trainable modules.

Author: Andrea Ladiana
"""

from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mtam_layers.mtam import mTAM_Core, build_adjacency, complete_graph
from mtam_layers.decoupled_engine import DecoupledEngine
from mtam_layers.entropy_gated_consensus import EntropyGatedConsensus
from mtam_layers.training.mtam_block import mTAMBlock, mTAMClassifier

# ─── Helpers ────────────────────────────────────────

def _random_setup(
    L: int = 3,
    d: int = 32,
    M: int = 16,
    B: int = 8,
    beta: float = 4.0,
    seed: int = 42,
    self_loops: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tuple[mTAM_Core, Tensor, Tensor]:
    """Create a random mTAM instance and random (Z, K)."""
    torch.manual_seed(seed)
    edges = complete_graph(L, self_loops=self_loops)
    engine = mTAM_Core(edges=edges, L=L, d=d, beta=beta, use_shift=True).to(dtype=dtype)
    K = torch.randn(L, d, M, dtype=dtype)
    Z = torch.randn(B, L, d, dtype=dtype)
    return engine, Z, K


def _run_message(name: str, passed: bool) -> None:
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  [{status}]  {name}")


# ─── Test 1: Row-Stochasticity ─────────────────────

def test_row_stochastic() -> bool:
    """Ā rows must sum to 1."""
    for L in [2, 3, 5]:
        edges = complete_graph(L)
        _, A_bar = build_adjacency(edges, L)
        row_sums = A_bar.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones(L), atol=1e-7):
            return False
    # Sparse graph
    edges_sparse = [(0, 1), (1, 0), (1, 2), (2, 0)]
    _, A_bar_s = build_adjacency(edges_sparse, 3)
    row_sums_s = A_bar_s.sum(dim=1)
    if not torch.allclose(row_sums_s, torch.ones(3), atol=1e-7):
        return False
    return True


# ─── Test 2: Energy at Origin ──────────────────────

def test_energy_at_origin() -> bool:
    """E(0) = 0 when use_shift is True (C_shift = β̃⁻¹ ln M)."""
    for L, d, M in [(2, 16, 8), (3, 64, 32), (5, 32, 100)]:
        engine, _, K = _random_setup(L=L, d=d, M=M, B=1)
        Z_zero = torch.zeros(1, L, d)
        E = engine.compute_energy(Z_zero, K=K)
        if not torch.allclose(E, torch.zeros(1), atol=1e-5):
            return False
    return True


# ─── Test 3: Monotonic Descent ─────────────────────

def test_energy_monotone() -> bool:
    """E(t+1) ≤ E(t) for the full CCCP step (damping=1)."""
    atol = 1e-10  # exact in infinite precision; tight in float64
    for seed in range(5):
        for L in [2, 3]:
            engine, Z, K = _random_setup(L=L, d=32, M=16, B=4,
                                         beta=4.0, seed=seed, dtype=torch.float64)
            T = 30
            _, _, traj = engine.forward(
                Z, K, T=T, damping=1.0, return_trajectory=True,
            )
            assert traj is not None
            energies = traj["energy"]  # T+1 entries
            for t in range(len(energies) - 1):
                Et = energies[t]
                Et1 = energies[t + 1]
                if (Et1 - Et).max().item() > atol:
                    print(f"    Violation at seed={seed}, L={L}, t={t}: "
                          f"ΔE_max = {(Et1 - Et).max().item():.2e}")
                    return False
    return True


# ─── Test 4: Chimera Elimination ───────────────────

def test_chimera_elimination() -> bool:
    """All layers share the SAME consensus p (by construction)."""
    engine, Z, K = _random_setup(L=4, d=32, M=20, B=8, beta=6.0)
    Z_out, p, _ = engine.step(Z, K, damping=1.0)
    mu_star = p.argmax(dim=-1)  # (B,)
    K_eff = engine.effective_keys(K)
    for a in range(engine.L):
        z_a_manual = torch.einsum("im, nm -> ni", K_eff[a], p)
        if not torch.allclose(Z_out[:, a, :], z_a_manual, atol=1e-6):
            return False
    return True


# ─── Test 5: Gradient Consistency ──────────────────

def test_gradient_consistency() -> bool:
    """Verify z_new = ∇_z V(z_old)  where V = β̃⁻¹ lse(β̃ S)."""
    engine, Z0, K = _random_setup(L=3, d=16, M=12, B=2, beta=3.0)
    Z = Z0.clone().requires_grad_(True)

    bt = engine.beta_tilde
    S = engine.compute_scores(Z, K)
    V = (1.0 / bt) * torch.logsumexp(bt * S, dim=-1)  # (B,)
    V_sum = V.sum()  # scalar for autograd

    grad_V = torch.autograd.grad(V_sum, Z, create_graph=False)[0]  # (B, L, d)
    Z_new, _, _ = engine.step(Z0, K, damping=1.0)

    if not torch.allclose(grad_V, Z_new, atol=1e-5):
        return False
    return True


# ─── Test 6: L=1 Reduces to Standard MHN ──────────

def test_l1_reduction() -> bool:
    """With L=1 and a self-loop, mTAM = standard softmax attention."""
    d, M, B = 32, 16, 4
    beta = 5.0
    torch.manual_seed(99)
    K_single = torch.randn(d, M)
    z = torch.randn(B, d)

    edges = [(0, 0)]
    engine = mTAM_Core(edges=edges, L=1, d=d, beta=beta, use_shift=True)
    Z = z.unsqueeze(1)
    K_3d = K_single.unsqueeze(0)
    Z_new, p_mtam, _ = engine.step(Z, K_3d)
    z_new_mtam = Z_new[:, 0, :]

    bt = beta / math.sqrt(d)
    logits = z @ K_single
    p_mhn = torch.softmax(bt * logits, dim=-1)
    z_new_mhn = p_mhn @ K_single.t()

    if not torch.allclose(p_mtam, p_mhn, atol=1e-6):
        return False
    if not torch.allclose(z_new_mtam, z_new_mhn, atol=1e-5):
        return False
    return True


# ─── Test 7: Fixed Point Stability ─────────────────

def test_fixed_point() -> bool:
    """At convergence, z* = update(z*): the state does not change."""
    engine, Z, K = _random_setup(L=3, d=32, M=16, B=4, beta=6.0)
    Z_conv, _, _ = engine.forward(Z, K, T=100, damping=1.0)
    Z_next, _, _ = engine.step(Z_conv, K, damping=1.0)
    diff = (Z_next - Z_conv).abs().max().item()
    return diff < 1e-4

# ─── Test 8: Effective Keys Matches Direct ─────────

def test_effective_keys_matches_direct() -> bool:
    """Verify K_eff[a,:,μ] = Σ_b Ā_{ab} K[b,:,μ] using direct computation."""
    engine, _, K = _random_setup(L=3, d=16, M=8, B=1)
    K_eff = engine.effective_keys(K)
    A_bar = engine.A_bar
    L, d, M = K.shape
    K_direct = torch.zeros_like(K)
    for a in range(L):
        for b in range(L):
            K_direct[a] += A_bar[a, b] * K[b]
    return torch.allclose(K_eff, K_direct, atol=1e-6)

# ─── Test 9: Sparse Graph Tests ────────────────────

def test_sparse_graph() -> bool:
    """Test on a non-complete (chain) graph."""
    L, d, M, B = 3, 16, 8, 4
    edges = [(0, 1), (1, 2), (2, 0)]
    torch.manual_seed(77)
    engine = mTAM_Core(edges=edges, L=L, d=d, beta=3.0)
    K = torch.randn(L, d, M)
    Z = torch.randn(B, L, d)

    A_bar = engine.A_bar
    expected = torch.zeros(3, 3)
    expected[0, 1] = 1.0
    expected[1, 2] = 1.0
    expected[2, 0] = 1.0
    if not torch.allclose(A_bar, expected, atol=1e-7):
        return False

    _, _, traj = engine.forward(Z, K, T=20, damping=1.0, return_trajectory=True)
    assert traj is not None
    energies = traj["energy"]
    for t in range(len(energies) - 1):
        if (energies[t + 1] - energies[t]).max().item() > 1e-5:
            return False
    return True

# ─── Test 10: NaNs / Inference Robustness ──────────

def test_nans_robustness() -> bool:
    """Checks if the engine avoids propagating internal NaNs on extreme input or zero norm."""
    engine, _, K = _random_setup(L=3, d=16, M=8, B=2, beta=100.0)
    Z = torch.zeros(2, 3, 16) # Exactly zero query
    K[:, :, 0] = 0.0          # Pattern 0 is entirely zero
    try:
        Z_new, p, S = engine.step(Z, K)
        # Without NaNs in output
        if torch.isnan(Z_new).any() or torch.isnan(p).any():
            return False
    except Exception as e:
        return False
    return True

# ─── Test 11: Mixed Precisions & Device Support ─────

def test_dtype_stability() -> bool:
    """Run steps in fp16, bf16 and check that it doesn't crash."""
    if not torch.cuda.is_available(): # Just test shapes / cpu support for bf16 where supported
        types = [torch.float32, torch.float64, torch.bfloat16]
    else:
        types = [torch.float32, torch.float16, torch.bfloat16]
        
    for t in types:
        try:
            engine, Z, K = _random_setup(L=2, d=16, M=4, B=2, dtype=t)
            Z_new, p, _ = engine.step(Z, K)
            assert not torch.isnan(Z_new).any()
        except RuntimeError as e: # Catch unsupported CPU ops for Half
            if "not implemented for" in str(e):
                pass
            else:
                return False
        except Exception:
            return False
    return True
    
# ─── Test 12: Scalability and Batch Sizes ──────────

def test_batch_scaling() -> bool:
    """Verifies that varied B doesn't share state and broadcasts correctly."""
    engine, _, K = _random_setup(L=3, d=16, M=8, B=1, beta=1.0)
    Z_1 = torch.randn(1, 3, 16)
    Z_100 = Z_1.expand(100, 3, 16)
    
    Z_out_1, p_1, _ = engine.step(Z_1, K)
    Z_out_100, p_100, _ = engine.step(Z_100, K)
    
    # All 100 batch outputs should be identical to the single 1-batch output
    if not torch.allclose(Z_out_1, Z_out_100[0:1], atol=1e-4): return False
    if not torch.allclose(p_1, p_100[0:1], atol=1e-4): return False
    return True

# ─── Test 13: Decoupled Engine Forward Pass ────────

def test_decoupled_engine() -> bool:
    """Ensure decoupled engine works and computes distinct edge logic."""
    edges = [(0, 1), (1, 0), (2, 0)]
    engine = DecoupledEngine(edges, beta=2.0)
    L, d, Kdim, B = 3, 16, 8, 2
    
    z_list = [torch.randn(B, d) for _ in range(L)]
    K_list = [torch.randn(d, Kdim) for _ in range(L)]
    
    # step decoupled
    z_next, p_edge = engine.step(z_list, K_list)
    if len(p_edge) != len(edges): 
        return False
        
    E = engine.energy(z_next, K_list)
    if E.shape != (B,): return False
    return True


def test_decoupled_engine_gradient_flow() -> bool:
    """Decoupled engine should remain differentiable for trainable ablations."""
    torch.manual_seed(7)
    edges = [(0, 1), (1, 0)]
    engine = DecoupledEngine(edges, beta=1.5)
    B, d, Kdim = 3, 12, 7

    z_list = [
        torch.randn(B, d, requires_grad=True),
        torch.randn(B, d, requires_grad=True),
    ]
    K_list = [
        torch.randn(d, Kdim, requires_grad=True),
        torch.randn(d, Kdim, requires_grad=True),
    ]

    z_next, p_edge, _ = engine.forward(z_list, K_list, T=2, damping=1.0, mode="jacobi", log=False)
    loss = sum((z ** 2).mean() for z in z_next) + sum((p ** 2).mean() for p in p_edge.values())
    loss.backward()

    for z in z_list:
        if z.grad is None:
            return False
    for Kb in K_list:
        if Kb.grad is None:
            return False
    return True


def test_decoupled_engine_validation() -> bool:
    """Decoupled engine should reject invalid damping and inconsistent dimensions."""
    edges = [(0, 1), (1, 0)]
    engine = DecoupledEngine(edges, beta=2.0)

    z_list = [torch.randn(2, 8), torch.randn(2, 8)]
    K_valid = [torch.randn(8, 4), torch.randn(8, 4)]

    try:
        _ = engine.step(z_list, K_valid, damping=0.0)
        return False
    except ValueError:
        pass

    K_invalid = [torch.randn(8, 4), torch.randn(7, 4)]
    try:
        _ = engine.step(z_list, K_invalid, damping=1.0)
        return False
    except ValueError:
        pass

    return True


def test_entropy_gated_closed_reduces_to_csa() -> bool:
    """With gate closed (g≈0), EG-CSA should match standard CSA update."""
    engine, Z, K = _random_setup(L=3, d=16, M=8, B=4, beta=4.0, seed=11)
    eg = EntropyGatedConsensus(core=engine, tau=1e6, kappa=20.0)

    Z_csa, p_csa, _ = engine.step(Z, K, damping=1.0)
    step = eg.step(Z, K, damping=1.0)

    if not torch.allclose(step.Z_new, Z_csa, atol=1e-6):
        return False

    p_broadcast = p_csa.unsqueeze(1).expand_as(step.p_gated)
    if not torch.allclose(step.p_gated, p_broadcast, atol=1e-6):
        return False

    if step.diagnostics.gate.max().item() > 1e-3:
        return False

    return True


def test_entropy_gated_open_tracks_local_posteriors() -> bool:
    """In discordant conflict with tiny tau, gate should open and follow q^(a)."""
    d = 2
    M = 2
    B = 1
    edges = [(0, 0), (1, 1)]
    engine = mTAM_Core(edges=edges, L=2, d=d, beta=8.0)
    eg = EntropyGatedConsensus(core=engine, tau=1e-6, kappa=50.0)

    K = torch.zeros(2, d, M)
    K[0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    K[1] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    Z = torch.zeros(B, 2, d)
    Z[0, 0] = torch.tensor([10.0, 0.0])
    Z[0, 1] = torch.tensor([0.0, 10.0])

    step = eg.step(Z, K, damping=1.0)

    if step.diagnostics.gate.item() < 0.99:
        return False

    if not torch.allclose(step.p_gated, step.q_local, atol=1e-3):
        return False

    if step.diagnostics.dissent_entropy.item() <= 0.0:
        return False

    return True


def test_entropy_gated_diagnostics_consistency() -> bool:
    """Validate EG-CSA diagnostics domain constraints and algebraic consistency."""
    engine, Z, K = _random_setup(L=4, d=12, M=9, B=5, beta=3.0, seed=123)
    eg = EntropyGatedConsensus(core=engine, tau=0.2, kappa=10.0)
    step = eg.step(Z, K, damping=1.0)

    diag = step.diagnostics

    if diag.dissent_entropy.min().item() < -1e-8:
        return False
    if diag.consensus_entropy.min().item() < -1e-8:
        return False
    if diag.weighted_modal_variance.min().item() < -1e-8:
        return False
    if diag.gate.min().item() < -1e-8 or diag.gate.max().item() > 1.0 + 1e-8:
        return False

    expected = (engine.beta_tilde ** 2) * diag.weighted_modal_variance
    if not torch.allclose(diag.dissent_upper_bound, expected, atol=1e-7):
        return False

    p_sum = step.p_gated.sum(dim=-1)
    if not torch.allclose(p_sum, torch.ones_like(p_sum), atol=1e-6):
        return False

    if not torch.isfinite(diag.synergy).all():
        return False
    return True

# ─── Test 14: Training Module (mTAM Block) ─────────

def test_training_block() -> bool:
    """Ensure training wrappers initialize and run forward/backward properly."""
    in_dims = [64, 32, 16]
    d = 24
    M = 10
    B = 2
    L = 3
    edges = complete_graph(L)
    
    block = mTAMBlock(in_dims=in_dims, d=d, num_patterns=M, edges=edges, beta=2.0, T_steps=2)
    x_list = [torch.randn(B, n, requires_grad=True) for n in in_dims]
    
    x_out, p = block(x_list, return_consensus=True)
    if p is None or p.shape != (B, M): return False
    if len(x_out) != L: return False
    
    # Check backprop
    loss = sum(x.sum() for x in x_out)
    loss.backward()
    
    # Should have gradients on learnable memory banks
    if block.K_banks.grad is None or block.V_banks.grad is None:
        return False
        
    # Should have gradients on inputs
    for x in x_list:
        if x.grad is None: return False
        
    return True

# ─── Test 15: Classifier Forward ───────────────────

def test_classifier() -> bool:
    """Ensure full end-to-end classifier can map inputs to logits."""
    in_dims = [64, 32]
    d = 16
    M = 8
    num_classes = 5
    B = 3
    
    model = mTAMClassifier(in_dims=in_dims, d=d, num_patterns=M, num_classes=num_classes)
    x_list = [torch.randn(B, n) for n in in_dims]
    
    logits = model(x_list)
    if logits.shape != (B, num_classes):
        return False
    return True


# ─── Pytest Compatibility Layer ───────────────────

# The suite is intentionally written in bool-returning style to support
# script-level reporting (`python test_mtam.py`). Pytest expects test
# functions to assert instead of returning bool, so we mark legacy tests
# as non-collectable and expose assert-based wrappers.

_LEGACY_BOOL_TESTS = [
    test_row_stochastic,
    test_energy_at_origin,
    test_energy_monotone,
    test_chimera_elimination,
    test_gradient_consistency,
    test_l1_reduction,
    test_fixed_point,
    test_effective_keys_matches_direct,
    test_sparse_graph,
    test_nans_robustness,
    test_dtype_stability,
    test_batch_scaling,
    test_decoupled_engine,
    test_decoupled_engine_gradient_flow,
    test_decoupled_engine_validation,
    test_entropy_gated_closed_reduces_to_csa,
    test_entropy_gated_open_tracks_local_posteriors,
    test_entropy_gated_diagnostics_consistency,
    test_training_block,
    test_classifier,
]

for _legacy_fn in _LEGACY_BOOL_TESTS:
    _legacy_fn.__test__ = False


def _make_pytest_wrapper(fn):
    def _wrapper() -> None:
        assert fn(), f"{fn.__name__} returned False"

    _wrapper.__name__ = f"{fn.__name__}_pytest"
    return _wrapper


for _legacy_fn in _LEGACY_BOOL_TESTS:
    globals()[f"{_legacy_fn.__name__}_pytest"] = _make_pytest_wrapper(_legacy_fn)


# ─── Main ──────────────────────────────────────────

def run_all_tests() -> bool:
    tests = [
        ("Row-stochasticity of Ā", test_row_stochastic),
        ("E(0) = 0 with C_shift", test_energy_at_origin),
        ("Monotonic energy descent (Theorem 4)", test_energy_monotone),
        ("Chimera elimination (Proposition 1)", test_chimera_elimination),
        ("Gradient consistency ∇V = z_new (Appendix)", test_gradient_consistency),
        ("L=1 reduces to standard MHN", test_l1_reduction),
        ("Fixed point stability", test_fixed_point),
        ("Effective keys match direct computation", test_effective_keys_matches_direct),
        ("Sparse (chain) graph: Ā + descent", test_sparse_graph),
        ("NaNs / Inference Robustness", test_nans_robustness),
        ("Mixed Precisions & Device Support", test_dtype_stability),
        ("Scalability and Batch Sizes", test_batch_scaling),
        ("Decoupled Engine Check", test_decoupled_engine),
        ("Decoupled Engine Gradient Flow", test_decoupled_engine_gradient_flow),
        ("Decoupled Engine Validation", test_decoupled_engine_validation),
        ("Entropy-Gated: closed gate = CSA", test_entropy_gated_closed_reduces_to_csa),
        ("Entropy-Gated: open gate follows local q", test_entropy_gated_open_tracks_local_posteriors),
        ("Entropy-Gated: diagnostics consistency", test_entropy_gated_diagnostics_consistency),
        ("Training Block (Forward/Backward)", test_training_block),
        ("Classifier End-to-End", test_classifier),
    ]

    print("=" * 60)
    print("  mTAM / CSA Comprehensive Verification Suite")
    print("=" * 60)

    all_passed = True
    for name, fn in tests:
        try:
            passed = fn()
        except Exception as e:
            passed = False
            print(f"    Exception: {e}")
        _run_message(name, passed)
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED.")
    else:
        print("  SOME TESTS FAILED.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
