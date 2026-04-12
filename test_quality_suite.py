from __future__ import annotations

import math

import pytest
import torch

from mtam_layers.decoupled_engine import DecoupledEngine
from mtam_layers.entropy_gated_consensus import EntropyGatedConsensus
from mtam_layers.metrics import (
    TrialLogs,
    accuracy_top1,
    check_energy_monotone,
    convergence_delta_z,
    cooperativity_index,
    entropy,
    margin_from_logits,
    mse_reconstruction,
    p_lower_bound,
    reconstruction_overlap,
    topk_mass,
)
from mtam_layers.mtam import build_adjacency, complete_graph, mTAM_Core, mTAM_Layer
from mtam_layers.training.mtam_block import mTAMBlock, mTAMClassifier, mTAMEncoder


def test_metrics_end_to_end() -> None:
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float32)
    S = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float32)

    h = entropy(p)
    assert h.shape == (2,)
    assert torch.all(h > 0)

    tk = topk_mass(p, k=2)
    assert torch.allclose(tk, torch.tensor([0.9, 0.9]))

    mu_star_vec = torch.tensor([0, 2], dtype=torch.long)
    mu_star_scalar = torch.tensor(0, dtype=torch.long)
    assert float(accuracy_top1(p, mu_star_vec).item()) == 1.0
    assert 0.0 <= float(accuracy_top1(p, mu_star_scalar).item()) <= 1.0

    margin = margin_from_logits(S, mu_star_vec)
    assert margin.shape == (2,)
    assert torch.all(margin > 0)
    margin_scalar = margin_from_logits(S, torch.tensor(0, dtype=torch.long))
    assert margin_scalar.shape == (2,)

    lb = p_lower_bound(beta=2.0, delta=margin, K=S.size(1))
    assert lb.shape == (2,)
    assert torch.all((lb > 0) & (lb <= 1))

    xa = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    Va = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mu_star_small = torch.tensor([0, 1], dtype=torch.long)
    overlap = reconstruction_overlap(xa, Va, mu_star_small)
    assert torch.allclose(overlap, torch.tensor([1.0, 1.0]), atol=1e-6)

    mse = mse_reconstruction(xa, Va, mu_star_small)
    assert torch.allclose(mse, torch.zeros_like(mse), atol=1e-6)

    overlap_scalar = reconstruction_overlap(xa, Va, torch.tensor(0, dtype=torch.long))
    mse_scalar = mse_reconstruction(xa, Va, torch.tensor(0, dtype=torch.long))
    assert overlap_scalar.shape == (2,)
    assert mse_scalar.shape == (2,)

    energies_ok = [
        torch.tensor([2.0, 1.5]),
        torch.tensor([1.8, 1.4]),
        torch.tensor([1.7, 1.2]),
    ]
    energies_bad = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.2, 1.1]),
    ]
    assert check_energy_monotone(energies_ok)
    assert check_energy_monotone(energies_ok, per_element=True)
    assert not check_energy_monotone(energies_bad)
    assert check_energy_monotone([torch.tensor([1.0])])

    z_hist = [
        [torch.zeros(2, 3), torch.ones(2, 3)],
        [torch.ones(2, 3), torch.ones(2, 3)],
        [torch.ones(2, 3) * 2, torch.ones(2, 3)],
    ]
    dz = convergence_delta_z(z_hist)
    assert dz.ndim == 0
    assert float(dz.item()) >= 0.0
    dz_short = convergence_delta_z([[torch.zeros(2, 3)]])
    assert float(dz_short.item()) == 0.0

    contribs = {
        (0, 1): torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        (1, 0): torch.tensor([[1.5, 0.5], [0.5, 1.5]]),
    }
    coop_default = cooperativity_index(contribs)
    coop_given = cooperativity_index(contribs, mu_hat=torch.tensor([0, 1]))
    assert 0.0 <= float(coop_default.item()) <= 1.0
    assert 0.0 <= float(coop_given.item()) <= 1.0

    logs = TrialLogs(
        energies=[torch.tensor([2.0]), torch.tensor([1.0])],
        ps=[p],
        z_hist=[[torch.zeros(2, 3)], [torch.ones(2, 3)]],
    )
    assert logs.is_energy_monotone()
    assert len(logs.mean_entropy()) == 1
    assert logs.last_p().shape == p.shape
    assert logs.last_energy().shape == torch.Size([1])
    assert not math.isnan(logs.convergence_score())

    logs_no_hist = TrialLogs(energies=[], ps=[], z_hist=None)
    assert logs_no_hist.last_p().numel() == 0
    assert logs_no_hist.last_energy().numel() == 0
    assert math.isnan(logs_no_hist.convergence_score())


def test_graph_and_core_validation_branches() -> None:
    with pytest.raises(ValueError):
        build_adjacency([(0, 0)], L=0)
    with pytest.raises(ValueError):
        build_adjacency([], L=2)
    with pytest.raises(ValueError):
        build_adjacency([(2, 0)], L=2)
    with pytest.raises(ValueError):
        build_adjacency([(0, 1)], L=2)

    with pytest.raises(ValueError):
        complete_graph(0)
    with pytest.raises(ValueError):
        complete_graph(1, self_loops=False)
    assert len(complete_graph(1, self_loops=True)) == 1

    with pytest.raises(ValueError):
        mTAM_Core(edges=[(0, 0)], L=0, d=4, beta=1.0)
    with pytest.raises(ValueError):
        mTAM_Core(edges=[(0, 0)], L=1, d=0, beta=1.0)
    with pytest.raises(ValueError):
        mTAM_Core(edges=[(0, 0)], L=1, d=4, beta=float("nan"))

    core = mTAM_Core(edges=[(0, 0), (1, 1)], L=2, d=4, beta=2.0)
    Z = torch.randn(3, 2, 4)
    K = torch.randn(2, 4, 5)

    with pytest.raises(ValueError):
        core.effective_keys(torch.randn(2, 4))

    with pytest.raises(ValueError):
        core.compute_scores(torch.randn(3, 2), K)
    with pytest.raises(ValueError):
        core.compute_scores(torch.randn(3, 1, 4), K)
    with pytest.raises(ValueError):
        core.compute_scores(Z, torch.randn(2, 4))
    with pytest.raises(ValueError):
        core.compute_scores(Z, torch.randn(1, 4, 5))
    with pytest.raises(ValueError):
        core.compute_scores(Z, torch.randn(2, 4, 0))
    with pytest.raises(ValueError):
        core.compute_scores(Z, K, K_eff=torch.randn(2, 4, 4))
    with pytest.raises(ValueError):
        core.compute_scores(Z, K, K_eff=torch.randn(2, 4))
    with pytest.raises(ValueError):
        core.compute_scores(Z, K, K_eff=torch.randn(1, 4, 5))

    with pytest.raises(ValueError):
        core.compute_consensus(torch.randn(3, 2, 5))
    with pytest.raises(ValueError):
        core.compute_consensus(torch.randn(3, 0))

    with pytest.raises(ValueError):
        core.compute_update(K, torch.randn(3, 5, 1))
    with pytest.raises(ValueError):
        core.compute_update(K, torch.randn(3, 0))
    with pytest.raises(ValueError):
        core.compute_update(K, torch.randn(3, 4))
    with pytest.raises(ValueError):
        core.compute_update(K, torch.randn(3, 5), K_eff=torch.randn(2, 4, 4))

    p_valid = torch.softmax(torch.randn(3, 5), dim=-1)
    z_from_update = core.compute_update(K, p_valid)
    assert z_from_update.shape == (3, 2, 4)

    with pytest.raises(ValueError):
        core.compute_energy(Z)
    with pytest.raises(ValueError):
        core.compute_energy(torch.randn(3, 2), K=K)
    with pytest.raises(ValueError):
        core.compute_energy(Z, S=torch.randn(3, 2, 5))

    S = core.compute_scores(Z, K)
    E = core.compute_energy(Z, S=S)
    assert E.shape == (3,)

    core_noshift = mTAM_Core(edges=[(0, 0), (1, 1)], L=2, d=4, beta=2.0, use_shift=False)
    E_noshift = core_noshift.compute_energy(Z, K=K)
    assert E_noshift.shape == (3,)

    with pytest.raises(ValueError):
        mTAM_Core.value_readout(torch.randn(2, 4), torch.randn(3, 5))
    with pytest.raises(ValueError):
        mTAM_Core.value_readout(torch.randn(2, 4, 5), torch.randn(3, 4))

    with pytest.raises(ValueError):
        core.step(Z, K, damping=0.0)
    with pytest.raises(ValueError):
        core.forward(Z, K, T=0)
    with pytest.raises(ValueError):
        core.forward(Z, K, T=2, damping=1.5)

    z_damped, _, _ = core.step(Z, K, damping=0.5)
    assert z_damped.shape == Z.shape


def test_mtam_layer_validation_branches() -> None:
    layer = mTAM_Layer(
        in_dims=[6, 4],
        d=4,
        num_patterns=5,
        edges=[(0, 0), (1, 1)],
        beta=2.0,
    )

    x0 = torch.randn(3, 6)
    x1 = torch.randn(3, 4)

    with pytest.raises(ValueError):
        layer.forward([x0])
    with pytest.raises(ValueError):
        layer.forward([x0, x1], T=0)
    with pytest.raises(ValueError):
        layer.forward([x0, x1], damping=0.0)
    with pytest.raises(ValueError):
        layer.forward([torch.randn(2, 6), torch.randn(3, 4)])

    x_out, p, traj = layer.forward([x0, x1], T=2, damping=1.0, return_trajectory=True)
    assert len(x_out) == 2
    assert p.shape == (3, 5)
    assert traj is not None


def test_entropy_gated_validation_and_trajectory() -> None:
    core = mTAM_Core(edges=[(0, 0), (1, 1)], L=2, d=4, beta=2.0)

    with pytest.raises(ValueError):
        EntropyGatedConsensus(core=core, tau=0.0)
    with pytest.raises(ValueError):
        EntropyGatedConsensus(core=core, tau=0.1, kappa=0.0)
    with pytest.raises(ValueError):
        EntropyGatedConsensus(core=core, tau=0.1, eps=0.0)

    eg = EntropyGatedConsensus(core=core, tau=0.1, kappa=8.0)
    Z = torch.randn(3, 2, 4)
    K = torch.randn(2, 4, 6)

    with pytest.raises(ValueError):
        eg.step(Z, K, damping=0.0)
    with pytest.raises(ValueError):
        eg.step(torch.randn(3, 2), K)
    with pytest.raises(ValueError):
        eg.step(torch.randn(3, 1, 4), K)
    with pytest.raises(ValueError):
        eg.step(Z, torch.randn(2, 4))
    with pytest.raises(ValueError):
        eg.step(Z, torch.randn(1, 4, 6))
    with pytest.raises(ValueError):
        eg.step(Z, torch.randn(2, 4, 0))

    local_no_keff = eg.per_modality_logits(Z, K)
    assert local_no_keff.shape == (3, 2, 6)

    with pytest.raises(ValueError):
        eg.local_posteriors(torch.randn(3, 2))

    with pytest.raises(ValueError):
        eg.dissent_entropy(torch.randn(3, 2), torch.randn(3, 6))
    with pytest.raises(ValueError):
        eg.dissent_entropy(torch.randn(3, 2, 6), torch.randn(3, 2, 6))

    step = eg.step(Z, K, damping=1.0)
    assert step.Z_new.shape == Z.shape
    assert step.p_gated.shape == (3, 2, 6)

    with pytest.raises(ValueError):
        eg.forward(Z, K, T=0)

    z_out, last, traj = eg.forward(Z, K, T=3, return_trajectory=True)
    assert z_out.shape == Z.shape
    assert last.Z_new.shape == Z.shape
    assert traj is not None
    assert len(traj["gate"]) == 3

    z_out_no_traj, last_no_traj, traj_none = eg.forward(Z, K, T=2, return_trajectory=False)
    assert z_out_no_traj.shape == Z.shape
    assert last_no_traj.Z_new.shape == Z.shape
    assert traj_none is None


def test_decoupled_engine_validation_and_modes() -> None:
    with pytest.raises(ValueError):
        DecoupledEngine(edges=[])
    with pytest.raises(ValueError):
        DecoupledEngine(edges=[(0, 0)], beta=float("nan"))
    with pytest.raises(ValueError):
        DecoupledEngine(edges=[(0, 0)], L=0)
    with pytest.raises(ValueError):
        DecoupledEngine(edges=[(1, 1)], L=1)
    with pytest.raises(ValueError):
        DecoupledEngine(edges=[(2, 0)], L=2)
    with pytest.raises(ValueError):
        DecoupledEngine(edges=[(-1, 0)], L=1)

    engine = DecoupledEngine(edges=[(0, 0), (1, 1)], beta=2.0, L=2)
    z_list = [torch.randn(3, 4), torch.randn(3, 4)]
    K_list = [torch.randn(4, 5), torch.randn(4, 5)]

    with pytest.raises(ValueError):
        engine.step(z_list, K_list, damping=0.0)
    with pytest.raises(ValueError):
        engine.step(z_list, K_list, mode="invalid")

    z_list_1d = [torch.randn(4), torch.randn(4)]
    z_next, _ = engine.step(z_list_1d, K_list)
    assert len(z_next) == 2
    assert z_next[0].shape == (1, 4)

    with pytest.raises(ValueError):
        engine.step([torch.randn(3, 4, 1), torch.randn(3, 4)], K_list)

    with pytest.raises(ValueError):
        engine.step([torch.randn(2, 4)], K_list)
    with pytest.raises(ValueError):
        engine.step(z_list, [torch.randn(4, 5)])
    with pytest.raises(ValueError):
        engine.step([torch.randn(2, 4), torch.randn(3, 4)], K_list)
    with pytest.raises(ValueError):
        engine.step(z_list, [torch.randn(4, 5, 1), torch.randn(4, 5)])
    with pytest.raises(ValueError):
        engine.step(z_list, [torch.randn(4, 5), torch.randn(4, 6)])
    with pytest.raises(ValueError):
        engine.step(z_list, [torch.randn(4, 0), torch.randn(4, 0)])
    with pytest.raises(ValueError):
        engine.step(z_list, [torch.randn(4, 5), torch.randn(3, 5)])

    bad_p_edge_missing = {(0, 0): torch.randn(3, 5)}
    with pytest.raises(ValueError):
        engine.update_z(z_list, K_list, bad_p_edge_missing)

    bad_p_edge_shape = {
        (0, 0): torch.randn(3, 5),
        (1, 1): torch.randn(3, 4),
    }
    with pytest.raises(ValueError):
        engine.update_z(z_list, K_list, bad_p_edge_shape)

    bad_p_edge_ndim = {
        (0, 0): torch.randn(3, 5, 1),
        (1, 1): torch.randn(3, 5),
    }
    with pytest.raises(ValueError):
        engine.update_z(z_list, K_list, bad_p_edge_ndim)

    with pytest.raises(ValueError):
        engine.forward(z_list, K_list, T=0)

    z_gs, p_gs, logs = engine.forward(z_list, K_list, T=2, mode="gauss-seidel", log=True)
    assert len(z_gs) == 2
    assert p_gs is not None
    assert logs is not None
    assert len(logs["E"]) == 2

    engine_no_shift = DecoupledEngine(edges=[(0, 0), (1, 1)], beta=2.0, use_shift=False, L=2)
    E = engine_no_shift.energy(z_list, K_list)
    assert E.shape == (3,)

    engine_sparse = DecoupledEngine(edges=[(0, 0)], beta=2.0, L=2)
    z_sparse = [torch.randn(2, 4), torch.randn(2, 4)]
    K_sparse = [torch.randn(4, 5), torch.randn(4, 5)]
    z_sparse_next, _ = engine_sparse.step(z_sparse, K_sparse)
    assert z_sparse_next[1].shape == (2, 4)


def test_training_encoder_consensus_paths() -> None:
    in_dims = [8, 6]
    x_list = [torch.randn(4, 8), torch.randn(4, 6)]

    block = mTAMBlock(
        in_dims=in_dims,
        d=6,
        num_patterns=7,
        edges=[(0, 0), (1, 1)],
        beta=2.0,
        T_steps=1,
        dropout=0.0,
    )

    with pytest.raises(ValueError):
        block([x_list[0]])

    out_no_p, p_none = block(x_list, return_consensus=False)
    assert len(out_no_p) == 2
    assert p_none is None

    encoder = mTAMEncoder(
        num_layers=2,
        in_dims=in_dims,
        d=6,
        num_patterns=7,
        edges=[(0, 0), (1, 1)],
        beta=2.0,
        T_steps=1,
        dropout=0.0,
    )

    out_embed, all_p = encoder(x_list, return_all_consensus=True)
    assert len(out_embed) == 2
    assert all_p is not None
    assert len(all_p) == 2

    _, all_p_none = encoder(x_list, return_all_consensus=False)
    assert all_p_none is None

    clf = mTAMClassifier(
        in_dims=in_dims,
        d=6,
        num_patterns=7,
        num_classes=3,
        edges=[(0, 0), (1, 1)],
        num_layers=1,
        dropout=0.0,
    )
    logits = clf(x_list)
    assert logits.shape == (4, 3)
