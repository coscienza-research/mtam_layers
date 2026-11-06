# mTAM: Multi-modal Transformer Associative Memory

[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-NFAM%20Workshop-darkblue.svg)](https://openreview.net/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](Thermodynamic_Binding_ICLR_NFAM_2026.pdf)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](test_mtam.py)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen.svg)](test_quality_suite.py)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!IMPORTANT]
> ### Official Reference Implementation
> This repository provides the official implementation and differentiable PyTorch library for the research paper:
>
> **Thermodynamic Binding: Freezing Chimeric States in Multi-Modal Associative Memories**  
> Elena Agliari, Adriano Barra, Andrea Ladiana, Andrea Lepre  
> *ICLR 2026 Workshop on New Frontiers in Associative Memories (NFAM)*.  
> **Proceedings:** [OpenReview](https://openreview.net/) | **Article PDF:** [`Thermodynamic_Binding_ICLR_NFAM_2026.pdf`](Thermodynamic_Binding_ICLR_NFAM_2026.pdf)

---

## Overview

Multi-modal inference requires heterogeneous perceptual streams (such as text, vision, and audio) to converge to a single, internally consistent interpretation. 

Standard cross-attention does not enforce this consistency: each modality computes its own softmax distribution $\mathbf{p}^{(a)}$ over a shared bank of $K$ memory prototypes. With $L$ modalities, this admits $K^L$ cross-modal assignments, of which only $K$ are internally coherent; the remaining configurations are **chimeric states**—stable configurations where different modalities lock onto incompatible prototypes from the shared memory bank. This represents a fundamental structural pathology rather than a mere training deficit.

The **Multi-modal Transformer Associative Memory (mTAM)** solves the binding problem *by construction* within an energy-based model (EBM). Its core mechanism, **Consensus Split-Bank Attention (CSA)**, aggregates query–key evidence across modalities into a unified global score, synthesizes a single shared distribution over memory, and broadcasts it synchronously to every modality.

---

## Scientific Scope & Mathematical Foundation

### 1. Consensus Split-Bank Attention (CSA)

Rather than maintaining $L$ decoupled softmax posteriors, CSA pools inter-modal evidence into a single global score $S_\mu = \sum_{a=1}^L \beta_a \langle \mathbf{q}^{(a)}, \mathbf{k}_\mu^{(a)} \rangle$. The network computes a joint consensus posterior:

$$p_\mu = \frac{\exp(S_\mu)}{\sum_{\nu=1}^K \exp(S_\nu)}$$

and uses $p_\mu$ to synchronously update all modality states via modern associative retrieval.

### 2. Difference-of-Convex Energy & CCCP Convergence

The network dynamics correspond to the **Concave–Convex Procedure (CCCP)** applied to a global Difference-of-Convex (DC) free-energy functional:

$$\mathcal{E}(\mathbf{X}) = \frac{1}{2} \sum_{a=1}^L \|\mathbf{x}^{(a)}\|^2 - \frac{1}{\beta} \log \sum_{\mu=1}^K \exp\left( \sum_{a=1}^L \beta \langle \mathbf{x}^{(a)}, \mathbf{w}_\mu^{(a)} \rangle \right)$$

This mathematically guarantees **monotonic energy descent** ($\mathcal{E}(\mathbf{X}_{t+1}) \le \mathcal{E}(\mathbf{X}_t)$) and strict convergence of every trajectory to a coherent stationary point.

### 3. Graph-Lifting & Storage Capacity

Through a graph-lifting construction, the model maps directly to a **Modern Hopfield Network (MHN)** on arbitrary interaction topologies. An extreme-value capacity analysis in the spirit of the Random Energy Model (REM) yields a rigorous, topology-dependent critical storage load that freezes out chimeric states.

---

## Core Components

- **`mtam.py`**:
  - `mTAM_Core`: Vectorized, batch-parallel CSA engine with temperature scaling and custom interaction topologies.
  - `mTAM_Layer`: Differentiable PyTorch layer with learnable query, key, and value parameter banks.
  - `build_adjacency`, `complete_graph`: Graph builders for arbitrary inter-modality coupling topologies.
- **`entropy_gated_consensus.py`**:
  - `EntropyGatedConsensus`: Entropy-Gated CSA (EG-CSA) with dissent-aware interpolation between global consensus and per-modality evidence.
  - `EGConsensusDiagnostics`: Real-time telemetry tracking consensus entropy, dissent entropy, gating factors, and free-energy synergy terms.
- **`training/mtam_block.py`**:
  - `mTAMBlock`, `mTAMEncoder`, `mTAMClassifier`: High-level modular building blocks ready for integration into modern deep learning architectures.
- **`decoupled_engine.py`**:
  - `DecoupledEngine`: Fully differentiable decoupled cross-attention baseline with identical parameter counts.
- **`metrics.py`**:
  - Energy diagnostics, entropy metrics, margin verification, and monotonic energy descent tracking.

---

## Installation

Install in editable development mode:

```bash
pip install -e .
```

To install with development, linting, and quality testing tools:

```bash
pip install -e .[dev]
```

### Core Requirements

- Python $\ge 3.10$
- `torch >= 2.0`
- `numpy >= 1.24`

---

## Quick Start (PyTorch)

### 1. Vectorized Consensus Split-Bank Attention (CSA)

```python
import torch
from mtam_layers import mTAM_Core
from mtam_layers.training import mTAMBlock

# Initialize a 2-modality CSA core engine
engine = mTAM_Core(
    edges=[(0, 1), (1, 0)],
    L=2,
    d=64,
    beta=2.0,
)

# Instantiate a high-level multimodal block with 64 shared prototypes
block = mTAMBlock(
    in_dims=[768, 512],
    d=128,
    num_patterns=64,
    edges=[(0, 1), (1, 0)],
    beta=4.0,
    T_steps=2,
)

# Forward pass on multimodal inputs: [B, D1] and [B, D2]
x1 = torch.randn(8, 768)
x2 = torch.randn(8, 512)
out1, out2 = block([x1, x2])
print("Retrieved representation shapes:", out1.shape, out2.shape)
```

### 2. Entropy-Gated Consensus (EG-CSA) with Dissent Telemetry

```python
import torch
from mtam_layers import EntropyGatedConsensus, mTAM_Core

engine = mTAM_Core(edges=[(0, 1), (1, 0)], L=2, d=64, beta=2.0)

# Configure entropy-gated consensus with threshold tau and steepness kappa
eg = EntropyGatedConsensus(core=engine, tau=0.15, kappa=12.0)

# Simulate batched state inputs and memory banks
query_states = torch.randn(4, 2, 64)
memory_banks = torch.randn(2, 64, 128)

step = eg.step(query_states, memory_banks)

# Inspect dissent-aware gating factor in [0, 1]
print("Consensus gate per batch element:", step.diagnostics.gate)
print("Consensus entropy:", step.diagnostics.entropy_consensus)
print("Dissent entropy across modalities:", step.diagnostics.entropy_dissent)
```

---

## Engineering Guarantees

- **Strict Tensor & Graph Contracts**: Rigorous runtime validation for tensor shapes, dimension alignment, damping ranges, and iteration counts.
- **Differentiability**: Fully differentiable computational graph enabling gradient backpropagation through unrolled CCCP iterations.
- **LogSumExp Numerical Stability**: All energy functions and denominator normalizations leverage numerically stable logsumexp kernels.
- **Backward Compatibility**: Stable public API with comprehensive interface coverage.

---

## Reproducibility & Provenance

Deterministic execution guidelines are provided in `REPRODUCIBILITY.md`.

Set deterministic seeds in Python:

```python
import torch

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
```

In PowerShell:

```powershell
$env:PYTHONHASHSEED = "0"
```

---

## Project Structure

```text
mtam_layers/
├── mtam.py                                  # Core CSA engine & trainable mTAM layers
├── entropy_gated_consensus.py               # Entropy-Gated CSA (EG-CSA) & dissent metrics
├── decoupled_engine.py                      # Decoupled cross-attention baseline
├── metrics.py                               # Free energy, consensus entropy & margins
├── version.py                               # Package semantic versioning
├── training/                                # Modular neural network layers
│   ├── __init__.py                          # Exposes mTAMBlock, mTAMEncoder, mTAMClassifier
│   └── mtam_block.py                        # Trainable multi-modal PyTorch modules
├── test_mtam.py                             # Verification suite for algebraic & dynamical properties
├── test_quality_suite.py                    # Quality gates & regression tests
├── Thermodynamic_Binding_ICLR_NFAM_2026.pdf # Full scientific workshop paper
├── CITATION.cff                             # Citation metadata (enables GitHub Cite button)
├── LICENSE                                  # Open-source MIT License
└── pyproject.toml                           # Tooling, linting, and coverage configuration
```

---

## Verification & Quality Gates

Run the verification test suite:

```bash
pytest test_mtam.py -v
```

Enforce coverage threshold ($\ge 95\%$):

```bash
pytest --cov=. --cov-fail-under=95
```

Run code formatting and lint checks:

```bash
ruff check .
ruff format --check .
```

Run static type checking:

```bash
mypy
```

---

## Citation

If you use this software, the mTAM architecture, or the theoretical derivations in your research, please cite our workshop paper:

```bibtex
@inproceedings{agliari2026thermodynamic,
  title={Thermodynamic Binding: Freezing Chimeric States in Multi-Modal Associative Memories},
  author={Agliari, Elena and Barra, Adriano and Ladiana, Andrea and Lepre, Andrea},
  booktitle={New Frontiers in Associative Memories - Workshop at ICLR 2026},
  year={2026},
  url={https://openreview.net/}
}
```

You can also use GitHub's **"Cite this repository"** button in the sidebar (powered by [`CITATION.cff`](CITATION.cff)).

---

## License

MIT License — Copyright (c) 2026 Andrea Ladiana, Scuderia "Cò'Scienza" (Sapienza University of Rome). See [`LICENSE`](LICENSE) for details.
