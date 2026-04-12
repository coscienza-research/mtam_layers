# mTAM Layers

Core, differentiable implementation of Multi-modal Transformer Associative Memory (mTAM), centered on the Consensus Split-Bank Attention (CSA) engine.

## Scope

This repository is the reusable library layer, not an experiments repository.
It focuses on:

- mathematically faithful implementation of CSA dynamics;
- strict runtime validation for graph and tensor contracts;
- differentiable training components for end-to-end PyTorch integration;
- reproducible verification tests for energy descent, fixed points, gradients, and numerical stability.

## Core Components

- `mtam.py`
    - `mTAM_Core`: vectorized CSA engine.
    - `mTAM_Layer`: trainable wrapper with learnable query/key/value banks.
    - `build_adjacency`, `complete_graph`: graph utilities.
- `decoupled_engine.py`
    - `DecoupledEngine`: per-edge decoupled attention baseline, now differentiable and input-validated.
- `training/mtam_block.py`
    - `mTAMBlock`, `mTAMEncoder`, `mTAMClassifier` for direct use in multimodal architectures.
- `metrics.py`
    - entropy, margins, overlap/reconstruction, convergence, and monotonic-energy checks.
- `test_mtam.py`
    - full verification suite for algebraic, dynamical, and gradient-level properties.

## Installation

From repository root:

```bash
pip install -e .
```

For development dependencies (pytest):

```bash
pip install -e .[dev]
```

## Minimal Usage

```python
from mtam_layers import mTAM_Core
from mtam_layers.training import mTAMBlock

engine = mTAM_Core(
        edges=[(0, 1), (1, 0)],
        L=2,
        d=64,
        beta=2.0,
)

block = mTAMBlock(
        in_dims=[768, 512],
        d=128,
        num_patterns=64,
        edges=[(0, 1), (1, 0)],
        beta=4.0,
        T_steps=2,
)
```

## Testing

Fast direct run:

```bash
python test_mtam.py
```

Pytest run:

```bash
pytest -v test_mtam.py
```

## Engineering Guarantees

- Explicit validation of graph dimensions, tensor shapes, damping range, and iteration counts.
- Differentiable computational graph preserved in both CSA and decoupled engines.
- Numerical stability through `logsumexp` formulation for energy terms.
- Backward compatibility for the public API used by current tests.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

Author: Andrea Ladiana
