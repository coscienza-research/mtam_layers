# mTAM Core Layers

This repository contains the simulative core of **mTAM** (Multi-modal Transformer Associative Memory). It provides a fully vectorized, autograd-compatible PyTorch implementation of the **Consensus Split-Bank Attention (CSA)** engine.

## Overview

mTAM extends Modern Hopfield Networks to heterogeneous, multi-modal settings by introducing a global consensus mechanism over a directed modality graph.

The repository is structured to be used directly as a Python module:
- `mtam.py`: The definitive implementation of the CSA engine.
- `decoupled_engine.py`: A decoupled variation of the engine provided for ablation studies and comparisons.
- `metrics.py`: Information-theoretic and optimization-based metrics for theoretical validation.
- `test_mtam.py`: A comprehensive, highly capillary test suite covering edge cases, math theorems, fixed point stability, gradients, precision boundaries, and the functional behavior of training components.
- `training/`: Contains end-to-end differentiable neural network modules (e.g., `mTAMBlock`, `mTAMEncoder`) ready for integration into deep learning pipelines.

## Requirements
- Python >= 3.8
- PyTorch >= 2.0.0

## Usage
Simply drop the `mtam_layers` directory into your project and import the core classes. 

```python
from mtam_layers.mtam import mTAM_Core
from mtam_layers.training import mTAMBlock

# Initialize your core engine
engine = mTAM_Core(
    edges=[(0, 1), (1, 0)], 
    L=2, 
    d=64, 
    beta=2.0
)

# ... or use fully trainable multi-modal transformer blocks
block = mTAMBlock(
    in_dims=[768, 512],
    d=128,
    num_patterns=64,
    edges=[(0, 1), (1, 0)],
    beta=4.0
)
```

## Running Tests
To verify the engine against algebraic, structural, and gradient consistency properties:
```bash
python test_mtam.py
# Or using pytest for verbose capillary outputs:
pytest test_mtam.py -v
```

## License
Provided under the Creative Commons Attribution 4.0 International (CC-BY 4.0) license.

**Author:** Andrea Ladiana
