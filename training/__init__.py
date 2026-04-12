"""
training — End-to-end trainable mTAM modules for integration into
           standard deep learning pipelines.

Provides:
  - mTAMBlock       : drop-in multi-modal Transformer block with CSA
  - mTAMEncoder     : stacked mTAMBlocks with residual connections
  - mTAMClassifier  : example classifier head for multi-modal tasks

These modules demonstrate that the mTAM/CSA mechanism is fully
differentiable and can be integrated into end-to-end trainable
architectures, following the same pattern as Modern Hopfield Layers
(Ramsauer et al., 2020) for the single-modality case.
"""

from .mtam_block import mTAMBlock, mTAMClassifier, mTAMEncoder

__all__ = ["mTAMBlock", "mTAMEncoder", "mTAMClassifier"]
