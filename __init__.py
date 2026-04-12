"""
mTAM Layers - Core Package
===========================
Multi-modal Transformer Associative Memory
"""

from .mtam import mTAM_Core, mTAM_Layer, build_adjacency, complete_graph
from .decoupled_engine import DecoupledEngine
from .entropy_gated_consensus import (
    EGConsensusDiagnostics,
    EGConsensusStep,
    EntropyGatedConsensus,
)
from .training import mTAMBlock, mTAMClassifier, mTAMEncoder

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "mTAM_Core",
    "mTAM_Layer",
    "build_adjacency",
    "complete_graph",
    "DecoupledEngine",
    "EntropyGatedConsensus",
    "EGConsensusDiagnostics",
    "EGConsensusStep",
    "mTAMBlock",
    "mTAMEncoder",
    "mTAMClassifier",
]
