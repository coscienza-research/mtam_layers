"""
mTAM Layers - Core Package
===========================
Multi-modal Transformer Associative Memory
"""

from .mtam import mTAM_Core, mTAM_Layer, build_adjacency, complete_graph
from .decoupled_engine import DecoupledEngine

__all__ = [
    "mTAM_Core",
    "mTAM_Layer",
    "build_adjacency",
    "complete_graph",
    "DecoupledEngine"
]
