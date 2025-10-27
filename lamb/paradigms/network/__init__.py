"""
Network paradigm implementation for graph-based models.

This module provides complete implementation of the Network paradigm for
graph-based agent-based models like SIR epidemics, opinion dynamics,
social networks, and information diffusion.

Based on Technical_Specification.md Section 2.3: Network Paradigm Specification.

Key features:
- Graph topology management (directed/undirected, weighted/unweighted)
- Agent positioning on network nodes
- O(1) direct neighbor queries, O(d^h) multi-hop queries
- Message passing and information diffusion
- Dynamic topology changes (edge creation/removal)
- Network topology generation and analysis utilities
"""

from .network_agent import NetworkAgent
from .network_environment import NetworkEnvironment
from .topology import NetworkTopology

__all__ = [
    "NetworkAgent",
    "NetworkEnvironment",
    "NetworkTopology"
]
