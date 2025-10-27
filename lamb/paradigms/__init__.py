"""
Paradigm implementations for the LAMB framework.

This module contains complete implementations of all three paradigms
supported by the LAMB framework, each optimized for different types
of agent-based models.

Based on Technical_Specification.md paradigm-specific sections.

Available paradigms:
- Grid: Discrete space models (Sugarscape, Schelling, Conway's Game of Life)
- Physics: Continuous space models (Boids, Social Force, particle systems)  
- Network: Graph-based models (SIR epidemics, opinion dynamics, social networks)

Each paradigm provides:
- Specialized agent implementations
- Optimized environment management
- Paradigm-specific utilities and algorithms
- Performance-tuned spatial indexing
"""

from .grid import GridAgent, GridEnvironment, GridSpace, NeighborhoodType
from .physics import PhysicsAgent, PhysicsEnvironment, ForceCalculator
from .network import NetworkAgent, NetworkEnvironment, NetworkTopology

__all__ = [
    # Grid paradigm
    "GridAgent", "GridEnvironment", "GridSpace", "NeighborhoodType",
    
    # Physics paradigm
    "PhysicsAgent", "PhysicsEnvironment", "ForceCalculator",
    
    # Network paradigm
    "NetworkAgent", "NetworkEnvironment", "NetworkTopology"
]
