"""
Physics paradigm implementation for continuous space models.

This module provides complete implementation of the Physics paradigm for
continuous space agent-based models like Boids flocking, Social Force Model,
particle systems, and crowd dynamics.

Based on Technical_Specification.md Section 2.2: Physics Paradigm Specification.

Key features:
- Continuous position and velocity tracking
- KD-tree spatial indexing for O(log n + k) neighbor queries
- Force-based interactions and collision detection
- Configurable world bounds and boundary conditions
- Physics simulation with configurable time step
"""

from .physics_agent import PhysicsAgent
from .physics_environment import PhysicsEnvironment
from .forces import ForceCalculator

__all__ = [
    "PhysicsAgent",
    "PhysicsEnvironment",
    "ForceCalculator"
]
