"""
Grid paradigm implementation for discrete space models.

This module provides complete implementation of the Grid paradigm for
discrete space agent-based models like Sugarscape, Schelling Segregation,
Conway's Game of Life, and Cellular Automata.

Based on Technical_Specification.md Section 2.1: Grid Paradigm Specification.

Key features:
- Discrete cell-based positioning
- Moore and Von Neumann neighborhoods  
- Configurable boundary conditions (WRAP, WALL, INFINITE)
- Resource management per cell
- O(1) agent updates, O(r²) neighbor queries
"""

from .grid_agent import GridAgent
from .grid_environment import GridEnvironment
from .grid_space import GridSpace, NeighborhoodType

__all__ = [
    "GridAgent",
    "GridEnvironment", 
    "GridSpace",
    "NeighborhoodType"
]
