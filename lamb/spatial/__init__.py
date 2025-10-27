"""
Spatial indexing structures for the LAMB framework.

This module provides efficient spatial data structures for different paradigms:
- GridIndex: Discrete space with O(r²) neighbor queries
- KDTreeIndex: Continuous space with O(log n + k) range queries  
- GraphIndex: Network space with O(1) direct neighbor lookup

Based on Technical_Specification.md paradigm-specific sections.
Performance characteristics validated from reconnaissance data.
"""

from .base_spatial import SpatialIndex, select_spatial_index
from .grid_index import GridIndex
from .kdtree_index import KDTreeIndex
from .graph_index import GraphIndex

__all__ = [
    "SpatialIndex",
    "select_spatial_index", 
    "GridIndex",
    "KDTreeIndex", 
    "GraphIndex"
]
