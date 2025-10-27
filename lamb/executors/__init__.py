"""
Paradigm-specific execution logic for LAMB framework.

This module contains executors that handle the specific execution patterns
for different simulation paradigms. Each executor implements the same
interface but uses paradigm-appropriate execution strategies.

Executors:
- GridExecutor: Discrete synchronous stepping for grid-based simulations
- PhysicsExecutor: Continuous dynamics with sub-stepping for physics simulations  
- NetworkExecutor: Event-driven propagation for network simulations
"""

from .base_executor import BaseExecutor
from .grid_executor import GridExecutor
from .physics_executor import PhysicsExecutor
from .network_executor import NetworkExecutor

__all__ = [
    "BaseExecutor",
    "GridExecutor", 
    "PhysicsExecutor",
    "NetworkExecutor"
]
