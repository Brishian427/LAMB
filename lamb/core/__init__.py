"""
Core abstractions for the LAMB framework.

This module contains the fundamental interfaces and types that all
paradigms and engines must implement.

Based on Technical_Specification.md Section 1: Core Architecture Specification.
"""

from .types import (
    # Core types
    AgentID, Position, GridCoord, PhysicsCoord, NodeID, Vector2D,
    EngineType, BoundaryCondition,
    
    # Data structures
    CircularBuffer, ObservationActionPair,
    Observation, Action, ActionResult, AgentState,
    PerformanceMetrics,
    
    # Exceptions
    LAMBError, AgentNotFoundError, InvalidEnvironmentError,
    InvalidObservationError, EngineTimeoutError, IncompatibleEngineError,
    InvalidActionError, ConflictError, EnvironmentConstraintError,
    StateConsistencyError
)

from .base_agent import BaseAgent
from .base_environment import BaseEnvironment
from .base_engine import BaseEngine, MockEngine
from .simulation import Simulation, SimulationResults, ComponentValidator

__all__ = [
    # Types
    "AgentID", "Position", "GridCoord", "PhysicsCoord", "NodeID", "Vector2D",
    "EngineType", "BoundaryCondition",
    
    # Data structures
    "CircularBuffer", "ObservationActionPair",
    "Observation", "Action", "ActionResult", "AgentState",
    "PerformanceMetrics",
    
    # Exceptions
    "LAMBError", "AgentNotFoundError", "InvalidEnvironmentError",
    "InvalidObservationError", "EngineTimeoutError", "IncompatibleEngineError",
    "InvalidActionError", "ConflictError", "EnvironmentConstraintError",
    "StateConsistencyError",
    
    # Core classes
    "BaseAgent", "BaseEnvironment", "BaseEngine", "MockEngine",
    "Simulation", "SimulationResults", "ComponentValidator"
]
