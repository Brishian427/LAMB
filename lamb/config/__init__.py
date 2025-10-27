"""
Configuration system for the LAMB framework.

This module provides comprehensive configuration management using Pydantic
for validation, type checking, and documentation. All configuration classes
are based on Technical_Specification.md requirements.

Key features:
- Type-safe configuration with validation
- Hierarchical configuration structure
- Paradigm and engine-specific settings
- Performance monitoring configuration
- JSON serialization support
"""

from .simulation_config import (
    SimulationConfig,
    ParadigmType,
    EngineType,
    BoundaryCondition,
    GridConfig,
    PhysicsConfig,
    NetworkConfig,
    LLMConfig,
    RuleConfig,
    HybridConfig,
    PerformanceConfig,
    SpatialConfig
)

__all__ = [
    "SimulationConfig",
    "ParadigmType",
    "EngineType", 
    "BoundaryCondition",
    "GridConfig",
    "PhysicsConfig",
    "NetworkConfig",
    "LLMConfig",
    "RuleConfig",
    "HybridConfig",
    "PerformanceConfig",
    "SpatialConfig"
]
