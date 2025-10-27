"""
LAMB: Large Language Model-enhanced Agent-Based Modeling Framework

A universal framework for agent-based modeling with first-class LLM integration.
Supports three paradigms: Grid (discrete), Physics (continuous), Network (graph-based).

Phase 1: Pure LLM architecture with OpenAI integration
Future phases: RULE and HYBRID modes
"""

__version__ = "0.1.0"
__author__ = "LAMB Development Team"

# Core imports for easy access
from .core.base_agent import BaseAgent
from .core.base_environment import BaseEnvironment
from .core.base_engine import BaseEngine
from .core.simulation import Simulation, SimulationResults

# API imports
from .api.research_api import ResearchAPI

# Factory imports
from .factories import SimulationFactory

__all__ = [
    "BaseAgent",
    "BaseEnvironment", 
    "BaseEngine",
    "Simulation",
    "SimulationResults",
    "ResearchAPI",
    "SimulationFactory",
]
