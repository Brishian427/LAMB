"""
API interfaces for the LAMB framework.

This module provides different API levels for different user types:
- ResearchAPI: High-level, research-friendly interface (Phase 1 primary)
- BaseAPI: Interface for future beginner and developer APIs

The APIs handle complexity abstraction and provide easy access to
LAMB framework capabilities.

Phase 1 focus: ResearchAPI for immediate scientific use
Future phases: BeginnerAPI and DeveloperAPI with progressive complexity
"""

from .research_api import ResearchAPI, SimulationResult

__all__ = [
    "ResearchAPI",
    "SimulationResult"
]
