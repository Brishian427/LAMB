"""
Decision engines for the LAMB framework.

This module contains all decision engines that agents can use to make decisions
based on their observations. Each engine implements the BaseEngine interface
and provides different decision-making strategies.

Based on Technical_Specification.md Section 1.3: Engine Decision Flow Algorithm.

Available engines:
- LLMEngine: Pure LLM decision making with OpenAI integration (Phase 1)
- MockEngine: Simple rule-based engine for testing (inherited from BaseEngine)

Future phases will add:
- RuleEngine: Traditional ABM rule-based decision making
- HybridEngine: LLM with rule-based fallback
"""

from ..core.base_engine import BaseEngine, MockEngine
from .llm_engine import LLMEngine
from .rule_engine import RuleEngine, BehavioralRule, SimpleRule, CooperationRules, FlockingRules, SocialNetworkRules
from .hybrid_engine import HybridEngine, HybridMode

__all__ = [
    "BaseEngine",
    "MockEngine", 
    "LLMEngine",
    "RuleEngine",
    "BehavioralRule",
    "SimpleRule",
    "CooperationRules",
    "FlockingRules",
    "SocialNetworkRules",
    "HybridEngine",
    "HybridMode"
]
