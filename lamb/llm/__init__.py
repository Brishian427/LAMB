"""
LLM integration module for the LAMB framework.

This module provides comprehensive LLM integration with OpenAI and other providers,
including circuit breaker patterns, response caching, batch processing, and
performance monitoring.

Based on Technical_Specification.md Section 1.3: Engine Decision Flow Algorithm.

Key components:
- OpenAI provider with robust error handling
- Circuit breaker pattern for reliability
- Response caching with LRU eviction
- Batch processing optimization
- Comprehensive performance monitoring

Phase 1 focus: Pure LLM architecture with OpenAI integration
Future phases: Additional providers (Anthropic, Google, local models)
"""

from .base_provider import BaseLLMProvider, LLMResponse
from .openai_provider import OpenAIProvider
from .circuit_breaker import CircuitBreaker, CircuitState
from .response_cache import ResponseCache
from .batch_processor import BatchProcessor
from .agent_prompts import (
    PromptManager, AgentPromptBuilder, AgentPersonality,
    PromptTemplate, PromptType, BehavioralRule, RuleType,
    COMMON_PERSONALITIES, create_research_personality, create_behavioral_rule
)

__all__ = [
    "BaseLLMProvider",
    "LLMResponse", 
    "OpenAIProvider",
    "CircuitBreaker",
    "CircuitState",
    "ResponseCache",
    "BatchProcessor",
    "PromptManager",
    "AgentPromptBuilder", 
    "AgentPersonality",
    "PromptTemplate",
    "PromptType",
    "BehavioralRule",
    "RuleType",
    "COMMON_PERSONALITIES",
    "create_research_personality",
    "create_behavioral_rule"
]
