"""
BaseEngine interface implementation based on Technical_Specification.md Section 1.3.

The BaseEngine class implements the observe-decide-act pipeline with LLM architecture,
circuit breaker pattern, and comprehensive error handling.

Phase 1: Pure LLM architecture (OpenAI integration)
Future phases: Extensible to RULE and HYBRID modes

Performance requirements (from reconnaissance validation):
- LLM decisions: <5s per batch (with 5s timeout)
- Fallback latency: <0.01s total (circuit breaker + rule execution)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import asyncio

from .types import (
    Observation, Action, EngineType, LAMBError,
    EngineTimeoutError, InvalidObservationError, AgentID
)
from .base_agent import BaseAgent


class BaseEngine(ABC):
    """
    Universal engine interface for all decision-making modes.
    
    Supports LLM, RULE, and HYBRID engine types as specified in
    Technical_Specification.md line 196: engine_type: EngineType  # LLM, RULE, or HYBRID
    
    Phase 1: Implements LLM mode only
    Future phases: Will add RULE and HYBRID implementations
    """
    
    def __init__(self, engine_type: EngineType):
        """
        Initialize BaseEngine.
        
        Args:
            engine_type: Type of engine (LLM, RULE, or HYBRID)
        """
        self.engine_type = engine_type
        
        # Performance monitoring
        self.metrics_collector = None  # Will be set by subclasses
        self.performance_monitor = None  # Will be set by subclasses
        
        # Performance tracking
        self._performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'avg_decision_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    @abstractmethod
    def process_single(self, observation: Observation) -> Action:
        """
        Process single observation to generate action.
        
        Performance targets:
        - LLM-based: <0.456s per call
        - Rule-based: <0.001s per call (future phase)
        
        Args:
            observation: Agent's observation
            
        Returns:
            Action to execute
            
        Raises:
            InvalidObservationError: For malformed observations
            EngineTimeoutError: For timeout (handled by circuit breaker)
        """
        pass
    
    @abstractmethod
    def process_batch(self, observations: List[Observation]) -> List[Action]:
        """
        Process batch of observations for performance optimization.
        
        Performance target: <5s per batch (with 5s timeout)
        Optimal batch size: 10-25 agents (from reconnaissance)
        
        Args:
            observations: List of agent observations
            
        Returns:
            List of actions corresponding to observations
            
        Raises:
            EngineTimeoutError: For batch timeout
        """
        pass
    
    def handle_failure(self, error: Exception, observation: Observation, fallback: bool = True) -> Action:
        """
        Handle engine failure with optional fallback.
        
        Performance target: <0.01s total (circuit breaker + fallback execution)
        
        Args:
            error: The error that occurred
            observation: Original observation
            fallback: Whether to use fallback (default True)
            
        Returns:
            Fallback action or re-raises error
        """
        self._performance_metrics['failed_decisions'] += 1
        
        if fallback and hasattr(self, '_fallback_action'):
            return self._fallback_action(observation)
        else:
            raise error
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get engine performance metrics.
        
        Returns comprehensive performance data for monitoring and optimization.
        """
        total_decisions = self._performance_metrics['total_decisions']
        success_rate = 0.0
        if total_decisions > 0:
            success_rate = self._performance_metrics['successful_decisions'] / total_decisions
        
        cache_total = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
        cache_hit_rate = 0.0
        if cache_total > 0:
            cache_hit_rate = self._performance_metrics['cache_hits'] / cache_total
        
        return {
            'engine_type': self.engine_type.value,
            'total_decisions': total_decisions,
            'success_rate': success_rate,
            'avg_decision_time': self._performance_metrics['avg_decision_time'],
            'cache_hit_rate': cache_hit_rate,
            'performance_status': self._get_performance_status()
        }
    
    def _get_performance_status(self) -> str:
        """Get performance status based on targets"""
        avg_time = self._performance_metrics['avg_decision_time']
        
        if self.engine_type == EngineType.LLM:
            target_time = 0.456  # LLM target from Technical_Specification.md
        else:
            target_time = 0.001  # Rule target (future phase)
        
        if avg_time <= target_time:
            return "good"
        elif avg_time <= target_time * 1.5:
            return "warning"
        else:
            return "critical"
    
    def _record_decision(self, decision_time: float, success: bool) -> None:
        """Record decision performance metrics"""
        self._performance_metrics['total_decisions'] += 1
        
        if success:
            self._performance_metrics['successful_decisions'] += 1
        else:
            self._performance_metrics['failed_decisions'] += 1
        
        # Update running average
        total = self._performance_metrics['total_decisions']
        current_avg = self._performance_metrics['avg_decision_time']
        self._performance_metrics['avg_decision_time'] = (
            (current_avg * (total - 1) + decision_time) / total
        )
    
    def _record_cache_hit(self) -> None:
        """Record cache hit"""
        self._performance_metrics['cache_hits'] += 1
    
    def _record_cache_miss(self) -> None:
        """Record cache miss"""
        self._performance_metrics['cache_misses'] += 1
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self._performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'avg_decision_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.engine_type.value})"


class MockEngine(BaseEngine):
    """
    Mock engine for testing and development.
    
    Provides simple rule-based behavior for testing the framework
    without requiring LLM integration.
    """
    
    def __init__(self):
        super().__init__(EngineType.LLM)  # Mock as LLM for testing
    
    def can_handle_agents(self, agents: List[BaseAgent]) -> bool:
        """Mock engine can handle any agents"""
        return True
    
    def decide(self, agent_id: AgentID, observation: Observation) -> Action:
        """Decide action for agent (alias for process_single)"""
        return self.process_single(observation)
    
    def process_single(self, observation: Observation) -> Action:
        """Generate mock action based on observation"""
        start_time = time.perf_counter()
        
        try:
            # Simple mock decision logic
            action = Action(
                agent_id=observation.agent_id,
                action_type="move",
                parameters={"direction": "random"}
            )
            
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, True)
            
            return action
            
        except Exception as e:
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, False)
            raise InvalidObservationError(f"Failed to process observation: {e}")
    
    def process_batch(self, observations: List[Observation]) -> List[Action]:
        """Process batch of observations using mock logic"""
        start_time = time.perf_counter()
        
        try:
            actions = []
            for obs in observations:
                action = self.process_single(obs)
                actions.append(action)
            
            batch_time = time.perf_counter() - start_time
            
            # Simulate batch processing benefit (should be faster than sequential)
            if len(observations) > 1:
                # Mock batch optimization - reduce time per agent
                effective_time = batch_time / len(observations) * 0.8
                for _ in observations:
                    self._record_decision(effective_time, True)
            
            return actions
            
        except Exception as e:
            batch_time = time.perf_counter() - start_time
            for _ in observations:
                self._record_decision(batch_time / len(observations), False)
            raise EngineTimeoutError(f"Batch processing failed: {e}")
    
    def _fallback_action(self, observation: Observation) -> Action:
        """Generate emergency fallback action"""
        return Action(
            agent_id=observation.agent_id,
            action_type="stay",
            parameters={}
        )
