"""
Hybrid decision engine combining LLM and rule-based approaches.

This engine intelligently combines LLM and rule-based decision making,
using rules as fallback when LLM fails or for specific scenarios.
Provides the flexibility of LLM with the reliability of rules.
"""

from typing import List, Dict, Any, Optional, Union
import time
import random
from enum import Enum

from ..core.base_engine import BaseEngine
from ..core.types import (
    Observation, Action, EngineType, AgentID,
    LAMBError, InvalidObservationError, EngineTimeoutError
)
from .llm_engine import LLMEngine
from .rule_engine import RuleEngine, BehavioralRule


class HybridMode(str, Enum):
    """Hybrid engine operation modes"""
    LLM_FIRST = "llm_first"  # Try LLM first, fallback to rules
    RULE_FIRST = "rule_first"  # Try rules first, use LLM for complex cases
    ADAPTIVE = "adaptive"  # Dynamically choose based on context
    PARALLEL = "parallel"  # Run both, choose best result


class HybridEngine(BaseEngine):
    """
    Hybrid decision engine combining LLM and rule-based approaches.
    
    Features:
    - Intelligent LLM/Rule combination
    - Multiple operation modes
    - Automatic fallback mechanisms
    - Performance optimization
    - Context-aware decision making
    """
    
    def __init__(self, 
                 llm_engine: Optional[LLMEngine] = None,
                 rule_engine: Optional[RuleEngine] = None,
                 mode: HybridMode = HybridMode.LLM_FIRST,
                 llm_threshold: float = 0.7,
                 rule_threshold: float = 0.3,
                 parallel_timeout: float = 2.0):
        """
        Initialize HybridEngine.
        
        Args:
            llm_engine: LLM engine instance (if None, creates default)
            rule_engine: Rule engine instance (if None, creates default)
            mode: Operation mode (LLM_FIRST, RULE_FIRST, ADAPTIVE, PARALLEL)
            llm_threshold: Confidence threshold for LLM decisions
            rule_threshold: Confidence threshold for rule decisions
            parallel_timeout: Timeout for parallel mode
        """
        super().__init__(EngineType.HYBRID)
        
        self.llm_engine = llm_engine or LLMEngine()
        self.rule_engine = rule_engine or RuleEngine()
        self.mode = mode
        self.llm_threshold = llm_threshold
        self.rule_threshold = rule_threshold
        self.parallel_timeout = parallel_timeout
        
        # Performance tracking
        self._hybrid_metrics = {
            'llm_decisions': 0,
            'rule_decisions': 0,
            'fallback_decisions': 0,
            'parallel_decisions': 0,
            'adaptive_switches': 0,
            'llm_failures': 0,
            'rule_failures': 0
        }
    
    def process_single(self, observation: Observation) -> Action:
        """Process single observation using hybrid approach"""
        start_time = time.perf_counter()
        
        try:
            if self.mode == HybridMode.LLM_FIRST:
                action = self._llm_first_decision(observation)
            elif self.mode == HybridMode.RULE_FIRST:
                action = self._rule_first_decision(observation)
            elif self.mode == HybridMode.ADAPTIVE:
                action = self._adaptive_decision(observation)
            elif self.mode == HybridMode.PARALLEL:
                action = self._parallel_decision(observation)
            else:
                raise ValueError(f"Unknown hybrid mode: {self.mode}")
            
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, True)
            return action
            
        except Exception as e:
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, False)
            raise InvalidObservationError(f"Failed to process observation: {e}")
    
    def process_batch(self, observations: List[Observation]) -> List[Action]:
        """Process batch of observations using hybrid approach"""
        start_time = time.perf_counter()
        
        try:
            actions = []
            for obs in observations:
                action = self.process_single(obs)
                actions.append(action)
            
            batch_time = time.perf_counter() - start_time
            
            # Record batch performance
            self._performance_metrics['total_batches'] += 1
            self._performance_metrics['successful_batches'] += 1
            self._performance_metrics['total_latency'] += batch_time
            
            return actions
            
        except Exception as e:
            batch_time = time.perf_counter() - start_time
            self._performance_metrics['total_batches'] += 1
            self._performance_metrics['failed_batches'] += 1
            self._performance_metrics['total_latency'] += batch_time
            raise InvalidObservationError(f"Failed to process batch: {e}")
    
    def _llm_first_decision(self, observation: Observation) -> Action:
        """Try LLM first, fallback to rules"""
        try:
            # Try LLM first
            action = self.llm_engine.process_single(observation)
            self._hybrid_metrics['llm_decisions'] += 1
            return action
            
        except Exception as e:
            # LLM failed, fallback to rules
            self._hybrid_metrics['llm_failures'] += 1
            self._hybrid_metrics['fallback_decisions'] += 1
            
            try:
                action = self.rule_engine.process_single(observation)
                self._hybrid_metrics['rule_decisions'] += 1
                return action
            except Exception as rule_error:
                self._hybrid_metrics['rule_failures'] += 1
                raise InvalidObservationError(f"Both LLM and rules failed: {e}, {rule_error}")
    
    def _rule_first_decision(self, observation: Observation) -> Action:
        """Try rules first, use LLM for complex cases"""
        try:
            # Try rules first
            action = self.rule_engine.process_single(observation)
            self._hybrid_metrics['rule_decisions'] += 1
            
            # Check if we should use LLM for complex cases
            if self._should_use_llm(observation, action):
                try:
                    llm_action = self.llm_engine.process_single(observation)
                    self._hybrid_metrics['llm_decisions'] += 1
                    return llm_action
                except Exception:
                    # LLM failed, keep rule action
                    pass
            
            return action
            
        except Exception as e:
            # Rules failed, try LLM
            self._hybrid_metrics['rule_failures'] += 1
            self._hybrid_metrics['fallback_decisions'] += 1
            
            try:
                action = self.llm_engine.process_single(observation)
                self._hybrid_metrics['llm_decisions'] += 1
                return action
            except Exception as llm_error:
                self._hybrid_metrics['llm_failures'] += 1
                raise InvalidObservationError(f"Both rules and LLM failed: {e}, {llm_error}")
    
    def _adaptive_decision(self, observation: Observation) -> Action:
        """Dynamically choose between LLM and rules based on context"""
        # Analyze observation complexity
        complexity = self._analyze_complexity(observation)
        
        if complexity > self.llm_threshold:
            # Complex case, use LLM
            try:
                action = self.llm_engine.process_single(observation)
                self._hybrid_metrics['llm_decisions'] += 1
                return action
            except Exception as e:
                # LLM failed, no fallback for pure testing
                self._hybrid_metrics['llm_failures'] += 1
                raise e  # Re-raise the exception instead of falling back
        else:
            # Simple case, use rules
            try:
                action = self.rule_engine.process_single(observation)
                self._hybrid_metrics['rule_decisions'] += 1
                return action
            except Exception as e:
                # Rules failed, no fallback for pure testing
                self._hybrid_metrics['rule_failures'] += 1
                raise e  # Re-raise the exception instead of falling back
    
    def _parallel_decision(self, observation: Observation) -> Action:
        """Run both engines in parallel, choose best result"""
        import asyncio
        import concurrent.futures
        
        def run_llm():
            try:
                return self.llm_engine.process_single(observation)
            except Exception:
                return None
        
        def run_rules():
            try:
                return self.rule_engine.process_single(observation)
            except Exception:
                return None
        
        # Run both engines in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = executor.submit(run_llm)
            rule_future = executor.submit(run_rules)
            
            try:
                # Wait for both with timeout
                llm_result = llm_future.result(timeout=self.parallel_timeout)
                rule_result = rule_future.result(timeout=self.parallel_timeout)
                
                # Choose best result
                if llm_result and rule_result:
                    # Both succeeded, choose based on confidence
                    llm_confidence = self._calculate_confidence(llm_result, observation)
                    rule_confidence = self._calculate_confidence(rule_result, observation)
                    
                    if llm_confidence > rule_confidence:
                        self._hybrid_metrics['llm_decisions'] += 1
                        return llm_result
                    else:
                        self._hybrid_metrics['rule_decisions'] += 1
                        return rule_result
                elif llm_result:
                    self._hybrid_metrics['llm_decisions'] += 1
                    return llm_result
                elif rule_result:
                    self._hybrid_metrics['rule_decisions'] += 1
                    return rule_result
                else:
                    raise InvalidObservationError("Both engines failed")
                    
            except concurrent.futures.TimeoutError:
                # Timeout, use whichever completed first
                if llm_future.done() and llm_future.result():
                    self._hybrid_metrics['llm_decisions'] += 1
                    return llm_future.result()
                elif rule_future.done() and rule_future.result():
                    self._hybrid_metrics['rule_decisions'] += 1
                    return rule_future.result()
                else:
                    raise EngineTimeoutError("Both engines timed out")
    
    def _should_use_llm(self, observation: Observation, rule_action: Action) -> bool:
        """Determine if LLM should be used for complex cases"""
        # Use LLM for complex scenarios
        if len(observation.neighbors) > 5:  # Many neighbors
            return True
        if observation.metadata.get("complexity", 0) > 0.7:  # High complexity
            return True
        if rule_action.action_type == "wait":  # Rule couldn't decide
            return True
        return False
    
    def _analyze_complexity(self, observation: Observation) -> float:
        """Analyze observation complexity (0.0 to 1.0)"""
        complexity = 0.0
        
        # Neighbor count complexity
        neighbor_count = len(observation.neighbors)
        if neighbor_count > 10:
            complexity += 0.4
        elif neighbor_count > 5:
            complexity += 0.2
        elif neighbor_count > 0:
            complexity += 0.1
        
        # Position complexity
        if hasattr(observation, 'position') and observation.position:
            if isinstance(observation.position, tuple) and len(observation.position) > 2:
                complexity += 0.2
        
        # Environment state complexity
        if observation.environment_state:
            complexity += min(len(observation.environment_state) * 0.1, 0.3)
        
        # Step complexity (later steps might be more complex)
        if hasattr(observation, 'step'):
            complexity += min(observation.step * 0.001, 0.1)
        
        return min(complexity, 1.0)
    
    def _calculate_confidence(self, action: Action, observation: Observation) -> float:
        """Calculate confidence score for an action"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on action type
        if action.action_type in ["cooperate", "defect", "compete"]:
            confidence += 0.2
        elif action.action_type in ["explore", "wait"]:
            confidence += 0.1
        
        # Increase confidence based on parameters
        if action.parameters and len(action.parameters) > 0:
            confidence += 0.1
        
        # Increase confidence based on observation context
        if len(observation.neighbors) > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def can_handle_agents(self, agents: List) -> bool:
        """Hybrid engine can handle any agents"""
        return True
    
    def decide(self, agent_id: AgentID, observation: Observation) -> Action:
        """Decide action for agent (alias for process_single)"""
        return self.process_single(observation)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get hybrid engine performance metrics"""
        base_metrics = super().get_performance_metrics()
        
        hybrid_metrics = {
            "mode": self.mode.value,
            "llm_threshold": self.llm_threshold,
            "rule_threshold": self.rule_threshold,
            "llm_decisions": self._hybrid_metrics['llm_decisions'],
            "rule_decisions": self._hybrid_metrics['rule_decisions'],
            "fallback_decisions": self._hybrid_metrics['fallback_decisions'],
            "parallel_decisions": self._hybrid_metrics['parallel_decisions'],
            "adaptive_switches": self._hybrid_metrics['adaptive_switches'],
            "llm_failures": self._hybrid_metrics['llm_failures'],
            "rule_failures": self._hybrid_metrics['rule_failures']
        }
        
        base_metrics.update(hybrid_metrics)
        return base_metrics
    
    def set_mode(self, mode: HybridMode):
        """Change hybrid mode"""
        self.mode = mode
        self._hybrid_metrics['adaptive_switches'] += 1
    
    def add_rule(self, rule: BehavioralRule):
        """Add rule to the rule engine"""
        self.rule_engine.add_rule(rule)
    
    def remove_rule(self, rule_name: str):
        """Remove rule from the rule engine"""
        self.rule_engine.remove_rule(rule_name)
