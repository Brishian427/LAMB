"""
LLM decision engine implementation with OpenAI integration.

Based on Technical_Specification.md Section 1.3: Engine Decision Flow Algorithm.
Implements pure LLM architecture for Phase 1 with circuit breaker pattern,
batch processing, and comprehensive error handling.

Performance requirements (from reconnaissance validation):
- LLM decisions: <0.456s per agent (single), <5s per batch (10-25 agents)
- Circuit breaker: 30% failure threshold, <0.01s fallback latency
- Response caching: 60% hit rate target with LRU eviction
- Timeout handling: 5s per batch with exponential backoff retry
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import asyncio
import json
import hashlib
from collections import OrderedDict
from enum import Enum

from ..core.base_engine import BaseEngine
from ..core.types import (
    Observation, Action, EngineType, EngineTimeoutError,
    InvalidObservationError, LAMBError, AgentID
)
from ..llm.openai_provider import OpenAIProvider
from ..llm.circuit_breaker import CircuitBreaker
from ..llm.response_cache import ResponseCache
from ..llm.batch_processor import BatchProcessor
from ..llm.agent_prompts import PromptManager, AgentPromptBuilder, AgentPersonality


class LLMEngine(BaseEngine):
    """
    LLM decision engine with OpenAI integration.
    
    Features:
    - OpenAI API integration with multiple model support
    - Circuit breaker pattern for reliability (30% failure threshold)
    - Batch processing optimization (10-25 agents per batch)
    - Response caching with LRU eviction (60% hit rate target)
    - Comprehensive error handling and fallback mechanisms
    - Performance monitoring and metrics collection
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 150,
        temperature: float = 0.7,
        batch_size: int = 15,
        cache_size: int = 1000,
        circuit_breaker_threshold: float = 0.3,
        timeout_seconds: float = 5.0,
        prompt_manager: Optional[PromptManager] = None,
        custom_prompt_template: Optional[str] = None
    ):
        """
        Initialize LLMEngine.
        
        Args:
            api_key: OpenAI API key (if None, will try environment variable)
            model: OpenAI model to use
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature (0.0 to 1.0)
            batch_size: Optimal batch size for processing
            cache_size: Maximum cache entries
            circuit_breaker_threshold: Failure rate threshold (0.0 to 1.0)
            timeout_seconds: Request timeout in seconds
        """
        super().__init__(EngineType.LLM)
        
        # Initialize components
        self.provider = OpenAIProvider(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds
        )
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=30.0,
            max_failures=10
        )
        
        self.response_cache = ResponseCache(
            max_size=cache_size,
            ttl_seconds=300.0  # 5 minutes
        )
        
        self.batch_processor = BatchProcessor(
            optimal_batch_size=batch_size,
            max_batch_size=25,
            timeout_seconds=timeout_seconds
        )
        
        # Configuration
        self.batch_size = batch_size
        self.prompt_manager = prompt_manager or PromptManager()
        self.prompt_builder = AgentPromptBuilder(self.prompt_manager)
        self.custom_prompt_template = custom_prompt_template
        self.timeout_seconds = timeout_seconds
        
        # Performance tracking
        self._batch_performance = {
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'avg_batch_time': 0.0,
            'avg_batch_size': 0.0
        }
    
    def process_single(self, observation: Observation) -> Action:
        """
        Process single observation to generate action.
        
        Performance target: <0.456s per call
        Uses caching and circuit breaker for reliability.
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(observation)
            cached_action = self.response_cache.get(cache_key)
            
            if cached_action is not None:
                self._record_cache_hit()
                decision_time = time.perf_counter() - start_time
                self._record_decision(decision_time, True)
                return cached_action
            
            self._record_cache_miss()
            
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                raise EngineTimeoutError("Circuit breaker is open")
            
            # Generate LLM prompt
            prompt = self._create_prompt(observation)
            
            # Call LLM provider
            try:
                response = self.provider.generate_response(prompt)
                action = self._parse_response(response, observation.agent_id)
                
                # Cache successful response
                self.response_cache.put(cache_key, action)
                
                # Record success
                self.circuit_breaker.record_success()
                decision_time = time.perf_counter() - start_time
                self._record_decision(decision_time, True)
                
                return action
                
            except Exception as e:
                # Record failure
                self.circuit_breaker.record_failure()
                decision_time = time.perf_counter() - start_time
                self._record_decision(decision_time, False)
                
                # Re-raise with context
                raise EngineTimeoutError(f"LLM call failed: {e}")
                
        except Exception as e:
            # Final fallback - create emergency action
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, False)
            
            return self._create_emergency_action(observation.agent_id)
    
    def process_batch(self, observations: List[Observation]) -> List[Action]:
        """
        Process batch of observations for performance optimization.
        
        Performance target: <5s per batch (with 5s timeout)
        Optimal batch size: 10-25 agents (from reconnaissance)
        """
        start_time = time.perf_counter()
        
        try:
            # Check if batch processing is beneficial
            if len(observations) == 1:
                return [self.process_single(observations[0])]
            
            # Process through batch processor
            actions = self.batch_processor.process_batch(
                observations, self._process_batch_internal
            )
            
            # Record batch performance
            batch_time = time.perf_counter() - start_time
            self._record_batch_performance(len(observations), batch_time, True)
            
            return actions
            
        except Exception as e:
            batch_time = time.perf_counter() - start_time
            self._record_batch_performance(len(observations), batch_time, False)
            
            # Fallback to individual processing
            return self._fallback_individual_processing(observations)
    
    def _process_batch_internal(self, observations: List[Observation]) -> List[Action]:
        """Internal batch processing with caching and circuit breaker"""
        # Check cache for all observations
        cached_actions = {}
        uncached_observations = []
        
        for obs in observations:
            cache_key = self._generate_cache_key(obs)
            cached_action = self.response_cache.get(cache_key)
            
            if cached_action is not None:
                cached_actions[obs.agent_id] = cached_action
                self._record_cache_hit()
            else:
                uncached_observations.append(obs)
                self._record_cache_miss()
        
        # Process uncached observations
        if uncached_observations:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                raise EngineTimeoutError("Circuit breaker is open")
            
            try:
                # Create batch prompt
                batch_prompt = self._create_batch_prompt(uncached_observations)
                
                # Call LLM provider
                batch_response = self.provider.generate_batch_response(
                    batch_prompt, len(uncached_observations)
                )
                
                # Parse responses
                new_actions = self._parse_batch_response(
                    batch_response, uncached_observations
                )
                
                # Cache new actions
                for obs, action in zip(uncached_observations, new_actions):
                    cache_key = self._generate_cache_key(obs)
                    self.response_cache.put(cache_key, action)
                    cached_actions[obs.agent_id] = action
                
                # Record success
                self.circuit_breaker.record_success()
                
            except Exception as e:
                # Record failure
                self.circuit_breaker.record_failure()
                
                # Create emergency actions for uncached observations
                for obs in uncached_observations:
                    cached_actions[obs.agent_id] = self._create_emergency_action(obs.agent_id)
        
        # Return actions in original order
        return [cached_actions[obs.agent_id] for obs in observations]
    
    def _create_prompt(self, observation: Observation, agent_personality: Optional[AgentPersonality] = None) -> str:
        """Create LLM prompt from observation using the prompt manager"""
        return self.prompt_builder.build_decision_prompt(
            observation=observation,
            agent_personality=agent_personality,
            custom_template=self.custom_prompt_template
        )
    
    def decide(self, agent_id: AgentID, observation: Observation, agent_personality: Optional[AgentPersonality] = None) -> Action:
        """
        Make a decision for an agent based on observation and personality.
        
        Args:
            observation: Agent's observation of the environment
            agent_personality: Optional personality traits for the agent
            
        Returns:
            Action to be taken by the agent
        """
        try:
            # Create prompt using the prompt manager
            prompt = self._create_prompt(observation, agent_personality)
            
            # Check cache first
            cache_key = self._create_cache_key(observation, agent_personality)
            cached_action = self.response_cache.get(cache_key)
            if cached_action:
                return cached_action
            
            # Check circuit breaker
            if self.circuit_breaker.is_open():
                return self._create_emergency_action(observation.agent_id)
            
            # Generate response from LLM
            response = self.provider.generate_response(prompt)
            action = self._parse_response(response)
            
            # Cache the response
            self.response_cache.set(cache_key, action)
            
            # Record success
            self.circuit_breaker.record_success()
            
            return action
            
        except Exception as e:
            # Record failure and return emergency action
            self.circuit_breaker.record_failure()
            return self._create_emergency_action(observation.agent_id)
    
    def _create_cache_key(self, observation: Observation, agent_personality: Optional[AgentPersonality] = None) -> str:
        """Create a cache key for the observation and personality"""
        key_data = {
            'agent_id': observation.agent_id,
            'position': observation.position,
            'neighbors': len(observation.neighbors),
            'paradigm': observation.paradigm,
            'personality': agent_personality.name if agent_personality else 'default'
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    
    def _parse_response(self, response: str) -> Action:
        """Parse LLM response into Action object"""
        try:
            # Extract JSON from response
            response_clean = response.strip()
            
            # Find JSON object in response
            start_idx = response_clean.find('{')
            end_idx = response_clean.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response_clean[start_idx:end_idx]
            action_data = json.loads(json_str)
            
            # Validate required fields
            if 'action_type' not in action_data:
                raise ValueError("Missing action_type in response")
            
            parameters = action_data.get('parameters', {})
            
            return Action(
                agent_id=agent_id,
                action_type=action_data['action_type'],
                parameters=parameters
            )
            
        except Exception as e:
            # Return emergency action if parsing fails
            return self._create_emergency_action(agent_id)
    
    def _parse_batch_response(self, response: str, observations: List[Observation]) -> List[Action]:
        """Parse batch LLM response into list of Action objects"""
        try:
            # Extract JSON array from response
            response_clean = response.strip()
            
            # Find JSON array in response
            start_idx = response_clean.find('[')
            end_idx = response_clean.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = response_clean[start_idx:end_idx]
            actions_data = json.loads(json_str)
            
            if not isinstance(actions_data, list):
                raise ValueError("Response is not a JSON array")
            
            if len(actions_data) != len(observations):
                raise ValueError(f"Expected {len(observations)} actions, got {len(actions_data)}")
            
            actions = []
            for i, (action_data, obs) in enumerate(zip(actions_data, observations)):
                if not isinstance(action_data, dict) or 'action_type' not in action_data:
                    actions.append(self._create_emergency_action(obs.agent_id))
                else:
                    parameters = action_data.get('parameters', {})
                    actions.append(Action(
                        agent_id=obs.agent_id,
                        action_type=action_data['action_type'],
                        parameters=parameters
                    ))
            
            return actions
            
        except Exception as e:
            # Return emergency actions for all observations
            return [self._create_emergency_action(obs.agent_id) for obs in observations]
    
    def _create_emergency_action(self, agent_id: int) -> Action:
        """Create emergency fallback action"""
        return Action(
            agent_id=agent_id,
            action_type="stay",
            parameters={}
        )
    
    def _generate_cache_key(self, observation: Observation) -> str:
        """Generate cache key for observation"""
        # Use observation's cache key method if available
        if hasattr(observation, 'to_cache_key'):
            base_key = observation.to_cache_key()
        else:
            # Fallback to simple key generation
            base_key = f"{observation.paradigm}_{len(observation.neighbors)}_{hash(str(observation.position))}"
        
        return f"llm_{self.provider.model}_{base_key}"
    
    def _fallback_individual_processing(self, observations: List[Observation]) -> List[Action]:
        """Fallback to individual processing when batch fails"""
        actions = []
        
        for obs in observations:
            try:
                action = self.process_single(obs)
                actions.append(action)
            except Exception:
                # Emergency action as last resort
                actions.append(self._create_emergency_action(obs.agent_id))
        
        return actions
    
    def _record_batch_performance(self, batch_size: int, batch_time: float, success: bool) -> None:
        """Record batch processing performance"""
        self._batch_performance['total_batches'] += 1
        
        if success:
            self._batch_performance['successful_batches'] += 1
        else:
            self._batch_performance['failed_batches'] += 1
        
        # Update running averages
        total = self._batch_performance['total_batches']
        current_avg_time = self._batch_performance['avg_batch_time']
        current_avg_size = self._batch_performance['avg_batch_size']
        
        self._batch_performance['avg_batch_time'] = (
            (current_avg_time * (total - 1) + batch_time) / total
        )
        self._batch_performance['avg_batch_size'] = (
            (current_avg_size * (total - 1) + batch_size) / total
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = super().get_performance_metrics()
        
        # Add LLM-specific metrics
        llm_metrics = {
            'provider_metrics': self.provider.get_metrics(),
            'circuit_breaker_metrics': self.circuit_breaker.get_metrics(),
            'cache_metrics': self.response_cache.get_metrics(),
            'batch_metrics': self._batch_performance.copy(),
            'batch_processor_metrics': self.batch_processor.get_metrics()
        }
        
        # Combine metrics
        base_metrics.update(llm_metrics)
        
        return base_metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        super().reset_metrics()
        
        # Reset component metrics
        self.provider.reset_metrics()
        self.circuit_breaker.reset_metrics()
        self.response_cache.reset_metrics()
        self.batch_processor.reset_metrics()
        
        # Reset batch performance
        self._batch_performance = {
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'avg_batch_time': 0.0,
            'avg_batch_size': 0.0
        }
    
    def __repr__(self) -> str:
        return (f"LLMEngine(model={self.provider.model}, "
                f"batch_size={self.batch_size}, "
                f"cache_size={self.response_cache.max_size})")
