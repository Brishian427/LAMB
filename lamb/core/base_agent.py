"""
BaseAgent interface implementation based on Technical_Specification.md Section 1.1.

The BaseAgent class serves as the universal interface for all agent types across paradigms,
providing a consistent observe-decide-act pipeline while maintaining paradigm-specific optimizations.

Memory target: <1KB per agent total
Performance targets: observe <0.0001s, decide <0.456s (LLM), act <0.0001s
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import pickle
import json

from .types import (
    AgentID, Position, Observation, Action, ActionResult, AgentState,
    CircularBuffer, ObservationActionPair, LAMBError
)


class BaseAgent(ABC):
    """
    Universal agent interface for all paradigms.
    
    Memory breakdown (target <1KB):
    - position: 16 bytes (2x float64 for 2D, or 1x int64 for node ID)
    - metadata: 512 bytes maximum (user-defined, validated at creation)
    - history: 100 * 50 bytes = 5000 bytes (circular buffer, configurable)
    - agent_id: 8 bytes (int64)
    - internal state: ~100 bytes (flags, counters, references)
    
    Total: ~636 bytes base + metadata + history
    Optimization: History can be reduced to 20 entries for <1KB target
    """
    
    def __init__(
        self,
        agent_id: AgentID,
        position: Position,
        metadata: Optional[Dict[str, Any]] = None,
        history_size: int = 100
    ):
        """
        Initialize BaseAgent.
        
        Args:
            agent_id: Unique 8-byte integer identifier
            position: Initial position (paradigm-specific)
            metadata: User-defined properties (max 512 bytes)
            history_size: Size of history buffer (default 100, can reduce to 20)
        """
        self.agent_id = agent_id
        self.position = position
        self.metadata = metadata or {}
        self.history = CircularBuffer(max_size=history_size)
        self._internal_state = {}
        
        # Validate metadata size
        if self._estimate_metadata_size() > 512:
            raise LAMBError(f"Metadata size exceeds 512 bytes limit")
    
    def _estimate_metadata_size(self) -> int:
        """Estimate metadata size in bytes"""
        try:
            return len(pickle.dumps(self.metadata))
        except:
            return len(str(self.metadata).encode('utf-8'))
    
    @abstractmethod
    def observe(self, environment: 'BaseEnvironment') -> Observation:
        """
        Generate agent's perception of environment state.
        
        Performance target: <0.0001s per call (spatial query dominates cost)
        Memory allocation: Observation object <200 bytes
        Thread safety: Read-only operation, thread-safe
        Serialization: Observation must be JSON-serializable for LLM prompts
        
        Args:
            environment: Environment to observe
            
        Returns:
            Observation object containing agent's perception
            
        Raises:
            AgentNotFoundError: If agent not in environment
            InvalidEnvironmentError: If environment corrupted
        """
        pass
    
    @abstractmethod
    def decide(self, observation: Observation, engine: 'BaseEngine') -> Action:
        """
        Transform observation into action using specified engine.
        
        Performance target:
        - LLM-based: <0.456s per call (batched: <5s per 15 agents)
        Memory allocation: Action object <100 bytes
        Thread safety: Stateless operation, thread-safe
        Fallback behavior: Engine handles fallback internally
        
        Args:
            observation: Agent's observation of environment
            engine: Decision engine to use
            
        Returns:
            Action to execute
            
        Raises:
            InvalidObservationError: For malformed observations
            EngineTimeoutError: For LLM timeout (handled by circuit breaker)
            IncompatibleEngineError: For paradigm mismatch
        """
        pass
    
    @abstractmethod
    def act(self, action: Action, environment: 'BaseEnvironment') -> ActionResult:
        """
        Execute action and update environment state.
        
        Performance target: <0.0001s per call (excluding conflict resolution)
        Memory allocation: ActionResult <50 bytes
        Thread safety: Requires environment lock for state modification
        Conflict resolution: Delegated to environment's resolve_conflicts method
        
        Args:
            action: Action to execute
            environment: Environment to act upon
            
        Returns:
            Result of action execution
            
        Raises:
            InvalidActionError: For impossible actions
            ConflictError: For simultaneous conflicting actions
            EnvironmentConstraintError: For boundary violations
        """
        pass
    
    def step(self, environment: 'BaseEnvironment', engine: 'BaseEngine') -> ActionResult:
        """
        Execute complete observe-decide-act cycle.
        
        This is the main simulation step method that orchestrates the
        agent's behavior cycle.
        """
        start_time = time.perf_counter()
        
        # Observe
        observation = self.observe(environment)
        observe_time = time.perf_counter() - start_time
        
        # Decide
        decide_start = time.perf_counter()
        action = self.decide(observation, engine)
        decide_time = time.perf_counter() - decide_start
        
        # Act
        act_start = time.perf_counter()
        result = self.act(action, environment)
        act_time = time.perf_counter() - act_start
        
        # Record in history
        self.history.append(ObservationActionPair(observation, action))
        
        # Update internal performance metrics
        self._internal_state.update({
            'last_observe_time': observe_time,
            'last_decide_time': decide_time,
            'last_act_time': act_time,
            'total_step_time': time.perf_counter() - start_time
        })
        
        return result
    
    def get_state(self) -> AgentState:
        """Get complete agent state for serialization"""
        return AgentState(
            agent_id=self.agent_id,
            position=self.position,
            metadata=self.metadata,
            history=self.history,
            internal_state=self._internal_state
        )
    
    def set_state(self, state: AgentState) -> None:
        """Set agent state from serialized data"""
        self.agent_id = state.agent_id
        self.position = state.position
        self.metadata = state.metadata
        self.history = state.history
        self._internal_state = state.internal_state
    
    def serialize(self) -> bytes:
        """
        Serialize agent state to bytes (pickle format - primary).
        Must support pickle (primary) + JSON (portable)
        """
        return pickle.dumps(self.get_state())
    
    def deserialize(self, data: bytes) -> None:
        """Deserialize agent state from bytes"""
        state = pickle.loads(data)
        self.set_state(state)
    
    def to_json(self) -> str:
        """
        Serialize to JSON format (portable).
        Note: History and complex objects may be simplified.
        """
        state_dict = {
            'agent_id': self.agent_id,
            'position': self.position,
            'metadata': self.metadata,
            'history_length': len(self.history),
            'internal_state': self._internal_state
        }
        return json.dumps(state_dict)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent's performance metrics"""
        return {
            'observe_time': self._internal_state.get('last_observe_time', 0.0),
            'decide_time': self._internal_state.get('last_decide_time', 0.0),
            'act_time': self._internal_state.get('last_act_time', 0.0),
            'total_step_time': self._internal_state.get('total_step_time', 0.0),
            'memory_estimate_bytes': self._estimate_memory_usage()
        }
    
    # ============================================================================
    # UNIVERSAL METHODS - Available to all paradigms
    # ============================================================================
    
    def get_position(self) -> Position:
        """Get agent's current position (universal method)"""
        return self.position
    
    def set_position(self, position: Position) -> None:
        """Set agent's position (universal method)"""
        self.position = position
    
    def get_metadata(self, key: str = None) -> Any:
        """Get agent metadata (universal method)"""
        if key is None:
            return self.metadata
        return self.metadata.get(key)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set agent metadata (universal method)"""
        self.metadata[key] = value
        # Validate metadata size after update
        if self._estimate_metadata_size() > 512:
            raise LAMBError(f"Metadata size exceeds 512 bytes limit after setting {key}")
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update multiple metadata fields (universal method)"""
        self.metadata.update(updates)
        if self._estimate_metadata_size() > 512:
            raise LAMBError(f"Metadata size exceeds 512 bytes limit after updates")
    
    def get_history(self) -> CircularBuffer:
        """Get agent's action history (universal method)"""
        return self.history
    
    def add_to_history(self, observation: Observation, action: Action) -> None:
        """Add observation-action pair to history (universal method)"""
        self.history.append(ObservationActionPair(observation, action))
    
    def get_internal_state(self, key: str = None) -> Any:
        """Get internal state (universal method)"""
        if key is None:
            return self._internal_state
        return self._internal_state.get(key)
    
    def set_internal_state(self, key: str, value: Any) -> None:
        """Set internal state (universal method)"""
        self._internal_state[key] = value
    
    def get_agent_id(self) -> AgentID:
        """Get agent ID (universal method)"""
        return self.agent_id
    
    def is_alive(self) -> bool:
        """Check if agent is alive (universal method)"""
        return self._internal_state.get('alive', True)
    
    def set_alive(self, alive: bool) -> None:
        """Set agent alive status (universal method)"""
        self._internal_state['alive'] = alive
    
    def get_age(self) -> int:
        """Get agent age in simulation steps (universal method)"""
        return self._internal_state.get('age', 0)
    
    def increment_age(self) -> None:
        """Increment agent age (universal method)"""
        self._internal_state['age'] = self._internal_state.get('age', 0) + 1
    
    def get_energy(self) -> float:
        """Get agent energy level (universal method)"""
        return self._internal_state.get('energy', 1.0)
    
    def set_energy(self, energy: float) -> None:
        """Set agent energy level (universal method)"""
        self._internal_state['energy'] = max(0.0, energy)
    
    def add_energy(self, amount: float) -> None:
        """Add energy to agent (universal method)"""
        current_energy = self.get_energy()
        self.set_energy(current_energy + amount)
    
    def consume_energy(self, amount: float) -> bool:
        """Consume energy from agent (universal method)"""
        current_energy = self.get_energy()
        if current_energy >= amount:
            self.set_energy(current_energy - amount)
            return True
        return False
    
    def get_resources(self) -> Dict[str, float]:
        """Get agent resources (universal method)"""
        return self._internal_state.get('resources', {})
    
    def set_resources(self, resources: Dict[str, float]) -> None:
        """Set agent resources (universal method)"""
        self._internal_state['resources'] = resources
    
    def add_resource(self, resource_type: str, amount: float) -> None:
        """Add resource to agent (universal method)"""
        resources = self.get_resources()
        resources[resource_type] = resources.get(resource_type, 0.0) + amount
        self.set_resources(resources)
    
    def consume_resource(self, resource_type: str, amount: float) -> bool:
        """Consume resource from agent (universal method)"""
        resources = self.get_resources()
        if resources.get(resource_type, 0.0) >= amount:
            resources[resource_type] = resources.get(resource_type, 0.0) - amount
            self.set_resources(resources)
            return True
        return False
    
    def has_resource(self, resource_type: str, amount: float = 0.0) -> bool:
        """Check if agent has sufficient resource (universal method)"""
        resources = self.get_resources()
        return resources.get(resource_type, 0.0) >= amount
    
    def get_behavior_state(self) -> str:
        """Get agent's current behavior state (universal method)"""
        return self._internal_state.get('behavior_state', 'idle')
    
    def set_behavior_state(self, state: str) -> None:
        """Set agent's behavior state (universal method)"""
        self._internal_state['behavior_state'] = state
    
    def get_last_action(self) -> Optional[Action]:
        """Get agent's last action (universal method)"""
        if len(self.history) > 0:
            return self.history[-1].action
        return None
    
    def get_last_observation(self) -> Optional[Observation]:
        """Get agent's last observation (universal method)"""
        if len(self.history) > 0:
            return self.history[-1].observation
        return None
    
    def clear_history(self) -> None:
        """Clear agent's history (universal method)"""
        self.history.clear()
    
    def get_history_length(self) -> int:
        """Get history length (universal method)"""
        return len(self.history)
    
    def _estimate_memory_usage(self) -> int:
        """Estimate agent's memory usage in bytes"""
        base_size = 16 + 8 + 100  # position + agent_id + internal_state
        metadata_size = self._estimate_metadata_size()
        history_size = len(self.history) * 50  # Rough estimate per entry
        
        return base_size + metadata_size + history_size
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, pos={self.position})"
