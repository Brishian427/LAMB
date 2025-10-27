"""
Core type definitions and interfaces for the LAMB framework.

Based on Technical_Specification.md Section 1: Core Architecture Specification.
All types are designed for LLM integration with future extensibility to RULE/HYBRID modes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque


# Agent ID type - 8-byte integer as specified
AgentID = int

# Position types for different paradigms
GridCoord = Tuple[int, int]
PhysicsCoord = Tuple[float, float]
NodeID = int

Position = Union[GridCoord, PhysicsCoord, NodeID]


class EngineType(Enum):
    """Engine types as specified in Technical_Specification.md line 196"""
    LLM = "llm"
    RULE = "rule"  # Future phase
    HYBRID = "hybrid"  # Future phase


class BoundaryCondition(Enum):
    """Boundary conditions for spatial environments"""
    WRAP = "wrap"
    WALL = "wall"
    REFLECT = "reflect"
    ABSORB = "absorb"
    INFINITE = "infinite"


@dataclass
class Vector2D:
    """2D vector for physics calculations"""
    x: float
    y: float
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


class CircularBuffer:
    """
    Circular buffer for agent history as specified in Technical_Specification.md.
    Size limit: 100 entries (configurable to 20 for <1KB target)
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def append(self, item: Any) -> None:
        self.buffer.append(item)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __iter__(self):
        return iter(self.buffer)
    
    def clear(self) -> None:
        """Clear all items from the buffer"""
        self.buffer.clear()
    
    def __getitem__(self, index: int) -> Any:
        """Get item at index"""
        return self.buffer[index]
    
    def __setitem__(self, index: int, value: Any) -> None:
        """Set item at index"""
        self.buffer[index] = value


@dataclass
class ObservationActionPair:
    """Pair of observation and action for agent history"""
    observation: 'Observation'
    action: 'Action'
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Observation:
    """
    Agent observation as specified in Technical_Specification.md.
    Must be JSON-serializable for LLM prompts.
    Memory allocation: <200 bytes
    """
    agent_id: AgentID
    position: Position
    neighbors: List[AgentID]
    environment_state: Dict[str, Any]
    paradigm: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "neighbors": self.neighbors,
            "environment_state": self.environment_state,
            "paradigm": self.paradigm,
            "timestamp": self.timestamp
        }
    
    def to_cache_key(self) -> str:
        """Generate cache key for response caching"""
        # Simplified key generation - can be enhanced
        return f"{self.paradigm}_{len(self.neighbors)}_{hash(str(self.position))}"


@dataclass
class Action:
    """
    Agent action as specified in Technical_Specification.md.
    Memory allocation: <100 bytes
    """
    agent_id: AgentID
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def copy(self) -> 'Action':
        """Create a copy of the action"""
        return Action(
            agent_id=self.agent_id,
            action_type=self.action_type,
            parameters=self.parameters.copy(),
            timestamp=self.timestamp
        )
    
    # ============================================================================
    # UNIVERSAL METHODS - Available to all paradigms
    # ============================================================================
    
    def get_agent_id(self) -> AgentID:
        """Get agent ID (universal method)"""
        return self.agent_id
    
    def get_action_type(self) -> str:
        """Get action type (universal method)"""
        return self.action_type
    
    def get_parameters(self, key: str = None) -> Any:
        """Get action parameters (universal method)"""
        if key is None:
            return self.parameters
        return self.parameters.get(key)
    
    def set_parameters(self, key: str, value: Any) -> None:
        """Set action parameter (universal method)"""
        self.parameters[key] = value
    
    def update_parameters(self, updates: Dict[str, Any]) -> None:
        """Update multiple parameters (universal method)"""
        self.parameters.update(updates)
    
    def get_timestamp(self) -> float:
        """Get action timestamp (universal method)"""
        return self.timestamp
    
    def set_timestamp(self, timestamp: float) -> None:
        """Set action timestamp (universal method)"""
        self.timestamp = timestamp
    
    def get_target_position(self) -> Optional[Position]:
        """Get target position if this is a movement action (universal method)"""
        return self.parameters.get('target_position')
    
    def set_target_position(self, position: Position) -> None:
        """Set target position (universal method)"""
        self.parameters['target_position'] = position
    
    def get_direction(self) -> Optional[Position]:
        """Get movement direction if this is a movement action (universal method)"""
        return self.parameters.get('direction')
    
    def set_direction(self, direction: Position) -> None:
        """Set movement direction (universal method)"""
        self.parameters['direction'] = direction
    
    def get_speed(self) -> Optional[float]:
        """Get movement speed if this is a movement action (universal method)"""
        return self.parameters.get('speed')
    
    def set_speed(self, speed: float) -> None:
        """Set movement speed (universal method)"""
        self.parameters['speed'] = speed
    
    def get_energy_cost(self) -> Optional[float]:
        """Get energy cost of this action (universal method)"""
        return self.parameters.get('energy_cost', 0.0)
    
    def set_energy_cost(self, cost: float) -> None:
        """Set energy cost (universal method)"""
        self.parameters['energy_cost'] = cost
    
    def get_resource_cost(self, resource_type: str) -> Optional[float]:
        """Get resource cost for specific resource type (universal method)"""
        costs = self.parameters.get('resource_costs', {})
        return costs.get(resource_type, 0.0)
    
    def set_resource_cost(self, resource_type: str, cost: float) -> None:
        """Set resource cost for specific resource type (universal method)"""
        if 'resource_costs' not in self.parameters:
            self.parameters['resource_costs'] = {}
        self.parameters['resource_costs'][resource_type] = cost
    
    def get_priority(self) -> Optional[float]:
        """Get action priority (universal method)"""
        return self.parameters.get('priority', 0.0)
    
    def set_priority(self, priority: float) -> None:
        """Set action priority (universal method)"""
        self.parameters['priority'] = priority
    
    def get_duration(self) -> Optional[float]:
        """Get action duration (universal method)"""
        return self.parameters.get('duration', 1.0)
    
    def set_duration(self, duration: float) -> None:
        """Set action duration (universal method)"""
        self.parameters['duration'] = duration
    
    def is_movement_action(self) -> bool:
        """Check if this is a movement action (universal method)"""
        return self.action_type in ['move', 'walk', 'run', 'fly', 'swim', 'jump']
    
    def is_interaction_action(self) -> bool:
        """Check if this is an interaction action (universal method)"""
        return self.action_type in ['interact', 'communicate', 'trade', 'fight', 'cooperate']
    
    def is_resource_action(self) -> bool:
        """Check if this is a resource-related action (universal method)"""
        return self.action_type in ['collect', 'consume', 'produce', 'store', 'trade']
    
    def is_communication_action(self) -> bool:
        """Check if this is a communication action (universal method)"""
        return self.action_type in ['communicate', 'signal', 'broadcast', 'whisper']
    
    def get_total_cost(self) -> Dict[str, float]:
        """Get total cost of this action (universal method)"""
        costs = {
            'energy': self.get_energy_cost(),
            'time': self.get_duration()
        }
        
        # Add resource costs
        resource_costs = self.parameters.get('resource_costs', {})
        costs.update(resource_costs)
        
        return costs
    
    def can_afford(self, agent_resources: Dict[str, float]) -> bool:
        """Check if agent can afford this action (universal method)"""
        total_costs = self.get_total_cost()
        
        for resource_type, cost in total_costs.items():
            if resource_type == 'energy':
                if agent_resources.get('energy', 0.0) < cost:
                    return False
            elif resource_type == 'time':
                # Time is always available
                continue
            else:
                if agent_resources.get(resource_type, 0.0) < cost:
                    return False
        
        return True
    
    def get_description(self) -> str:
        """Get human-readable description of action (universal method)"""
        desc = f"Agent {self.agent_id} {self.action_type}"
        
        if self.get_target_position():
            desc += f" to {self.get_target_position()}"
        
        if self.get_direction():
            desc += f" in direction {self.get_direction()}"
        
        if self.get_speed():
            desc += f" at speed {self.get_speed()}"
        
        return desc
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary (universal method)"""
        return {
            'agent_id': self.agent_id,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Create action from dictionary (universal method)"""
        return cls(
            agent_id=data['agent_id'],
            action_type=data['action_type'],
            parameters=data['parameters'],
            timestamp=data.get('timestamp')
        )
    
    def __str__(self) -> str:
        """String representation of action"""
        return self.get_description()
    
    def __eq__(self, other) -> bool:
        """Check if two actions are equal"""
        if not isinstance(other, Action):
            return False
        return (self.agent_id == other.agent_id and 
                self.action_type == other.action_type and 
                self.parameters == other.parameters)


@dataclass
class ActionResult:
    """
    Result of action execution.
    Memory allocation: <50 bytes
    """
    agent_id: AgentID
    success: bool
    new_position: Optional[Position] = None
    error_message: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class AgentState:
    """Complete agent state for serialization"""
    agent_id: AgentID
    position: Position
    metadata: Dict[str, Any]
    history: CircularBuffer
    internal_state: Dict[str, Any]


# Exception types for error handling
class LAMBError(Exception):
    """Base exception for LAMB framework"""
    pass


class AgentNotFoundError(LAMBError):
    """Agent not found in environment"""
    pass


class InvalidEnvironmentError(LAMBError):
    """Environment is corrupted or invalid"""
    pass


class InvalidObservationError(LAMBError):
    """Observation is malformed"""
    pass


class EngineTimeoutError(LAMBError):
    """Engine operation timed out"""
    pass


class IncompatibleEngineError(LAMBError):
    """Engine incompatible with paradigm"""
    pass


class InvalidActionError(LAMBError):
    """Action is invalid or impossible"""
    pass


class ConflictError(LAMBError):
    """Conflicting actions detected"""
    pass


class EnvironmentConstraintError(LAMBError):
    """Action violates environment constraints"""
    pass


class StateConsistencyError(LAMBError):
    """State validation failed"""
    pass


# Performance monitoring types
@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    agent_throughput: float  # agents/second
    memory_usage_mb: float
    decision_latency_ms: float
    cache_hit_rate: float
    error_rate: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
