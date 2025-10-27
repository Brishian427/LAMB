"""
BaseEnvironment interface implementation based on Technical_Specification.md Section 1.2.

The BaseEnvironment class manages spatial representation, agent registry, and state updates
with paradigm-specific optimizations for performance.

Performance targets (validated from reconnaissance):
- Agent lookup: O(1) - <0.0001s
- Neighbor query: O(log n) average, O(r²) for grid - <0.01s
- State update: O(1) per agent - <0.0001s
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import time
import threading
import pickle
import json

from .types import (
    AgentID, Position, Observation, Action, ActionResult,
    LAMBError, AgentNotFoundError, InvalidEnvironmentError,
    ConflictError, EnvironmentConstraintError
)


class BaseEnvironment(ABC):
    """
    Universal environment interface for all paradigms.
    
    Manages spatial representation, agent registry, and state updates
    with paradigm-specific optimizations for performance.
    """
    
    def __init__(self):
        """Initialize BaseEnvironment"""
        # Agent registry for O(1) agent lookup
        self.agent_registry: Dict[AgentID, 'BaseAgent'] = {}
        
        # Global environment state
        self.global_state: Dict[str, Any] = {}
        
        # Simulation step counter
        self.step_counter: int = 0
        
        # Thread lock for state modification
        self._lock = threading.Lock()
        
        # Performance tracking
        self._performance_metrics = {
            'total_agents': 0,
            'total_queries': 0,
            'total_updates': 0,
            'avg_query_time': 0.0,
            'avg_update_time': 0.0
        }
        
        # Spatial index will be set by subclasses
        self.spatial_index = None
    
    @abstractmethod
    def get_neighbors(self, agent_id: AgentID, radius: float) -> List[AgentID]:
        """
        Get neighboring agents within radius.
        
        Performance targets:
        - Grid: O(r²) - <0.001s for radius ≤5 cells
        - Physics: O(log n + k) - <0.01s for radius ≤10 units  
        - Network: O(1) - <0.001s for 1-hop neighbors
        
        Args:
            agent_id: Agent to find neighbors for
            radius: Search radius (paradigm-specific units)
            
        Returns:
            List of neighboring agent IDs
            
        Raises:
            AgentNotFoundError: If agent not in environment
        """
        pass
    
    def add_agent(self, agent: 'BaseAgent') -> None:
        """
        Add agent to environment.
        
        Performance target: <0.0001s per agent
        Thread safety: Uses lock for registry modification
        """
        with self._lock:
            if agent.agent_id in self.agent_registry:
                raise LAMBError(f"Agent {agent.agent_id} already exists")
            
            self.agent_registry[agent.agent_id] = agent
            self._performance_metrics['total_agents'] += 1
            
            # Add to spatial index (implemented by subclasses)
            if self.spatial_index is not None:
                self.spatial_index.add_agent(agent.agent_id, agent.position)
    
    def remove_agent(self, agent_id: AgentID) -> None:
        """
        Remove agent from environment.
        
        Performance target: <0.0001s per agent
        Thread safety: Uses lock for registry modification
        """
        with self._lock:
            if agent_id not in self.agent_registry:
                raise AgentNotFoundError(f"Agent {agent_id} not found")
            
            agent = self.agent_registry[agent_id]
            
            # Remove from spatial index
            if self.spatial_index:
                self.spatial_index.remove_agent(agent_id)
            
            del self.agent_registry[agent_id]
            self._performance_metrics['total_agents'] -= 1
    
    def get_agent(self, agent_id: AgentID) -> 'BaseAgent':
        """
        Get agent by ID.
        
        Performance target: O(1) - <0.0001s
        Thread safety: Read-only operation, thread-safe
        """
        if agent_id not in self.agent_registry:
            raise AgentNotFoundError(f"Agent {agent_id} not found")
        
        return self.agent_registry[agent_id]
    
    def update_agent_state(self, agent_id: AgentID, new_position: Position, _lock_held: bool = False) -> bool:
        """
        Update agent's position in environment.
        
        Performance target: O(1) per agent - <0.0001s
        Thread safety: Uses lock for state modification unless _lock_held is True
        """
        start_time = time.perf_counter()
        
        def _update_internal():
            if agent_id not in self.agent_registry:
                raise AgentNotFoundError(f"Agent {agent_id} not found")
            
            agent = self.agent_registry[agent_id]
            old_position = agent.position
            
            # Validate new position (implemented by subclasses)
            if not self._is_valid_position(new_position):
                return False
            
            # Update spatial index
            if self.spatial_index is not None:
                self.spatial_index.update_agent(agent_id, old_position, new_position)
            
            # Update agent position
            agent.position = new_position
            
            # Update performance metrics
            update_time = time.perf_counter() - start_time
            self._update_performance_metrics('update', update_time)
            
            return True
        
        if _lock_held:
            return _update_internal()
        else:
            with self._lock:
                return _update_internal()
    
    @abstractmethod
    def _is_valid_position(self, position: Position) -> bool:
        """Validate if position is valid for this environment"""
        pass
    
    def resolve_conflicts(self, actions: List[Action]) -> List[ActionResult]:
        """
        Resolve conflicting actions.
        
        Performance target: <0.01s per conflict set
        Thread safety: Uses lock for conflict resolution
        
        Args:
            actions: List of potentially conflicting actions
            
        Returns:
            List of action results with conflicts resolved
        """
        with self._lock:
            results = []
            
            # Group actions by target position to detect conflicts
            position_conflicts = {}
            for action in actions:
                target_pos = self._get_target_position(action)
                if target_pos not in position_conflicts:
                    position_conflicts[target_pos] = []
                position_conflicts[target_pos].append(action)
            
            # Resolve conflicts
            for target_pos, conflicting_actions in position_conflicts.items():
                if len(conflicting_actions) == 1:
                    # No conflict
                    action = conflicting_actions[0]
                    results.append(self._execute_action(action))
                else:
                    # Conflict - resolve using priority or random selection
                    resolved_results = self._resolve_position_conflict(conflicting_actions)
                    results.extend(resolved_results)
            
            return results
    
    @abstractmethod
    def _get_target_position(self, action: Action) -> Position:
        """Get target position for action (paradigm-specific)"""
        pass
    
    @abstractmethod
    def _execute_action(self, action: Action) -> ActionResult:
        """Execute single action (paradigm-specific)"""
        pass
    
    @abstractmethod
    def _resolve_position_conflict(self, actions: List[Action]) -> List[ActionResult]:
        """Resolve conflict between actions targeting same position"""
        pass
    
    def get_all_observations(self) -> Dict[AgentID, Observation]:
        """
        Get observations for all agents (bulk operation for performance).
        
        Performance target: Linear scaling up to 10,000 agents
        """
        observations = {}
        
        for agent_id, agent in self.agent_registry.items():
            try:
                observations[agent_id] = agent.observe(self)
            except Exception as e:
                # Log error but continue with other agents
                print(f"Warning: Failed to get observation for agent {agent_id}: {e}")
        
        return observations
    
    def apply_all_actions(self, actions: Dict[AgentID, Action]) -> Dict[AgentID, ActionResult]:
        """
        Apply actions for all agents (bulk operation for performance).
        
        Performance target: Linear scaling up to 10,000 agents
        """
        # Convert to list for conflict resolution
        action_list = list(actions.values())
        
        # Resolve conflicts
        results = self.resolve_conflicts(action_list)
        
        # Convert back to dictionary
        result_dict = {}
        for result in results:
            result_dict[result.agent_id] = result
        
        return result_dict
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get global environment state"""
        return self.global_state.copy()
    
    def set_global_state(self, key: str, value: Any) -> None:
        """
        Set global environment state.
        
        Performance target: <0.001s per change
        Thread safety: Uses lock for state modification
        """
        with self._lock:
            self.global_state[key] = value
    
    def step(self) -> None:
        """
        Advance environment by one simulation step.
        
        Updates step counter and performs any environment-specific updates.
        """
        with self._lock:
            self.step_counter += 1
            self._update_environment_state()
    
    @abstractmethod
    def _update_environment_state(self) -> None:
        """Update environment-specific state (implemented by subclasses)"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            Dictionary containing environment state information
        """
        with self._lock:
            return {
                "step_counter": self.step_counter,
                "num_agents": len(self.agent_registry),
                "global_state": self.global_state.copy(),
                "agent_ids": list(self.agent_registry.keys())
            }
    
    def serialize_state(self) -> bytes:
        """
        Serialize environment state to bytes.
        
        Performance targets:
        - Pickle format: <0.1s for 10,000 agents (fastest)
        - Memory overhead: 2x during serialization (temporary)
        """
        state = {
            'global_state': self.global_state,
            'step_counter': self.step_counter,
            'agent_states': {
                agent_id: agent.get_state() 
                for agent_id, agent in self.agent_registry.items()
            }
        }
        
        return pickle.dumps(state)
    
    def restore_state(self, data: bytes) -> None:
        """Restore environment state from bytes"""
        state = pickle.loads(data)
        
        with self._lock:
            self.global_state = state['global_state']
            self.step_counter = state['step_counter']
            
            # Restore agents
            self.agent_registry.clear()
            for agent_id, agent_state in state['agent_states'].items():
                # This would need agent factory - simplified for now
                # agent = create_agent_from_state(agent_state)
                # self.agent_registry[agent_id] = agent
                pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get environment performance metrics"""
        return {
            'total_agents': self._performance_metrics['total_agents'],
            'total_queries': self._performance_metrics['total_queries'],
            'total_updates': self._performance_metrics['total_updates'],
            'avg_query_time': self._performance_metrics['avg_query_time'],
            'avg_update_time': self._performance_metrics['avg_update_time'],
            'step_counter': self.step_counter
        }
    
    def _update_performance_metrics(self, operation: str, time_taken: float) -> None:
        """Update performance metrics"""
        if operation == 'query':
            self._performance_metrics['total_queries'] += 1
            # Running average
            total = self._performance_metrics['total_queries']
            current_avg = self._performance_metrics['avg_query_time']
            self._performance_metrics['avg_query_time'] = (
                (current_avg * (total - 1) + time_taken) / total
            )
        elif operation == 'update':
            self._performance_metrics['total_updates'] += 1
            total = self._performance_metrics['total_updates']
            current_avg = self._performance_metrics['avg_update_time']
            self._performance_metrics['avg_update_time'] = (
                (current_avg * (total - 1) + time_taken) / total
            )
    
    def validate_state(self) -> bool:
        """
        Validate environment state consistency.
        
        Returns True if state is consistent, False otherwise.
        Used for debugging and testing.
        """
        try:
            # Check agent registry consistency
            for agent_id, agent in self.agent_registry.items():
                if agent.agent_id != agent_id:
                    return False
                
                # Check if agent position is valid
                if not self._is_valid_position(agent.position):
                    return False
            
            # Additional paradigm-specific validation
            return self._validate_paradigm_state()
            
        except Exception:
            return False
    
    @abstractmethod
    def _validate_paradigm_state(self) -> bool:
        """Validate paradigm-specific state consistency"""
        pass
    
    def __len__(self) -> int:
        """Return number of agents in environment"""
        return len(self.agent_registry)
    
    def __contains__(self, agent_id: AgentID) -> bool:
        """Check if agent exists in environment"""
        return agent_id in self.agent_registry
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agents={len(self.agent_registry)}, step={self.step_counter})"
    
    # ============================================================================
    # UNIVERSAL METHODS - Available to all paradigms
    # ============================================================================
    
    def get_agents(self) -> List['BaseAgent']:
        """Get all agents in environment (universal method)"""
        return list(self.agent_registry.values())
    
    def get_agent(self, agent_id: AgentID) -> Optional['BaseAgent']:
        """Get specific agent by ID (universal method)"""
        return self.agent_registry.get(agent_id)
    
    def get_agent_count(self) -> int:
        """Get total number of agents (universal method)"""
        return len(self.agent_registry)
    
    def get_agent_ids(self) -> List[AgentID]:
        """Get all agent IDs (universal method)"""
        return list(self.agent_registry.keys())
    
    def get_agents_at_position(self, position: Position) -> List['BaseAgent']:
        """Get all agents at specific position (universal method)"""
        agents = []
        for agent in self.agent_registry.values():
            if self._positions_equal(agent.position, position):
                agents.append(agent)
        return agents
    
    def get_agent_at_position(self, position: Position) -> Optional['BaseAgent']:
        """Get first agent at specific position (universal method)"""
        agents = self.get_agents_at_position(position)
        return agents[0] if agents else None
    
    def is_position_occupied(self, position: Position) -> bool:
        """Check if position is occupied (universal method)"""
        return len(self.get_agents_at_position(position)) > 0
    
    def get_empty_positions(self) -> List[Position]:
        """Get all empty positions (universal method)"""
        # This is paradigm-specific, but we provide a default implementation
        return self._get_empty_positions_default()
    
    def get_random_empty_position(self) -> Optional[Position]:
        """Get a random empty position (universal method)"""
        empty_positions = self.get_empty_positions()
        if empty_positions:
            import random
            return random.choice(empty_positions)
        return None
    
    def get_neighbors(self, agent_id: AgentID, radius: float = 1.0) -> List[AgentID]:
        """Get neighbors of agent (universal method - delegates to paradigm-specific)"""
        return self._get_neighbors_impl(agent_id, radius)
    
    def get_adjacent_agents(self, agent_id: AgentID, radius: float = 1.0) -> List[AgentID]:
        """Get adjacent agents (universal method - same as get_neighbors)"""
        return self.get_neighbors(agent_id, radius)
    
    def get_agents_in_radius(self, position: Position, radius: float) -> List[AgentID]:
        """Get all agents within radius of position (universal method)"""
        agents_in_radius = []
        for agent_id, agent in self.agent_registry.items():
            if self._distance(agent.position, position) <= radius:
                agents_in_radius.append(agent_id)
        return agents_in_radius
    
    def get_closest_agent(self, position: Position, exclude_agent: AgentID = None) -> Optional[AgentID]:
        """Get closest agent to position (universal method)"""
        closest_agent = None
        closest_distance = float('inf')
        
        for agent_id, agent in self.agent_registry.items():
            if exclude_agent and agent_id == exclude_agent:
                continue
                
            distance = self._distance(agent.position, position)
            if distance < closest_distance:
                closest_distance = distance
                closest_agent = agent_id
        
        return closest_agent
    
    def get_global_state(self, key: str = None) -> Any:
        """Get global environment state (universal method)"""
        if key is None:
            return self.global_state
        return self.global_state.get(key)
    
    def set_global_state(self, key: str, value: Any) -> None:
        """Set global environment state (universal method)"""
        self.global_state[key] = value
    
    def update_global_state(self, updates: Dict[str, Any]) -> None:
        """Update global environment state (universal method)"""
        self.global_state.update(updates)
    
    def get_step_counter(self) -> int:
        """Get current step counter (universal method)"""
        return self.step_counter
    
    def increment_step(self) -> None:
        """Increment step counter (universal method)"""
        self.step_counter += 1
    
    def reset_step_counter(self) -> None:
        """Reset step counter to 0 (universal method)"""
        self.step_counter = 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get environment performance metrics (universal method)"""
        return self._performance_metrics.copy()
    
    def get_agent_positions(self) -> Dict[AgentID, Position]:
        """Get all agent positions (universal method)"""
        return {agent_id: agent.position for agent_id, agent in self.agent_registry.items()}
    
    def get_position_bounds(self) -> Dict[str, float]:
        """Get environment bounds (universal method)"""
        return self._get_position_bounds_impl()
    
    def is_position_valid(self, position: Position) -> bool:
        """Check if position is valid (universal method)"""
        return self._is_valid_position(position)
    
    def get_environment_size(self) -> Dict[str, Any]:
        """Get environment size information (universal method)"""
        return self._get_environment_size_impl()
    
    def get_resource_at_position(self, position: Position, resource_type: str = None) -> Any:
        """Get resource at position (universal method)"""
        return self._get_resource_at_position_impl(position, resource_type)
    
    def set_resource_at_position(self, position: Position, resource_type: str, amount: Any) -> None:
        """Set resource at position (universal method)"""
        self._set_resource_at_position_impl(position, resource_type, amount)
    
    def add_resource_at_position(self, position: Position, resource_type: str, amount: Any) -> None:
        """Add resource at position (universal method)"""
        current = self.get_resource_at_position(position, resource_type)
        if current is not None:
            self.set_resource_at_position(position, resource_type, current + amount)
        else:
            self.set_resource_at_position(position, resource_type, amount)
    
    def remove_resource_at_position(self, position: Position, resource_type: str, amount: Any) -> bool:
        """Remove resource at position (universal method)"""
        current = self.get_resource_at_position(position, resource_type)
        if current is not None and current >= amount:
            self.set_resource_at_position(position, resource_type, current - amount)
            return True
        return False
    
    def get_all_resources(self) -> Dict[Position, Dict[str, Any]]:
        """Get all resources in environment (universal method)"""
        return self._get_all_resources_impl()
    
    def clear_resources(self) -> None:
        """Clear all resources (universal method)"""
        self._clear_resources_impl()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information (universal method)"""
        return {
            'agent_count': self.get_agent_count(),
            'step_counter': self.get_step_counter(),
            'global_state': self.get_global_state(),
            'position_bounds': self.get_position_bounds(),
            'environment_size': self.get_environment_size(),
            'performance_metrics': self.get_performance_metrics()
        }
    
    # ============================================================================
    # HELPER METHODS - Default implementations that can be overridden
    # ============================================================================
    
    def _positions_equal(self, pos1: Position, pos2: Position) -> bool:
        """Check if two positions are equal (default implementation)"""
        return pos1 == pos2
    
    def _distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate distance between positions (default implementation)"""
        # This should be overridden by paradigm-specific classes
        if hasattr(pos1, '__len__') and hasattr(pos2, '__len__'):
            if len(pos1) == 2 and len(pos2) == 2:
                return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        return 0.0
    
    def _get_empty_positions_default(self) -> List[Position]:
        """Default implementation for getting empty positions"""
        # This should be overridden by paradigm-specific classes
        return []
    
    def _get_position_bounds_impl(self) -> Dict[str, float]:
        """Get position bounds (to be implemented by paradigm classes)"""
        return {}
    
    def _get_environment_size_impl(self) -> Dict[str, Any]:
        """Get environment size (to be implemented by paradigm classes)"""
        return {}
    
    def _get_resource_at_position_impl(self, position: Position, resource_type: str = None) -> Any:
        """Get resource at position (to be implemented by paradigm classes)"""
        return None
    
    def _set_resource_at_position_impl(self, position: Position, resource_type: str, amount: Any) -> None:
        """Set resource at position (to be implemented by paradigm classes)"""
        pass
    
    def _get_all_resources_impl(self) -> Dict[Position, Dict[str, Any]]:
        """Get all resources (to be implemented by paradigm classes)"""
        return {}
    
    def _clear_resources_impl(self) -> None:
        """Clear all resources (to be implemented by paradigm classes)"""
        pass
