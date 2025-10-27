"""
Base spatial index interface for the LAMB framework.

Based on Technical_Specification.md Section 1.2 and paradigm-specific sections.
Provides common interface for Grid, KD-tree, and Graph spatial indexing.

Performance characteristics by paradigm:
- GridIndex (discrete space): O(r²) neighbor queries, O(1) updates
- KDTreeIndex (continuous space): O(log n + k) queries, O(n log n) rebuilds  
- GraphIndex (network space): O(1) direct neighbors, O(d^h) for h-hop
"""

from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Any, Optional
import time

from ..core.types import AgentID, Position


class SpatialIndex(ABC):
    """
    Abstract base class for spatial indexing structures.
    
    All spatial indices must implement these core operations
    with paradigm-specific performance characteristics.
    """
    
    def __init__(self):
        """Initialize spatial index"""
        self.agent_positions = {}  # AgentID -> Position mapping
        self.performance_metrics = {
            'total_queries': 0,
            'total_updates': 0,
            'avg_query_time': 0.0,
            'avg_update_time': 0.0,
            'last_rebuild_time': 0.0
        }
    
    @abstractmethod
    def add_agent(self, agent_id: AgentID, position: Position) -> None:
        """
        Add agent to spatial index.
        
        Args:
            agent_id: Unique agent identifier
            position: Agent's position
        """
        pass
    
    @abstractmethod
    def remove_agent(self, agent_id: AgentID) -> None:
        """
        Remove agent from spatial index.
        
        Args:
            agent_id: Agent to remove
        """
        pass
    
    @abstractmethod
    def update_agent(self, agent_id: AgentID, old_position: Position, new_position: Position) -> None:
        """
        Update agent's position in spatial index.
        
        Args:
            agent_id: Agent to update
            old_position: Previous position
            new_position: New position
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, agent_id: AgentID, radius: float) -> List[AgentID]:
        """
        Get neighboring agents within radius.
        
        Args:
            agent_id: Agent to find neighbors for
            radius: Search radius (paradigm-specific units)
            
        Returns:
            List of neighboring agent IDs
        """
        pass
    
    @abstractmethod
    def get_neighbors_at_position(self, position: Position, radius: float) -> List[AgentID]:
        """
        Get neighboring agents at specific position.
        
        Args:
            position: Position to search around
            radius: Search radius
            
        Returns:
            List of neighboring agent IDs
        """
        pass
    
    def get_agent_position(self, agent_id: AgentID) -> Optional[Position]:
        """Get agent's current position"""
        return self.agent_positions.get(agent_id)
    
    def get_all_agents(self) -> List[AgentID]:
        """Get all agents in the spatial index"""
        return list(self.agent_positions.keys())
    
    def get_agent_count(self) -> int:
        """Get total number of agents"""
        return len(self.agent_positions)
    
    def clear(self) -> None:
        """Clear all agents from spatial index"""
        self.agent_positions.clear()
        self._clear_internal_structures()
    
    @abstractmethod
    def _clear_internal_structures(self) -> None:
        """Clear paradigm-specific internal structures"""
        pass
    
    def get_performance_metrics(self) -> dict:
        """Get spatial index performance metrics"""
        return {
            'total_queries': self.performance_metrics['total_queries'],
            'total_updates': self.performance_metrics['total_updates'],
            'avg_query_time': self.performance_metrics['avg_query_time'],
            'avg_update_time': self.performance_metrics['avg_update_time'],
            'last_rebuild_time': self.performance_metrics['last_rebuild_time'],
            'agent_count': self.get_agent_count(),
            'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    @abstractmethod
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        pass
    
    def _record_query_time(self, query_time: float) -> None:
        """Record query performance"""
        self.performance_metrics['total_queries'] += 1
        total = self.performance_metrics['total_queries']
        current_avg = self.performance_metrics['avg_query_time']
        self.performance_metrics['avg_query_time'] = (
            (current_avg * (total - 1) + query_time) / total
        )
    
    def _record_update_time(self, update_time: float) -> None:
        """Record update performance"""
        self.performance_metrics['total_updates'] += 1
        total = self.performance_metrics['total_updates']
        current_avg = self.performance_metrics['avg_update_time']
        self.performance_metrics['avg_update_time'] = (
            (current_avg * (total - 1) + update_time) / total
        )
    
    def validate_consistency(self) -> bool:
        """
        Validate spatial index consistency.
        
        Returns True if index is consistent, False otherwise.
        Used for debugging and testing.
        """
        try:
            # Check that all agents in positions are in internal structures
            for agent_id in self.agent_positions:
                if not self._agent_exists_in_structures(agent_id):
                    return False
            
            # Check paradigm-specific consistency
            return self._validate_internal_consistency()
            
        except Exception:
            return False
    
    @abstractmethod
    def _agent_exists_in_structures(self, agent_id: AgentID) -> bool:
        """Check if agent exists in internal structures"""
        pass
    
    @abstractmethod
    def _validate_internal_consistency(self) -> bool:
        """Validate paradigm-specific consistency"""
        pass
    
    def __len__(self) -> int:
        """Return number of agents in spatial index"""
        return len(self.agent_positions)
    
    def __contains__(self, agent_id: AgentID) -> bool:
        """Check if agent exists in spatial index"""
        return agent_id in self.agent_positions
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agents={len(self.agent_positions)})"


def select_spatial_index(agent_count: int, paradigm: str) -> str:
    """
    Automatic algorithm selection based on reconnaissance findings.
    
    From Technical_Specification.md Section 1.2:
    
    Args:
        agent_count: Number of agents in simulation
        paradigm: Paradigm type ("grid", "physics", "network")
        
    Returns:
        Recommended spatial index class name
    """
    if paradigm == "grid":
        # Grid paradigm: Always use GridIndex for discrete space
        return "GridIndex"
    
    elif paradigm == "physics":
        # Physics paradigm: KD-tree for continuous space
        if agent_count < 1000:
            return "SimpleKDTree"  # Lower overhead for small simulations
        else:
            return "OptimizedKDTree"  # Bulk operations for large simulations
    
    elif paradigm == "network":
        # Network paradigm: Graph-based adjacency
        return "GraphIndex"
    
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")
