"""
Grid environment implementation for discrete space models.

Based on Technical_Specification.md Section 2.1: Grid Paradigm Specification.
Manages discrete grid space with cell-based agent positioning and interactions.

Performance characteristics:
- Agent lookup: O(1) - <0.0001s
- Neighbor query: O(r²) - <0.001s for radius ≤5 cells
- State update: O(1) per agent - <0.0001s
- Memory: 100 bytes per agent + 8 bytes per occupied cell
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import random
from collections import defaultdict

from ...core.base_environment import BaseEnvironment
from ...core.types import (
    AgentID, GridCoord, Action, ActionResult, BoundaryCondition,
    ConflictError, EnvironmentConstraintError
)
from ...spatial.grid_index import GridIndex
from .grid_agent import GridAgent


class GridEnvironment(BaseEnvironment):
    """
    Environment for discrete grid-based models.
    
    Features:
    - Discrete cell-based positioning
    - Moore and Von Neumann neighborhoods
    - Configurable boundary conditions (WRAP, WALL, INFINITE)
    - Resource management per cell
    - Conflict resolution for movement
    """
    
    def __init__(
        self,
        dimensions: Tuple[int, int] = (100, 100),
        boundary_condition: BoundaryCondition = BoundaryCondition.WRAP,
        cell_size: float = 1.0,
        max_agents_per_cell: int = 1
    ):
        """
        Initialize GridEnvironment.
        
        Args:
            dimensions: Grid size (width, height)
            boundary_condition: How to handle boundaries
            cell_size: Size of each cell (usually 1.0 for discrete)
            max_agents_per_cell: Maximum agents allowed per cell
        """
        super().__init__()
        
        self.dimensions = dimensions
        self.boundary_condition = boundary_condition
        self.cell_size = cell_size
        self.max_agents_per_cell = max_agents_per_cell
        
        # Initialize spatial index
        self.spatial_index = GridIndex(
            dimensions=dimensions,
            boundary_condition=boundary_condition,
            cell_size=cell_size
        )
        
        # Cell resources and properties
        self.cell_resources: Dict[GridCoord, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cell_properties: Dict[GridCoord, Dict[str, Any]] = defaultdict(dict)
        
        # Movement conflicts tracking
        self._movement_conflicts: Dict[GridCoord, List[AgentID]] = defaultdict(list)
    
    def get_neighbors(self, agent_id: AgentID, radius: float) -> List[AgentID]:
        """
        Get neighboring agents within radius.
        
        Performance target: <0.001s for radius ≤5 cells
        Uses GridIndex for O(r²) Moore neighborhood queries
        """
        return self.spatial_index.get_neighbors(agent_id, radius)
    
    def get_neighbors_von_neumann(self, agent_id: AgentID, radius: float) -> List[AgentID]:
        """Get neighbors using Von Neumann topology (4-connected)"""
        if agent_id not in self.agent_registry:
            return []
        
        position = self.agent_registry[agent_id].position
        return self.spatial_index.get_neighbors_von_neumann(position, radius)
    
    def _is_valid_position(self, position: GridCoord) -> bool:
        """Validate if position is valid for this grid"""
        if not isinstance(position, tuple) or len(position) != 2:
            return False
        
        x, y = position
        
        if self.boundary_condition == BoundaryCondition.WALL:
            return (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1])
        elif self.boundary_condition == BoundaryCondition.WRAP:
            return True  # All positions valid with wrapping
        elif self.boundary_condition == BoundaryCondition.INFINITE:
            return True  # All positions valid in infinite grid
        else:
            return True
    
    def _get_target_position(self, action: Action) -> GridCoord:
        """Get target position for action"""
        if action.action_type == "move":
            target_pos = action.parameters.get('target_position')
            if target_pos:
                return tuple(target_pos)
            
            # Calculate from direction
            agent = self.get_agent(action.agent_id)
            direction = action.parameters.get('direction', 'stay')
            
            if direction == 'stay':
                return agent.position
            
            # Use agent's movement calculation
            if isinstance(agent, GridAgent):
                return agent._calculate_new_position(direction)
        
        # For non-movement actions, target is current position
        agent = self.get_agent(action.agent_id)
        return agent.position
    
    def _execute_action(self, action: Action) -> ActionResult:
        """Execute single action"""
        agent = self.get_agent(action.agent_id)
        return agent.act(action, self)
    
    def _resolve_position_conflict(self, actions: List[Action]) -> List[ActionResult]:
        """
        Resolve conflict between actions targeting same position.
        
        Strategy: Random selection for movement conflicts
        """
        results = []
        
        if len(actions) <= self.max_agents_per_cell:
            # No conflict - execute all actions
            for action in actions:
                results.append(self._execute_action(action))
        else:
            # Conflict - randomly select winners
            winners = random.sample(actions, self.max_agents_per_cell)
            
            for action in actions:
                if action in winners:
                    results.append(self._execute_action(action))
                else:
                    # Blocked action
                    results.append(ActionResult(
                        agent_id=action.agent_id,
                        success=False,
                        error_message="Movement blocked by conflict"
                    ))
        
        return results
    
    def _update_environment_state(self) -> None:
        """Update grid-specific environment state"""
        # Update resource regeneration
        self._regenerate_resources()
        
        # Update cell properties based on occupancy
        self._update_cell_properties()
        
        # Clear movement conflicts from previous step
        self._movement_conflicts.clear()
    
    def _regenerate_resources(self) -> None:
        """Regenerate resources in cells (simple implementation)"""
        # This would be customized based on specific model needs
        for cell_pos in list(self.cell_resources.keys()):
            resources = self.cell_resources[cell_pos]
            for resource_type in resources:
                # Simple regeneration: increase by 1% per step, max 100
                current = resources[resource_type]
                if current < 100:
                    resources[resource_type] = min(100, current * 1.01)
    
    def _update_cell_properties(self) -> None:
        """Update cell properties based on current state"""
        # Update occupancy information
        for cell_pos in self.spatial_index.occupied_cells:
            agents_in_cell = self.spatial_index.cells.get(cell_pos, set())
            self.cell_properties[cell_pos]['occupancy'] = len(agents_in_cell)
            self.cell_properties[cell_pos]['agents'] = list(agents_in_cell)
    
    def get_cell_info(self, position: GridCoord) -> Optional[Dict[str, Any]]:
        """Get information about a specific cell"""
        if not self._is_valid_position(position):
            return None
        
        # Apply boundary condition to get actual position
        actual_pos = self._apply_boundary_condition(position)
        if actual_pos is None:
            return None
        
        agents_in_cell = list(self.spatial_index.cells.get(actual_pos, set()))
        
        return {
            'position': actual_pos,
            'agents': agents_in_cell,
            'agent_count': len(agents_in_cell),
            'resources': dict(self.cell_resources[actual_pos]),
            'properties': dict(self.cell_properties[actual_pos])
        }
    
    def _apply_boundary_condition(self, position: GridCoord) -> Optional[GridCoord]:
        """Apply boundary condition to position"""
        x, y = position
        
        if self.boundary_condition == BoundaryCondition.WRAP:
            # Toroidal topology
            wrapped_x = x % self.dimensions[0]
            wrapped_y = y % self.dimensions[1]
            return (wrapped_x, wrapped_y)
        
        elif self.boundary_condition == BoundaryCondition.WALL:
            # Reflective boundaries
            if x < 0 or x >= self.dimensions[0] or y < 0 or y >= self.dimensions[1]:
                return None  # Out of bounds
            return (x, y)
        
        elif self.boundary_condition == BoundaryCondition.INFINITE:
            # Infinite grid
            return (x, y)
        
        else:
            return (x, y)
    
    def consume_resource(self, agent_id: AgentID, resource_type: str, amount: float) -> bool:
        """
        Allow agent to consume resource from current cell.
        
        Args:
            agent_id: Agent consuming resource
            resource_type: Type of resource to consume
            amount: Amount to consume
            
        Returns:
            True if consumption successful, False otherwise
        """
        if agent_id not in self.agent_registry:
            return False
        
        agent_pos = self.agent_registry[agent_id].position
        available = self.cell_resources[agent_pos][resource_type]
        
        if available >= amount:
            self.cell_resources[agent_pos][resource_type] -= amount
            return True
        
        return False
    
    def produce_resource(self, agent_id: AgentID, resource_type: str, amount: float) -> bool:
        """
        Allow agent to produce resource in current cell.
        
        Args:
            agent_id: Agent producing resource
            resource_type: Type of resource to produce
            amount: Amount to produce
            
        Returns:
            True if production successful, False otherwise
        """
        if agent_id not in self.agent_registry:
            return False
        
        agent_pos = self.agent_registry[agent_id].position
        self.cell_resources[agent_pos][resource_type] += amount
        return True
    
    def process_interaction(self, agent_id: AgentID, target: Any, interaction_type: str) -> Dict[str, Any]:
        """
        Process interaction between agent and target.
        
        Args:
            agent_id: Agent initiating interaction
            target: Target of interaction (agent_id, cell position, etc.)
            interaction_type: Type of interaction
            
        Returns:
            Result of interaction
        """
        # Simple interaction implementation
        result = {
            'success': True,
            'interaction_type': interaction_type,
            'agent_id': agent_id,
            'target': target,
            'timestamp': self.step_counter
        }
        
        # Log interaction in global state
        if 'interactions' not in self.global_state:
            self.global_state['interactions'] = []
        
        self.global_state['interactions'].append(result)
        
        return result
    
    def set_cell_resource(self, position: GridCoord, resource_type: str, amount: float) -> None:
        """Set resource amount in specific cell"""
        actual_pos = self._apply_boundary_condition(position)
        if actual_pos:
            self.cell_resources[actual_pos][resource_type] = amount
    
    def get_cell_resource(self, position: GridCoord, resource_type: str) -> float:
        """Get resource amount in specific cell"""
        actual_pos = self._apply_boundary_condition(position)
        if actual_pos:
            return self.cell_resources[actual_pos][resource_type]
        return 0.0
    
    def set_cell_property(self, position: GridCoord, property_name: str, value: Any) -> None:
        """Set property of specific cell"""
        actual_pos = self._apply_boundary_condition(position)
        if actual_pos:
            self.cell_properties[actual_pos][property_name] = value
    
    def get_cell_property(self, position: GridCoord, property_name: str) -> Any:
        """Get property of specific cell"""
        actual_pos = self._apply_boundary_condition(position)
        if actual_pos:
            return self.cell_properties[actual_pos].get(property_name)
        return None
    
    def _validate_paradigm_state(self) -> bool:
        """Validate grid-specific state consistency"""
        try:
            # Validate spatial index consistency
            if not self.spatial_index.validate_consistency():
                return False
            
            # Check that all agents are in valid positions
            for agent_id, agent in self.agent_registry.items():
                if not self._is_valid_position(agent.position):
                    return False
                
                # Check that agent is in spatial index
                if agent_id not in self.spatial_index:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_grid_statistics(self) -> Dict[str, Any]:
        """Get grid-specific statistics"""
        stats = self.spatial_index.get_grid_statistics()
        
        # Add environment-specific stats
        stats.update({
            'total_resources': sum(
                sum(resources.values()) 
                for resources in self.cell_resources.values()
            ),
            'resource_types': set(
                resource_type 
                for resources in self.cell_resources.values()
                for resource_type in resources.keys()
            ),
            'max_agents_per_cell': self.max_agents_per_cell,
            'step_counter': self.step_counter
        })
        
        return stats
    
    def create_agent(self, agent_id: AgentID, position: GridCoord, **kwargs) -> GridAgent:
        """
        Create and add a new GridAgent to the environment.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial grid position
            **kwargs: Additional agent properties
            
        Returns:
            Created GridAgent instance
        """
        # Validate position
        if not self._is_valid_position(position):
            position = self._apply_boundary_condition(position)
            if position is None:
                raise EnvironmentConstraintError(f"Invalid position for agent {agent_id}")
        
        # Create agent
        agent = GridAgent(agent_id, position, metadata=kwargs)
        
        # Add to environment
        self.add_agent(agent)
        
        return agent
    
    def __repr__(self) -> str:
        return (f"GridEnvironment(dims={self.dimensions}, agents={len(self.agent_registry)}, "
                f"boundary={self.boundary_condition.value}, step={self.step_counter})")
