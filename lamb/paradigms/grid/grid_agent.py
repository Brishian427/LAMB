"""
Grid paradigm agent implementation for discrete space models.

Based on Technical_Specification.md Section 2.1: Grid Paradigm Specification.
Designed for discrete space models where agents occupy distinct cells
(Sugarscape, Schelling Segregation, Conway's Game of Life, Cellular Automata).

Performance characteristics:
- Memory usage: 100 bytes per agent + 8 bytes per occupied cell
- Optimal: 100-5,000 agents, radius ≤ 10 cells
- Movement: O(1) cell updates
- Observation: O(r²) for Moore neighborhood
"""

from typing import Dict, List, Tuple, Any, Optional
import random

from ...core.base_agent import BaseAgent
from ...core.base_environment import BaseEnvironment
from ...core.base_engine import BaseEngine
from ...core.types import AgentNotFoundError
from ...core.types import (
    AgentID, GridCoord, Observation, Action, ActionResult,
    InvalidActionError, EnvironmentConstraintError
)


class GridAgent(BaseAgent):
    """
    Agent for discrete grid-based models.
    
    Operates on discrete grid cells with Moore or Von Neumann neighborhoods.
    Supports typical grid-based actions: move, stay, interact with cell contents.
    """
    
    def __init__(
        self,
        agent_id: AgentID,
        position: GridCoord,
        metadata: Optional[Dict[str, Any]] = None,
        history_size: int = 100
    ):
        """
        Initialize GridAgent.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial grid position (x, y)
            metadata: Agent-specific properties
            history_size: Size of action history buffer
        """
        super().__init__(agent_id, position, metadata, history_size)
        
        # Grid-specific properties
        self.cell_contents = metadata.get('cell_contents', {}) if metadata else {}
        self.movement_range = metadata.get('movement_range', 1) if metadata else 1
        self.vision_range = metadata.get('vision_range', 1) if metadata else 1
    
    def observe(self, environment: 'GridEnvironment') -> Observation:
        """
        Generate observation of grid environment.
        
        Performance target: <0.0001s per call
        Includes: neighbors, cell contents, local grid state
        """
        if self.agent_id not in environment:
            raise AgentNotFoundError(f"Agent {self.agent_id} not in environment")
        
        # Get neighbors within vision range
        neighbors = environment.get_neighbors(self.agent_id, self.vision_range)
        
        # Get local environment state
        local_state = self._observe_local_state(environment)
        
        # Create observation
        observation = Observation(
            agent_id=self.agent_id,
            position=self.position,
            neighbors=neighbors,
            environment_state=local_state,
            paradigm="grid"
        )
        
        return observation
    
    def _observe_local_state(self, environment: 'GridEnvironment') -> Dict[str, Any]:
        """Observe local grid state around agent"""
        local_state = {
            'grid_dimensions': environment.dimensions,
            'boundary_condition': environment.boundary_condition.value,
            'cell_contents': self.cell_contents.copy(),
            'nearby_cells': {},
            'agent_count': len(environment)
        }
        
        # Observe nearby cells within vision range
        x, y = self.position
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                if dx == 0 and dy == 0:
                    continue
                
                cell_pos = (x + dx, y + dy)
                cell_info = environment.get_cell_info(cell_pos)
                if cell_info:
                    local_state['nearby_cells'][cell_pos] = cell_info
        
        return local_state
    
    def decide(self, observation: Observation, engine: BaseEngine) -> Action:
        """
        Make decision based on observation using specified engine.
        
        Performance target: <0.456s per call (LLM mode)
        """
        if observation.paradigm != "grid":
            raise IncompatibleEngineError("GridAgent requires grid paradigm observation")
        
        # Use engine to make decision
        action = engine.process_single(observation)
        
        # Validate action is appropriate for grid paradigm
        if not self._is_valid_grid_action(action):
            # Create fallback action
            action = self._create_fallback_action()
        
        return action
    
    def _is_valid_grid_action(self, action: Action) -> bool:
        """Validate that action is valid for grid paradigm"""
        valid_actions = {'move', 'stay', 'interact', 'consume', 'produce'}
        return action.action_type in valid_actions
    
    def _create_fallback_action(self) -> Action:
        """Create safe fallback action"""
        return Action(
            agent_id=self.agent_id,
            action_type="stay",
            parameters={}
        )
    
    def act(self, action: Action, environment: 'GridEnvironment') -> ActionResult:
        """
        Execute action in grid environment.
        
        Performance target: <0.0001s per call
        """
        if action.agent_id != self.agent_id:
            raise InvalidActionError("Action agent_id doesn't match agent")
        
        try:
            if action.action_type == "move":
                return self._execute_move(action, environment)
            elif action.action_type == "stay":
                return self._execute_stay(action, environment)
            elif action.action_type == "interact":
                return self._execute_interact(action, environment)
            elif action.action_type == "consume":
                return self._execute_consume(action, environment)
            elif action.action_type == "produce":
                return self._execute_produce(action, environment)
            else:
                raise InvalidActionError(f"Unknown action type: {action.action_type}")
                
        except Exception as e:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    def _execute_move(self, action: Action, environment: 'GridEnvironment') -> ActionResult:
        """Execute movement action"""
        direction = action.parameters.get('direction')
        target_pos = action.parameters.get('target_position')
        
        if target_pos:
            # Direct position specification
            new_position = tuple(target_pos)
        elif direction:
            # Direction-based movement
            new_position = self._calculate_new_position(direction)
        else:
            # Random movement within range
            new_position = self._get_random_adjacent_position()
        
        # Validate movement range
        if not self._is_within_movement_range(new_position):
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Target position outside movement range"
            )
        
        # Attempt to update position in environment
        success = environment.update_agent_state(self.agent_id, new_position, _lock_held=True)
        
        if success:
            return ActionResult(
                agent_id=self.agent_id,
                success=True,
                new_position=new_position
            )
        else:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Movement blocked or invalid position"
            )
    
    def _execute_stay(self, action: Action, environment: 'GridEnvironment') -> ActionResult:
        """Execute stay action (no movement)"""
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_interact(self, action: Action, environment: 'GridEnvironment') -> ActionResult:
        """Execute interaction with environment or other agents"""
        target = action.parameters.get('target')
        interaction_type = action.parameters.get('interaction_type', 'default')
        
        # Simple interaction implementation
        result_data = environment.process_interaction(
            self.agent_id, target, interaction_type
        )
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_consume(self, action: Action, environment: 'GridEnvironment') -> ActionResult:
        """Execute resource consumption"""
        resource_type = action.parameters.get('resource_type', 'default')
        amount = action.parameters.get('amount', 1)
        
        # Attempt to consume resource from current cell
        success = environment.consume_resource(self.agent_id, resource_type, amount)
        
        return ActionResult(
            agent_id=self.agent_id,
            success=success,
            new_position=self.position,
            error_message=None if success else f"Cannot consume {resource_type}"
        )
    
    def _execute_produce(self, action: Action, environment: 'GridEnvironment') -> ActionResult:
        """Execute resource production"""
        resource_type = action.parameters.get('resource_type', 'default')
        amount = action.parameters.get('amount', 1)
        
        # Produce resource in current cell
        success = environment.produce_resource(self.agent_id, resource_type, amount)
        
        return ActionResult(
            agent_id=self.agent_id,
            success=success,
            new_position=self.position,
            error_message=None if success else f"Cannot produce {resource_type}"
        )
    
    def _calculate_new_position(self, direction: str) -> GridCoord:
        """Calculate new position based on direction"""
        x, y = self.position
        
        direction_map = {
            'north': (0, 1),
            'south': (0, -1),
            'east': (1, 0),
            'west': (-1, 0),
            'northeast': (1, 1),
            'northwest': (-1, 1),
            'southeast': (1, -1),
            'southwest': (-1, -1),
            'up': (0, 1),      # Alias for north
            'down': (0, -1),   # Alias for south
            'right': (1, 0),   # Alias for east
            'left': (-1, 0),   # Alias for west
            'random': self._get_random_direction()
        }
        
        if direction in direction_map:
            if direction == 'random':
                dx, dy = direction_map[direction]
            else:
                dx, dy = direction_map[direction]
            return (x + dx, y + dy)
        else:
            # Unknown direction - stay in place
            return self.position
    
    def _get_random_direction(self) -> Tuple[int, int]:
        """Get random direction vector"""
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        return random.choice(directions)
    
    def _get_random_adjacent_position(self) -> GridCoord:
        """Get random adjacent position within movement range"""
        x, y = self.position
        
        # Generate possible positions within movement range
        possible_positions = []
        for dx in range(-self.movement_range, self.movement_range + 1):
            for dy in range(-self.movement_range, self.movement_range + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                possible_positions.append((x + dx, y + dy))
        
        return random.choice(possible_positions) if possible_positions else self.position
    
    def _is_within_movement_range(self, new_position: GridCoord) -> bool:
        """Check if new position is within movement range"""
        x, y = self.position
        new_x, new_y = new_position
        
        distance = max(abs(new_x - x), abs(new_y - y))  # Chebyshev distance
        return distance <= self.movement_range
    
    def get_grid_properties(self) -> Dict[str, Any]:
        """Get grid-specific agent properties"""
        return {
            'position': self.position,
            'movement_range': self.movement_range,
            'vision_range': self.vision_range,
            'cell_contents': self.cell_contents,
            'memory_usage_bytes': self._estimate_memory_usage()
        }
    
    def set_cell_contents(self, contents: Dict[str, Any]) -> None:
        """Update agent's cell contents"""
        self.cell_contents.update(contents)
    
    def get_cell_contents(self, key: str = None) -> Any:
        """Get cell contents"""
        if key:
            return self.cell_contents.get(key)
        return self.cell_contents.copy()
    
    def __repr__(self) -> str:
        return f"GridAgent(id={self.agent_id}, pos={self.position}, range={self.movement_range})"
