"""
Physics environment implementation for continuous space models.

Based on Technical_Specification.md Section 2.2: Physics Paradigm Specification.
Manages continuous space with force-based interactions and KD-tree spatial indexing.

Performance characteristics:
- Agent lookup: O(1) - <0.0001s
- Neighbor query: O(log n + k) - <0.01s for radius ≤10 units
- State update: O(1) per agent - <0.0001s
- Physics update: O(n) for all agents
- Memory: 200 bytes per agent (tree + positions + velocities)
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import math
import time

from ...core.base_environment import BaseEnvironment
from ...core.types import (
    AgentID, PhysicsCoord, Vector2D, Action, ActionResult, BoundaryCondition,
    ConflictError, EnvironmentConstraintError
)
from ...spatial.kdtree_index import KDTreeIndex
from .physics_agent import PhysicsAgent


class PhysicsEnvironment(BaseEnvironment):
    """
    Environment for continuous space physics-based models.
    
    Features:
    - Continuous position and velocity tracking
    - KD-tree spatial indexing for efficient neighbor queries
    - Force-based interactions and collision detection
    - Configurable world bounds and boundary conditions
    - Physics simulation with configurable time step
    """
    
    def __init__(
        self,
        world_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-100.0, -100.0), (100.0, 100.0)),
        boundary_condition: BoundaryCondition = BoundaryCondition.REFLECT,
        dt: float = 0.1,
        enable_collisions: bool = True,
        collision_damping: float = 0.8
    ):
        """
        Initialize PhysicsEnvironment.
        
        Args:
            world_bounds: ((min_x, min_y), (max_x, max_y)) world boundaries
            boundary_condition: How to handle boundaries
            dt: Physics simulation time step
            enable_collisions: Whether to enable collision detection
            collision_damping: Energy loss factor in collisions (0-1)
        """
        super().__init__()
        
        self.world_bounds = world_bounds
        self.boundary_condition = boundary_condition
        self.dt = dt
        self.enable_collisions = enable_collisions
        self.collision_damping = collision_damping
        
        # Initialize KD-tree spatial index
        self.spatial_index = KDTreeIndex(
            rebuild_threshold=100,
            rebuild_interval=1.0,
            track_velocities=True
        )
        
        # Physics simulation state
        self.total_kinetic_energy = 0.0
        self.collision_count = 0
        
        # Performance tracking
        self.physics_update_time = 0.0
        self.collision_detection_time = 0.0
    
    def get_neighbors(self, agent_id: AgentID, radius: float) -> List[AgentID]:
        """
        Get neighboring agents within radius using KD-tree.
        
        Performance target: <0.01s for radius ≤10 units
        Uses KDTreeIndex for O(log n + k) range queries
        """
        return self.spatial_index.get_neighbors(agent_id, radius)
    
    def _is_valid_position(self, position: PhysicsCoord) -> bool:
        """Validate if position is within world bounds"""
        if not isinstance(position, tuple) or len(position) != 2:
            return False
        
        x, y = position
        (min_x, min_y), (max_x, max_y) = self.world_bounds
        
        if self.boundary_condition == BoundaryCondition.INFINITE:
            return True  # All positions valid in infinite world
        else:
            return min_x <= x <= max_x and min_y <= y <= max_y
    
    def _get_target_position(self, action: Action) -> PhysicsCoord:
        """Get target position for action (used for conflict detection)"""
        agent = self.get_agent(action.agent_id)
        
        if action.action_type == "move_to":
            target_x = action.parameters.get('target_x', agent.position[0])
            target_y = action.parameters.get('target_y', agent.position[1])
            return (target_x, target_y)
        
        # For other actions, predict position after physics update
        if isinstance(agent, PhysicsAgent):
            # Simulate one physics step to predict position
            predicted_pos = agent.update_physics(self.dt)
            return predicted_pos
        
        return agent.position
    
    def _execute_action(self, action: Action) -> ActionResult:
        """Execute single action"""
        agent = self.get_agent(action.agent_id)
        return agent.act(action, self)
    
    def _resolve_position_conflict(self, actions: List[Action]) -> List[ActionResult]:
        """
        Resolve conflicts in continuous space.
        
        For physics paradigm, conflicts are resolved through collision detection
        rather than discrete position blocking.
        """
        results = []
        
        # Execute all actions - conflicts handled by collision detection
        for action in actions:
            results.append(self._execute_action(action))
        
        return results
    
    def _update_environment_state(self) -> None:
        """Update physics-specific environment state"""
        start_time = time.perf_counter()
        
        # Update physics for all agents
        self._update_physics()
        
        # Handle collisions if enabled
        if self.enable_collisions:
            collision_start = time.perf_counter()
            self._handle_collisions()
            self.collision_detection_time = time.perf_counter() - collision_start
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Update performance metrics
        self.physics_update_time = time.perf_counter() - start_time
        
        # Calculate total kinetic energy
        self._calculate_kinetic_energy()
    
    def _update_physics(self) -> None:
        """Update physics simulation for all agents"""
        new_positions = {}
        
        # Update physics for each agent
        for agent_id, agent in self.agent_registry.items():
            if isinstance(agent, PhysicsAgent):
                new_position = agent.update_physics(self.dt)
                new_positions[agent_id] = new_position
        
        # Update spatial index with new positions
        for agent_id, new_position in new_positions.items():
            old_position = self.agent_registry[agent_id].position
            self.spatial_index.update_agent(agent_id, old_position, new_position)
            self.agent_registry[agent_id].position = new_position
    
    def _handle_collisions(self) -> None:
        """Handle collisions between agents"""
        collision_pairs = self._detect_collisions()
        
        for agent1_id, agent2_id in collision_pairs:
            self._resolve_collision(agent1_id, agent2_id)
            self.collision_count += 1
    
    def _detect_collisions(self) -> List[Tuple[AgentID, AgentID]]:
        """Detect collisions between agents"""
        collisions = []
        processed_pairs = set()
        
        for agent_id, agent in self.agent_registry.items():
            if not isinstance(agent, PhysicsAgent):
                continue
            
            # Get nearby agents
            nearby = self.get_neighbors(agent_id, agent.radius * 3)  # Search wider area
            
            for neighbor_id in nearby:
                if neighbor_id == agent_id:
                    continue
                
                # Avoid duplicate pair processing
                pair = tuple(sorted([agent_id, neighbor_id]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                neighbor = self.agent_registry[neighbor_id]
                if isinstance(neighbor, PhysicsAgent):
                    distance = self._calculate_distance(agent.position, neighbor.position)
                    min_distance = agent.radius + neighbor.radius
                    
                    if distance < min_distance:
                        collisions.append((agent_id, neighbor_id))
        
        return collisions
    
    def _resolve_collision(self, agent1_id: AgentID, agent2_id: AgentID) -> None:
        """Resolve collision between two agents"""
        agent1 = self.agent_registry[agent1_id]
        agent2 = self.agent_registry[agent2_id]
        
        if not (isinstance(agent1, PhysicsAgent) and isinstance(agent2, PhysicsAgent)):
            return
        
        # Calculate collision normal
        dx = agent2.position[0] - agent1.position[0]
        dy = agent2.position[1] - agent1.position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            # Agents at same position - separate randomly
            dx, dy = 1.0, 0.0
            distance = 1.0
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Separate agents to prevent overlap
        overlap = (agent1.radius + agent2.radius) - distance
        if overlap > 0:
            separation = overlap * 0.5
            agent1.position = (
                agent1.position[0] - nx * separation,
                agent1.position[1] - ny * separation
            )
            agent2.position = (
                agent2.position[0] + nx * separation,
                agent2.position[1] + ny * separation
            )
        
        # Calculate relative velocity
        rel_vx = agent2.velocity.x - agent1.velocity.x
        rel_vy = agent2.velocity.y - agent1.velocity.y
        
        # Calculate relative velocity along collision normal
        vel_along_normal = rel_vx * nx + rel_vy * ny
        
        # Don't resolve if velocities are separating
        if vel_along_normal > 0:
            return
        
        # Calculate collision impulse
        restitution = self.collision_damping
        impulse_magnitude = -(1 + restitution) * vel_along_normal
        impulse_magnitude /= (1 / agent1.mass + 1 / agent2.mass)
        
        # Apply impulse to velocities
        impulse_x = impulse_magnitude * nx
        impulse_y = impulse_magnitude * ny
        
        agent1.velocity = Vector2D(
            agent1.velocity.x - impulse_x / agent1.mass,
            agent1.velocity.y - impulse_y / agent1.mass
        )
        
        agent2.velocity = Vector2D(
            agent2.velocity.x + impulse_x / agent2.mass,
            agent2.velocity.y + impulse_y / agent2.mass
        )
    
    def _apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to all agents"""
        (min_x, min_y), (max_x, max_y) = self.world_bounds
        
        for agent_id, agent in self.agent_registry.items():
            if not isinstance(agent, PhysicsAgent):
                continue
            
            x, y = agent.position
            new_x, new_y = x, y
            velocity_changed = False
            
            if self.boundary_condition == BoundaryCondition.REFLECT:
                # Reflect off boundaries
                if x < min_x:
                    new_x = min_x
                    agent.velocity = Vector2D(-agent.velocity.x, agent.velocity.y)
                    velocity_changed = True
                elif x > max_x:
                    new_x = max_x
                    agent.velocity = Vector2D(-agent.velocity.x, agent.velocity.y)
                    velocity_changed = True
                
                if y < min_y:
                    new_y = min_y
                    agent.velocity = Vector2D(agent.velocity.x, -agent.velocity.y)
                    velocity_changed = True
                elif y > max_y:
                    new_y = max_y
                    agent.velocity = Vector2D(agent.velocity.x, -agent.velocity.y)
                    velocity_changed = True
            
            elif self.boundary_condition == BoundaryCondition.WRAP:
                # Wrap around boundaries (toroidal)
                if x < min_x:
                    new_x = max_x
                elif x > max_x:
                    new_x = min_x
                
                if y < min_y:
                    new_y = max_y
                elif y > max_y:
                    new_y = min_y
            
            elif self.boundary_condition == BoundaryCondition.ABSORB:
                # Remove agents that go out of bounds
                if x < min_x or x > max_x or y < min_y or y > max_y:
                    self.remove_agent(agent_id)
                    continue
            
            # Update position if changed
            if new_x != x or new_y != y:
                old_position = agent.position
                new_position = (new_x, new_y)
                agent.position = new_position
                self.spatial_index.update_agent(agent_id, old_position, new_position)
    
    def _calculate_distance(self, pos1: PhysicsCoord, pos2: PhysicsCoord) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def _calculate_kinetic_energy(self) -> None:
        """Calculate total kinetic energy of the system"""
        total_energy = 0.0
        
        for agent in self.agent_registry.values():
            if isinstance(agent, PhysicsAgent):
                speed_squared = agent.velocity.x ** 2 + agent.velocity.y ** 2
                kinetic_energy = 0.5 * agent.mass * speed_squared
                total_energy += kinetic_energy
        
        self.total_kinetic_energy = total_energy
    
    def _validate_paradigm_state(self) -> bool:
        """Validate physics-specific state consistency"""
        try:
            # Validate spatial index consistency
            if not self.spatial_index.validate_consistency():
                return False
            
            # Check that all agents have valid physics properties
            for agent_id, agent in self.agent_registry.items():
                if isinstance(agent, PhysicsAgent):
                    # Check position bounds
                    if not self._is_valid_position(agent.position):
                        if self.boundary_condition != BoundaryCondition.INFINITE:
                            return False
                    
                    # Check physics properties
                    if agent.mass <= 0 or agent.radius <= 0:
                        return False
                    
                    if math.isnan(agent.velocity.x) or math.isnan(agent.velocity.y):
                        return False
            
            return True
            
        except Exception:
            return False
    
    def apply_global_force(self, force: Vector2D) -> None:
        """Apply force to all agents (e.g., gravity, wind)"""
        for agent in self.agent_registry.values():
            if isinstance(agent, PhysicsAgent):
                agent.forces = agent.forces + force
    
    def apply_radial_force(self, center: PhysicsCoord, strength: float, max_radius: float) -> None:
        """Apply radial force from center point (attraction/repulsion)"""
        for agent in self.agent_registry.values():
            if isinstance(agent, PhysicsAgent):
                distance = self._calculate_distance(agent.position, center)
                
                if distance < max_radius and distance > 0:
                    # Calculate force direction
                    dx = agent.position[0] - center[0]
                    dy = agent.position[1] - center[1]
                    
                    # Normalize
                    dx /= distance
                    dy /= distance
                    
                    # Calculate force magnitude (inverse square law)
                    force_magnitude = strength / (distance * distance)
                    
                    # Apply force
                    force = Vector2D(dx * force_magnitude, dy * force_magnitude)
                    agent.forces = agent.forces + force
    
    def get_physics_statistics(self) -> Dict[str, Any]:
        """Get physics-specific statistics"""
        stats = self.spatial_index.get_tree_statistics()
        
        # Add physics-specific stats
        stats.update({
            'world_bounds': self.world_bounds,
            'boundary_condition': self.boundary_condition.value,
            'dt': self.dt,
            'total_kinetic_energy': self.total_kinetic_energy,
            'collision_count': self.collision_count,
            'enable_collisions': self.enable_collisions,
            'collision_damping': self.collision_damping,
            'physics_update_time': self.physics_update_time,
            'collision_detection_time': self.collision_detection_time,
            'step_counter': self.step_counter
        })
        
        return stats
    
    def create_agent(self, agent_id: AgentID, position: PhysicsCoord, **kwargs) -> PhysicsAgent:
        """
        Create and add a new PhysicsAgent to the environment.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial position in continuous space
            **kwargs: Additional agent properties
            
        Returns:
            Created PhysicsAgent instance
        """
        # Validate position
        if not self._is_valid_position(position):
            raise EnvironmentConstraintError(f"Invalid position {position} for agent {agent_id}")
        
        # Set default physics properties
        kwargs.setdefault('dt', self.dt)
        
        # Create agent
        agent = PhysicsAgent(agent_id, position, metadata=kwargs)
        
        # Add to environment
        self.add_agent(agent)
        
        return agent
    
    def __repr__(self) -> str:
        return (f"PhysicsEnvironment(bounds={self.world_bounds}, agents={len(self.agent_registry)}, "
                f"boundary={self.boundary_condition.value}, dt={self.dt}, step={self.step_counter})")
