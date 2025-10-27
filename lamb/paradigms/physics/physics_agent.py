"""
Physics paradigm agent implementation for continuous space models.

Based on Technical_Specification.md Section 2.2: Physics Paradigm Specification.
Designed for continuous space models with force-based interactions
(Boids flocking, Social Force Model, particle systems, crowd dynamics).

Performance characteristics:
- Memory usage: 200 bytes per agent (position + velocity + forces + history)
- Optimal: 1,000-50,000 agents in continuous space
- Movement: O(1) position updates
- Observation: O(log n + k) using KD-tree spatial indexing
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import random

from ...core.base_agent import BaseAgent
from ...core.base_environment import BaseEnvironment
from ...core.base_engine import BaseEngine
from ...core.types import (
    AgentID, PhysicsCoord, Vector2D, Observation, Action, ActionResult,
    InvalidActionError, EnvironmentConstraintError
)


class PhysicsAgent(BaseAgent):
    """
    Agent for continuous space physics-based models.
    
    Features:
    - Continuous position and velocity
    - Force-based movement and interactions
    - Configurable mass, radius, and physical properties
    - Support for acceleration, friction, and collision detection
    """
    
    def __init__(
        self,
        agent_id: AgentID,
        position: PhysicsCoord,
        metadata: Optional[Dict[str, Any]] = None,
        history_size: int = 100
    ):
        """
        Initialize PhysicsAgent.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial position (x, y) in continuous space
            metadata: Agent-specific properties
            history_size: Size of action history buffer
        """
        super().__init__(agent_id, position, metadata, history_size)
        
        # Physics properties
        self.velocity = Vector2D(
            metadata.get('velocity_x', 0.0) if metadata else 0.0,
            metadata.get('velocity_y', 0.0) if metadata else 0.0
        )
        
        self.mass = metadata.get('mass', 1.0) if metadata else 1.0
        self.radius = metadata.get('radius', 1.0) if metadata else 1.0
        self.max_speed = metadata.get('max_speed', 10.0) if metadata else 10.0
        self.max_force = metadata.get('max_force', 5.0) if metadata else 5.0
        
        # Vision and interaction ranges
        self.vision_range = metadata.get('vision_range', 5.0) if metadata else 5.0
        self.interaction_range = metadata.get('interaction_range', 2.0) if metadata else 2.0
        
        # Accumulated forces for current step
        self.forces = Vector2D(0.0, 0.0)
        
        # Physics simulation parameters
        self.friction = metadata.get('friction', 0.01) if metadata else 0.01
        self.dt = metadata.get('dt', 0.1) if metadata else 0.1  # Time step
    
    def observe(self, environment: 'PhysicsEnvironment') -> Observation:
        """
        Generate observation of physics environment.
        
        Performance target: <0.0001s per call
        Includes: neighbors, forces, velocities, local physics state
        """
        if self.agent_id not in environment:
            raise AgentNotFoundError(f"Agent {self.agent_id} not in environment")
        
        # Get neighbors within vision range
        neighbors = environment.get_neighbors(self.agent_id, self.vision_range)
        
        # Get local physics state
        local_state = self._observe_local_state(environment, neighbors)
        
        # Create observation
        observation = Observation(
            agent_id=self.agent_id,
            position=self.position,
            neighbors=neighbors,
            environment_state=local_state,
            paradigm="physics"
        )
        
        return observation
    
    def _observe_local_state(self, environment: 'PhysicsEnvironment', neighbors: List[AgentID]) -> Dict[str, Any]:
        """Observe local physics state around agent"""
        # Get neighbor information
        neighbor_info = []
        for neighbor_id in neighbors:
            if neighbor_id in environment.agent_registry:
                neighbor = environment.agent_registry[neighbor_id]
                if isinstance(neighbor, PhysicsAgent):
                    distance = self._calculate_distance(self.position, neighbor.position)
                    neighbor_info.append({
                        'id': neighbor_id,
                        'position': neighbor.position,
                        'velocity': (neighbor.velocity.x, neighbor.velocity.y),
                        'distance': distance,
                        'mass': neighbor.mass,
                        'radius': neighbor.radius
                    })
        
        local_state = {
            'world_bounds': environment.world_bounds,
            'boundary_condition': environment.boundary_condition.value,
            'agent_velocity': (self.velocity.x, self.velocity.y),
            'agent_mass': self.mass,
            'agent_radius': self.radius,
            'current_forces': (self.forces.x, self.forces.y),
            'neighbors': neighbor_info,
            'agent_count': len(environment),
            'dt': self.dt
        }
        
        return local_state
    
    def decide(self, observation: Observation, engine: BaseEngine) -> Action:
        """
        Make decision based on observation using specified engine.
        
        Performance target: <0.456s per call (LLM mode)
        """
        if observation.paradigm != "physics":
            raise IncompatibleEngineError("PhysicsAgent requires physics paradigm observation")
        
        # Use engine to make decision
        action = engine.process_single(observation)
        
        # Validate action is appropriate for physics paradigm
        if not self._is_valid_physics_action(action):
            # Create fallback action
            action = self._create_fallback_action()
        
        return action
    
    def _is_valid_physics_action(self, action: Action) -> bool:
        """Validate that action is valid for physics paradigm"""
        valid_actions = {'apply_force', 'set_velocity', 'move_to', 'brake', 'accelerate'}
        return action.action_type in valid_actions
    
    def _create_fallback_action(self) -> Action:
        """Create safe fallback action"""
        return Action(
            agent_id=self.agent_id,
            action_type="brake",
            parameters={'factor': 0.9}
        )
    
    def act(self, action: Action, environment: 'PhysicsEnvironment') -> ActionResult:
        """
        Execute action in physics environment.
        
        Performance target: <0.0001s per call
        """
        if action.agent_id != self.agent_id:
            raise InvalidActionError("Action agent_id doesn't match agent")
        
        try:
            if action.action_type == "apply_force":
                return self._execute_apply_force(action, environment)
            elif action.action_type == "set_velocity":
                return self._execute_set_velocity(action, environment)
            elif action.action_type == "move_to":
                return self._execute_move_to(action, environment)
            elif action.action_type == "brake":
                return self._execute_brake(action, environment)
            elif action.action_type == "accelerate":
                return self._execute_accelerate(action, environment)
            else:
                raise InvalidActionError(f"Unknown action type: {action.action_type}")
                
        except Exception as e:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    def _execute_apply_force(self, action: Action, environment: 'PhysicsEnvironment') -> ActionResult:
        """Execute force application"""
        force_x = action.parameters.get('force_x', 0.0)
        force_y = action.parameters.get('force_y', 0.0)
        
        # Create force vector
        force = Vector2D(force_x, force_y)
        
        # Limit force magnitude
        force_magnitude = force.magnitude()
        if force_magnitude > self.max_force:
            force = force * (self.max_force / force_magnitude)
        
        # Add to accumulated forces
        self.forces = self.forces + force
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_set_velocity(self, action: Action, environment: 'PhysicsEnvironment') -> ActionResult:
        """Execute velocity setting"""
        velocity_x = action.parameters.get('velocity_x', 0.0)
        velocity_y = action.parameters.get('velocity_y', 0.0)
        
        # Create velocity vector
        new_velocity = Vector2D(velocity_x, velocity_y)
        
        # Limit velocity magnitude
        speed = new_velocity.magnitude()
        if speed > self.max_speed:
            new_velocity = new_velocity * (self.max_speed / speed)
        
        self.velocity = new_velocity
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_move_to(self, action: Action, environment: 'PhysicsEnvironment') -> ActionResult:
        """Execute direct movement to target position"""
        target_x = action.parameters.get('target_x', self.position[0])
        target_y = action.parameters.get('target_y', self.position[1])
        target_pos = (target_x, target_y)
        
        # Calculate desired velocity to reach target
        dx = target_x - self.position[0]
        dy = target_y - self.position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            # Normalize and scale by max speed
            desired_velocity = Vector2D(
                (dx / distance) * self.max_speed,
                (dy / distance) * self.max_speed
            )
            
            # Apply steering force
            steering_force = desired_velocity + (self.velocity * -1)
            steering_magnitude = steering_force.magnitude()
            
            if steering_magnitude > self.max_force:
                steering_force = steering_force * (self.max_force / steering_magnitude)
            
            self.forces = self.forces + steering_force
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_brake(self, action: Action, environment: 'PhysicsEnvironment') -> ActionResult:
        """Execute braking (reduce velocity)"""
        brake_factor = action.parameters.get('factor', 0.9)
        brake_factor = max(0.0, min(1.0, brake_factor))  # Clamp to [0, 1]
        
        self.velocity = self.velocity * brake_factor
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_accelerate(self, action: Action, environment: 'PhysicsEnvironment') -> ActionResult:
        """Execute acceleration in current direction"""
        acceleration = action.parameters.get('acceleration', 1.0)
        
        # Accelerate in current velocity direction
        if self.velocity.magnitude() > 0:
            velocity_unit = Vector2D(
                self.velocity.x / self.velocity.magnitude(),
                self.velocity.y / self.velocity.magnitude()
            )
            force = velocity_unit * acceleration
            
            # Limit force
            if force.magnitude() > self.max_force:
                force = force * (self.max_force / force.magnitude())
            
            self.forces = self.forces + force
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def update_physics(self, dt: Optional[float] = None) -> PhysicsCoord:
        """
        Update physics simulation for this agent.
        
        Applies forces, updates velocity and position using Euler integration.
        
        Args:
            dt: Time step (uses agent's dt if not provided)
            
        Returns:
            New position after physics update
        """
        if dt is None:
            dt = self.dt
        
        # Apply forces to velocity (F = ma, so a = F/m)
        acceleration = Vector2D(self.forces.x / self.mass, self.forces.y / self.mass)
        self.velocity = self.velocity + (acceleration * dt)
        
        # Apply friction
        friction_force = self.velocity * (-self.friction)
        self.velocity = self.velocity + (friction_force * dt)
        
        # Limit velocity
        speed = self.velocity.magnitude()
        if speed > self.max_speed:
            self.velocity = self.velocity * (self.max_speed / speed)
        
        # Update position
        new_x = self.position[0] + self.velocity.x * dt
        new_y = self.position[1] + self.velocity.y * dt
        new_position = (new_x, new_y)
        
        # Reset forces for next step
        self.forces = Vector2D(0.0, 0.0)
        
        return new_position
    
    def apply_separation_force(self, neighbors: List['PhysicsAgent'], strength: float = 1.0) -> None:
        """Apply separation force to avoid crowding neighbors"""
        if not neighbors:
            return
        
        separation = Vector2D(0.0, 0.0)
        count = 0
        
        for neighbor in neighbors:
            distance = self._calculate_distance(self.position, neighbor.position)
            desired_separation = self.radius + neighbor.radius
            
            if 0 < distance < desired_separation:
                # Calculate repulsion vector
                diff_x = self.position[0] - neighbor.position[0]
                diff_y = self.position[1] - neighbor.position[1]
                
                # Normalize and weight by distance
                if distance > 0:
                    diff_x /= distance
                    diff_y /= distance
                    diff_x /= distance  # Weight by distance
                    diff_y /= distance
                
                separation = separation + Vector2D(diff_x, diff_y)
                count += 1
        
        if count > 0:
            separation = separation * (1.0 / count)  # Average
            separation = separation * strength
            
            # Limit force
            if separation.magnitude() > self.max_force:
                separation = separation * (self.max_force / separation.magnitude())
            
            self.forces = self.forces + separation
    
    def apply_alignment_force(self, neighbors: List['PhysicsAgent'], strength: float = 1.0) -> None:
        """Apply alignment force to match neighbors' velocities"""
        if not neighbors:
            return
        
        average_velocity = Vector2D(0.0, 0.0)
        count = 0
        
        for neighbor in neighbors:
            distance = self._calculate_distance(self.position, neighbor.position)
            if distance < self.vision_range:
                average_velocity = average_velocity + neighbor.velocity
                count += 1
        
        if count > 0:
            average_velocity = average_velocity * (1.0 / count)  # Average
            
            # Calculate steering force
            steering = average_velocity + (self.velocity * -1)
            steering = steering * strength
            
            # Limit force
            if steering.magnitude() > self.max_force:
                steering = steering * (self.max_force / steering.magnitude())
            
            self.forces = self.forces + steering
    
    def apply_cohesion_force(self, neighbors: List['PhysicsAgent'], strength: float = 1.0) -> None:
        """Apply cohesion force to move toward neighbors' center"""
        if not neighbors:
            return
        
        center = Vector2D(0.0, 0.0)
        count = 0
        
        for neighbor in neighbors:
            distance = self._calculate_distance(self.position, neighbor.position)
            if distance < self.vision_range:
                center = center + Vector2D(neighbor.position[0], neighbor.position[1])
                count += 1
        
        if count > 0:
            center = center * (1.0 / count)  # Average position
            
            # Calculate desired velocity toward center
            desired = Vector2D(
                center.x - self.position[0],
                center.y - self.position[1]
            )
            
            # Normalize and scale
            if desired.magnitude() > 0:
                desired = desired * (self.max_speed / desired.magnitude())
            
            # Calculate steering force
            steering = desired + (self.velocity * -1)
            steering = steering * strength
            
            # Limit force
            if steering.magnitude() > self.max_force:
                steering = steering * (self.max_force / steering.magnitude())
            
            self.forces = self.forces + steering
    
    def _calculate_distance(self, pos1: PhysicsCoord, pos2: PhysicsCoord) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_physics_properties(self) -> Dict[str, Any]:
        """Get physics-specific agent properties"""
        return {
            'position': self.position,
            'velocity': (self.velocity.x, self.velocity.y),
            'speed': self.velocity.magnitude(),
            'mass': self.mass,
            'radius': self.radius,
            'max_speed': self.max_speed,
            'max_force': self.max_force,
            'vision_range': self.vision_range,
            'interaction_range': self.interaction_range,
            'friction': self.friction,
            'current_forces': (self.forces.x, self.forces.y),
            'memory_usage_bytes': self._estimate_memory_usage()
        }
    
    def set_physics_properties(self, **kwargs) -> None:
        """Update physics properties"""
        if 'mass' in kwargs:
            self.mass = max(0.1, kwargs['mass'])  # Minimum mass
        if 'radius' in kwargs:
            self.radius = max(0.1, kwargs['radius'])  # Minimum radius
        if 'max_speed' in kwargs:
            self.max_speed = max(0.0, kwargs['max_speed'])
        if 'max_force' in kwargs:
            self.max_force = max(0.0, kwargs['max_force'])
        if 'friction' in kwargs:
            self.friction = max(0.0, min(1.0, kwargs['friction']))  # Clamp [0, 1]
    
    def __repr__(self) -> str:
        return (f"PhysicsAgent(id={self.agent_id}, pos={self.position}, "
                f"vel=({self.velocity.x:.2f},{self.velocity.y:.2f}), mass={self.mass})")
