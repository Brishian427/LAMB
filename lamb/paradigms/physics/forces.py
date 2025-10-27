"""
Force calculation utilities for physics-based models.

Based on Technical_Specification.md Section 2.2: Physics Paradigm Specification.
Provides common force calculations for flocking, social forces, particle systems,
and other physics-based agent behaviors.

Performance characteristics:
- Force calculations: O(1) per force per agent
- Neighbor-based forces: O(k) where k = number of neighbors
- Global forces: O(n) for all agents
"""

from typing import List, Tuple, Optional, Callable
import math
import random

from ...core.types import PhysicsCoord, Vector2D
from .physics_agent import PhysicsAgent


class ForceCalculator:
    """
    Utility class for calculating various forces in physics-based models.
    
    Provides implementations of common force models:
    - Boids flocking forces (separation, alignment, cohesion)
    - Social forces (attraction, repulsion, goal-seeking)
    - Physical forces (gravity, friction, spring forces)
    - Obstacle avoidance forces
    """
    
    @staticmethod
    def separation_force(
        agent: PhysicsAgent,
        neighbors: List[PhysicsAgent],
        desired_separation: float = 2.0,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate separation force to avoid crowding neighbors.
        
        Based on Craig Reynolds' boids algorithm.
        
        Args:
            agent: Agent to calculate force for
            neighbors: List of neighboring agents
            desired_separation: Minimum desired distance
            max_force: Maximum force magnitude (uses agent's max_force if None)
            
        Returns:
            Separation force vector
        """
        if not neighbors:
            return Vector2D(0.0, 0.0)
        
        if max_force is None:
            max_force = agent.max_force
        
        separation = Vector2D(0.0, 0.0)
        count = 0
        
        for neighbor in neighbors:
            distance = ForceCalculator._calculate_distance(agent.position, neighbor.position)
            
            if 0 < distance < desired_separation:
                # Calculate repulsion vector
                diff_x = agent.position[0] - neighbor.position[0]
                diff_y = agent.position[1] - neighbor.position[1]
                
                # Normalize and weight by distance
                if distance > 0:
                    diff_x /= distance
                    diff_y /= distance
                    diff_x /= distance  # Weight by distance (closer = stronger)
                    diff_y /= distance
                
                separation = separation + Vector2D(diff_x, diff_y)
                count += 1
        
        if count > 0:
            # Average the separation vectors
            separation = separation * (1.0 / count)
            
            # Scale to desired speed
            if separation.magnitude() > 0:
                separation = separation * (agent.max_speed / separation.magnitude())
            
            # Calculate steering force
            steering = separation + (agent.velocity * -1)
            
            # Limit force magnitude
            if steering.magnitude() > max_force:
                steering = steering * (max_force / steering.magnitude())
            
            return steering
        
        return Vector2D(0.0, 0.0)
    
    @staticmethod
    def alignment_force(
        agent: PhysicsAgent,
        neighbors: List[PhysicsAgent],
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate alignment force to match neighbors' velocities.
        
        Based on Craig Reynolds' boids algorithm.
        
        Args:
            agent: Agent to calculate force for
            neighbors: List of neighboring agents
            max_force: Maximum force magnitude
            
        Returns:
            Alignment force vector
        """
        if not neighbors:
            return Vector2D(0.0, 0.0)
        
        if max_force is None:
            max_force = agent.max_force
        
        average_velocity = Vector2D(0.0, 0.0)
        count = 0
        
        for neighbor in neighbors:
            average_velocity = average_velocity + neighbor.velocity
            count += 1
        
        if count > 0:
            # Calculate average velocity
            average_velocity = average_velocity * (1.0 / count)
            
            # Scale to desired speed
            if average_velocity.magnitude() > 0:
                average_velocity = average_velocity * (agent.max_speed / average_velocity.magnitude())
            
            # Calculate steering force
            steering = average_velocity + (agent.velocity * -1)
            
            # Limit force magnitude
            if steering.magnitude() > max_force:
                steering = steering * (max_force / steering.magnitude())
            
            return steering
        
        return Vector2D(0.0, 0.0)
    
    @staticmethod
    def cohesion_force(
        agent: PhysicsAgent,
        neighbors: List[PhysicsAgent],
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate cohesion force to move toward neighbors' center.
        
        Based on Craig Reynolds' boids algorithm.
        
        Args:
            agent: Agent to calculate force for
            neighbors: List of neighboring agents
            max_force: Maximum force magnitude
            
        Returns:
            Cohesion force vector
        """
        if not neighbors:
            return Vector2D(0.0, 0.0)
        
        if max_force is None:
            max_force = agent.max_force
        
        center = Vector2D(0.0, 0.0)
        count = 0
        
        for neighbor in neighbors:
            center = center + Vector2D(neighbor.position[0], neighbor.position[1])
            count += 1
        
        if count > 0:
            # Calculate center of mass
            center = center * (1.0 / count)
            
            # Calculate desired velocity toward center
            desired = Vector2D(
                center.x - agent.position[0],
                center.y - agent.position[1]
            )
            
            # Scale to desired speed
            if desired.magnitude() > 0:
                desired = desired * (agent.max_speed / desired.magnitude())
            
            # Calculate steering force
            steering = desired + (agent.velocity * -1)
            
            # Limit force magnitude
            if steering.magnitude() > max_force:
                steering = steering * (max_force / steering.magnitude())
            
            return steering
        
        return Vector2D(0.0, 0.0)
    
    @staticmethod
    def seek_force(
        agent: PhysicsAgent,
        target: PhysicsCoord,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate force to seek toward target position.
        
        Args:
            agent: Agent to calculate force for
            target: Target position to seek
            max_force: Maximum force magnitude
            
        Returns:
            Seek force vector
        """
        if max_force is None:
            max_force = agent.max_force
        
        # Calculate desired velocity
        desired = Vector2D(
            target[0] - agent.position[0],
            target[1] - agent.position[1]
        )
        
        # Scale to max speed
        if desired.magnitude() > 0:
            desired = desired * (agent.max_speed / desired.magnitude())
        
        # Calculate steering force
        steering = desired + (agent.velocity * -1)
        
        # Limit force magnitude
        if steering.magnitude() > max_force:
            steering = steering * (max_force / steering.magnitude())
        
        return steering
    
    @staticmethod
    def flee_force(
        agent: PhysicsAgent,
        threat: PhysicsCoord,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate force to flee from threat position.
        
        Args:
            agent: Agent to calculate force for
            threat: Threat position to flee from
            max_force: Maximum force magnitude
            
        Returns:
            Flee force vector
        """
        if max_force is None:
            max_force = agent.max_force
        
        # Calculate desired velocity (opposite of seek)
        desired = Vector2D(
            agent.position[0] - threat[0],
            agent.position[1] - threat[1]
        )
        
        # Scale to max speed
        if desired.magnitude() > 0:
            desired = desired * (agent.max_speed / desired.magnitude())
        
        # Calculate steering force
        steering = desired + (agent.velocity * -1)
        
        # Limit force magnitude
        if steering.magnitude() > max_force:
            steering = steering * (max_force / steering.magnitude())
        
        return steering
    
    @staticmethod
    def arrive_force(
        agent: PhysicsAgent,
        target: PhysicsCoord,
        slowing_radius: float = 10.0,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate force to arrive at target with deceleration.
        
        Args:
            agent: Agent to calculate force for
            target: Target position to arrive at
            slowing_radius: Radius at which to start slowing down
            max_force: Maximum force magnitude
            
        Returns:
            Arrive force vector
        """
        if max_force is None:
            max_force = agent.max_force
        
        # Calculate desired velocity
        desired = Vector2D(
            target[0] - agent.position[0],
            target[1] - agent.position[1]
        )
        
        distance = desired.magnitude()
        
        if distance > 0:
            # Scale speed based on distance
            if distance < slowing_radius:
                # Slow down as we approach
                speed = agent.max_speed * (distance / slowing_radius)
            else:
                speed = agent.max_speed
            
            desired = desired * (speed / distance)
        
        # Calculate steering force
        steering = desired + (agent.velocity * -1)
        
        # Limit force magnitude
        if steering.magnitude() > max_force:
            steering = steering * (max_force / steering.magnitude())
        
        return steering
    
    @staticmethod
    def wander_force(
        agent: PhysicsAgent,
        wander_radius: float = 2.0,
        wander_distance: float = 4.0,
        wander_angle_change: float = 0.3,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate wandering force for random movement.
        
        Args:
            agent: Agent to calculate force for
            wander_radius: Radius of wander circle
            wander_distance: Distance to wander circle center
            wander_angle_change: Maximum change in wander angle
            max_force: Maximum force magnitude
            
        Returns:
            Wander force vector
        """
        if max_force is None:
            max_force = agent.max_force
        
        # Get or initialize wander angle
        if not hasattr(agent, '_wander_angle'):
            agent._wander_angle = random.uniform(0, 2 * math.pi)
        
        # Change wander angle randomly
        agent._wander_angle += random.uniform(-wander_angle_change, wander_angle_change)
        
        # Calculate circle center in front of agent
        velocity_normalized = agent.velocity
        if velocity_normalized.magnitude() > 0:
            velocity_normalized = Vector2D(
                velocity_normalized.x / velocity_normalized.magnitude(),
                velocity_normalized.y / velocity_normalized.magnitude()
            )
        else:
            velocity_normalized = Vector2D(1.0, 0.0)  # Default forward direction
        
        circle_center = Vector2D(
            agent.position[0] + velocity_normalized.x * wander_distance,
            agent.position[1] + velocity_normalized.y * wander_distance
        )
        
        # Calculate target on circle
        target = Vector2D(
            circle_center.x + math.cos(agent._wander_angle) * wander_radius,
            circle_center.y + math.sin(agent._wander_angle) * wander_radius
        )
        
        # Calculate steering force toward target
        desired = Vector2D(
            target.x - agent.position[0],
            target.y - agent.position[1]
        )
        
        if desired.magnitude() > 0:
            desired = desired * (agent.max_speed / desired.magnitude())
        
        steering = desired + (agent.velocity * -1)
        
        # Limit force magnitude
        if steering.magnitude() > max_force:
            steering = steering * (max_force / steering.magnitude())
        
        return steering
    
    @staticmethod
    def obstacle_avoidance_force(
        agent: PhysicsAgent,
        obstacles: List[Tuple[PhysicsCoord, float]],  # (position, radius) pairs
        avoidance_distance: float = 5.0,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate force to avoid circular obstacles.
        
        Args:
            agent: Agent to calculate force for
            obstacles: List of (position, radius) obstacle tuples
            avoidance_distance: Distance at which to start avoiding
            max_force: Maximum force magnitude
            
        Returns:
            Obstacle avoidance force vector
        """
        if not obstacles:
            return Vector2D(0.0, 0.0)
        
        if max_force is None:
            max_force = agent.max_force
        
        avoidance = Vector2D(0.0, 0.0)
        
        for obstacle_pos, obstacle_radius in obstacles:
            distance = ForceCalculator._calculate_distance(agent.position, obstacle_pos)
            total_radius = obstacle_radius + agent.radius
            
            if distance < avoidance_distance + total_radius:
                # Calculate repulsion vector
                if distance > 0:
                    repulsion_x = (agent.position[0] - obstacle_pos[0]) / distance
                    repulsion_y = (agent.position[1] - obstacle_pos[1]) / distance
                else:
                    # Agent inside obstacle - push in random direction
                    angle = random.uniform(0, 2 * math.pi)
                    repulsion_x = math.cos(angle)
                    repulsion_y = math.sin(angle)
                
                # Scale by proximity (closer = stronger)
                strength = (avoidance_distance + total_radius - distance) / avoidance_distance
                strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
                
                avoidance = avoidance + Vector2D(
                    repulsion_x * strength,
                    repulsion_y * strength
                )
        
        # Scale to max speed
        if avoidance.magnitude() > 0:
            avoidance = avoidance * (agent.max_speed / avoidance.magnitude())
        
        # Calculate steering force
        steering = avoidance + (agent.velocity * -1)
        
        # Limit force magnitude
        if steering.magnitude() > max_force:
            steering = steering * (max_force / steering.magnitude())
        
        return steering
    
    @staticmethod
    def social_force(
        agent: PhysicsAgent,
        neighbors: List[PhysicsAgent],
        attraction_strength: float = 1.0,
        repulsion_strength: float = 2.0,
        personal_space: float = 1.5,
        max_force: Optional[float] = None
    ) -> Vector2D:
        """
        Calculate social force combining attraction and repulsion.
        
        Based on Helbing's social force model for pedestrian dynamics.
        
        Args:
            agent: Agent to calculate force for
            neighbors: List of neighboring agents
            attraction_strength: Strength of attraction to others
            repulsion_strength: Strength of repulsion from others
            personal_space: Radius of personal space
            max_force: Maximum force magnitude
            
        Returns:
            Social force vector
        """
        if not neighbors:
            return Vector2D(0.0, 0.0)
        
        if max_force is None:
            max_force = agent.max_force
        
        total_force = Vector2D(0.0, 0.0)
        
        for neighbor in neighbors:
            distance = ForceCalculator._calculate_distance(agent.position, neighbor.position)
            
            if distance > 0:
                # Unit vector from neighbor to agent
                dx = (agent.position[0] - neighbor.position[0]) / distance
                dy = (agent.position[1] - neighbor.position[1]) / distance
                
                if distance < personal_space:
                    # Repulsion when too close
                    repulsion_magnitude = repulsion_strength * (personal_space - distance) / personal_space
                    total_force = total_force + Vector2D(
                        dx * repulsion_magnitude,
                        dy * repulsion_magnitude
                    )
                else:
                    # Weak attraction when farther away
                    attraction_magnitude = attraction_strength / (distance * distance)
                    total_force = total_force + Vector2D(
                        -dx * attraction_magnitude,
                        -dy * attraction_magnitude
                    )
        
        # Limit force magnitude
        if total_force.magnitude() > max_force:
            total_force = total_force * (max_force / total_force.magnitude())
        
        return total_force
    
    @staticmethod
    def spring_force(
        agent: PhysicsAgent,
        anchor: PhysicsCoord,
        rest_length: float = 0.0,
        spring_constant: float = 1.0,
        damping: float = 0.1
    ) -> Vector2D:
        """
        Calculate spring force toward anchor point.
        
        Args:
            agent: Agent to calculate force for
            anchor: Anchor position
            rest_length: Rest length of spring
            spring_constant: Spring stiffness
            damping: Damping coefficient
            
        Returns:
            Spring force vector
        """
        # Calculate displacement
        displacement = Vector2D(
            anchor[0] - agent.position[0],
            anchor[1] - agent.position[1]
        )
        
        distance = displacement.magnitude()
        
        if distance > 0:
            # Spring force: F = -k * (distance - rest_length)
            spring_magnitude = spring_constant * (distance - rest_length)
            
            # Damping force: F = -c * velocity
            velocity_along_spring = (
                agent.velocity.x * displacement.x + 
                agent.velocity.y * displacement.y
            ) / distance
            
            damping_magnitude = damping * velocity_along_spring
            
            # Total force magnitude
            total_magnitude = spring_magnitude - damping_magnitude
            
            # Apply force along displacement direction
            force = Vector2D(
                (displacement.x / distance) * total_magnitude,
                (displacement.y / distance) * total_magnitude
            )
            
            return force
        
        return Vector2D(0.0, 0.0)
    
    @staticmethod
    def _calculate_distance(pos1: PhysicsCoord, pos2: PhysicsCoord) -> float:
        """Calculate Euclidean distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def combine_forces(forces: List[Vector2D], weights: Optional[List[float]] = None) -> Vector2D:
        """
        Combine multiple forces with optional weights.
        
        Args:
            forces: List of force vectors
            weights: Optional weights for each force (default: equal weights)
            
        Returns:
            Combined force vector
        """
        if not forces:
            return Vector2D(0.0, 0.0)
        
        if weights is None:
            weights = [1.0] * len(forces)
        
        if len(weights) != len(forces):
            raise ValueError("Number of weights must match number of forces")
        
        combined = Vector2D(0.0, 0.0)
        
        for force, weight in zip(forces, weights):
            combined = combined + (force * weight)
        
        return combined
