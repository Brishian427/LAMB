"""
Boids Flocking Model Implementation using LAMB Framework

This example implements the classic Boids flocking model by Craig Reynolds,
demonstrating emergent collective behavior from simple individual rules.

Key Features:
- Separation, alignment, and cohesion behaviors
- Physics-based continuous movement
- Emergent flocking patterns
- LLM-driven decision making
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from typing import List, Dict, Any, Tuple
import math

from lamb import Simulation, SimulationFactory
from lamb.paradigms.physics import PhysicsAgent, PhysicsEnvironment
from lamb.engines import LLMEngine, RuleEngine, HybridEngine, HybridMode
from lamb.executors import PhysicsExecutor
from lamb.llm import PromptManager, create_research_personality
from lamb.core.types import Vector2D


class BoidAgent(PhysicsAgent):
    """Boid agent with flocking behavior"""
    
    def __init__(self, agent_id: int, position: Tuple[float, float], 
                 velocity: Tuple[float, float] = None, max_speed: float = 2.0):
        # Create metadata with velocity
        metadata = {
            'velocity_x': velocity[0] if velocity else 0.0,
            'velocity_y': velocity[1] if velocity else 0.0,
            'max_speed': max_speed
        }
        super().__init__(agent_id, position, metadata)
        
        # Boid-specific properties
        self.max_speed = max_speed
        self.max_force = 0.1
        self.separation_distance = 25.0
        self.alignment_distance = 50.0
        self.cohesion_distance = 50.0
        
        # Behavior weights
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        
        # Update metadata
        self.metadata.update({
            "max_speed": max_speed,
            "separation_distance": self.separation_distance,
            "alignment_distance": self.alignment_distance,
            "cohesion_distance": self.cohesion_distance,
            "boid_type": "flocker"
        })
    
    def observe(self, environment):
        """Observe environment including nearby boids"""
        # Get basic observation
        observation = super().observe(environment)
        
        # Get nearby boids
        neighbors = environment.get_neighbors(self.agent_id, radius=self.cohesion_distance)
        
        # Calculate flocking information
        separation_neighbors = [n for n in neighbors if self._distance_to(n) < self.separation_distance]
        alignment_neighbors = [n for n in neighbors if self._distance_to(n) < self.alignment_distance]
        cohesion_neighbors = [n for n in neighbors if self._distance_to(n) < self.cohesion_distance]
        
        # Calculate forces
        separation_force = self._calculate_separation_force(separation_neighbors)
        alignment_force = self._calculate_alignment_force(alignment_neighbors)
        cohesion_force = self._calculate_cohesion_force(cohesion_neighbors)
        
        # Update observation with boid-specific data
        observation.environment_state.update({
            "neighbor_count": len(neighbors),
            "separation_neighbors": len(separation_neighbors),
            "alignment_neighbors": len(alignment_neighbors),
            "cohesion_neighbors": len(cohesion_neighbors),
            "separation_force": separation_force,
            "alignment_force": alignment_force,
            "cohesion_force": cohesion_force,
            "current_speed": np.linalg.norm([self.velocity.x, self.velocity.y]),
            "max_speed": self.max_speed
        })
        
        return observation
    
    def act(self, action, environment):
        """Execute action and update boid state"""
        action_type = action.action_type
        
        if action_type == "flock":
            # Apply flocking forces
            separation_force = action.parameters.get("separation_force", (0, 0))
            alignment_force = action.parameters.get("alignment_force", (0, 0))
            cohesion_force = action.parameters.get("cohesion_force", (0, 0))
            
            # Combine forces
            total_force = (
                separation_force[0] * self.separation_weight +
                alignment_force[0] * self.alignment_weight +
                cohesion_force[0] * self.cohesion_weight,
                separation_force[1] * self.separation_weight +
                alignment_force[1] * self.alignment_weight +
                cohesion_force[1] * self.cohesion_weight
            )
            
            # Limit force
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > self.max_force:
                total_force = (
                    total_force[0] * self.max_force / force_magnitude,
                    total_force[1] * self.max_force / force_magnitude
                )
            
            # Apply force to velocity
            self.velocity = Vector2D(
                self.velocity.x + total_force[0],
                self.velocity.y + total_force[1]
            )
            
            # Limit speed
            speed = np.linalg.norm([self.velocity.x, self.velocity.y])
            if speed > self.max_speed:
                self.velocity = Vector2D(
                    self.velocity.x * self.max_speed / speed,
                    self.velocity.y * self.max_speed / speed
                )
            
            # Update position
            self.position = (
                self.position[0] + self.velocity.x,
                self.position[1] + self.velocity.y
            )
        
        elif action_type == "wander":
            # Random wandering behavior
            wander_force = action.parameters.get("wander_force", (0, 0))
            self.velocity = Vector2D(
                self.velocity.x + wander_force[0],
                self.velocity.y + wander_force[1]
            )
            
            # Limit speed
            speed = np.linalg.norm([self.velocity.x, self.velocity.y])
            if speed > self.max_speed:
                self.velocity = Vector2D(
                    self.velocity.x * self.max_speed / speed,
                    self.velocity.y * self.max_speed / speed
                )
            
            # Update position
            self.position = (
                self.position[0] + self.velocity.x,
                self.position[1] + self.velocity.y
            )
    
    def _distance_to(self, other_agent):
        """Calculate distance to another agent"""
        return np.sqrt(
            (self.position[0] - other_agent.position[0])**2 +
            (self.position[1] - other_agent.position[1])**2
        )
    
    def _calculate_separation_force(self, neighbors):
        """Calculate separation force to avoid crowding"""
        if not neighbors:
            return (0, 0)
        
        steer = (0, 0)
        count = 0
        
        for neighbor in neighbors:
            distance = self._distance_to(neighbor)
            if distance > 0:
                # Calculate vector pointing away from neighbor
                diff = (
                    self.position[0] - neighbor.position[0],
                    self.position[1] - neighbor.position[1]
                )
                # Weight by distance
                diff = (diff[0] / distance, diff[1] / distance)
                steer = (steer[0] + diff[0], steer[1] + diff[1])
                count += 1
        
        if count > 0:
            steer = (steer[0] / count, steer[1] / count)
            # Normalize and scale
            magnitude = np.linalg.norm(steer)
            if magnitude > 0:
                steer = (steer[0] / magnitude, steer[1] / magnitude)
                steer = (steer[0] * self.max_force, steer[1] * self.max_force)
        
        return steer
    
    def _calculate_alignment_force(self, neighbors):
        """Calculate alignment force to match neighbor velocities"""
        if not neighbors:
            return (0, 0)
        
        avg_velocity = (0, 0)
        count = 0
        
        for neighbor in neighbors:
            avg_velocity = (
                avg_velocity[0] + neighbor.velocity[0],
                avg_velocity[1] + neighbor.velocity[1]
            )
            count += 1
        
        if count > 0:
            avg_velocity = (avg_velocity[0] / count, avg_velocity[1] / count)
            # Normalize and scale
            magnitude = np.linalg.norm(avg_velocity)
            if magnitude > 0:
                avg_velocity = (avg_velocity[0] / magnitude, avg_velocity[1] / magnitude)
                avg_velocity = (avg_velocity[0] * self.max_force, avg_velocity[1] * self.max_force)
        
        return avg_velocity
    
    def _calculate_cohesion_force(self, neighbors):
        """Calculate cohesion force to move toward group center"""
        if not neighbors:
            return (0, 0)
        
        center = (0, 0)
        count = 0
        
        for neighbor in neighbors:
            center = (
                center[0] + neighbor.position[0],
                center[1] + neighbor.position[1]
            )
            count += 1
        
        if count > 0:
            center = (center[0] / count, center[1] / count)
            # Calculate steering force toward center
            steer = (
                center[0] - self.position[0],
                center[1] - self.position[1]
            )
            # Normalize and scale
            magnitude = np.linalg.norm(steer)
            if magnitude > 0:
                steer = (steer[0] / magnitude, steer[1] / magnitude)
                steer = (steer[0] * self.max_force, steer[1] * self.max_force)
        
        return steer


class BoidsEnvironment(PhysicsEnvironment):
    """Environment for Boids flocking simulation"""
    
    def __init__(self, world_bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                 dt: float = 0.1, enable_wrapping: bool = True):
        from lamb.core.types import BoundaryCondition
        
        boundary_condition = BoundaryCondition.WRAP if enable_wrapping else BoundaryCondition.REFLECT
        super().__init__(world_bounds, boundary_condition, dt)
        
        self.enable_wrapping = enable_wrapping
        self.min_x, self.min_y = world_bounds[0]
        self.max_x, self.max_y = world_bounds[1]
    
    def step_physics(self, dt: float):
        """Update physics including boundary conditions"""
        # Update agent positions based on velocities
        for agent in self.agent_registry.values():
            if hasattr(agent, 'position') and hasattr(agent, 'velocity'):
                # Update position
                agent.position = (
                    agent.position[0] + agent.velocity.x * dt,
                    agent.position[1] + agent.velocity.y * dt
                )
                
                # Apply boundary conditions
                if self.enable_wrapping:
                    # Wrap around edges
                    if agent.position[0] < self.min_x:
                        agent.position = (self.max_x, agent.position[1])
                    elif agent.position[0] > self.max_x:
                        agent.position = (self.min_x, agent.position[1])
                    
                    if agent.position[1] < self.min_y:
                        agent.position = (agent.position[0], self.max_y)
                    elif agent.position[1] > self.max_y:
                        agent.position = (agent.position[0], self.min_y)
                else:
                    # Bounce off edges
                    if agent.position[0] < self.min_x or agent.position[0] > self.max_x:
                        agent.velocity = Vector2D(-agent.velocity.x, agent.velocity.y)
                        agent.position = (
                            max(self.min_x, min(self.max_x, agent.position[0])),
                            agent.position[1]
                        )
                    
                    if agent.position[1] < self.min_y or agent.position[1] > self.max_y:
                        agent.velocity = Vector2D(agent.velocity.x, -agent.velocity.y)
                        agent.position = (
                            agent.position[0],
                            max(self.min_y, min(self.max_y, agent.position[1]))
                        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get boids statistics"""
        agents = list(self.agent_registry.values())
        
        if not agents:
            return {"total_boids": 0, "average_speed": 0, "average_separation": 0}
        
        # Calculate average speed
        speeds = [np.linalg.norm([agent.velocity.x, agent.velocity.y]) for agent in agents]
        average_speed = np.mean(speeds)
        
        # Calculate average separation
        separations = []
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                distance = np.sqrt(
                    (agent1.position[0] - agent2.position[0])**2 +
                    (agent1.position[1] - agent2.position[1])**2
                )
                separations.append(distance)
        
        average_separation = np.mean(separations) if separations else 0
        
        return {
            "total_boids": len(agents),
            "average_speed": average_speed,
            "average_separation": average_separation,
            "world_bounds": self.world_bounds
        }


def create_boids_simulation(num_boids: int = 50, 
                           world_size: float = 200.0,
                           use_llm: bool = True,
                           engine_type: str = "llm") -> Simulation:
    """Create a Boids flocking simulation"""
    
    # Create environment
    environment = BoidsEnvironment(
        world_bounds=((-world_size/2, -world_size/2), (world_size/2, world_size/2)),
        dt=0.1,
        enable_wrapping=True
    )
    
    # Create boids with random positions and velocities
    boids = []
    for i in range(num_boids):
        # Random position
        position = (
            random.uniform(-world_size/2, world_size/2),
            random.uniform(-world_size/2, world_size/2)
        )
        
        # Random velocity
        velocity = (
            random.uniform(-2, 2),
            random.uniform(-2, 2)
        )
        
        # Random properties
        max_speed = random.uniform(1.5, 2.5)
        
        boid = BoidAgent(
            agent_id=i,
            position=position,
            velocity=velocity,
            max_speed=max_speed
        )
        boids.append(boid)
        environment.add_agent(boid)
    
    # Create engine based on type
    if engine_type == "llm":
        prompt_manager = PromptManager()
        engine = LLMEngine(prompt_manager=prompt_manager)
    elif engine_type == "rule":
        from lamb.engines import FlockingRules
        rules = FlockingRules.create_flocking_rules()
        engine = RuleEngine(rules=rules)
    elif engine_type == "hybrid":
        prompt_manager = PromptManager()
        llm_engine = LLMEngine(prompt_manager=prompt_manager)
        rule_engine = RuleEngine()
        engine = HybridEngine(llm_engine=llm_engine, rule_engine=rule_engine, mode=HybridMode.ADAPTIVE)
    else:
        from lamb.engines import MockEngine
        engine = MockEngine()
    
    # Create executor
    executor = PhysicsExecutor()
    
    return Simulation(environment, boids, engine, executor)


def run_boids_experiment():
    """Run Boids experiment with different engine types"""
    
    print("🐦 Boids Flocking Experiment")
    print("=" * 50)
    
    # Test different engine types
    engine_types = ["mock", "rule", "llm", "hybrid"]
    
    for engine_type in engine_types:
        print(f"\n--- Testing {engine_type.upper()} Engine ---")
        
        # Create simulation
        simulation = create_boids_simulation(
            num_boids=30,
            world_size=150.0,
            engine_type=engine_type
        )
        
        # Run simulation
        print(f"Running simulation with {len(simulation.agents)} boids...")
        results = simulation.run(max_steps=50)
        
        # Get statistics
        stats = simulation.environment.get_statistics()
        
        print(f"Results after {results.step_count} steps:")
        print(f"  - Boids: {stats['total_boids']}")
        print(f"  - Average speed: {stats['average_speed']:.2f}")
        print(f"  - Average separation: {stats['average_separation']:.2f}")
        print(f"  - Total simulation time: {results.total_time:.2f}s")
        print(f"  - Average step time: {results.total_time / results.step_count:.4f}s")


def run_boids_parameter_sweep():
    """Run parameter sweep for Boids model"""
    
    print("\n🔬 Boids Parameter Sweep")
    print("=" * 50)
    
    # Test different boid counts
    boid_counts = [10, 20, 30, 50, 100]
    
    for num_boids in boid_counts:
        print(f"\n--- Testing with {num_boids} boids ---")
        
        simulation = create_boids_simulation(
            num_boids=num_boids,
            world_size=200.0,
            engine_type="rule"  # Use rule engine for speed
        )
        
        results = simulation.run(max_steps=30)
        stats = simulation.environment.get_statistics()
        
        print(f"  - Boids: {stats['total_boids']}")
        print(f"  - Average speed: {stats['average_speed']:.2f}")
        print(f"  - Average separation: {stats['average_separation']:.2f}")
        print(f"  - Simulation time: {results.total_time:.2f}s")


if __name__ == "__main__":
    print("LAMB Framework: Boids Flocking Model Example")
    print("=" * 60)
    
    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OpenAI API key found!")
        print("   Set OPENAI_API_KEY environment variable to run with real LLM")
        print("   Examples will use mock engines for demonstration")
        print()
    
    # Run experiments
    run_boids_experiment()
    run_boids_parameter_sweep()
    
    print("\n🎉 Boids experiment completed!")
    print("\nKey Insights:")
    print("✅ Emergent flocking behavior from simple rules")
    print("✅ Separation, alignment, and cohesion create complex patterns")
    print("✅ Different engines produce different flocking dynamics")
    print("✅ LAMB enables easy physics-based simulations")
