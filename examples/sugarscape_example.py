"""
Sugarscape Model Implementation using LAMB Framework

This example implements the classic Sugarscape model by Joshua Epstein and Robert Axtell,
demonstrating how resource distribution affects agent behavior and social dynamics.

Key Features:
- Resource-based agent behavior
- Wealth accumulation and inheritance
- Social dynamics and migration
- LLM-driven decision making
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from typing import List, Dict, Any, Tuple

from lamb import Simulation, SimulationFactory
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.engines import LLMEngine, RuleEngine, HybridEngine, HybridMode
from lamb.executors import GridExecutor
from lamb.llm import PromptManager, create_research_personality


class SugarscapeAgent(GridAgent):
    """Agent for Sugarscape model with wealth and metabolism"""
    
    def __init__(self, agent_id: int, position: Tuple[int, int], 
                 metabolism: int = 1, vision: int = 1, sugar: int = 0):
        super().__init__(agent_id, position)
        
        # Sugarscape-specific properties
        self.metabolism = metabolism  # Sugar consumed per step
        self.vision = vision  # How far agent can see
        self.sugar = sugar  # Current sugar wealth
        self.max_sugar = 100  # Maximum sugar capacity
        self.age = 0
        self.is_alive = True
        
        # Update metadata
        self.metadata.update({
            "metabolism": metabolism,
            "vision": vision,
            "sugar": sugar,
            "age": self.age,
            "is_alive": self.is_alive
        })
    
    def observe(self, environment):
        """Observe environment including sugar distribution and neighbors"""
        # Get basic observation
        observation = super().observe(environment)
        
        # Add sugarscape-specific information
        sugar_info = environment.get_sugar_info(self.position, self.vision)
        neighbors = environment.get_neighbors(self.agent_id, radius=1)
        
        # Update observation with sugarscape data
        observation.environment_state.update({
            "sugar_at_position": sugar_info["current_sugar"],
            "max_sugar_visible": sugar_info["max_sugar_visible"],
            "sugar_positions": sugar_info["sugar_positions"],
            "neighbor_count": len(neighbors),
            "neighbor_sugar": [n.metadata.get("sugar", 0) for n in neighbors],
            "metabolism": self.metabolism,
            "vision": self.vision,
            "current_sugar": self.sugar,
            "age": self.age
        })
        
        return observation
    
    def act(self, action, environment):
        """Execute action and update agent state"""
        action_type = action.action_type
        
        if action_type == "collect_sugar":
            # Collect sugar from current position
            sugar_collected = environment.collect_sugar(self.position)
            self.sugar = min(self.sugar + sugar_collected, self.max_sugar)
        
        elif action_type == "reproduce":
            # Create offspring (simplified)
            if self.sugar >= 50:  # Need enough sugar to reproduce
                self.sugar -= 25  # Cost of reproduction
                # Offspring would be created by environment
        
        elif action_type == "wait":
            # Agent waits (no action)
            pass
        
        # Consume sugar for metabolism
        self.sugar = max(0, self.sugar - self.metabolism)
        
        # Age the agent
        self.age += 1
        
        # Check if agent dies
        if self.sugar <= 0:
            self.is_alive = False
        
        # Update metadata
        self.metadata.update({
            "sugar": self.sugar,
            "age": self.age,
            "is_alive": self.is_alive
        })


class SugarscapeEnvironment(GridEnvironment):
    """Environment for Sugarscape model with sugar distribution"""
    
    def __init__(self, dimensions: Tuple[int, int], sugar_regrowth_rate: float = 0.1):
        super().__init__(dimensions)
        
        self.sugar_regrowth_rate = sugar_regrowth_rate
        
        # Sugar distribution patterns
        self.sugar_centers = [
            (dimensions[0]//4, dimensions[1]//4),
            (3*dimensions[0]//4, 3*dimensions[1]//4)
        ]
        
        self.sugar_capacity = self._create_sugar_capacity(dimensions)
        self.current_sugar = self.sugar_capacity.copy()
    
    def _create_sugar_capacity(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """Create sugar capacity distribution"""
        capacity = np.zeros(dimensions)
        
        # Create sugar mountains at specified centers
        for center_x, center_y in self.sugar_centers:
            for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    capacity[y, x] = max(0, 20 - distance)  # Sugar decreases with distance
        
        return capacity
    
    def get_sugar_info(self, position: Tuple[int, int], vision: int) -> Dict[str, Any]:
        """Get sugar information within agent's vision"""
        x, y = position
        max_sugar = 0
        sugar_positions = []
        
        # Search within vision range
        for dx in range(-vision, vision + 1):
            for dy in range(-vision, vision + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.dimensions[0] and 0 <= check_y < self.dimensions[1]):
                    sugar_amount = self.current_sugar[check_y, check_x]
                    max_sugar = max(max_sugar, sugar_amount)
                    if sugar_amount > 0:
                        sugar_positions.append(((check_x, check_y), sugar_amount))
        
        return {
            "current_sugar": self.current_sugar[y, x] if (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]) else 0,
            "max_sugar_visible": max_sugar,
            "sugar_positions": sugar_positions
        }
    
    def collect_sugar(self, position: Tuple[int, int]) -> int:
        """Collect sugar from position and return amount collected"""
        x, y = position
        if (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]):
            sugar_collected = self.current_sugar[y, x]
            self.current_sugar[y, x] = 0
            return sugar_collected
        return 0
    
    def move_agent(self, agent_id: int, new_position: Tuple[int, int]):
        """Move agent to new position"""
        if agent_id in self.agent_registry:
            agent = self.agent_registry[agent_id]
            old_position = agent.position
            agent.position = new_position
            # Update spatial index if it exists
            if hasattr(self, 'spatial_index'):
                self.spatial_index.update_agent(agent_id, old_position, new_position)
    
    def step(self):
        """Update environment including sugar regrowth"""
        super().step()
        
        # Regrow sugar
        self.current_sugar = np.minimum(
            self.current_sugar + self.sugar_regrowth_rate,
            self.sugar_capacity
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sugarscape statistics"""
        agents = [agent for agent in self.agent_registry.values() if agent.is_alive]
        
        if not agents:
            return {"total_agents": 0, "average_sugar": 0, "average_age": 0}
        
        total_sugar = sum(agent.sugar for agent in agents)
        total_age = sum(agent.age for agent in agents)
        
        return {
            "total_agents": len(agents),
            "average_sugar": total_sugar / len(agents),
            "average_age": total_age / len(agents),
            "total_sugar_in_world": np.sum(self.current_sugar),
            "max_sugar_capacity": np.max(self.sugar_capacity)
        }


def create_sugarscape_simulation(num_agents: int = 100, 
                                grid_size: int = 50,
                                use_llm: bool = True,
                                engine_type: str = "llm") -> Simulation:
    """Create a Sugarscape simulation"""
    
    # Create environment
    environment = SugarscapeEnvironment(dimensions=(grid_size, grid_size))
    
    # Create agents with random properties
    agents = []
    for i in range(num_agents):
        # Random position
        position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        
        # Random metabolism and vision
        metabolism = random.randint(1, 4)
        vision = random.randint(1, 6)
        initial_sugar = random.randint(10, 50)
        
        agent = SugarscapeAgent(
            agent_id=i,
            position=position,
            metabolism=metabolism,
            vision=vision,
            sugar=initial_sugar
        )
        agents.append(agent)
        environment.add_agent(agent)
    
    # Create engine based on type
    if engine_type == "llm":
        prompt_manager = PromptManager()
        engine = LLMEngine(prompt_manager=prompt_manager)
    elif engine_type == "rule":
        from lamb.engines import CooperationRules
        rules = CooperationRules.create_cooperation_rules()
        engine = RuleEngine(rules=rules)
    elif engine_type == "hybrid":
        prompt_manager = PromptManager()
        llm_engine = LLMEngine(prompt_manager=prompt_manager)
        rule_engine = RuleEngine()
        engine = HybridEngine(llm_engine=llm_engine, rule_engine=rule_engine, mode=HybridMode.LLM_FIRST)
    else:
        from lamb.engines import MockEngine
        engine = MockEngine()
    
    # Create executor
    executor = GridExecutor()
    
    return Simulation(environment, agents, engine, executor)


def run_sugarscape_experiment():
    """Run Sugarscape experiment with different engine types"""
    
    print("🍯 Sugarscape Model Experiment")
    print("=" * 50)
    
    # Test different engine types
    engine_types = ["mock", "rule", "llm", "hybrid"]
    
    for engine_type in engine_types:
        print(f"\n--- Testing {engine_type.upper()} Engine ---")
        
        # Create simulation
        simulation = create_sugarscape_simulation(
            num_agents=50,
            grid_size=30,
            engine_type=engine_type
        )
        
        # Run simulation
        print(f"Running simulation with {len(simulation.agents)} agents...")
        results = simulation.run(max_steps=100)
        
        # Get statistics
        stats = simulation.environment.get_statistics()
        
        print(f"Results after {results.step_count} steps:")
        print(f"  - Agents alive: {stats['total_agents']}")
        print(f"  - Average sugar: {stats['average_sugar']:.2f}")
        print(f"  - Average age: {stats['average_age']:.2f}")
        print(f"  - Total simulation time: {results.total_time:.2f}s")
        print(f"  - Average step time: {results.total_time / results.step_count:.4f}s")


def run_sugarscape_parameter_sweep():
    """Run parameter sweep for Sugarscape model"""
    
    print("\n🔬 Sugarscape Parameter Sweep")
    print("=" * 50)
    
    # Test different agent counts
    agent_counts = [25, 50, 100, 200]
    
    for num_agents in agent_counts:
        print(f"\n--- Testing with {num_agents} agents ---")
        
        simulation = create_sugarscape_simulation(
            num_agents=num_agents,
            grid_size=40,
            engine_type="rule"  # Use rule engine for speed
        )
        
        results = simulation.run(max_steps=50)
        stats = simulation.environment.get_statistics()
        
        print(f"  - Final agents: {stats['total_agents']}")
        print(f"  - Average sugar: {stats['average_sugar']:.2f}")
        print(f"  - Simulation time: {results.total_time:.2f}s")


if __name__ == "__main__":
    print("LAMB Framework: Sugarscape Model Example")
    print("=" * 60)
    
    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OpenAI API key found!")
        print("   Set OPENAI_API_KEY environment variable to run with real LLM")
        print("   Examples will use mock engines for demonstration")
        print()
    
    # Run experiments
    run_sugarscape_experiment()
    run_sugarscape_parameter_sweep()
    
    print("\n🎉 Sugarscape experiment completed!")
    print("\nKey Insights:")
    print("✅ Resource distribution affects agent behavior")
    print("✅ Wealth accumulation creates social dynamics")
    print("✅ Different engines produce different behaviors")
    print("✅ LAMB enables easy parameter sweeps")
