"""
Researcher's Sugarscape Implementation

This shows exactly what a researcher needs to implement using LAMB,
separating LAMB's universal framework from domain-specific logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
from typing import List, Dict, Any, Tuple

# ============================================================================
# LAMB PROVIDES: Universal Framework (No domain-specific logic)
# ============================================================================
from lamb.core import BaseAgent, BaseEnvironment, Simulation
from lamb.core.types import Observation, Action, AgentID
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.engines import RuleEngine, BehavioralRule
from lamb.executors import GridExecutor


# ============================================================================
# RESEARCHER IMPLEMENTS: Domain-Specific Agent
# ============================================================================
class SugarscapeAgent(GridAgent):
    """
    Researcher's domain-specific agent implementation.
    
    LAMB provides: GridAgent base class, observation/action interface
    Researcher implements: Sugarscape-specific properties and behaviors
    """
    
    def __init__(self, agent_id: int, position: Tuple[int, int], 
                 metabolism: int = 1, vision: int = 1, sugar: int = 0):
        # LAMB provides: Base agent structure
        super().__init__(agent_id, position)
        
        # Researcher implements: Sugarscape-specific properties
        self.metabolism = metabolism
        self.vision = vision
        self.sugar = sugar
        self.max_sugar = 100
        self.age = 0
        self.is_alive = True
        
        # Update metadata for rule evaluation
        self.metadata.update({
            "metabolism": metabolism,
            "vision": vision,
            "sugar": sugar,
            "age": self.age,
            "is_alive": self.is_alive
        })
    
    def observe(self, environment):
        """
        Researcher implements: Sugarscape-specific observation logic
        
        LAMB provides: Standard observation interface
        Researcher implements: How to observe sugar distribution
        """
        # LAMB provides: Basic observation structure
        observation = super().observe(environment)
        
        # Researcher implements: Sugarscape-specific observation
        sugar_info = environment.get_sugar_info(self.position, self.vision)
        neighbors = environment.get_neighbors(self.agent_id, radius=1)
        
        # Add domain-specific data to observation
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
        """
        Researcher implements: Sugarscape-specific action execution
        
        LAMB provides: Action structure and environment interface
        Researcher implements: How to execute Sugarscape actions
        """
        action_type = action.action_type
        
        if action_type == "collect_sugar":
            # Researcher implements: Sugar collection logic
            sugar_collected = environment.collect_sugar(self.position)
            self.sugar = min(self.sugar + sugar_collected, self.max_sugar)
        
        elif action_type == "reproduce":
            # Researcher implements: Reproduction logic
            if self.sugar >= 50:
                self.sugar -= 25  # Cost of reproduction
        
        elif action_type == "wait":
            # Researcher implements: Wait behavior
            pass
        
        # Researcher implements: Metabolism and aging
        self.sugar = max(0, self.sugar - self.metabolism)
        self.age += 1
        
        if self.sugar <= 0:
            self.is_alive = False
        
        # Update metadata
        self.metadata.update({
            "sugar": self.sugar,
            "age": self.age,
            "is_alive": self.is_alive
        })


# ============================================================================
# RESEARCHER IMPLEMENTS: Domain-Specific Environment
# ============================================================================
class SugarscapeEnvironment(GridEnvironment):
    """
    Researcher's domain-specific environment implementation.
    
    LAMB provides: GridEnvironment base class, spatial indexing
    Researcher implements: Sugar distribution and dynamics
    """
    
    def __init__(self, dimensions: Tuple[int, int], sugar_regrowth_rate: float = 0.1):
        # LAMB provides: Grid environment structure
        super().__init__(dimensions)
        
        # Researcher implements: Sugarscape-specific properties
        self.sugar_regrowth_rate = sugar_regrowth_rate
        
        # Sugar distribution patterns (researcher's domain knowledge)
        self.sugar_centers = [
            (dimensions[0]//4, dimensions[1]//4),
            (3*dimensions[0]//4, 3*dimensions[1]//4)
        ]
        
        self.sugar_capacity = self._create_sugar_capacity(dimensions)
        self.current_sugar = self.sugar_capacity.copy()
    
    def _create_sugar_capacity(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Researcher implements: Sugar distribution algorithm
        
        This is domain-specific knowledge about how sugar is distributed
        """
        capacity = np.zeros(dimensions)
        
        for center_x, center_y in self.sugar_centers:
            for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    capacity[y, x] = max(0, 20 - distance)
        
        return capacity
    
    def get_sugar_info(self, position: Tuple[int, int], vision: int) -> Dict[str, Any]:
        """
        Researcher implements: Sugar observation logic
        
        LAMB provides: Environment interface
        Researcher implements: How to observe sugar distribution
        """
        x, y = position
        max_sugar = 0
        sugar_positions = []
        
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
        """
        Researcher implements: Sugar collection logic
        
        LAMB provides: Environment interface
        Researcher implements: How sugar is collected
        """
        x, y = position
        if (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]):
            sugar_collected = self.current_sugar[y, x]
            self.current_sugar[y, x] = 0
            return sugar_collected
        return 0
    
    def step(self):
        """
        Researcher implements: Sugarscape environment dynamics
        
        LAMB provides: Base step functionality
        Researcher implements: Sugar regrowth
        """
        # LAMB provides: Basic environment step
        super().step()
        
        # Researcher implements: Sugar regrowth dynamics
        self.current_sugar = np.minimum(
            self.current_sugar + self.sugar_regrowth_rate,
            self.sugar_capacity
        )


# ============================================================================
# RESEARCHER IMPLEMENTS: Domain-Specific Rules
# ============================================================================
class SugarscapeRule(BehavioralRule):
    """
    Researcher's domain-specific rule implementation.
    
    LAMB provides: BehavioralRule interface
    Researcher implements: Sugarscape-specific rule logic
    """
    
    def __init__(self, name: str, priority: int, condition_func, action_func):
        self._name = name
        self._priority = priority
        self.condition_func = condition_func  # Researcher's domain logic
        self.action_func = action_func        # Researcher's domain logic
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def priority(self) -> int:
        return self._priority
    
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        """
        LAMB provides: Rule evaluation interface
        Researcher implements: Domain-specific conditions
        """
        return self.condition_func(observation, agent_state)
    
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        """
        LAMB provides: Action creation interface
        Researcher implements: Domain-specific actions
        """
        return self.action_func(observation, agent_state)


# ============================================================================
# RESEARCHER IMPLEMENTS: Domain-Specific Rule Set
# ============================================================================
class SugarscapeRules:
    """
    Researcher's collection of Sugarscape-specific rules.
    
    LAMB provides: RuleEngine that can use any BehavioralRule
    Researcher implements: Specific rules for Sugarscape simulation
    """
    
    @staticmethod
    def create_sugarscape_rules() -> List[BehavioralRule]:
        """
        Researcher implements: Sugarscape-specific rule definitions
        
        These are the actual behavioral rules that make Sugarscape work
        """
        
        # Researcher's domain knowledge: Sugarscape conditions
        def has_low_sugar(obs, state):
            return state.get("current_sugar", 0) < 10
        
        def has_high_sugar(obs, state):
            return state.get("current_sugar", 0) > 50
        
        def can_see_sugar(obs, state):
            return state.get("max_sugar_visible", 0) > state.get("current_sugar", 0)
        
        # Researcher's domain knowledge: Sugarscape actions
        def move_to_sugar(obs, state):
            sugar_positions = state.get("sugar_positions", [])
            if sugar_positions:
                best_pos = max(sugar_positions, key=lambda x: x[1])
                return Action(
                    agent_id=obs.agent_id,
                    action_type="move",
                    parameters={"position": best_pos[0]}
                )
            return Action(agent_id=obs.agent_id, action_type="wait", parameters={})
        
        def collect_sugar(obs, state):
            return Action(
                agent_id=obs.agent_id,
                action_type="collect_sugar",
                parameters={}
            )
        
        def reproduce(obs, state):
            return Action(
                agent_id=obs.agent_id,
                action_type="reproduce",
                parameters={}
            )
        
        # Researcher creates specific rules for Sugarscape
        return [
            SugarscapeRule(
                name="move_to_sugar",
                priority=10,
                condition_func=lambda obs, state: has_low_sugar(obs, state) and can_see_sugar(obs, state),
                action_func=move_to_sugar
            ),
            SugarscapeRule(
                name="collect_sugar",
                priority=9,
                condition_func=lambda obs, state: not has_low_sugar(obs, state),
                action_func=collect_sugar
            ),
            SugarscapeRule(
                name="reproduce",
                priority=8,
                condition_func=lambda obs, state: has_high_sugar(obs, state),
                action_func=reproduce
            )
        ]


# ============================================================================
# RESEARCHER USES: LAMB's Universal Framework
# ============================================================================
def create_researcher_sugarscape_simulation():
    """
    Researcher uses LAMB's universal framework to create their simulation.
    
    LAMB provides: All the infrastructure
    Researcher provides: Domain-specific components
    """
    
    # Researcher creates domain-specific components
    environment = SugarscapeEnvironment(dimensions=(30, 30))
    
    agents = []
    for i in range(50):
        position = (random.randint(0, 29), random.randint(0, 29))
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
    
    # Researcher creates domain-specific rules
    rules = SugarscapeRules.create_sugarscape_rules()
    
    # LAMB provides: Universal engine and executor
    engine = RuleEngine(rules=rules)
    executor = GridExecutor()
    
    # LAMB provides: Universal simulation framework
    simulation = Simulation(environment, agents, engine, executor)
    
    return simulation


# ============================================================================
# DEMONSTRATION
# ============================================================================
if __name__ == "__main__":
    print("🔬 Researcher's Sugarscape Implementation")
    print("=" * 60)
    
    print("\n📦 LAMB PROVIDES (Universal Framework):")
    print("  ✅ BaseAgent, BaseEnvironment, BaseEngine")
    print("  ✅ GridAgent, GridEnvironment, GridExecutor")
    print("  ✅ RuleEngine, BehavioralRule interface")
    print("  ✅ Simulation, Observation, Action")
    print("  ✅ Spatial indexing, performance monitoring")
    
    print("\n🎯 RESEARCHER IMPLEMENTS (Domain-Specific):")
    print("  ✅ SugarscapeAgent: metabolism, vision, sugar, aging")
    print("  ✅ SugarscapeEnvironment: sugar distribution, regrowth")
    print("  ✅ SugarscapeRule: move_to_sugar, collect_sugar, reproduce")
    print("  ✅ SugarscapeRules: domain-specific rule collection")
    print("  ✅ Domain logic: sugar dynamics, agent behaviors")
    
    print("\n🚀 RUNNING SIMULATION:")
    simulation = create_researcher_sugarscape_simulation()
    results = simulation.run(max_steps=50)
    
    print(f"  - Steps completed: {results.step_count}")
    print(f"  - Total time: {results.total_time:.2f}s")
    print(f"  - Average step time: {results.total_time / results.step_count:.4f}s")
    
    print("\n✅ SUCCESS: LAMB provides universal framework!")
    print("✅ SUCCESS: Researcher implements domain-specific logic!")
    print("✅ SUCCESS: Clean separation of concerns!")
