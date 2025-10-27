"""
Advanced Rule-Based Engine Examples

This demonstrates how to create domain-specific rule sets for different
simulation types using the LAMB rule engine framework.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
import random
import math

from lamb.engines import RuleEngine, BehavioralRule
from lamb.core.types import Observation, Action, AgentID


class DomainSpecificRule(BehavioralRule):
    """Base class for domain-specific rules with custom evaluation logic"""
    
    def __init__(self, name: str, priority: int, domain: str):
        self._name = name
        self._priority = priority
        self._domain = domain
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def priority(self) -> int:
        return self._priority


# ============================================================================
# SUGARSCAPE-SPECIFIC RULES
# ============================================================================

class SugarscapeRule(DomainSpecificRule):
    """Rule for Sugarscape resource-based simulation"""
    
    def __init__(self, name: str, priority: int, condition_func, action_func):
        super().__init__(name, priority, "sugarscape")
        self.condition_func = condition_func
        self.action_func = action_func
    
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        """Evaluate using domain-specific logic"""
        return self.condition_func(observation, agent_state)
    
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        """Execute using domain-specific logic"""
        return self.action_func(observation, agent_state)


class SugarscapeRules:
    """Domain-specific rules for Sugarscape simulation"""
    
    @staticmethod
    def create_sugarscape_rules() -> List[BehavioralRule]:
        """Create rules specific to Sugarscape resource dynamics"""
        
        def has_low_sugar(obs, state):
            return state.get("current_sugar", 0) < 10
        
        def has_high_sugar(obs, state):
            return state.get("current_sugar", 0) > 50
        
        def can_see_sugar(obs, state):
            return state.get("max_sugar_visible", 0) > state.get("current_sugar", 0)
        
        def move_to_sugar(obs, state):
            sugar_positions = state.get("sugar_positions", [])
            if sugar_positions:
                # Move to position with highest sugar
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
# BOIDS-SPECIFIC RULES
# ============================================================================

class BoidsRule(DomainSpecificRule):
    """Rule for Boids flocking simulation"""
    
    def __init__(self, name: str, priority: int, condition_func, action_func):
        super().__init__(name, priority, "boids")
        self.condition_func = condition_func
        self.action_func = action_func
    
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        return self.condition_func(observation, agent_state)
    
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        return self.action_func(observation, agent_state)


class BoidsRules:
    """Domain-specific rules for Boids flocking simulation"""
    
    @staticmethod
    def create_boids_rules() -> List[BehavioralRule]:
        """Create rules specific to Boids flocking behavior"""
        
        def has_separation_neighbors(obs, state):
            return state.get("separation_neighbors", 0) > 0
        
        def has_alignment_neighbors(obs, state):
            return state.get("alignment_neighbors", 0) > 0
        
        def has_cohesion_neighbors(obs, state):
            return state.get("cohesion_neighbors", 0) > 0
        
        def is_alone(obs, state):
            return state.get("neighbor_count", 0) == 0
        
        def apply_separation(obs, state):
            separation_force = state.get("separation_force", (0, 0))
            return Action(
                agent_id=obs.agent_id,
                action_type="flock",
                parameters={
                    "separation_force": separation_force,
                    "alignment_force": (0, 0),
                    "cohesion_force": (0, 0)
                }
            )
        
        def apply_alignment(obs, state):
            alignment_force = state.get("alignment_force", (0, 0))
            return Action(
                agent_id=obs.agent_id,
                action_type="flock",
                parameters={
                    "separation_force": (0, 0),
                    "alignment_force": alignment_force,
                    "cohesion_force": (0, 0)
                }
            )
        
        def apply_cohesion(obs, state):
            cohesion_force = state.get("cohesion_force", (0, 0))
            return Action(
                agent_id=obs.agent_id,
                action_type="flock",
                parameters={
                    "separation_force": (0, 0),
                    "alignment_force": (0, 0),
                    "cohesion_force": cohesion_force
                }
            )
        
        def wander(obs, state):
            # Random wandering when alone
            wander_force = (
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1)
            )
            return Action(
                agent_id=obs.agent_id,
                action_type="wander",
                parameters={"wander_force": wander_force}
            )
        
        return [
            BoidsRule(
                name="separate",
                priority=10,
                condition_func=lambda obs, state: has_separation_neighbors(obs, state),
                action_func=apply_separation
            ),
            BoidsRule(
                name="align",
                priority=9,
                condition_func=lambda obs, state: has_alignment_neighbors(obs, state),
                action_func=apply_alignment
            ),
            BoidsRule(
                name="cohere",
                priority=8,
                condition_func=lambda obs, state: has_cohesion_neighbors(obs, state),
                action_func=apply_cohesion
            ),
            BoidsRule(
                name="wander",
                priority=5,
                condition_func=lambda obs, state: is_alone(obs, state),
                action_func=wander
            )
        ]


# ============================================================================
# EPIDEMIOLOGY-SPECIFIC RULES (SIR Model)
# ============================================================================

class SIRRule(DomainSpecificRule):
    """Rule for SIR epidemiological simulation"""
    
    def __init__(self, name: str, priority: int, condition_func, action_func):
        super().__init__(name, priority, "sir")
        self.condition_func = condition_func
        self.action_func = action_func
    
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        return self.condition_func(observation, agent_state)
    
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        return self.action_func(observation, agent_state)


class SIRRules:
    """Domain-specific rules for SIR epidemiological simulation"""
    
    @staticmethod
    def create_sir_rules() -> List[BehavioralRule]:
        """Create rules specific to SIR disease spread"""
        
        def is_susceptible(obs, state):
            return state.get("health_status") == "susceptible"
        
        def is_infected(obs, state):
            return state.get("health_status") == "infected"
        
        def is_recovered(obs, state):
            return state.get("health_status") == "recovered"
        
        def has_infected_neighbors(obs, state):
            neighbors = state.get("neighbor_health", [])
            return "infected" in neighbors
        
        def infection_probability(obs, state):
            infected_count = state.get("neighbor_health", []).count("infected")
            return random.random() < (infected_count * 0.1)  # 10% per infected neighbor
        
        def become_infected(obs, state):
            return Action(
                agent_id=obs.agent_id,
                action_type="become_infected",
                parameters={}
            )
        
        def recover(obs, state):
            return Action(
                agent_id=obs.agent_id,
                action_type="recover",
                parameters={}
            )
        
        def social_distance(obs, state):
            return Action(
                agent_id=obs.agent_id,
                action_type="social_distance",
                parameters={"distance": 2.0}
            )
        
        return [
            SIRRule(
                name="become_infected",
                priority=10,
                condition_func=lambda obs, state: is_susceptible(obs, state) and 
                                                has_infected_neighbors(obs, state) and 
                                                infection_probability(obs, state),
                action_func=become_infected
            ),
            SIRRule(
                name="recover",
                priority=9,
                condition_func=lambda obs, state: is_infected(obs, state) and 
                                                random.random() < 0.05,  # 5% recovery chance
                action_func=recover
            ),
            SIRRule(
                name="social_distance",
                priority=8,
                condition_func=lambda obs, state: is_susceptible(obs, state) and 
                                                has_infected_neighbors(obs, state),
                action_func=social_distance
            )
        ]


# ============================================================================
# SCHELLING-SPECIFIC RULES (Segregation Model)
# ============================================================================

class SchellingRule(DomainSpecificRule):
    """Rule for Schelling segregation simulation"""
    
    def __init__(self, name: str, priority: int, condition_func, action_func):
        super().__init__(name, priority, "schelling")
        self.condition_func = condition_func
        self.action_func = action_func
    
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        return self.condition_func(observation, agent_state)
    
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        return self.action_func(observation, agent_state)


class SchellingRules:
    """Domain-specific rules for Schelling segregation simulation"""
    
    @staticmethod
    def create_schelling_rules() -> List[BehavioralRule]:
        """Create rules specific to Schelling segregation dynamics"""
        
        def is_unhappy(obs, state):
            agent_type = state.get("agent_type")
            neighbors = state.get("neighbor_types", [])
            if not neighbors:
                return False
            
            # Calculate similarity ratio
            similar_count = neighbors.count(agent_type)
            total_count = len(neighbors)
            similarity_ratio = similar_count / total_count if total_count > 0 else 0
            
            # Unhappy if similarity ratio < threshold
            threshold = state.get("similarity_threshold", 0.5)
            return similarity_ratio < threshold
        
        def find_empty_location(obs, state):
            # Find empty location (simplified)
            empty_positions = state.get("empty_positions", [])
            if empty_positions:
                return Action(
                    agent_id=obs.agent_id,
                    action_type="move",
                    parameters={"position": random.choice(empty_positions)}
                )
            return Action(agent_id=obs.agent_id, action_type="wait", parameters={})
        
        def stay_put(obs, state):
            return Action(
                agent_id=obs.agent_id,
                action_type="wait",
                parameters={}
            )
        
        return [
            SchellingRule(
                name="move_if_unhappy",
                priority=10,
                condition_func=lambda obs, state: is_unhappy(obs, state),
                action_func=find_empty_location
            ),
            SchellingRule(
                name="stay_if_happy",
                priority=5,
                condition_func=lambda obs, state: not is_unhappy(obs, state),
                action_func=stay_put
            )
        ]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demonstrate_domain_specific_rules():
    """Demonstrate how to use domain-specific rules"""
    
    print("🔧 Domain-Specific Rule Engine Examples")
    print("=" * 50)
    
    # Example 1: Sugarscape Rules
    print("\n1. Sugarscape Rules:")
    sugarscape_rules = SugarscapeRules.create_sugarscape_rules()
    sugarscape_engine = RuleEngine(rules=sugarscape_rules)
    print(f"   - Created {len(sugarscape_rules)} Sugarscape-specific rules")
    print(f"   - Rules: {[rule.name for rule in sugarscape_rules]}")
    
    # Example 2: Boids Rules
    print("\n2. Boids Rules:")
    boids_rules = BoidsRules.create_boids_rules()
    boids_engine = RuleEngine(rules=boids_rules)
    print(f"   - Created {len(boids_rules)} Boids-specific rules")
    print(f"   - Rules: {[rule.name for rule in boids_rules]}")
    
    # Example 3: SIR Rules
    print("\n3. SIR Epidemiological Rules:")
    sir_rules = SIRRules.create_sir_rules()
    sir_engine = RuleEngine(rules=sir_rules)
    print(f"   - Created {len(sir_rules)} SIR-specific rules")
    print(f"   - Rules: {[rule.name for rule in sir_rules]}")
    
    # Example 4: Schelling Rules
    print("\n4. Schelling Segregation Rules:")
    schelling_rules = SchellingRules.create_schelling_rules()
    schelling_engine = RuleEngine(rules=schelling_rules)
    print(f"   - Created {len(schelling_rules)} Schelling-specific rules")
    print(f"   - Rules: {[rule.name for rule in schelling_rules]}")
    
    print("\n✅ Each simulation type has completely different rules!")
    print("✅ Rules are domain-specific and highly expressive!")
    print("✅ Same RuleEngine class handles all simulation types!")


if __name__ == "__main__":
    demonstrate_domain_specific_rules()
