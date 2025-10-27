"""
Rule-based decision engine for LAMB framework.

This engine implements traditional ABM rule-based decision making,
providing a fast, deterministic alternative to LLM-based decisions.
Suitable for well-defined behavioral rules and high-performance simulations.
"""

from typing import List, Dict, Any, Optional, Union
import time
import random
from abc import ABC, abstractmethod

from ..core.base_engine import BaseEngine
from ..core.types import (
    Observation, Action, EngineType, AgentID,
    LAMBError, InvalidObservationError
)


class BehavioralRule(ABC):
    """Abstract base class for behavioral rules"""
    
    @abstractmethod
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        """Evaluate if this rule should be applied"""
        pass
    
    @abstractmethod
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        """Execute the rule and return an action"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Rule priority (higher = more important)"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name for identification"""
        pass


class SimpleRule(BehavioralRule):
    """Simple rule implementation"""
    
    def __init__(self, name: str, condition: str, action_type: str, 
                 parameters: Dict[str, Any] = None, priority: int = 5):
        self._name = name
        self._condition = condition
        self._action_type = action_type
        self._parameters = parameters or {}
        self._priority = priority
    
    def evaluate(self, observation: Observation, agent_state: Dict[str, Any]) -> bool:
        """Evaluate rule condition"""
        try:
            # Simple condition evaluation
            if self._condition == "always":
                return True
            elif self._condition == "never":
                return False
            elif self._condition == "random":
                return random.random() < 0.5
            elif self._condition == "has_neighbors":
                return len(observation.neighbors) > 0
            elif self._condition == "no_neighbors":
                return len(observation.neighbors) == 0
            elif self._condition == "cooperative_personality":
                return agent_state.get("personality") == "cooperative"
            elif self._condition == "selfish_personality":
                return agent_state.get("personality") == "selfish"
            else:
                return False
        except Exception:
            return False
    
    def execute(self, observation: Observation, agent_state: Dict[str, Any]) -> Action:
        """Execute rule and return action"""
        action_params = self._parameters.copy()
        
        # Add observation-based parameters
        if "neighbor_count" in action_params:
            action_params["neighbor_count"] = len(observation.neighbors)
        
        if "position" in action_params:
            action_params["position"] = observation.position
        
        return Action(
            agent_id=observation.agent_id,
            action_type=self._action_type,
            parameters=action_params
        )
    
    @property
    def priority(self) -> int:
        return self._priority
    
    @property
    def name(self) -> str:
        return self._name


class RuleEngine(BaseEngine):
    """
    Rule-based decision engine for traditional ABM behavior.
    
    Features:
    - Fast, deterministic decision making
    - Rule-based behavior definition
    - Priority-based rule evaluation
    - Extensible rule system
    - High performance (no API calls)
    """
    
    def __init__(self, rules: Optional[List[BehavioralRule]] = None, 
                 default_action: str = "wait"):
        """
        Initialize RuleEngine.
        
        Args:
            rules: List of behavioral rules (if None, uses default rules)
            default_action: Default action when no rules match
        """
        super().__init__(EngineType.RULE)
        
        self.rules = rules or self._create_default_rules()
        self.default_action = default_action
        
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def _create_default_rules(self) -> List[BehavioralRule]:
        """Create default behavioral rules"""
        return [
            SimpleRule(
                name="cooperate_with_neighbors",
                condition="cooperative_personality",
                action_type="cooperate",
                parameters={"target": "neighbors", "strength": 1.0},
                priority=10
            ),
            SimpleRule(
                name="compete_with_neighbors",
                condition="selfish_personality",
                action_type="compete",
                parameters={"target": "neighbors", "strength": 0.8},
                priority=9
            ),
            SimpleRule(
                name="explore_when_alone",
                condition="no_neighbors",
                action_type="explore",
                parameters={"direction": "random", "distance": 1},
                priority=8
            ),
            SimpleRule(
                name="socialize_when_crowded",
                condition="has_neighbors",
                action_type="socialize",
                parameters={"target": "neighbors", "duration": 1},
                priority=7
            ),
            SimpleRule(
                name="random_action",
                condition="random",
                action_type="wait",
                parameters={"reason": "random"},
                priority=1
            )
        ]
    
    def process_single(self, observation: Observation) -> Action:
        """Process single observation using rule-based logic"""
        start_time = time.perf_counter()
        
        try:
            # Get agent state from observation
            agent_state = {
                "personality": observation.environment_state.get("personality", "neutral"),
                "position": observation.position,
                "neighbors": observation.neighbors,
                "step": observation.environment_state.get("step", 0)
            }
            
            # Evaluate rules in priority order
            for rule in self.rules:
                if rule.evaluate(observation, agent_state):
                    action = rule.execute(observation, agent_state)
                    decision_time = time.perf_counter() - start_time
                    self._record_decision(decision_time, True)
                    return action
            
            # No rules matched, use default action
            action = Action(
                agent_id=observation.agent_id,
                action_type=self.default_action,
                parameters={"reason": "no_rules_matched"}
            )
            
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, True)
            return action
            
        except Exception as e:
            decision_time = time.perf_counter() - start_time
            self._record_decision(decision_time, False)
            raise InvalidObservationError(f"Failed to process observation: {e}")
    
    def process_batch(self, observations: List[Observation]) -> List[Action]:
        """Process batch of observations using rule-based logic"""
        start_time = time.perf_counter()
        
        try:
            actions = []
            for obs in observations:
                action = self.process_single(obs)
                actions.append(action)
            
            batch_time = time.perf_counter() - start_time
            
            # Record batch performance
            self._performance_metrics['total_batches'] += 1
            self._performance_metrics['successful_batches'] += 1
            self._performance_metrics['total_latency'] += batch_time
            
            return actions
            
        except Exception as e:
            batch_time = time.perf_counter() - start_time
            self._performance_metrics['total_batches'] += 1
            self._performance_metrics['failed_batches'] += 1
            self._performance_metrics['total_latency'] += batch_time
            raise InvalidObservationError(f"Failed to process batch: {e}")
    
    def add_rule(self, rule: BehavioralRule):
        """Add a new behavioral rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, rule_name: str):
        """Remove a rule by name"""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def get_rules(self) -> List[BehavioralRule]:
        """Get all rules"""
        return self.rules.copy()
    
    def can_handle_agents(self, agents: List) -> bool:
        """Rule engine can handle any agents"""
        return True
    
    def decide(self, agent_id: AgentID, observation: Observation) -> Action:
        """Decide action for agent (alias for process_single)"""
        return self.process_single(observation)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get rule engine performance metrics"""
        base_metrics = super().get_performance_metrics()
        
        rule_metrics = {
            "num_rules": len(self.rules),
            "rule_names": [rule.name for rule in self.rules],
            "default_action": self.default_action
        }
        
        base_metrics.update(rule_metrics)
        return base_metrics


# Predefined rule sets for common behaviors
class CooperationRules:
    """Predefined rules for cooperation studies"""
    
    @staticmethod
    def create_cooperation_rules() -> List[BehavioralRule]:
        return [
            SimpleRule(
                name="cooperate_with_cooperators",
                condition="cooperative_personality",
                action_type="cooperate",
                parameters={"target": "neighbors", "strength": 1.0},
                priority=10
            ),
            SimpleRule(
                name="defect_against_defectors",
                condition="selfish_personality",
                action_type="defect",
                parameters={"target": "neighbors", "strength": 0.9},
                priority=9
            ),
            SimpleRule(
                name="punish_defectors",
                condition="cooperative_personality",
                action_type="punish",
                parameters={"target": "defectors", "strength": 0.5},
                priority=8
            )
        ]


class FlockingRules:
    """Predefined rules for flocking behavior"""
    
    @staticmethod
    def create_flocking_rules() -> List[BehavioralRule]:
        return [
            SimpleRule(
                name="align_with_neighbors",
                condition="has_neighbors",
                action_type="align",
                parameters={"target": "neighbors", "strength": 0.1},
                priority=10
            ),
            SimpleRule(
                name="cohere_to_center",
                condition="has_neighbors",
                action_type="cohere",
                parameters={"target": "neighbors", "strength": 0.05},
                priority=9
            ),
            SimpleRule(
                name="separate_from_neighbors",
                condition="has_neighbors",
                action_type="separate",
                parameters={"target": "neighbors", "strength": 0.2},
                priority=8
            ),
            SimpleRule(
                name="random_wander",
                condition="no_neighbors",
                action_type="wander",
                parameters={"strength": 0.1},
                priority=5
            )
        ]


class SocialNetworkRules:
    """Predefined rules for social network behavior"""
    
    @staticmethod
    def create_social_rules() -> List[BehavioralRule]:
        return [
            SimpleRule(
                name="share_information",
                condition="has_neighbors",
                action_type="share",
                parameters={"content": "information", "target": "neighbors"},
                priority=10
            ),
            SimpleRule(
                name="form_connections",
                condition="random",
                action_type="connect",
                parameters={"target": "random_neighbor", "strength": 0.5},
                priority=8
            ),
            SimpleRule(
                name="break_weak_connections",
                condition="random",
                action_type="disconnect",
                parameters={"target": "weak_connections", "threshold": 0.3},
                priority=6
            )
        ]
