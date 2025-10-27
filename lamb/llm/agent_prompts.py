"""
Agent Prompting and Rule System for LAMB Framework.

This module provides flexible, researcher-controlled prompting and rule-setting
for LLM-driven agents. Researchers can define custom prompts, behavioral rules,
and personality templates for their agents.

Key Features:
- Custom prompt templates with variable substitution
- Behavioral rule definitions
- Personality and trait systems
- Context-aware prompt generation
- Multi-language support
- Prompt versioning and A/B testing
"""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from string import Template
from abc import ABC, abstractmethod

from ..core.types import Observation, Action, AgentID


class PromptType(str, Enum):
    """Types of prompts available for agents"""
    DECISION = "decision"           # Main decision-making prompt
    OBSERVATION = "observation"     # How to interpret observations
    PERSONALITY = "personality"     # Personality and traits
    RULES = "rules"                # Behavioral rules and constraints
    CONTEXT = "context"            # Environmental context
    MEMORY = "memory"              # Memory and learning prompts


class RuleType(str, Enum):
    """Types of behavioral rules"""
    CONSTRAINT = "constraint"       # Hard constraints (must follow)
    PREFERENCE = "preference"       # Soft preferences (should follow)
    GOAL = "goal"                  # Long-term goals
    STRATEGY = "strategy"          # Decision strategies
    INTERACTION = "interaction"     # How to interact with others


@dataclass
class PromptTemplate:
    """
    A prompt template with variable substitution and context awareness.
    
    Variables are enclosed in ${variable_name} and can be:
    - Agent properties: ${agent_id}, ${position}, ${metadata}
    - Environment state: ${environment_state}, ${neighbors}
    - Observation data: ${observation}, ${sensory_input}
    - Custom variables: ${custom_var}
    """
    name: str
    prompt_type: PromptType
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with given context"""
        try:
            template = Template(self.template)
            return template.safe_substitute(context)
        except Exception as e:
            raise ValueError(f"Error rendering template '{self.name}': {e}")


@dataclass
class BehavioralRule:
    """
    A behavioral rule that constrains or guides agent behavior.
    """
    name: str
    rule_type: RuleType
    condition: str  # Python expression that evaluates to boolean
    action: str    # What to do when condition is true
    priority: int = 0  # Higher priority rules are checked first
    description: str = ""
    enabled: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the rule condition in the given context"""
        try:
            # Safe evaluation with limited context
            safe_context = {
                'agent_id': context.get('agent_id'),
                'position': context.get('position'),
                'neighbors': context.get('neighbors', []),
                'environment_state': context.get('environment_state', {}),
                'metadata': context.get('metadata', {}),
                'len': len,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum
            }
            return eval(self.condition, {"__builtins__": {}}, safe_context)
        except Exception as e:
            print(f"Warning: Rule '{self.name}' evaluation failed: {e}")
            return False


@dataclass
class AgentPersonality:
    """
    Defines an agent's personality traits and behavioral tendencies.
    """
    name: str
    traits: Dict[str, float] = field(default_factory=dict)  # trait_name: strength (0-1)
    preferences: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    description: str = ""
    
    def get_trait(self, trait_name: str) -> float:
        """Get the strength of a personality trait"""
        return self.traits.get(trait_name, 0.5)  # Default neutral


class PromptManager:
    """
    Manages prompt templates and behavioral rules for agents.
    
    This is the main interface for researchers to define how their agents
    should behave and what prompts they should use.
    """
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.rules: Dict[str, BehavioralRule] = {}
        self.personalities: Dict[str, AgentPersonality] = {}
        self.default_templates = self._create_default_templates()
        self._load_defaults()
    
    def _create_default_templates(self) -> Dict[str, PromptTemplate]:
        """Create default prompt templates for common use cases"""
        return {
            "grid_decision": PromptTemplate(
                name="grid_decision",
                prompt_type=PromptType.DECISION,
                template="""You are an agent in a grid-based simulation.

Agent ID: ${agent_id}
Position: ${position}
Grid Size: ${grid_dimensions}
Neighbors: ${neighbor_count} agents nearby

Your goal: ${goal}
Your personality: ${personality}
Your constraints: ${constraints}

Available actions:
- move: Move to adjacent cell (north/south/east/west/northeast/northwest/southeast/southwest/random)
- stay: Remain in current position
- interact: Interact with environment or neighbors
- consume: Consume resources from current cell
- produce: Produce resources in current cell

Current situation:
${situation_description}

Based on your personality, goals, and constraints, choose the best action.
Respond with JSON: {"action_type": "move", "parameters": {"direction": "north"}}

Your response:""",
                variables=["agent_id", "position", "grid_dimensions", "neighbor_count", 
                          "goal", "personality", "constraints", "situation_description"],
                description="Default decision prompt for grid-based agents"
            ),
            
            "physics_decision": PromptTemplate(
                name="physics_decision",
                prompt_type=PromptType.DECISION,
                template="""You are an agent in a physics-based simulation.

Agent ID: ${agent_id}
Position: ${position}
Velocity: ${velocity}
Mass: ${mass}
World Bounds: ${world_bounds}

Your goal: ${goal}
Your personality: ${personality}
Your constraints: ${constraints}

Available actions:
- apply_force: Apply force in a direction (specify force vector)
- change_velocity: Change velocity directly
- interact: Interact with nearby agents
- observe: Make detailed observations

Current situation:
${situation_description}

Based on your personality, goals, and constraints, choose the best action.
Respond with JSON: {"action_type": "apply_force", "parameters": {"force": [1.0, 0.0]}}

Your response:""",
                variables=["agent_id", "position", "velocity", "mass", "world_bounds",
                          "goal", "personality", "constraints", "situation_description"],
                description="Default decision prompt for physics-based agents"
            ),
            
            "network_decision": PromptTemplate(
                name="network_decision",
                prompt_type=PromptType.DECISION,
                template="""You are an agent in a network-based simulation.

Agent ID: ${agent_id}
Node ID: ${node_id}
Neighbors: ${neighbor_count} connected agents
Network Type: ${network_type}

Your goal: ${goal}
Your personality: ${personality}
Your constraints: ${constraints}

Available actions:
- move: Move to a connected node
- interact: Interact with neighbors
- broadcast: Send message to all neighbors
- observe: Observe network structure

Current situation:
${situation_description}

Based on your personality, goals, and constraints, choose the best action.
Respond with JSON: {"action_type": "move", "parameters": {"target_node": 5}}

Your response:""",
                variables=["agent_id", "node_id", "neighbor_count", "network_type",
                          "goal", "personality", "constraints", "situation_description"],
                description="Default decision prompt for network-based agents"
            )
        }
    
    def _load_defaults(self):
        """Load default templates and rules"""
        for template in self.default_templates.values():
            self.add_template(template)
    
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template"""
        self.templates[template.name] = template
    
    def add_rule(self, rule: BehavioralRule):
        """Add a new behavioral rule"""
        self.rules[rule.name] = rule
    
    def add_personality(self, personality: AgentPersonality):
        """Add a new agent personality"""
        self.personalities[personality.name] = personality
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(name)
    
    def get_rule(self, name: str) -> Optional[BehavioralRule]:
        """Get a behavioral rule by name"""
        return self.rules.get(name)
    
    def get_personality(self, name: str) -> Optional[AgentPersonality]:
        """Get an agent personality by name"""
        return self.personalities.get(name)
    
    def render_prompt(
        self, 
        template_name: str, 
        context: Dict[str, Any],
        agent_personality: Optional[AgentPersonality] = None
    ) -> str:
        """Render a prompt template with context and personality"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Add personality information to context
        if agent_personality:
            context.update({
                'personality': agent_personality.name,
                'traits': agent_personality.traits,
                'goals': ', '.join(agent_personality.goals),
                'constraints': ', '.join(agent_personality.constraints)
            })
        
        return template.render(context)
    
    def evaluate_rules(self, context: Dict[str, Any]) -> List[BehavioralRule]:
        """Evaluate all enabled rules and return applicable ones"""
        applicable_rules = []
        for rule in sorted(self.rules.values(), key=lambda r: r.priority, reverse=True):
            if rule.enabled and rule.evaluate(context):
                applicable_rules.append(rule)
        return applicable_rules
    
    def create_custom_template(
        self,
        name: str,
        template: str,
        prompt_type: PromptType = PromptType.DECISION,
        description: str = ""
    ) -> PromptTemplate:
        """Create a custom prompt template from a string"""
        # Extract variables from template
        variables = re.findall(r'\$\{([^}]+)\}', template)
        
        prompt_template = PromptTemplate(
            name=name,
            prompt_type=prompt_type,
            template=template,
            variables=variables,
            description=description
        )
        
        self.add_template(prompt_template)
        return prompt_template


class AgentPromptBuilder:
    """
    Builder class for creating agent prompts with context and personality.
    
    This class helps researchers build complex prompts by combining
    templates, rules, and personality traits.
    """
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def build_decision_prompt(
        self,
        observation: Observation,
        agent_personality: Optional[AgentPersonality] = None,
        custom_template: Optional[str] = None
    ) -> str:
        """Build a decision prompt for an agent"""
        
        # Determine template to use
        if custom_template:
            template_name = custom_template
        else:
            # Choose template based on paradigm
            paradigm = observation.paradigm
            if paradigm == "grid":
                template_name = "grid_decision"
            elif paradigm == "physics":
                template_name = "physics_decision"
            elif paradigm == "network":
                template_name = "network_decision"
            else:
                template_name = "grid_decision"  # Default fallback
        
        # Build context
        context = {
            'agent_id': observation.agent_id,
            'position': observation.position,
            'neighbor_count': len(observation.neighbors),
            'environment_state': observation.environment_state,
            'metadata': observation.metadata,
            'situation_description': self._describe_situation(observation)
        }
        
        # Add paradigm-specific context
        if observation.paradigm == "grid":
            context.update({
                'grid_dimensions': observation.environment_state.get('grid_dimensions', 'unknown'),
                'boundary_condition': observation.environment_state.get('boundary_condition', 'unknown')
            })
        elif observation.paradigm == "physics":
            context.update({
                'velocity': observation.metadata.get('velocity', [0, 0]),
                'mass': observation.metadata.get('mass', 1.0),
                'world_bounds': observation.environment_state.get('world_bounds', 'unknown')
            })
        elif observation.paradigm == "network":
            context.update({
                'node_id': observation.metadata.get('node_id', observation.agent_id),
                'network_type': observation.environment_state.get('network_type', 'unknown')
            })
        
        # Render the prompt
        return self.prompt_manager.render_prompt(
            template_name, 
            context, 
            agent_personality
        )
    
    def _describe_situation(self, observation: Observation) -> str:
        """Create a natural language description of the current situation"""
        paradigm = observation.paradigm
        neighbors = observation.neighbors
        
        if paradigm == "grid":
            return f"You are at position {observation.position} in a grid world. " \
                   f"You can see {len(neighbors)} other agents nearby."
        elif paradigm == "physics":
            return f"You are at position {observation.position} with velocity {observation.metadata.get('velocity', [0, 0])}. " \
                   f"You can sense {len(neighbors)} other agents in your vicinity."
        elif paradigm == "network":
            return f"You are at node {observation.metadata.get('node_id', observation.agent_id)} in a network. " \
                   f"You are connected to {len(neighbors)} other agents."
        else:
            return f"You are in an unknown environment with {len(neighbors)} other agents nearby."


# Predefined personality templates for common research scenarios
COMMON_PERSONALITIES = {
    "cooperative": AgentPersonality(
        name="cooperative",
        traits={"cooperation": 0.9, "trust": 0.8, "altruism": 0.7},
        goals=["help others", "build relationships", "create harmony"],
        constraints=["avoid conflict", "be fair"],
        description="A cooperative agent that prioritizes helping others"
    ),
    
    "competitive": AgentPersonality(
        name="competitive",
        traits={"competition": 0.9, "ambition": 0.8, "aggression": 0.6},
        goals=["win", "gain advantage", "outperform others"],
        constraints=["follow rules", "maintain reputation"],
        description="A competitive agent that seeks to outperform others"
    ),
    
    "cautious": AgentPersonality(
        name="cautious",
        traits={"caution": 0.9, "risk_aversion": 0.8, "patience": 0.7},
        goals=["avoid danger", "preserve resources", "stay safe"],
        constraints=["never take unnecessary risks", "conserve energy"],
        description="A cautious agent that avoids risks and conserves resources"
    ),
    
    "explorer": AgentPersonality(
        name="explorer",
        traits={"curiosity": 0.9, "adventure": 0.8, "independence": 0.7},
        goals=["discover new areas", "learn about environment", "find resources"],
        constraints=["avoid getting lost", "maintain contact with others"],
        description="An exploratory agent that seeks new experiences"
    ),
    
    "leader": AgentPersonality(
        name="leader",
        traits={"leadership": 0.9, "confidence": 0.8, "charisma": 0.7},
        goals=["guide others", "make decisions", "build consensus"],
        constraints=["be responsible", "consider group welfare"],
        description="A leadership-oriented agent that guides others"
    )
}


def create_research_personality(
    name: str,
    traits: Dict[str, float],
    goals: List[str],
    constraints: List[str],
    description: str = ""
) -> AgentPersonality:
    """Helper function to create custom research personalities"""
    return AgentPersonality(
        name=name,
        traits=traits,
        goals=goals,
        constraints=constraints,
        description=description
    )


def create_behavioral_rule(
    name: str,
    condition: str,
    action: str,
    rule_type: RuleType = RuleType.CONSTRAINT,
    priority: int = 0,
    description: str = ""
) -> BehavioralRule:
    """Helper function to create behavioral rules"""
    return BehavioralRule(
        name=name,
        rule_type=rule_type,
        condition=condition,
        action=action,
        priority=priority,
        description=description
    )
