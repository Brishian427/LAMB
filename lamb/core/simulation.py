"""
Composition-based Simulation architecture for LAMB framework.

This module implements the core Simulation class that uses composition rather than
inheritance. The Simulation acts as a lightweight coordinator that combines
independent components: Environment, Agents, Engine, and Executor.

Key Design Principles:
- Composition over inheritance
- Components own their state
- Simulation only coordinates
- Easy engine swapping
- AI-friendly configuration
- Paradigm-agnostic design
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass

from .base_agent import BaseAgent
from .base_environment import BaseEnvironment
from .base_engine import BaseEngine
from .types import AgentID, Observation, Action
from ..config.simulation_config import SimulationConfig


@dataclass
class SimulationResults:
    """Results from a simulation run"""
    step_count: int
    total_time: float
    step_metrics: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    agent_decisions: List[Dict[AgentID, Action]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of simulation results"""
        return {
            "total_steps": self.step_count,
            "total_time": self.total_time,
            "avg_step_time": self.total_time / self.step_count if self.step_count > 0 else 0,
            "final_agent_count": len(self.final_state.get("agents", [])),
            "decision_patterns": self._analyze_decisions()
        }
    
    def _analyze_decisions(self) -> Dict[str, int]:
        """Analyze decision patterns across all steps"""
        decision_counts = {}
        for step_decisions in self.agent_decisions:
            for action in step_decisions.values():
                action_type = action.get("action_type", "unknown")
                decision_counts[action_type] = decision_counts.get(action_type, 0) + 1
        return decision_counts


class BaseExecutor(ABC):
    """Abstract base class for paradigm-specific execution logic"""
    
    @abstractmethod
    def run(self, 
            environment: BaseEnvironment,
            agents: List[BaseAgent],
            engine: BaseEngine,
            max_steps: int,
            config: Optional[SimulationConfig] = None) -> SimulationResults:
        """
        Execute the simulation using paradigm-specific logic.
        
        Args:
            environment: The simulation environment
            agents: List of agents in the simulation
            engine: Decision-making engine
            max_steps: Maximum number of steps to run
            config: Optional simulation configuration
            
        Returns:
            SimulationResults containing all simulation data
        """
        pass
    
    @abstractmethod
    def get_paradigm(self) -> str:
        """Return the paradigm this executor handles"""
        pass


class ComponentValidator:
    """Validates component compatibility and performance bounds"""
    
    @staticmethod
    def validate_paradigm_compatibility(env: BaseEnvironment, 
                                      agents: List[BaseAgent], 
                                      executor: BaseExecutor) -> bool:
        """Ensure all components use the same paradigm"""
        if not agents:
            return True
        
        env_paradigm = getattr(env, 'paradigm', 'unknown')
        agent_paradigm = getattr(agents[0], 'paradigm', 'unknown')
        executor_paradigm = executor.get_paradigm()
        
        # Check if all paradigms match
        paradigms = {env_paradigm, agent_paradigm, executor_paradigm}
        if len(paradigms) > 1 and 'unknown' not in paradigms:
            return False
        
        return True
    
    @staticmethod
    def validate_engine_capabilities(engine: BaseEngine, agents: List[BaseAgent]) -> bool:
        """Check if engine can handle the given agents"""
        if hasattr(engine, 'can_handle_agents'):
            return engine.can_handle_agents(agents)
        
        # Default check: engine should have decide method
        return hasattr(engine, 'decide') and callable(engine.decide)
    
    @staticmethod
    def validate_performance_bounds(agents: List[BaseAgent], engine: BaseEngine) -> List[str]:
        """Check performance constraints and return warnings"""
        warnings = []
        
        # Check agent count limits
        if len(agents) > 1000:
            warnings.append(f"Large agent count ({len(agents)}) may impact performance")
        
        # Check LLM engine with many agents
        if len(agents) > 500 and hasattr(engine, '__class__') and 'LLM' in engine.__class__.__name__:
            warnings.append(f"LLM engine with {len(agents)} agents may be slow and expensive")
        
        # Check memory usage estimate
        estimated_memory = len(agents) * 1024  # 1KB per agent estimate
        if estimated_memory > 1024 * 1024:  # 1MB
            warnings.append(f"Estimated memory usage: {estimated_memory / 1024 / 1024:.1f}MB")
        
        return warnings


class Simulation:
    """
    Lightweight coordinator that composes independent components.
    
    This class uses composition rather than inheritance. Users create simulation
    instances by combining four independent components: Environment, Agents, 
    Engine, and Executor. Each component has its own interface and can be 
    mixed and matched freely.
    
    Key Features:
    - No inheritance required
    - Easy engine swapping
    - Component validation
    - Paradigm-agnostic design
    - AI-friendly interface
    """
    
    def __init__(self, 
                 environment: BaseEnvironment,
                 agents: List[BaseAgent], 
                 engine: BaseEngine,
                 executor: BaseExecutor,
                 config: Optional[SimulationConfig] = None):
        """
        Initialize simulation with composed components.
        
        Args:
            environment: The simulation environment
            agents: List of agents in the simulation
            engine: Decision-making engine
            executor: Paradigm-specific execution logic
            config: Optional simulation configuration
        """
        self.environment = environment
        self.agents = agents
        self.engine = engine
        self.executor = executor
        self.config = config or SimulationConfig(
            name="LAMB Simulation",
            paradigm="grid",  # Default paradigm
            num_agents=len(agents),
            max_steps=1000
        )
        
        # Validate component compatibility
        self._validate_components()
        
        # Minimal coordinator state
        self.step_count = 0
        self.running = True
        self.results = None
    
    def _validate_components(self):
        """Validate that all components are compatible"""
        validator = ComponentValidator()
        
        # Check paradigm compatibility
        if not validator.validate_paradigm_compatibility(
            self.environment, self.agents, self.executor
        ):
            raise ValueError(
                "Component paradigm mismatch: Environment, agents, and executor "
                "must all use the same paradigm"
            )
        
        # Check engine capabilities
        if not validator.validate_engine_capabilities(self.engine, self.agents):
            raise ValueError(
                "Engine cannot handle the provided agents. Check engine capabilities."
            )
        
        # Check performance bounds
        warnings = validator.validate_performance_bounds(self.agents, self.engine)
        for warning in warnings:
            warnings.warn(warning, UserWarning)
    
    def run(self, max_steps: int = 1000) -> SimulationResults:
        """
        Run the simulation using the paradigm-specific executor.
        
        Args:
            max_steps: Maximum number of steps to run
            
        Returns:
            SimulationResults containing all simulation data
        """
        if not self.agents:
            raise ValueError("Cannot run simulation with no agents")
        
        # Delegate execution to paradigm-specific executor
        self.results = self.executor.run(
            environment=self.environment,
            agents=self.agents,
            engine=self.engine,
            max_steps=max_steps,
            config=self.config
        )
        
        return self.results
    
    def swap_engine(self, new_engine: BaseEngine):
        """
        Swap the decision engine (one-line engine swapping).
        
        Args:
            new_engine: New engine to use for decision making
            
        Raises:
            ValueError: If new engine is incompatible with current agents
        """
        if not ComponentValidator.validate_engine_capabilities(new_engine, self.agents):
            raise ValueError("New engine incompatible with current agents")
        
        self.engine = new_engine
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            "step_count": self.step_count,
            "running": self.running,
            "num_agents": len(self.agents),
            "environment_state": self.environment.get_state(),
            "has_results": self.results is not None
        }
    
    def get_results(self) -> Optional[SimulationResults]:
        """Get simulation results if available"""
        return self.results
    
    def reset(self):
        """Reset simulation to initial state"""
        self.step_count = 0
        self.running = True
        self.results = None
        
        # Reset environment if it has a reset method
        if hasattr(self.environment, 'reset'):
            self.environment.reset()
        
        # Reset agents if they have a reset method
        for agent in self.agents:
            if hasattr(agent, 'reset'):
                agent.reset()
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the simulation"""
        if not ComponentValidator.validate_engine_capabilities(self.engine, [agent]):
            raise ValueError("Engine cannot handle this agent type")
        
        self.agents.append(agent)
        self.environment.add_agent(agent)
    
    def remove_agent(self, agent_id: AgentID):
        """Remove an agent from the simulation"""
        # Find and remove agent
        agent_to_remove = None
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent_to_remove = agent
                break
        
        if agent_to_remove:
            self.agents.remove(agent_to_remove)
            self.environment.remove_agent(agent_id)
        else:
            raise ValueError(f"Agent with ID {agent_id} not found")
    
    def __repr__(self) -> str:
        """String representation of the simulation"""
        return (f"Simulation(agents={len(self.agents)}, "
                f"engine={self.engine.__class__.__name__}, "
                f"executor={self.executor.__class__.__name__})")
