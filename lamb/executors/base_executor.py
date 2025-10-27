"""
Base executor for paradigm-specific execution logic.

This module defines the abstract base class that all executors must implement.
Executors handle the specific execution patterns for different simulation
paradigms while maintaining a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

from ..core.base_agent import BaseAgent
from ..core.base_environment import BaseEnvironment
from ..core.base_engine import BaseEngine
from ..core.simulation import SimulationResults
from ..core.types import AgentID, Observation, Action
from ..config.simulation_config import SimulationConfig


class BaseExecutor(ABC):
    """
    Abstract base class for paradigm-specific execution logic.
    
    Each paradigm (Grid, Physics, Network) has different execution requirements:
    - Grid: Discrete synchronous stepping
    - Physics: Continuous dynamics with sub-stepping
    - Network: Event-driven propagation
    
    Executors implement these patterns while maintaining a consistent interface.
    """
    
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
    
    def _create_observation(self, agent: BaseAgent, environment: BaseEnvironment) -> Observation:
        """Create observation for an agent"""
        return agent.observe(environment)
    
    def _create_observations(self, agents: List[BaseAgent], environment: BaseEnvironment) -> List[Observation]:
        """Create observations for all agents"""
        return [self._create_observation(agent, environment) for agent in agents]
    
    def _get_agent_decisions(self, observations: List[Observation], engine: BaseEngine) -> List[Action]:
        """Get decisions from engine for all observations"""
        if hasattr(engine, 'decide_batch'):
            return engine.decide_batch(observations)
        else:
            # Fallback to individual decisions
            return [engine.decide(obs.agent_id, obs) for obs in observations]
    
    def _execute_agent_actions(self, agents: List[BaseAgent], 
                              decisions: List[Action], 
                              environment: BaseEnvironment):
        """Execute actions for all agents"""
        for agent, decision in zip(agents, decisions):
            if hasattr(agent, 'act'):
                agent.act(decision, environment)
            elif hasattr(agent, 'step'):
                # Fallback to step method
                agent.step(environment)
    
    def _record_step_metrics(self, step: int, environment: BaseEnvironment, 
                           agents: List[BaseAgent], decisions: List[Action]) -> Dict[str, Any]:
        """Record metrics for a single step"""
        return {
            "step": step,
            "num_agents": len(agents),
            "environment_state": environment.get_state(),
            "agent_decisions": {agent.agent_id: decision for agent, decision in zip(agents, decisions)},
            "timestamp": time.time()
        }
