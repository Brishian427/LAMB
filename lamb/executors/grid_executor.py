"""
Grid executor for discrete synchronous stepping.

This executor handles grid-based simulations where all agents act
synchronously in discrete time steps. This is the most common pattern
for cellular automata and simple agent-based models.
"""

from typing import List, Dict, Any, Optional
import time

from .base_executor import BaseExecutor
from ..core.base_agent import BaseAgent
from ..core.base_environment import BaseEnvironment
from ..core.base_engine import BaseEngine
from ..core.simulation import SimulationResults
from ..core.types import Observation, Action
from ..config.simulation_config import SimulationConfig


class GridExecutor(BaseExecutor):
    """
    Executor for grid-based simulations with discrete synchronous stepping.
    
    This executor implements the standard ABM pattern:
    1. Environment updates
    2. All agents observe simultaneously
    3. All agents decide simultaneously
    4. All agents act simultaneously
    5. Repeat for next step
    
    This is suitable for:
    - Cellular automata
    - Schelling segregation models
    - Game of Life variants
    - Simple spatial agent models
    """
    
    def run(self, 
            environment: BaseEnvironment,
            agents: List[BaseAgent],
            engine: BaseEngine,
            max_steps: int,
            config: Optional[SimulationConfig] = None) -> SimulationResults:
        """
        Execute grid-based simulation with discrete synchronous stepping.
        
        Args:
            environment: The grid environment
            agents: List of agents in the simulation
            engine: Decision-making engine
            max_steps: Maximum number of steps to run
            config: Optional simulation configuration
            
        Returns:
            SimulationResults containing all simulation data
        """
        start_time = time.time()
        step_metrics = []
        agent_decisions = []
        
        # Run simulation steps
        for step in range(max_steps):
            step_start_time = time.time()
            
            # 1. Environment updates (spatial indexing, resource updates, etc.)
            environment.step()
            
            # 2. All agents observe simultaneously
            observations = self._create_observations(agents, environment)
            
            # 3. All agents decide simultaneously (can be batched for LLM)
            decisions = self._get_agent_decisions(observations, engine)
            
            # 4. All agents act simultaneously
            self._execute_agent_actions(agents, decisions, environment)
            
            # Record step metrics
            step_metrics.append(self._record_step_metrics(step, environment, agents, decisions))
            agent_decisions.append({agent.agent_id: decision for agent, decision in zip(agents, decisions)})
            
            # Log progress for long simulations
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"Grid simulation: Step {step}/{max_steps} completed in {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        
        # Create results
        results = SimulationResults(
            step_count=max_steps,
            total_time=total_time,
            step_metrics=step_metrics,
            final_state=environment.get_state(),
            agent_decisions=agent_decisions
        )
        
        return results
    
    def get_paradigm(self) -> str:
        """Return the paradigm this executor handles"""
        return "grid"
    
    def _record_step_metrics(self, step: int, environment: BaseEnvironment, 
                           agents: List[BaseAgent], decisions: List[Action]) -> Dict[str, Any]:
        """Record metrics specific to grid simulations"""
        base_metrics = super()._record_step_metrics(step, environment, agents, decisions)
        
        # Add grid-specific metrics
        grid_metrics = {
            "grid_dimensions": getattr(environment, 'grid_dimensions', None),
            "agent_positions": [getattr(agent, 'position', None) for agent in agents],
            "decision_summary": self._summarize_decisions(decisions)
        }
        
        base_metrics.update(grid_metrics)
        return base_metrics
    
    def _summarize_decisions(self, decisions: List[Action]) -> Dict[str, int]:
        """Summarize decision patterns for this step"""
        decision_counts = {}
        for decision in decisions:
            action_type = decision.action_type
            decision_counts[action_type] = decision_counts.get(action_type, 0) + 1
        return decision_counts
