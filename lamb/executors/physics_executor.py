"""
Physics executor for continuous dynamics with sub-stepping.

This executor handles physics-based simulations where agents move continuously
in space and require sub-stepping for numerical stability. This is suitable
for flocking, traffic, and other continuous-space models.
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


class PhysicsExecutor(BaseExecutor):
    """
    Executor for physics-based simulations with continuous dynamics.
    
    This executor implements continuous simulation with sub-stepping:
    1. Multiple sub-steps for physics accuracy
    2. Agents decide less frequently than physics updates
    3. Forces and velocities are integrated continuously
    4. Collision detection and response
    
    This is suitable for:
    - Flocking models (Boids)
    - Traffic simulations
    - Particle systems
    - Continuous-space social models
    """
    
    def run(self, 
            environment: BaseEnvironment,
            agents: List[BaseAgent],
            engine: BaseEngine,
            max_steps: int,
            config: Optional[SimulationConfig] = None) -> SimulationResults:
        """
        Execute physics-based simulation with continuous dynamics.
        
        Args:
            environment: The physics environment
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
        
        # Get physics configuration
        dt = getattr(environment, 'dt', 0.1)  # Time step
        sub_steps = getattr(environment, 'sub_steps', 1)  # Sub-steps per main step
        decision_interval = getattr(environment, 'decision_interval', 1)  # Decision frequency
        
        # Run simulation steps
        for step in range(max_steps):
            step_start_time = time.time()
            current_decisions = []
            
            # Multiple sub-steps for physics accuracy
            for sub_step in range(sub_steps):
                # Apply physics updates (forces, velocities, positions)
                if hasattr(environment, 'step_physics'):
                    environment.step_physics(dt)
                else:
                    # Fallback to regular step
                    environment.step()
                
                # Agents decide less frequently than physics updates
                if sub_step % decision_interval == 0:
                    # All agents observe
                    observations = self._create_observations(agents, environment)
                    
                    # All agents decide
                    decisions = self._get_agent_decisions(observations, engine)
                    current_decisions = decisions
                    
                    # All agents act
                    self._execute_agent_actions(agents, decisions, environment)
            
            # Record step metrics (only for main steps)
            step_metrics.append(self._record_step_metrics(step, environment, agents, current_decisions))
            agent_decisions.append({agent.agent_id: decision for agent, decision in zip(agents, current_decisions)})
            
            # Log progress for long simulations
            if step % 50 == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"Physics simulation: Step {step}/{max_steps} completed in {elapsed:.2f}s")
        
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
        return "physics"
    
    def _record_step_metrics(self, step: int, environment: BaseEnvironment, 
                           agents: List[BaseAgent], decisions: List[Action]) -> Dict[str, Any]:
        """Record metrics specific to physics simulations"""
        base_metrics = super()._record_step_metrics(step, environment, agents, decisions)
        
        # Add physics-specific metrics
        physics_metrics = {
            "agent_positions": [getattr(agent, 'position', None) for agent in agents],
            "agent_velocities": [getattr(agent, 'velocity', None) for agent in agents],
            "agent_masses": [getattr(agent, 'mass', None) for agent in agents],
            "world_bounds": getattr(environment, 'world_bounds', None),
            "dt": getattr(environment, 'dt', 0.1),
            "decision_summary": self._summarize_decisions(decisions)
        }
        
        base_metrics.update(physics_metrics)
        return base_metrics
    
    def _summarize_decisions(self, decisions: List[Action]) -> Dict[str, int]:
        """Summarize decision patterns for this step"""
        decision_counts = {}
        for decision in decisions:
            action_type = decision.action_type
            decision_counts[action_type] = decision_counts.get(action_type, 0) + 1
        return decision_counts
