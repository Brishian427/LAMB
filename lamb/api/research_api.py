"""
Research-focused API for the LAMB framework.

Based on Technical_Specification.md Section 9: Configuration Schema and API Specification.
Provides a high-level, research-friendly interface for creating and running
agent-based simulations with LLM integration.

This is the primary API for Phase 1, designed for research use cases.
Future phases will add beginner and developer APIs with different complexity levels.

Key features:
- Simple, intuitive interface for researchers
- Automatic configuration and optimization
- Built-in data collection and analysis
- Performance monitoring and reporting
- Easy integration with research workflows
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import random

from ..config.simulation_config import SimulationConfig, ParadigmType, EngineType
from ..core.base_agent import BaseAgent
from ..core.base_environment import BaseEnvironment
from ..core.base_engine import BaseEngine
from ..engines.llm_engine import LLMEngine
from ..paradigms.grid import GridAgent, GridEnvironment
from ..paradigms.physics import PhysicsAgent, PhysicsEnvironment
from ..paradigms.network import NetworkAgent, NetworkEnvironment
from ..monitoring.performance_monitor import PerformanceMonitor
from ..monitoring.metrics_collector import MetricsCollector


@dataclass
class SimulationResult:
    """Results from a completed simulation"""
    config: SimulationConfig
    steps_completed: int
    total_time: float
    performance_metrics: Dict[str, Any]
    agent_data: List[Dict[str, Any]]
    environment_data: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class ResearchAPI:
    """
    Research-focused API for LAMB simulations.
    
    This API is designed for researchers who want to:
    - Quickly set up and run ABM simulations with LLM agents
    - Collect and analyze simulation data
    - Monitor performance and optimize parameters
    - Export results for further analysis
    
    The API handles all the complexity of paradigm selection,
    engine configuration, and performance optimization.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize Research API.
        
        Args:
            config: Simulation configuration (if None, will be set later)
        """
        self.config = config
        self.environment: Optional[BaseEnvironment] = None
        self.engine: Optional[BaseEngine] = None
        self.agents: List[BaseAgent] = []
        
        # Monitoring components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Simulation state
        self.current_step = 0
        self.is_running = False
        self.start_time = 0.0
        
        # Data collection
        self.step_data: List[Dict[str, Any]] = []
        self.agent_trajectories: Dict[int, List[Dict[str, Any]]] = {}
    
    def create_simulation(
        self,
        paradigm: str,
        num_agents: int,
        engine_type: str = "llm",
        **kwargs
    ) -> 'ResearchAPI':
        """
        Create a new simulation with automatic configuration.
        
        Args:
            paradigm: Simulation paradigm ("grid", "physics", "network")
            num_agents: Number of agents
            engine_type: Decision engine ("llm", "mock")
            **kwargs: Additional configuration parameters
            
        Returns:
            Self for method chaining
        """
        # Create configuration
        config_dict = {
            "paradigm": paradigm,
            "engine_type": engine_type,
            "num_agents": num_agents,
            **kwargs
        }
        
        self.config = SimulationConfig(**config_dict)
        
        # Initialize components
        self._initialize_simulation()
        
        return self
    
    def load_config(self, config_path: str) -> 'ResearchAPI':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Self for method chaining
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        self.config = SimulationConfig(**config_dict)
        self._initialize_simulation()
        
        return self
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        if not self.config:
            raise ValueError("No configuration to save")
        
        with open(config_path, 'w') as f:
            json.dump(self.config.dict(), f, indent=2)
    
    def _initialize_simulation(self) -> None:
        """Initialize simulation components based on configuration"""
        if not self.config:
            raise ValueError("No configuration provided")
        
        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        # Initialize environment
        self.environment = self._create_environment()
        
        # Initialize engine
        self.engine = self._create_engine()
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Initialize monitoring
        if self.config.performance_config.enable_monitoring:
            self.performance_monitor = PerformanceMonitor(self.config.performance_config)
            self.metrics_collector = MetricsCollector()
        
        # Reset simulation state
        self.current_step = 0
        self.is_running = False
        self.step_data.clear()
        self.agent_trajectories.clear()
    
    def _create_environment(self) -> BaseEnvironment:
        """Create environment based on paradigm"""
        paradigm_config = self.config.get_paradigm_config()
        
        if self.config.paradigm == ParadigmType.GRID:
            return GridEnvironment(
                dimensions=paradigm_config.dimensions,
                boundary_condition=paradigm_config.boundary_condition,
                cell_size=paradigm_config.cell_size,
                max_agents_per_cell=paradigm_config.max_agents_per_cell
            )
        
        elif self.config.paradigm == ParadigmType.PHYSICS:
            return PhysicsEnvironment(
                world_bounds=paradigm_config.world_bounds,
                boundary_condition=paradigm_config.boundary_condition,
                dt=paradigm_config.dt,
                enable_collisions=paradigm_config.enable_collisions,
                collision_damping=paradigm_config.collision_damping
            )
        
        elif self.config.paradigm == ParadigmType.NETWORK:
            return NetworkEnvironment(
                is_directed=paradigm_config.is_directed,
                weighted=paradigm_config.weighted,
                default_node_capacity=paradigm_config.default_node_capacity,
                enable_message_passing=paradigm_config.enable_message_passing
            )
        
        else:
            raise ValueError(f"Unknown paradigm: {self.config.paradigm}")
    
    def _create_engine(self) -> BaseEngine:
        """Create decision engine based on configuration"""
        if self.config.engine_type == EngineType.LLM:
            engine_config = self.config.llm_config
            return LLMEngine(
                api_key=engine_config.api_key,
                model=engine_config.model,
                max_tokens=engine_config.max_tokens,
                temperature=engine_config.temperature,
                batch_size=engine_config.batch_size,
                cache_size=engine_config.cache_size,
                circuit_breaker_threshold=engine_config.circuit_breaker_threshold,
                timeout_seconds=engine_config.timeout_seconds
            )
        
        elif self.config.engine_type == EngineType.MOCK:
            from ..core.base_engine import MockEngine
            return MockEngine()
        
        else:
            raise ValueError(f"Engine type {self.config.engine_type} not implemented in Phase 1")
    
    def _create_agents(self) -> List[BaseAgent]:
        """Create agents based on paradigm and configuration"""
        agents = []
        
        for i in range(self.config.num_agents):
            agent_id = i
            position = self._generate_initial_position(agent_id)
            metadata = self.config.agent_config.copy()
            
            if self.config.paradigm == ParadigmType.GRID:
                agent = GridAgent(agent_id, position, metadata)
            elif self.config.paradigm == ParadigmType.PHYSICS:
                agent = PhysicsAgent(agent_id, position, metadata)
            elif self.config.paradigm == ParadigmType.NETWORK:
                agent = NetworkAgent(agent_id, position, metadata)
            else:
                raise ValueError(f"Unknown paradigm: {self.config.paradigm}")
            
            # Add agent to environment
            self.environment.add_agent(agent)
            agents.append(agent)
        
        return agents
    
    def _generate_initial_position(self, agent_id: int):
        """Generate initial position for agent based on paradigm"""
        if self.config.paradigm == ParadigmType.GRID:
            grid_config = self.config.grid_config
            x = random.randint(0, grid_config.dimensions[0] - 1)
            y = random.randint(0, grid_config.dimensions[1] - 1)
            return (x, y)
        
        elif self.config.paradigm == ParadigmType.PHYSICS:
            physics_config = self.config.physics_config
            (min_x, min_y), (max_x, max_y) = physics_config.world_bounds
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            return (x, y)
        
        elif self.config.paradigm == ParadigmType.NETWORK:
            # For network, use agent_id as node_id initially
            return agent_id
        
        else:
            return (0, 0)
    
    def run_simulation(
        self,
        steps: Optional[int] = None,
        step_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SimulationResult:
        """
        Run the complete simulation.
        
        Args:
            steps: Number of steps to run (uses config.max_steps if None)
            step_callback: Called after each step with (step, data)
            progress_callback: Called with (current_step, total_steps)
            
        Returns:
            SimulationResult with all collected data
        """
        if not self.config or not self.environment or not self.engine:
            raise ValueError("Simulation not properly initialized")
        
        max_steps = steps or self.config.max_steps
        self.is_running = True
        self.start_time = time.time()
        
        try:
            for step in range(max_steps):
                self.current_step = step
                
                # Execute simulation step
                step_data = self._execute_step()
                
                # Collect data
                self.step_data.append(step_data)
                self._collect_agent_trajectories(step)
                
                # Call callbacks
                if step_callback:
                    step_callback(step, step_data)
                
                if progress_callback:
                    progress_callback(step + 1, max_steps)
                
                # Check for early termination conditions
                if self._should_terminate():
                    break
                
                # Sleep if step interval is specified
                if self.config.step_interval > 0:
                    time.sleep(self.config.step_interval)
            
            # Simulation completed successfully
            total_time = time.time() - self.start_time
            
            return SimulationResult(
                config=self.config,
                steps_completed=self.current_step + 1,
                total_time=total_time,
                performance_metrics=self._get_performance_metrics(),
                agent_data=self._get_agent_data(),
                environment_data=self._get_environment_data(),
                success=True
            )
            
        except Exception as e:
            # Simulation failed
            total_time = time.time() - self.start_time
            
            return SimulationResult(
                config=self.config,
                steps_completed=self.current_step,
                total_time=total_time,
                performance_metrics=self._get_performance_metrics(),
                agent_data=self._get_agent_data(),
                environment_data=self._get_environment_data(),
                success=False,
                error_message=str(e)
            )
            
        finally:
            self.is_running = False
    
    def _execute_step(self) -> Dict[str, Any]:
        """Execute a single simulation step"""
        step_start_time = time.time()
        
        # Get observations for all agents
        observations = self.environment.get_all_observations()
        
        # Process decisions (batch processing for efficiency)
        observation_list = list(observations.values())
        actions = self.engine.process_batch(observation_list)
        
        # Create action dictionary
        action_dict = {action.agent_id: action for action in actions}
        
        # Apply actions
        results = self.environment.apply_all_actions(action_dict)
        
        # Update environment
        self.environment.step()
        
        # Collect step metrics
        step_time = time.time() - step_start_time
        
        step_data = {
            'step': self.current_step,
            'step_time': step_time,
            'num_agents': len(self.agents),
            'successful_actions': sum(1 for r in results.values() if r.success),
            'failed_actions': sum(1 for r in results.values() if not r.success),
            'environment_metrics': self.environment.get_performance_metrics()
        }
        
        # Add engine metrics
        if hasattr(self.engine, 'get_performance_metrics'):
            step_data['engine_metrics'] = self.engine.get_performance_metrics()
        
        # Update performance monitoring
        if self.performance_monitor:
            self.performance_monitor.record_step(step_data)
        
        return step_data
    
    def _collect_agent_trajectories(self, step: int) -> None:
        """Collect agent trajectory data"""
        for agent in self.agents:
            if agent.agent_id not in self.agent_trajectories:
                self.agent_trajectories[agent.agent_id] = []
            
            trajectory_point = {
                'step': step,
                'position': agent.position,
                'metadata': agent.metadata.copy()
            }
            
            # Add paradigm-specific data
            if hasattr(agent, 'get_physics_properties'):
                trajectory_point.update(agent.get_physics_properties())
            elif hasattr(agent, 'get_network_properties'):
                trajectory_point.update(agent.get_network_properties())
            elif hasattr(agent, 'get_grid_properties'):
                trajectory_point.update(agent.get_grid_properties())
            
            self.agent_trajectories[agent.agent_id].append(trajectory_point)
    
    def _should_terminate(self) -> bool:
        """Check if simulation should terminate early"""
        # Add custom termination conditions here
        # For now, just continue until max_steps
        return False
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {}
        
        if self.performance_monitor:
            metrics['performance_monitor'] = self.performance_monitor.get_summary()
        
        if self.metrics_collector:
            metrics['metrics_collector'] = self.metrics_collector.get_all_metrics()
        
        if self.engine:
            metrics['engine'] = self.engine.get_performance_metrics()
        
        if self.environment:
            metrics['environment'] = self.environment.get_performance_metrics()
        
        return metrics
    
    def _get_agent_data(self) -> List[Dict[str, Any]]:
        """Get final agent data"""
        agent_data = []
        
        for agent in self.agents:
            data = {
                'agent_id': agent.agent_id,
                'final_position': agent.position,
                'metadata': agent.metadata,
                'trajectory': self.agent_trajectories.get(agent.agent_id, []),
                'performance_metrics': agent.get_performance_metrics()
            }
            
            agent_data.append(data)
        
        return agent_data
    
    def _get_environment_data(self) -> Dict[str, Any]:
        """Get environment data"""
        data = {
            'paradigm': self.config.paradigm,
            'final_step': self.current_step,
            'performance_metrics': self.environment.get_performance_metrics()
        }
        
        # Add paradigm-specific data
        if hasattr(self.environment, 'get_grid_statistics'):
            data['grid_statistics'] = self.environment.get_grid_statistics()
        elif hasattr(self.environment, 'get_physics_statistics'):
            data['physics_statistics'] = self.environment.get_physics_statistics()
        elif hasattr(self.environment, 'get_network_statistics'):
            data['network_statistics'] = self.environment.get_network_statistics()
        
        return data
    
    def export_results(self, result: SimulationResult, output_dir: str) -> None:
        """
        Export simulation results to files.
        
        Args:
            result: Simulation result to export
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(output_path / "config.json", 'w') as f:
            json.dump(result.config.dict(), f, indent=2)
        
        # Save summary
        summary = {
            'success': result.success,
            'steps_completed': result.steps_completed,
            'total_time': result.total_time,
            'error_message': result.error_message
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save performance metrics
        with open(output_path / "performance_metrics.json", 'w') as f:
            json.dump(result.performance_metrics, f, indent=2)
        
        # Save agent data
        with open(output_path / "agent_data.json", 'w') as f:
            json.dump(result.agent_data, f, indent=2)
        
        # Save environment data
        with open(output_path / "environment_data.json", 'w') as f:
            json.dump(result.environment_data, f, indent=2)
        
        # Save step data
        with open(output_path / "step_data.json", 'w') as f:
            json.dump(self.step_data, f, indent=2)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'current_step': self.current_step,
            'is_running': self.is_running,
            'num_agents': len(self.agents),
            'environment_state': self.environment.get_performance_metrics() if self.environment else None,
            'engine_state': self.engine.get_performance_metrics() if self.engine else None
        }
    
    def reset_simulation(self) -> None:
        """Reset simulation to initial state"""
        if self.config:
            self._initialize_simulation()
    
    def __repr__(self) -> str:
        if self.config:
            return (f"ResearchAPI(paradigm={self.config.paradigm}, "
                    f"engine={self.config.engine_type}, "
                    f"agents={self.config.num_agents})")
        else:
            return "ResearchAPI(not configured)"
