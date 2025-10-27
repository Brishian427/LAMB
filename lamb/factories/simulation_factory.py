"""
Main factory for creating LAMB simulations.

This factory provides high-level methods for creating complete simulations
from AI-friendly configuration dictionaries. It handles component creation,
validation, and assembly automatically.
"""

from typing import Dict, Any, List, Optional
import random

from ..core.simulation import Simulation
from ..core.base_agent import BaseAgent
from ..core.base_environment import BaseEnvironment
from ..core.base_engine import BaseEngine
from ..executors import GridExecutor, PhysicsExecutor, NetworkExecutor
from ..config.simulation_config import SimulationConfig, ParadigmType, EngineType
from .environment_factory import EnvironmentFactory
from .agent_factory import AgentFactory
from .engine_factory import EngineFactory
from .executor_factory import ExecutorFactory


class SimulationFactory:
    """
    Factory for creating complete LAMB simulations.
    
    This factory provides methods for creating simulations from:
    1. AI-friendly configuration dictionaries
    2. Preset simulation types
    3. Component specifications
    """
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Simulation:
        """
        Create simulation from AI-friendly configuration dictionary.
        
        Args:
            config: Configuration dictionary with semantic keys
            
        Returns:
            Complete simulation ready to run
            
        Example:
            config = {
                "world_type": "grid_cooperation_study",
                "grid_size": 20,
                "num_agents": 100,
                "agent_personalities": ["cooperative", "selfish", "conditional"],
                "personality_distribution": [0.4, 0.3, 0.3],
                "execution_mode": "llm",
                "llm_model": "gpt-3.5-turbo",
                "simulation_duration": 1000
            }
        """
        # Extract configuration
        world_type = config.get("world_type", "grid_simple")
        num_agents = config.get("num_agents", 100)
        max_steps = config.get("simulation_duration", 1000)
        
        # Create components
        environment = EnvironmentFactory.create(world_type, config)
        agents = AgentFactory.create_batch(num_agents, config)
        
        # Add agents to environment
        for agent in agents:
            environment.add_agent(agent)
        
        engine = EngineFactory.create(config.get("execution_mode", "llm"), config)
        executor = ExecutorFactory.create(world_type, config)
        
        # Create simulation configuration
        sim_config = SimulationConfig(
            name=config.get("name", "LAMB Simulation"),
            paradigm=ParadigmType.GRID,  # Will be determined by world_type
            num_agents=num_agents,
            max_steps=max_steps
        )
        
        return Simulation(environment, agents, engine, executor, sim_config)
    
    @staticmethod
    def create_cooperation_study(
        num_agents: int = 100,
        grid_size: int = 20,
        cooperation_rate: float = 0.6,
        use_llm: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ) -> Simulation:
        """
        Create a cooperation study simulation.
        
        Args:
            num_agents: Number of agents in the simulation
            grid_size: Size of the grid world
            cooperation_rate: Proportion of cooperative agents
            use_llm: Whether to use LLM or rule-based engine
            llm_model: LLM model to use if use_llm is True
            
        Returns:
            Cooperation study simulation
        """
        # Create grid environment
        environment = EnvironmentFactory.create("grid_simple", {
            "grid_size": grid_size,
            "boundary_condition": "wrap"
        })
        
        # Create agents with different personalities
        agents = []
        for i in range(num_agents):
            if i < num_agents * cooperation_rate:
                personality = "cooperative"
            else:
                personality = "selfish"
            
            agent = AgentFactory.create_agent(
                agent_type="cooperation_agent",
                agent_id=i,
                personality=personality,
                position=(random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            )
            agents.append(agent)
            environment.add_agent(agent)
        
        # Create engine
        if use_llm:
            engine = EngineFactory.create("llm", {
                "model": llm_model,
                "temperature": 0.7,
                "prompt_template": "cooperation_study"
            })
        else:
            engine = EngineFactory.create("rule", {
                "rule_type": "cooperation_rules"
            })
        
        # Create executor
        executor = ExecutorFactory.create("grid", {})
        
        return Simulation(environment, agents, engine, executor)
    
    @staticmethod
    def create_flocking_simulation(
        num_agents: int = 50,
        world_size: float = 100.0,
        use_llm: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ) -> Simulation:
        """
        Create a flocking simulation (Boids-like).
        
        Args:
            num_agents: Number of agents in the simulation
            world_size: Size of the world (width and height)
            use_llm: Whether to use LLM or rule-based engine
            llm_model: LLM model to use if use_llm is True
            
        Returns:
            Flocking simulation
        """
        # Create physics environment
        environment = EnvironmentFactory.create("physics_flocking", {
            "world_bounds": ((-world_size/2, -world_size/2), (world_size/2, world_size/2)),
            "dt": 0.1,
            "enable_collisions": True
        })
        
        # Create flocking agents
        agents = []
        for i in range(num_agents):
            agent = AgentFactory.create_agent(
                agent_type="flocking_agent",
                agent_id=i,
                personality="flocker",
                position=(random.uniform(-world_size/2, world_size/2), 
                         random.uniform(-world_size/2, world_size/2))
            )
            agents.append(agent)
        
        # Create engine
        if use_llm:
            engine = EngineFactory.create("llm", {
                "model": llm_model,
                "temperature": 0.5,
                "prompt_template": "flocking_behavior"
            })
        else:
            engine = EngineFactory.create("rule", {
                "rule_type": "boids_rules"
            })
        
        # Create executor
        executor = ExecutorFactory.create("physics", {})
        
        return Simulation(environment, agents, engine, executor)
    
    @staticmethod
    def create_social_network_study(
        num_agents: int = 200,
        network_type: str = "random",
        use_llm: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ) -> Simulation:
        """
        Create a social network study simulation.
        
        Args:
            num_agents: Number of agents in the network
            network_type: Type of network ("random", "scale_free", "small_world")
            use_llm: Whether to use LLM or rule-based engine
            llm_model: LLM model to use if use_llm is True
            
        Returns:
            Social network simulation
        """
        # Create network environment
        environment = EnvironmentFactory.create("network_social", {
            "network_type": network_type,
            "num_nodes": num_agents,
            "is_directed": False
        })
        
        # Create social agents
        agents = []
        for i in range(num_agents):
            agent = AgentFactory.create_agent(
                agent_type="social_agent",
                agent_id=i,
                personality="social",
                position=i  # Node ID in network
            )
            agents.append(agent)
        
        # Create engine
        if use_llm:
            engine = EngineFactory.create("llm", {
                "model": llm_model,
                "temperature": 0.7,
                "prompt_template": "social_network"
            })
        else:
            engine = EngineFactory.create("rule", {
                "rule_type": "social_rules"
            })
        
        # Create executor
        executor = ExecutorFactory.create("network", {})
        
        return Simulation(environment, agents, engine, executor)
    
    @staticmethod
    def create_traffic_simulation(
        num_vehicles: int = 100,
        road_length: int = 1000,
        use_llm: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ) -> Simulation:
        """
        Create a traffic simulation.
        
        Args:
            num_vehicles: Number of vehicles in the simulation
            road_length: Length of the road
            use_llm: Whether to use LLM or rule-based engine
            llm_model: LLM model to use if use_llm is True
            
        Returns:
            Traffic simulation
        """
        # Create physics environment for traffic
        environment = EnvironmentFactory.create("physics_traffic", {
            "road_length": road_length,
            "num_lanes": 3,
            "speed_limit": 30.0,
            "dt": 0.1
        })
        
        # Create vehicle agents
        agents = []
        for i in range(num_vehicles):
            agent = AgentFactory.create_agent(
                agent_type="vehicle_agent",
                agent_id=i,
                personality="driver",
                position=(random.uniform(0, road_length), random.randint(0, 2))
            )
            agents.append(agent)
        
        # Create engine
        if use_llm:
            engine = EngineFactory.create("llm", {
                "model": llm_model,
                "temperature": 0.3,
                "prompt_template": "traffic_driving"
            })
        else:
            engine = EngineFactory.create("rule", {
                "rule_type": "traffic_rules"
            })
        
        # Create executor
        executor = ExecutorFactory.create("physics", {})
        
        return Simulation(environment, agents, engine, executor)
    
    @staticmethod
    def create_disease_spread_model(
        num_agents: int = 500,
        network_type: str = "scale_free",
        initial_infected: int = 10,
        use_llm: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ) -> Simulation:
        """
        Create a disease spread (SIR) model.
        
        Args:
            num_agents: Number of agents in the population
            network_type: Type of contact network
            initial_infected: Number of initially infected agents
            use_llm: Whether to use LLM or rule-based engine
            llm_model: LLM model to use if use_llm is True
            
        Returns:
            Disease spread simulation
        """
        # Create network environment
        environment = EnvironmentFactory.create("network_disease", {
            "network_type": network_type,
            "num_nodes": num_agents,
            "is_directed": False
        })
        
        # Create population agents
        agents = []
        for i in range(num_agents):
            # Determine initial health status
            if i < initial_infected:
                health_status = "infected"
            else:
                health_status = "susceptible"
            
            agent = AgentFactory.create_agent(
                agent_type="disease_agent",
                agent_id=i,
                personality="health_conscious",
                position=i,  # Node ID
                metadata={"health_status": health_status}
            )
            agents.append(agent)
        
        # Create engine
        if use_llm:
            engine = EngineFactory.create("llm", {
                "model": llm_model,
                "temperature": 0.4,
                "prompt_template": "disease_spread"
            })
        else:
            engine = EngineFactory.create("rule", {
                "rule_type": "sir_model"
            })
        
        # Create executor
        executor = ExecutorFactory.create("network", {})
        
        return Simulation(environment, agents, engine, executor)
