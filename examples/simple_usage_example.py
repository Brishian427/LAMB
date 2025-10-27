"""
Simple Usage Example: How to Use the LAMB Framework

This example demonstrates the typical workflow for researchers using LAMB.
It shows how easy it is to create and run agent-based simulations with LLM-driven agents.
"""

import asyncio
import os
from lamb.api import ResearchAPI
from lamb.config import SimulationConfig, ParadigmType, EngineType
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.paradigms.physics import PhysicsAgent, PhysicsEnvironment
from lamb.paradigms.network import NetworkAgent, NetworkEnvironment


async def example_1_simple_grid_simulation():
    """
    Example 1: Simple Grid-based Simulation
    Shows the most basic usage pattern.
    """
    print("=== Example 1: Simple Grid Simulation ===")
    
    # Step 1: Create a configuration
    config = SimulationConfig(
        name="Simple Grid Demo",
        paradigm=ParadigmType.GRID,
        engine_type=EngineType.LLM,
        num_agents=10,
        max_steps=50,
        grid_config={
            "dimensions": (20, 20),
            "boundary_condition": "wrap"
        },
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    # Step 2: Create the simulation
    api = ResearchAPI()
    environment, engine = api.create_simulation(
        config=config,
        agent_class=GridAgent,
        environment_class=GridEnvironment
    )
    
    # Step 3: Run the simulation
    print(f"Running simulation with {len(environment.get_all_agents())} agents...")
    metrics = await api.run_simulation(environment, engine)
    
    # Step 4: Analyze results
    api.analyze_results(metrics)
    print("Grid simulation completed!\n")


async def example_2_physics_flocking():
    """
    Example 2: Physics-based Flocking Simulation
    Shows how to create a Boids-like simulation.
    """
    print("=== Example 2: Physics Flocking Simulation ===")
    
    config = SimulationConfig(
        name="Boids Flocking",
        paradigm=ParadigmType.PHYSICS,
        engine_type=EngineType.LLM,
        num_agents=20,
        max_steps=100,
        physics_config={
            "world_bounds": ((-50, -50), (50, 50)),
            "dt": 0.1,
            "enable_collisions": True,
            "boundary_condition": "reflect"
        },
        llm_config={
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "max_tokens": 150
        }
    )
    
    api = ResearchAPI()
    environment, engine = api.create_simulation(
        config=config,
        agent_class=PhysicsAgent,
        environment_class=PhysicsEnvironment
    )
    
    print(f"Running flocking simulation with {len(environment.get_all_agents())} agents...")
    metrics = await api.run_simulation(environment, engine)
    api.analyze_results(metrics)
    print("Flocking simulation completed!\n")


async def example_3_network_disease_spread():
    """
    Example 3: Network-based Disease Spread
    Shows how to model disease transmission on a social network.
    """
    print("=== Example 3: Network Disease Spread ===")
    
    config = SimulationConfig(
        name="SIR Disease Model",
        paradigm=ParadigmType.NETWORK,
        engine_type=EngineType.LLM,
        num_agents=50,
        max_steps=200,
        network_config={
            "is_directed": False,
            "weighted": True,
            "enable_message_passing": True
        },
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 100
        }
    )
    
    api = ResearchAPI()
    environment, engine = api.create_simulation(
        config=config,
        agent_class=NetworkAgent,
        environment_class=NetworkEnvironment
    )
    
    print(f"Running disease spread simulation with {len(environment.get_all_agents())} agents...")
    metrics = await api.run_simulation(environment, engine)
    api.analyze_results(metrics)
    print("Disease spread simulation completed!\n")


def example_4_advanced_configuration():
    """
    Example 4: Advanced Configuration
    Shows how to customize the framework for specific research needs.
    """
    print("=== Example 4: Advanced Configuration ===")
    
    # Custom configuration with performance tuning
    config = SimulationConfig(
        name="High-Performance Simulation",
        paradigm=ParadigmType.PHYSICS,
        engine_type=EngineType.LLM,
        num_agents=1000,  # Large population
        max_steps=500,
        
        # Physics settings
        physics_config={
            "world_bounds": ((-100, -100), (100, 100)),
            "dt": 0.05,  # Smaller time step for accuracy
            "enable_collisions": True,
            "collision_damping": 0.8
        },
        
        # LLM optimization
        llm_config={
            "model": "gpt-4o-mini",
            "batch_size": 20,  # Larger batches for efficiency
            "temperature": 0.7,
            "max_tokens": 200,
            "circuit_breaker_threshold": 0.2,  # Stricter failure threshold
            "cache_size": 2000,  # Larger cache
            "cache_ttl_seconds": 600  # Longer cache lifetime
        },
        
        # Performance monitoring
        performance_config={
            "enable_monitoring": True,
            "target_agent_throughput": 15.0,  # Target 15 agents/second
            "max_decision_time": 0.3,  # Max 300ms per decision
            "memory_tracking": True
        },
        
        # Spatial optimization
        spatial_config={
            "auto_select": True,
            "rebuild_threshold": 50,  # Rebuild spatial index every 50 moves
            "rebuild_interval": 0.5  # Or every 0.5 seconds
        }
    )
    
    print("Advanced configuration created!")
    print(f"Target: {config.num_agents} agents, {config.max_steps} steps")
    print(f"Performance target: {config.performance_config.target_agent_throughput} agents/sec")
    print("Advanced configuration example completed!\n")


def example_5_research_workflow():
    """
    Example 5: Typical Research Workflow
    Shows how a researcher would use LAMB for a study.
    """
    print("=== Example 5: Research Workflow ===")
    
    # Step 1: Define research question
    print("Research Question: How does agent density affect cooperation in a grid world?")
    
    # Step 2: Design experiment
    densities = [0.1, 0.2, 0.3, 0.4, 0.5]  # Agent density levels
    results = {}
    
    for density in densities:
        print(f"\nTesting density: {density}")
        
        # Calculate number of agents for this density
        grid_size = 20 * 20  # 20x20 grid
        num_agents = int(grid_size * density)
        
        # Create configuration
        config = SimulationConfig(
            name=f"Cooperation Study - Density {density}",
            paradigm=ParadigmType.GRID,
            engine_type=EngineType.LLM,
            num_agents=num_agents,
            max_steps=100,
            grid_config={
                "dimensions": (20, 20),
                "boundary_condition": "wrap"
            },
            llm_config={
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        )
        
        # Run simulation
        api = ResearchAPI()
        environment, engine = api.create_simulation(
            config=config,
            agent_class=GridAgent,
            environment_class=GridEnvironment
        )
        
        # Store results
        results[density] = {
            "num_agents": num_agents,
            "config": config
        }
    
    print(f"\nExperiment designed: {len(densities)} conditions")
    print("Each condition tests different agent density levels")
    print("Results would be collected and analyzed for cooperation patterns")
    print("Research workflow example completed!\n")


async def main():
    """
    Main function demonstrating all usage patterns.
    """
    print("LAMB Framework Usage Examples")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Set your API key to run actual LLM simulations")
        print("   Examples will show configuration only\n")
    
    # Run examples
    await example_1_simple_grid_simulation()
    await example_2_physics_flocking()
    await example_3_network_disease_spread()
    example_4_advanced_configuration()
    example_5_research_workflow()
    
    print("All examples completed!")
    print("\nKey Benefits of LAMB:")
    print("✅ Simple API - Just 3 lines to create and run simulations")
    print("✅ LLM Integration - Agents make intelligent decisions")
    print("✅ Multiple Paradigms - Grid, Physics, Network support")
    print("✅ Performance Optimized - Built-in caching, batching, monitoring")
    print("✅ Research Ready - Easy to run experiments and collect data")


if __name__ == "__main__":
    asyncio.run(main())
