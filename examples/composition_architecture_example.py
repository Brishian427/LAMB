"""
Composition Architecture Example

This example demonstrates the new composition-based Simulation architecture.
It shows how to create simulations by combining independent components rather
than using inheritance. This approach is more flexible and AI-friendly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lamb.core import Simulation
from lamb.factories import SimulationFactory
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.engines import LLMEngine
from lamb.executors import GridExecutor
from lamb.llm import PromptManager, create_research_personality


def example_1_direct_composition():
    """
    Example 1: Direct component composition
    Shows how to create a simulation by directly composing components.
    """
    print("=== Example 1: Direct Component Composition ===")
    
    # Create components directly
    environment = GridEnvironment(dimensions=(10, 10))
    
    # Create agents with different personalities
    agents = []
    for i in range(10):
        personality = "cooperative" if i < 5 else "selfish"
        agent = GridAgent(
            agent_id=i,
            position=(i % 10, i // 10),
            metadata={"personality": personality}
        )
        agents.append(agent)
        environment.add_agent(agent)
    
    # Create engine (use mock if no API key)
    if os.getenv("OPENAI_API_KEY"):
        prompt_manager = PromptManager()
        engine = LLMEngine(prompt_manager=prompt_manager)
    else:
        from lamb.engines import MockEngine
        engine = MockEngine()
    
    # Create executor
    executor = GridExecutor()
    
    # Compose simulation
    simulation = Simulation(
        environment=environment,
        agents=agents,
        engine=engine,
        executor=executor
    )
    
    print(f"Created simulation: {simulation}")
    print(f"Number of agents: {len(simulation.agents)}")
    print(f"Environment type: {type(simulation.environment).__name__}")
    print(f"Engine type: {type(simulation.engine).__name__}")
    print(f"Executor type: {type(simulation.executor).__name__}")
    print()


def example_2_factory_methods():
    """
    Example 2: Using factory methods
    Shows how to create simulations using high-level factory methods.
    """
    print("=== Example 2: Factory Methods ===")
    
    # Create cooperation study using factory
    simulation = SimulationFactory.create_cooperation_study(
        num_agents=20,
        grid_size=10,
        cooperation_rate=0.7,
        use_llm=False  # Use rule-based engine for demo
    )
    
    print(f"Created cooperation study: {simulation}")
    print(f"Number of agents: {len(simulation.agents)}")
    print(f"Grid size: {simulation.environment.dimensions}")
    print()

def example_3_ai_friendly_config():
    """
    Example 3: AI-friendly configuration
    Shows how to create simulations from semantic configuration dictionaries.
    """
    print("=== Example 3: AI-Friendly Configuration ===")
    
    # AI-generated configuration
    config = {
        "world_type": "grid_cooperation_study",
        "grid_size": 15,
        "num_agents": 30,
        "agent_personalities": ["cooperative", "selfish", "conditional"],
        "personality_distribution": [0.5, 0.3, 0.2],
        "execution_mode": "llm",
        "llm_model": "gpt-3.5-turbo",
        "simulation_duration": 100
    }
    
    # Create simulation from config
    simulation = SimulationFactory.create_from_config(config)
    
    print(f"Created simulation from AI config: {simulation}")
    print(f"Configuration: {config}")
    print()


def example_4_engine_swapping():
    """
    Example 4: Engine swapping
    Shows how easy it is to swap engines with the composition architecture.
    """
    print("=== Example 4: Engine Swapping ===")
    
    # Create initial simulation
    simulation = SimulationFactory.create_cooperation_study(
        num_agents=10,
        grid_size=8,
        use_llm=True
    )
    
    print(f"Initial simulation: {simulation}")
    print(f"Engine type: {type(simulation.engine).__name__}")
    
    # Swap to rule-based engine (one line!)
    from lamb.engines import MockEngine
    simulation.swap_engine(MockEngine())
    
    print(f"After engine swap: {simulation}")
    print(f"New engine type: {type(simulation.engine).__name__}")
    print()


def example_5_validation_system():
    """
    Example 5: Component validation
    Shows how the validation system catches incompatible components.
    """
    print("=== Example 5: Component Validation ===")
    
    try:
        # Try to create incompatible simulation
        from lamb.paradigms.physics import PhysicsEnvironment
        from lamb.paradigms.grid import GridAgent
        from lamb.engines import MockEngine
        
        # Grid agent in physics environment (should fail)
        environment = PhysicsEnvironment(world_bounds=((-10, -10), (10, 10)))
        agents = [GridAgent(agent_id=0, position=(0, 0))]  # Grid agent
        engine = MockEngine()
        executor = GridExecutor()  # Grid executor
        
        simulation = Simulation(environment, agents, engine, executor)
        print("ERROR: Validation should have caught this!")
        
    except ValueError as e:
        print(f"Validation caught error: {e}")
        print("This is expected - components must be compatible!")
    print()


def example_6_different_paradigms():
    """
    Example 6: Different paradigms
    Shows how the same architecture works for different paradigms.
    """
    print("=== Example 6: Different Paradigms ===")
    
    # Grid simulation
    grid_sim = SimulationFactory.create_cooperation_study(num_agents=10)
    print(f"Grid simulation: {grid_sim}")
    print(f"Executor paradigm: {grid_sim.executor.get_paradigm()}")
    
    # Physics simulation
    physics_sim = SimulationFactory.create_flocking_simulation(num_agents=10)
    print(f"Physics simulation: {physics_sim}")
    print(f"Executor paradigm: {physics_sim.executor.get_paradigm()}")
    
    # Network simulation
    network_sim = SimulationFactory.create_social_network_study(num_agents=20)
    print(f"Network simulation: {network_sim}")
    print(f"Executor paradigm: {network_sim.executor.get_paradigm()}")
    print()


def example_7_simulation_state():
    """
    Example 7: Simulation state management
    Shows how to manage simulation state and results.
    """
    print("=== Example 7: Simulation State Management ===")
    
    # Create simulation
    simulation = SimulationFactory.create_cooperation_study(
        num_agents=5,
        grid_size=5,
        use_llm=False  # Use mock engine for demo
    )
    
    # Check initial state
    state = simulation.get_state()
    print(f"Initial state: {state}")
    
    # Add an agent
    from lamb.paradigms.grid import GridAgent
    new_agent = GridAgent(agent_id=999, position=(2, 2))
    simulation.add_agent(new_agent)
    
    # Check state after adding agent
    state = simulation.get_state()
    print(f"State after adding agent: {state}")
    
    # Remove an agent (use the first agent's ID)
    if simulation.agents:
        simulation.remove_agent(simulation.agents[0].agent_id)
    
    # Check state after removing agent
    state = simulation.get_state()
    print(f"State after removing agent: {state}")
    print()


def main():
    """
    Main function demonstrating composition architecture.
    """
    print("LAMB Framework: Composition Architecture Examples")
    print("=" * 60)
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OpenAI API key found!")
        print("   Set OPENAI_API_KEY environment variable to run with real LLM")
        print("   Examples will use mock engines for demonstration")
        print()
    
    # Run examples
    example_1_direct_composition()
    example_2_factory_methods()
    example_3_ai_friendly_config()
    example_4_engine_swapping()
    example_5_validation_system()
    example_6_different_paradigms()
    example_7_simulation_state()
    
    print("🎉 All composition architecture examples completed!")
    print()
    print("Key Benefits of Composition Architecture:")
    print("✅ No inheritance required")
    print("✅ Easy engine swapping")
    print("✅ Component reusability")
    print("✅ AI-friendly configuration")
    print("✅ Paradigm-agnostic design")
    print("✅ Clear separation of concerns")
    print()
    print("This architecture makes LAMB more flexible and easier to use!")


if __name__ == "__main__":
    main()
