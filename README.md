# LAMB: LLM Agent Model Base

[![PyPI version](https://img.shields.io/pypi/v/lamb-abm.svg?style=flat&color=blue)](https://pypi.org/project/lamb-abm/)
[![Python versions](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/Brishian427/LAMB/workflows/Tests/badge.svg)](https://github.com/Brishian427/LAMB/actions)
[![Development Status](https://img.shields.io/badge/status-development-orange.svg)](https://github.com/Brishian427/LAMB)

A unified framework for building agent-based models with Large Language Model integration, supporting multiple simulation paradigms and behavioral engines.

**Note**: This is a work-in-progress project by the OASIS-Fudan Complex System AI Social Scientist Team. Documentation and features are being actively developed.

## Installation

```bash
pip install lamb-abm
```

### Optional Dependencies

For LLM integration:
```bash
pip install lamb-abm[llm]
```

For full functionality including visualization:
```bash
pip install lamb-abm[all]
```

## Quick Start

```python
from lamb import ResearchAPI, SimulationConfig

# Create a simple grid simulation
api = ResearchAPI()
api.create_simulation(
    paradigm="grid",
    num_agents=100,
    engine_type="rule",
    max_steps=1000
)

# Run simulation
results = api.run_simulation()
print(f"Simulation completed with {len(results)} steps")
```

### 5-Minute Example: Sugarscape Model

```python
from lamb import ResearchAPI, SimulationConfig
import matplotlib.pyplot as plt

# Create Sugarscape simulation
api = ResearchAPI()
config = SimulationConfig(
    paradigm="grid",
    dimensions=(50, 50),
    num_agents=200,
    engine_type="rule",
    max_steps=500,
    model_name="sugarscape"
)

api.create_simulation(config=config)
results = api.run_simulation()

# Visualize results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(results[0]['environment'], cmap='YlOrRd')
plt.title('Initial Sugar Distribution')

plt.subplot(1, 3, 2)
plt.imshow(results[-1]['environment'], cmap='YlOrRd')
plt.title('Final Sugar Distribution')

plt.subplot(1, 3, 3)
agent_counts = [step['agent_count'] for step in results]
plt.plot(agent_counts)
plt.title('Agent Population Over Time')
plt.xlabel('Step')
plt.ylabel('Number of Agents')

plt.tight_layout()
plt.show()
```

## Key Features

- **Multi-Paradigm Support**: Grid, Physics, and Network simulation paradigms
- **LLM Integration**: Seamless Large Language Model agent behavior
- **Composition Architecture**: Modular design for easy extension
- **High Performance**: Optimized for 10,000+ agents
- **Research Ready**: Built-in metrics, visualization, and analysis tools
- **Academic Focus**: Designed for social science and complexity research

## Core Concepts

### Paradigms
- **Grid**: Discrete space models (Sugarscape, Schelling)
- **Physics**: Continuous space models (Boids, Social Force)
- **Network**: Graph-based models (SIR, Opinion Dynamics)

### Engines
- **Rule Engine**: Traditional rule-based behavior
- **LLM Engine**: Large Language Model decision making
- **Hybrid Engine**: Combines multiple approaches

### Composition Pattern
LAMB uses a composition-based architecture where simulations are built by combining independent components (Environment, Agents, Engine, Executor) rather than inheritance hierarchies.

## Detailed Usage Examples

### 1. Beginner: Schelling Segregation Model

```python
from lamb import ResearchAPI, SimulationConfig
import matplotlib.pyplot as plt
import numpy as np

# Create Schelling segregation simulation
api = ResearchAPI()
config = SimulationConfig(
    paradigm="grid",
    dimensions=(30, 30),
    num_agents=400,
    engine_type="rule",
    max_steps=200,
    model_name="schelling",
    model_params={
        "similarity_threshold": 0.3,  # Agents want 30% similar neighbors
        "agent_types": 2,  # Two types of agents
        "occupancy_rate": 0.4  # 40% of cells occupied
    }
)

api.create_simulation(config=config)
results = api.run_simulation()

# Calculate segregation index over time
segregation_indices = []
for step in results:
    if 'segregation_index' in step:
        segregation_indices.append(step['segregation_index'])

# Visualize results
plt.figure(figsize=(15, 5))

# Initial state
plt.subplot(1, 3, 1)
env_initial = results[0]['environment']
plt.imshow(env_initial, cmap='tab10')
plt.title('Initial State')
plt.colorbar()

# Final state
plt.subplot(1, 3, 2)
env_final = results[-1]['environment']
plt.imshow(env_final, cmap='tab10')
plt.title('Final State (Segregated)')
plt.colorbar()

# Segregation over time
plt.subplot(1, 3, 3)
plt.plot(segregation_indices)
plt.title('Segregation Index Over Time')
plt.xlabel('Step')
plt.ylabel('Segregation Index')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final segregation index: {segregation_indices[-1]:.3f}")
```

### 2. Intermediate: SIR Epidemic Model

```python
from lamb import ResearchAPI, SimulationConfig
import matplotlib.pyplot as plt

# Create SIR epidemic simulation
api = ResearchAPI()
config = SimulationConfig(
    paradigm="network",
    num_agents=1000,
    engine_type="rule",
    max_steps=100,
    model_name="sir",
    model_params={
        "infection_rate": 0.3,
        "recovery_rate": 0.1,
        "initial_infected": 10,
        "network_type": "random",
        "network_params": {"p": 0.01}  # Random network with 1% connection probability
    }
)

api.create_simulation(config=config)
results = api.run_simulation()

# Extract SIR counts over time
s_counts = [step['s_count'] for step in results]
i_counts = [step['i_count'] for step in results]
r_counts = [step['r_count'] for step in results]

# Plot epidemic curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(s_counts, label='Susceptible', color='blue')
plt.plot(i_counts, label='Infected', color='red')
plt.plot(r_counts, label='Recovered', color='green')
plt.title('SIR Epidemic Model')
plt.xlabel('Time Steps')
plt.ylabel('Number of Agents')
plt.legend()
plt.grid(True)

# Network visualization (final state)
plt.subplot(1, 2, 2)
# This would show the network structure
plt.title('Network Structure')
plt.text(0.5, 0.5, f'Network with {len(results[-1].get("network_nodes", []))} nodes', 
         ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

print(f"Peak infected: {max(i_counts)} agents")
print(f"Final recovered: {r_counts[-1]} agents")
```

### 3. Advanced: Physics-Based Boids Simulation

```python
from lamb import ResearchAPI, SimulationConfig
import matplotlib.pyplot as plt
import numpy as np

# Create Boids flocking simulation
api = ResearchAPI()
config = SimulationConfig(
    paradigm="physics",
    num_agents=100,
    engine_type="rule",
    max_steps=300,
    model_name="boids",
    model_params={
        "world_bounds": [(-50, -50), (50, 50)],
        "separation_weight": 1.5,
        "alignment_weight": 1.0,
        "cohesion_weight": 1.0,
        "max_speed": 2.0,
        "perception_radius": 10.0
    }
)

api.create_simulation(config=config)
results = api.run_simulation()

# Extract agent positions over time
positions_history = []
for step in results:
    if 'agent_positions' in step:
        positions_history.append(step['agent_positions'])

# Create animation frames
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, frame_idx in enumerate([0, 50, 100, 150, 200, 299]):
    if frame_idx < len(positions_history):
        positions = positions_history[frame_idx]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        axes[i].scatter(x_coords, y_coords, alpha=0.6, s=20)
        axes[i].set_xlim(-50, 50)
        axes[i].set_ylim(-50, 50)
        axes[i].set_title(f'Step {frame_idx}')
        axes[i].grid(True)

plt.suptitle('Boids Flocking Simulation Over Time')
plt.tight_layout()
plt.show()

# Calculate flocking metrics
def calculate_cohesion(positions):
    if len(positions) < 2:
        return 0
    center = np.mean(positions, axis=0)
    distances = [np.linalg.norm(pos - center) for pos in positions]
    return np.mean(distances)

cohesion_values = [calculate_cohesion(positions) for positions in positions_history]
plt.figure(figsize=(10, 4))
plt.plot(cohesion_values)
plt.title('Flock Cohesion Over Time')
plt.xlabel('Step')
plt.ylabel('Average Distance to Center')
plt.grid(True)
plt.show()
```

### 4. Expert: Custom Agent with LLM Integration

```python
from lamb import ResearchAPI, SimulationConfig, GridAgent, Action
from lamb.engines import LLMEngine
import openai

# Custom agent with LLM decision making
class LLMAgent(GridAgent):
    def __init__(self, agent_id, position, agent_type="trader"):
        super().__init__(agent_id, position)
        self.agent_type = agent_type
        self.wealth = 100.0
        self.satisfaction = 0.5
        
    def decide(self, observation, engine):
        # Get context about the environment
        context = {
            "agent_type": self.agent_type,
            "wealth": self.wealth,
            "satisfaction": self.satisfaction,
            "position": self.position,
            "neighbors": observation.get("neighbors", [])
        }
        
        # Use LLM to decide action
        prompt = f"""
        You are a {self.agent_type} agent in a simulation.
        Current context: {context}
        
        Available actions: move, trade, rest
        Choose the most appropriate action and explain why.
        """
        
        response = engine.generate_response(prompt)
        
        # Parse LLM response and create action
        if "move" in response.lower():
            return Action(agent_id=self.agent_id, action_type="move", 
                         target_position=self._choose_move_target(observation))
        elif "trade" in response.lower():
            return Action(agent_id=self.agent_id, action_type="trade")
        else:
            return Action(agent_id=self.agent_id, action_type="rest")
    
    def _choose_move_target(self, observation):
        # Simple movement logic
        current_pos = self.position
        possible_moves = [
            (current_pos[0] + 1, current_pos[1]),
            (current_pos[0] - 1, current_pos[1]),
            (current_pos[0], current_pos[1] + 1),
            (current_pos[0], current_pos[1] - 1)
        ]
        return possible_moves[0]  # Move right

# Create LLM-powered simulation
api = ResearchAPI()
config = SimulationConfig(
    paradigm="grid",
    dimensions=(20, 20),
    num_agents=50,
    engine_type="llm",
    max_steps=100,
    llm_config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100
    }
)

# Set up OpenAI API key (you'll need to set this)
# openai.api_key = "your-api-key-here"

api.create_simulation(config=config)
results = api.run_simulation()

print("LLM-powered simulation completed!")
print(f"Total steps: {len(results)}")
```

### 5. Research: Batch Processing and Analysis

```python
from lamb import ResearchAPI, SimulationConfig
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def run_single_simulation(params):
    """Run a single simulation with given parameters"""
    api = ResearchAPI()
    config = SimulationConfig(
        paradigm="grid",
        dimensions=(30, 30),
        num_agents=200,
        engine_type="rule",
        max_steps=300,
        model_name="schelling",
        model_params=params
    )
    
    api.create_simulation(config=config)
    results = api.run_simulation()
    
    # Extract final metrics
    final_step = results[-1]
    return {
        'similarity_threshold': params['similarity_threshold'],
        'final_segregation': final_step.get('segregation_index', 0),
        'final_agents': final_step.get('agent_count', 0),
        'convergence_step': len(results)
    }

# Parameter sweep
similarity_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
param_sets = [
    {"similarity_threshold": threshold, "agent_types": 2, "occupancy_rate": 0.4}
    for threshold in similarity_thresholds
]

# Run simulations in parallel
print("Running parameter sweep...")
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_single_simulation, param_sets))

# Convert to DataFrame for analysis
df = pd.DataFrame(results)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df['similarity_threshold'], df['final_segregation'], 'o-')
plt.xlabel('Similarity Threshold')
plt.ylabel('Final Segregation Index')
plt.title('Segregation vs Similarity Threshold')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df['similarity_threshold'], df['convergence_step'], 'o-', color='red')
plt.xlabel('Similarity Threshold')
plt.ylabel('Convergence Time (Steps)')
plt.title('Convergence Time vs Similarity Threshold')
plt.grid(True)

plt.tight_layout()
plt.show()

# Statistical analysis
print("\nParameter Sweep Results:")
print(df.describe())
print(f"\nCorrelation between threshold and segregation: {df['similarity_threshold'].corr(df['final_segregation']):.3f}")
```

## Real-World Applications

### Social Science Research
- **Urban Segregation**: Study how housing preferences lead to neighborhood segregation
- **Opinion Dynamics**: Model how ideas spread through social networks
- **Economic Behavior**: Simulate market dynamics and trading patterns
- **Crowd Behavior**: Analyze pedestrian movement and crowd formation

### Epidemiology and Public Health
- **Disease Spread**: Model how infections propagate through populations
- **Vaccination Strategies**: Test different immunization approaches
- **Social Distancing**: Study the effects of behavioral interventions
- **Contact Tracing**: Simulate network-based disease tracking

### Economics and Finance
- **Market Microstructure**: Model trading behavior and price formation
- **Financial Contagion**: Study how shocks spread through financial networks
- **Behavioral Economics**: Simulate decision-making under uncertainty
- **Cryptocurrency Dynamics**: Model blockchain-based economic systems

### Environmental Science
- **Ecosystem Dynamics**: Study predator-prey relationships and food webs
- **Climate Adaptation**: Model how populations respond to environmental changes
- **Resource Management**: Simulate sustainable resource extraction
- **Migration Patterns**: Study animal and human migration behaviors

### Technology and AI
- **Multi-Agent Systems**: Design intelligent agent coordination
- **Robotics**: Simulate swarm robotics and collective behavior
- **Social Media**: Model information spread and echo chambers
- **Autonomous Vehicles**: Study traffic flow and coordination

## Performance Benchmarks

LAMB is optimized for high-performance simulations:

```python
# Performance test
import time
from lamb import ResearchAPI, SimulationConfig

def benchmark_simulation(num_agents, max_steps):
    api = ResearchAPI()
    config = SimulationConfig(
        paradigm="grid",
        dimensions=(100, 100),
        num_agents=num_agents,
        engine_type="rule",
        max_steps=max_steps
    )
    
    start_time = time.time()
    api.create_simulation(config=config)
    results = api.run_simulation()
    end_time = time.time()
    
    return {
        'agents': num_agents,
        'steps': max_steps,
        'time': end_time - start_time,
        'steps_per_second': max_steps / (end_time - start_time)
    }

# Run benchmarks
benchmarks = [
    benchmark_simulation(1000, 1000),
    benchmark_simulation(5000, 1000),
    benchmark_simulation(10000, 1000),
    benchmark_simulation(1000, 5000),
]

for b in benchmarks:
    print(f"{b['agents']} agents, {b['steps']} steps: {b['time']:.2f}s ({b['steps_per_second']:.0f} steps/sec)")
```

**Typical Performance:**
- **1,000 agents**: ~10,000 steps/second
- **10,000 agents**: ~1,000 steps/second
- **Memory usage**: ~1MB per 1,000 agents
- **Scalability**: Tested up to 100,000 agents

## Documentation

Documentation is currently being developed. For now, please refer to the examples in the `examples/` directory and the inline code documentation.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OASIS-Fudan Complex System AI Social Scientist Team
- The agent-based modeling community
- OpenAI for LLM API access
- Contributors and users

## Support

- Issues: [GitHub Issues](https://github.com/Brishian427/LAMB/issues)
- Discussions: [GitHub Discussions](https://github.com/Brishian427/LAMB/discussions)
