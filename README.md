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

## Basic Usage

### Beginner: Simple Grid Simulation

```python
from lamb import ResearchAPI

api = ResearchAPI()
api.create_simulation(
    paradigm="grid",
    num_agents=50,
    engine_type="rule",
    max_steps=500
)

results = api.run_simulation()
```

### Intermediate: LLM-Powered Agents

```python
from lamb import ResearchAPI, SimulationConfig

config = SimulationConfig(
    paradigm="grid",
    num_agents=20,
    engine_type="llm",
    llm_config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 150
    }
)

api = ResearchAPI()
api.create_simulation(config=config)
results = api.run_simulation()
```

### Advanced: Custom Model

```python
from lamb import ResearchAPI, GridAgent, GridEnvironment
from lamb.engines import RuleEngine

class CustomAgent(GridAgent):
    def decide(self, observation, engine):
        # Custom decision logic
        return Action(agent_id=self.agent_id, action_type="move")

# Build custom simulation
environment = GridEnvironment(dimensions=(100, 100))
agents = [CustomAgent(i, (i%10, i//10)) for i in range(100)]
engine = RuleEngine()

simulation = Simulation(environment, agents, engine)
results = simulation.run(max_steps=1000)
```

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
