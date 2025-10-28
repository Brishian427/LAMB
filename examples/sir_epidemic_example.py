#!/usr/bin/env python3
"""
LAMB Framework: SIR Rumor Spread Model Example

This example implements the SIR (Susceptible-Infected-Recovered) model for rumor spread
using the LAMB framework's Network paradigm. The model simulates how rumors spread
through a social network over time.

Key Features:
- Network-based rumor transmission
- Realistic rumor spread dynamics
- Visualization of rumor propagation
- Parameter sensitivity analysis
- Multiple network topologies

The SIR model for rumors demonstrates:
- How network structure affects rumor spread
- The role of "super-spreaders" in information diffusion
- Rumor extinction vs persistence
- Social influence patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lamb.config import SimulationConfig, ParadigmType, EngineType
from lamb.api import ResearchAPI
from lamb.paradigms.network import NetworkAgent, NetworkEnvironment
from lamb.core import Action, Observation, AgentID, Position
from lamb.engines import RuleEngine


class SIRState(Enum):
    """SIR rumor spread model states"""
    SUSCEPTIBLE = "ignorant"      # Haven't heard the rumor
    INFECTED = "spreader"         # Actively spreading the rumor
    RECOVERED = "stifler"         # Heard it but stopped spreading


@dataclass
class SIRParameters:
    """SIR rumor spread model parameters"""
    spread_rate: float = 0.3         # Probability of rumor spread per contact
    stifle_rate: float = 0.1         # Probability of becoming stifler per time step
    initial_spreaders: int = 5       # Number of initially spreading agents
    network_type: str = "random"     # Network topology type
    network_params: Dict = None      # Network-specific parameters
    
    def __post_init__(self):
        if self.network_params is None:
            self.network_params = {"p": 0.01}  # Random network connection probability


class SIRAgent(NetworkAgent):
    """Agent for SIR rumor spread model"""
    
    def __init__(self, agent_id: AgentID, position: Position, 
                 state: SIRState = SIRState.SUSCEPTIBLE, **kwargs):
        super().__init__(agent_id, position, **kwargs)
        self.state = state
        self.spread_time = 0     # Time when agent started spreading
        self.stifle_time = None  # Time when agent became stifler
        
        # Add state to metadata
        self.metadata["sir_state"] = state.value
        self.metadata["spread_time"] = self.spread_time
        self.metadata["stifle_time"] = self.stifle_time
    
    def observe(self, environment: NetworkEnvironment) -> Observation:
        """Observe neighbors and their states"""
        neighbors = environment.get_neighbors(self.agent_id, radius=1.0)
        
        # Count neighbors by state
        neighbor_states = {"ignorant": 0, "spreader": 0, "stifler": 0}
        for neighbor_id in neighbors:
            if neighbor_id in environment.agents:
                neighbor = environment.agents[neighbor_id]
                state = neighbor.metadata.get("sir_state", "ignorant")
                neighbor_states[state] += 1
        
        return Observation(
            agent_id=self.agent_id,
            position=self.position,
            neighbors=neighbors,
            paradigm="network",
            environment_state={
                "my_state": self.state.value,
                "neighbor_states": neighbor_states,
                "total_neighbors": len(neighbors)
            }
        )
    
    def decide(self, observation: Observation, engine) -> Action:
        """Decide action based on current state and neighbors"""
        if self.state == SIRState.SUSCEPTIBLE:
            return self._decide_susceptible(observation)
        elif self.state == SIRState.INFECTED:
            return self._decide_infected(observation)
        else:  # RECOVERED
            return self._decide_recovered(observation)
    
    def _decide_susceptible(self, observation: Observation) -> Action:
        """Ignorant agents can become spreaders when they hear the rumor"""
        spreader_neighbors = observation.environment_state["neighbor_states"]["spreader"]
        
        if spreader_neighbors > 0:
            # Probability of rumor spread increases with spreading neighbors
            spread_prob = 1 - (1 - 0.3) ** spreader_neighbors  # 30% base rate
            if np.random.random() < spread_prob:
                return Action(
                    agent_id=self.agent_id,
                    action_type="become_spreader",
                    parameters={"spread_time": self.environment.step_count}
                )
        
        return Action(agent_id=self.agent_id, action_type="wait", parameters={})
    
    def _decide_infected(self, observation: Observation) -> Action:
        """Spreader agents can become stiflers (lose interest)"""
        # Stifle probability (10% per time step)
        if np.random.random() < 0.1:
            return Action(
                agent_id=self.agent_id,
                action_type="become_stifler",
                parameters={"stifle_time": self.environment.step_count}
            )
        
        return Action(agent_id=self.agent_id, action_type="wait", parameters={})
    
    def _decide_recovered(self, observation: Observation) -> Action:
        """Stifler agents don't spread rumors anymore"""
        return Action(agent_id=self.agent_id, action_type="wait", parameters={})
    
    def execute_action(self, action: Action, environment: NetworkEnvironment) -> bool:
        """Execute the decided action"""
        if action.action_type == "become_spreader":
            self.state = SIRState.INFECTED
            self.spread_time = action.parameters.get("spread_time", 0)
            self.metadata["sir_state"] = self.state.value
            self.metadata["spread_time"] = self.spread_time
            return True
            
        elif action.action_type == "become_stifler":
            self.state = SIRState.RECOVERED
            self.stifle_time = action.parameters.get("stifle_time", 0)
            self.metadata["sir_state"] = self.state.value
            self.metadata["stifle_time"] = self.stifle_time
            return True
        
        return True  # "wait" action always succeeds


class SIREnvironment(NetworkEnvironment):
    """Environment for SIR rumor spread simulation"""
    
    def __init__(self, num_agents: int = 1000, sir_params: SIRParameters = None, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.sir_params = sir_params or SIRParameters()
        self.step_count = 0
        self.agents = {}  # Initialize agents dictionary
        
        # Initialize network
        self._create_network()
        
        # Initialize agents
        self._create_agents()
        
        # Statistics tracking
        self.sir_counts_history = []
        self.rumor_metrics = {}
    
    def _create_network(self):
        """Create network topology based on parameters"""
        if self.sir_params.network_type == "random":
            # Random network (Erdős–Rényi)
            p = self.sir_params.network_params.get("p", 0.01)
            self.network = nx.erdos_renyi_graph(self.num_agents, p)
            
        elif self.sir_params.network_type == "small_world":
            # Small-world network (Watts-Strogatz)
            k = self.sir_params.network_params.get("k", 4)
            p = self.sir_params.network_params.get("p", 0.1)
            self.network = nx.watts_strogatz_graph(self.num_agents, k, p)
            
        elif self.sir_params.network_type == "scale_free":
            # Scale-free network (Barabási-Albert)
            m = self.sir_params.network_params.get("m", 2)
            self.network = nx.barabasi_albert_graph(self.num_agents, m)
            
        else:
            # Default to random
            self.network = nx.erdos_renyi_graph(self.num_agents, 0.01)
        
        # Add edges to spatial index
        for edge in self.network.edges():
            self.create_edge(edge[0], edge[1])
    
    def _create_agents(self):
        """Create agents with initial SIR states"""
        # Create all agents as susceptible
        for i in range(self.num_agents):
            agent = SIRAgent(i, (0, 0))  # Position doesn't matter for network
            self.add_agent(agent)
        
        # Randomly select initial spreaders
        spreader_indices = np.random.choice(
            len(self.agents), 
            size=min(self.sir_params.initial_spreaders, len(self.agents)), 
            replace=False
        )
        
        for idx in spreader_indices:
            agent_id = list(self.agents.keys())[idx]
            agent = self.agents[agent_id]
            agent.state = SIRState.INFECTED
            agent.metadata["sir_state"] = SIRState.INFECTED.value
            agent.metadata["spread_time"] = 0
    
    def update_state(self):
        """Update environment state and collect statistics"""
        self.step_count += 1
        
        # Count agents by state
        sir_counts = {"ignorant": 0, "spreader": 0, "stifler": 0}
        for agent in self.agents.values():
            state = agent.metadata.get("sir_state", "ignorant")
            sir_counts[state] += 1
        
        # Store history
        self.sir_counts_history.append(sir_counts.copy())
        
        # Calculate rumor metrics
        self._calculate_rumor_metrics()
    
    def _calculate_rumor_metrics(self):
        """Calculate key rumor spread metrics"""
        if not self.sir_counts_history:
            return
        
        current_counts = self.sir_counts_history[-1]
        total_agents = sum(current_counts.values())
        
        # Basic ratios
        if total_agents > 0:
            self.rumor_metrics = {
                "s_ratio": current_counts["ignorant"] / total_agents,
                "i_ratio": current_counts["spreader"] / total_agents,
                "r_ratio": current_counts["stifler"] / total_agents,
                "total_agents": total_agents,
                "step": self.step_count
            }
        else:
            self.rumor_metrics = {
                "s_ratio": 0.0,
                "i_ratio": 0.0,
                "r_ratio": 0.0,
                "total_agents": 0,
                "step": self.step_count
            }
        
        # Peak spreading
        if len(self.sir_counts_history) > 1:
            max_spreaders = max(step["spreader"] for step in self.sir_counts_history)
            self.rumor_metrics["peak_spreaders"] = max_spreaders
            if total_agents > 0:
                self.rumor_metrics["peak_spreaders_ratio"] = max_spreaders / total_agents
            else:
                self.rumor_metrics["peak_spreaders_ratio"] = 0.0
        
        # Rumor duration (time from first spread to no more spreaders)
        if current_counts["spreader"] == 0 and len(self.sir_counts_history) > 1:
            # Find when rumor ended
            for i, step in enumerate(reversed(self.sir_counts_history)):
                if step["spreader"] > 0:
                    self.rumor_metrics["rumor_duration"] = len(self.sir_counts_history) - i
                    break
    
    def get_rumor_summary(self) -> Dict:
        """Get comprehensive rumor spread summary"""
        if not self.sir_counts_history:
            return {}
        
        final_counts = self.sir_counts_history[-1]
        total_agents = sum(final_counts.values())
        
        if total_agents > 0:
            return {
                "final_ignorant": final_counts["ignorant"],
                "final_spreaders": final_counts["spreader"],
                "final_stiflers": final_counts["stifler"],
                "total_agents": total_agents,
                "total_steps": self.step_count,
                "peak_spreaders": self.rumor_metrics.get("peak_spreaders", 0),
                "peak_spreaders_ratio": self.rumor_metrics.get("peak_spreaders_ratio", 0),
                "rumor_duration": self.rumor_metrics.get("rumor_duration", self.step_count),
                "final_s_ratio": final_counts["ignorant"] / total_agents,
                "final_i_ratio": final_counts["spreader"] / total_agents,
                "final_r_ratio": final_counts["stifler"] / total_agents
            }
        else:
            return {
                "final_ignorant": 0,
                "final_spreaders": 0,
                "final_stiflers": 0,
                "total_agents": 0,
                "total_steps": self.step_count,
                "peak_spreaders": 0,
                "peak_spreaders_ratio": 0.0,
                "rumor_duration": self.step_count,
                "final_s_ratio": 0.0,
                "final_i_ratio": 0.0,
                "final_r_ratio": 0.0
            }


def run_sir_simulation(sir_params: SIRParameters, max_steps: int = 200) -> Tuple[SIREnvironment, List[Dict]]:
    """Run a complete SIR rumor spread simulation"""
    print(f"📢 Running SIR Rumor Spread Simulation")
    print(f"   Network: {sir_params.network_type} ({sir_params.network_params})")
    print(f"   Agents: {sir_params.initial_spreaders} spreaders out of 1000")
    print(f"   Parameters: β={sir_params.spread_rate}, γ={sir_params.stifle_rate}")
    
    # Create environment
    environment = SIREnvironment(num_agents=1000, sir_params=sir_params)
    
    # Create engine
    engine = RuleEngine()
    
    # Run simulation
    results = []
    for step in range(max_steps):
        # Collect observations
        observations = []
        for agent in environment.agents.values():
            obs = agent.observe(environment)
            observations.append(obs)
        
        # Make decisions
        actions = []
        for agent, obs in zip(environment.agents.values(), observations):
            action = agent.decide(obs, engine)
            actions.append(action)
        
        # Execute actions
        for agent, action in zip(environment.agents.values(), actions):
            agent.execute_action(action, environment)
        
        # Update environment
        environment.update_state()
        
        # Store step results
        step_data = {
            "step": step,
            "s_count": environment.rumor_metrics["s_ratio"] * 1000,
            "i_count": environment.rumor_metrics["i_ratio"] * 1000,
            "r_count": environment.rumor_metrics["r_ratio"] * 1000,
            "s_ratio": environment.rumor_metrics["s_ratio"],
            "i_ratio": environment.rumor_metrics["i_ratio"],
            "r_ratio": environment.rumor_metrics["r_ratio"]
        }
        results.append(step_data)
        
        # Check for rumor end
        if environment.rumor_metrics["i_ratio"] == 0 and step > 10:
            print(f"   Rumor ended at step {step}")
            break
    
    return environment, results


def visualize_sir_results(environment: SIREnvironment, results: List[Dict], 
                         title: str = "SIR Rumor Spread Model"):
    """Visualize SIR rumor spread simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    steps = [r["step"] for r in results]
    s_counts = [r["s_count"] for r in results]
    i_counts = [r["i_count"] for r in results]
    r_counts = [r["r_count"] for r in results]
    
    # Plot 1: SIR curves over time
    axes[0, 0].plot(steps, s_counts, label='Ignorant', color='blue', linewidth=2)
    axes[0, 0].plot(steps, i_counts, label='Spreader', color='red', linewidth=2)
    axes[0, 0].plot(steps, r_counts, label='Stifler', color='green', linewidth=2)
    axes[0, 0].set_title(f'{title} - Population Over Time')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Number of Agents')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Network structure (sample)
    axes[0, 1].set_title('Network Structure (Sample)')
    if hasattr(environment, 'network') and environment.network.number_of_nodes() > 0:
        # Create a subgraph for visualization
        sample_size = min(50, environment.network.number_of_nodes())
        sample_nodes = list(environment.network.nodes())[:sample_size]
        subgraph = environment.network.subgraph(sample_nodes)
        
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        nx.draw(subgraph, pos, ax=axes[0, 1], node_size=20, node_color='lightblue', 
                edge_color='gray', alpha=0.7)
    else:
        axes[0, 1].text(0.5, 0.5, 'No network data', ha='center', va='center', 
                        transform=axes[0, 1].transAxes)
    
    # Plot 3: Spreading rate over time
    spreading_rates = [r["i_ratio"] for r in results]
    axes[1, 0].plot(steps, spreading_rates, color='red', linewidth=2)
    axes[1, 0].set_title('Spreading Rate Over Time')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Spreading Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rumor metrics
    summary = environment.get_rumor_summary()
    metrics_text = f"""
    Peak Spreaders: {summary.get('peak_spreaders', 0):.0f}
    Peak Rate: {summary.get('peak_spreaders_ratio', 0):.1%}
    Duration: {summary.get('rumor_duration', 0)} steps
    Final S: {summary.get('final_ignorant', 0):.0f}
    Final I: {summary.get('final_spreaders', 0):.0f}
    Final R: {summary.get('final_stiflers', 0):.0f}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Rumor Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sir_epidemic_results.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'sir_epidemic_results.png'")
    
    # Show the plot
    plt.show()
    
    # Keep the plot open for a moment
    import time
    time.sleep(2)


def compare_network_topologies():
    """Compare SIR dynamics across different network topologies"""
    print("\n🔬 Comparing Network Topologies")
    
    topologies = [
        ("Random", SIRParameters(network_type="random", network_params={"p": 0.01})),
        ("Small World", SIRParameters(network_type="small_world", network_params={"k": 4, "p": 0.1})),
        ("Scale Free", SIRParameters(network_type="scale_free", network_params={"m": 2}))
    ]
    
    results = {}
    for name, params in topologies:
        print(f"\n   Testing {name} network...")
        env, sim_results = run_sir_simulation(params, max_steps=150)
        results[name] = (env, sim_results)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    for i, (name, (env, sim_results)) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        steps = [r["step"] for r in sim_results]
        s_counts = [r["s_count"] for r in sim_results]
        i_counts = [r["i_count"] for r in sim_results]
        r_counts = [r["r_count"] for r in sim_results]
        
        plt.plot(steps, s_counts, label='Ignorant', color='blue', alpha=0.7)
        plt.plot(steps, i_counts, label='Spreader', color='red', alpha=0.7)
        plt.plot(steps, r_counts, label='Stifler', color='green', alpha=0.7)
        
        plt.title(f'{name} Network')
        plt.xlabel('Time Steps')
        plt.ylabel('Number of Agents')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n📊 Network Comparison Summary:")
    for name, (env, sim_results) in results.items():
        summary = env.get_rumor_summary()
        print(f"   {name}:")
        print(f"     Peak Spreaders: {summary.get('peak_spreaders', 0):.0f} ({summary.get('peak_spreaders_ratio', 0):.1%})")
        print(f"     Duration: {summary.get('rumor_duration', 0)} steps")
        print(f"     Final Stiflers: {summary.get('final_stiflers', 0):.0f}")


def main():
    """Main function to run SIR rumor spread examples"""
    print("📢 LAMB Framework: SIR Rumor Spread Model Example")
    print("=" * 60)
    
    # Example 1: Basic SIR simulation
    print("\n1️⃣ Basic SIR Rumor Spread Simulation")
    sir_params = SIRParameters(
        spread_rate=0.3,
        stifle_rate=0.1,
        initial_spreaders=10,
        network_type="random",
        network_params={"p": 0.01}
    )
    
    environment, results = run_sir_simulation(sir_params, max_steps=200)
    visualize_sir_results(environment, results, "Basic SIR Rumor Model")
    
    # Print summary
    summary = environment.get_rumor_summary()
    print(f"\n📈 Rumor Summary:")
    print(f"   Peak Spreaders: {summary.get('peak_spreaders', 0):.0f} agents")
    print(f"   Peak Rate: {summary.get('peak_spreaders_ratio', 0):.1%}")
    print(f"   Duration: {summary.get('rumor_duration', 0)} steps")
    print(f"   Final Stiflers: {summary.get('final_stiflers', 0):.0f} agents")
    
    # Example 2: Compare network topologies
    print("\n2️⃣ Network Topology Comparison")
    compare_network_topologies()
    
    # Example 3: Parameter sensitivity
    print("\n3️⃣ Parameter Sensitivity Analysis")
    print("   Testing different spread rates...")
    
    spread_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    peak_spreadings = []
    
    for rate in spread_rates:
        params = SIRParameters(spread_rate=rate, stifle_rate=0.1)
        env, sim_results = run_sir_simulation(params, max_steps=100)
        summary = env.get_rumor_summary()
        peak_spreadings.append(summary.get('peak_spreaders_ratio', 0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(spread_rates, peak_spreadings, 'o-', linewidth=2, markersize=8)
    plt.title('Peak Spreading Rate vs Spread Rate Parameter')
    plt.xlabel('Spread Rate (β)')
    plt.ylabel('Peak Spreading Rate')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n✅ SIR Rumor Spread Model Examples Complete!")
    print("   - Demonstrated network-based rumor transmission")
    print("   - Compared different network topologies")
    print("   - Analyzed parameter sensitivity")
    print("   - Showed realistic rumor spread dynamics")


if __name__ == "__main__":
    main()
