#!/usr/bin/env python3
"""
LAMB Framework: Schelling Segregation Model Example

This example implements the classic Schelling Segregation Model using the LAMB framework's
Grid paradigm. The model demonstrates how individual preferences for similar neighbors
can lead to large-scale segregation, even when people are only mildly prejudiced.

Key Features:
- Grid-based spatial segregation
- Agent happiness calculation
- Segregation index measurement
- Multiple agent types
- Visualization of segregation patterns
- Parameter sensitivity analysis

The Schelling model is fundamental in social science and demonstrates:
- Emergent segregation from individual preferences
- The power of small preferences to create large effects
- How spatial patterns emerge from local interactions
- The difference between individual and collective outcomes
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from lamb.config import SimulationConfig, ParadigmType, EngineType
from lamb.api import ResearchAPI
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.core import Action, Observation, AgentID, Position
from lamb.engines import RuleEngine


class AgentType(Enum):
    """Agent types in Schelling model"""
    TYPE_A = "A"
    TYPE_B = "B"
    EMPTY = "empty"


@dataclass
class SchellingParameters:
    """Schelling model parameters"""
    width: int = 30
    height: int = 30
    num_agents: int = 400
    agent_types: int = 2
    similarity_threshold: float = 0.3  # Agents want 30% similar neighbors
    occupancy_rate: float = 0.4  # 40% of cells occupied
    max_moves: int = 10  # Maximum moves per agent per step
    random_seed: int = None
    
    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


class SchellingAgent(GridAgent):
    """Agent for Schelling segregation model"""
    
    def __init__(self, agent_id: AgentID, position: Position, 
                 agent_type: AgentType = AgentType.TYPE_A, **kwargs):
        super().__init__(agent_id, position, **kwargs)
        self.agent_type = agent_type
        self.happiness = 0.0
        self.moves_made = 0
        self.max_moves = 10
        
        # Add type to metadata
        self.metadata["agent_type"] = agent_type.value
        self.metadata["happiness"] = self.happiness
        self.metadata["moves_made"] = self.moves_made
    
    def observe(self, environment: GridEnvironment) -> Observation:
        """Observe neighborhood and calculate happiness"""
        # Get neighbors in 8-cell neighborhood
        neighbors = environment.get_neighbors(self.agent_id, radius=1.5)
        
        # Count neighbors by type
        neighbor_types = {"A": 0, "B": 0, "empty": 0}
        total_neighbors = 0
        
        for neighbor_id in neighbors:
            if neighbor_id in environment.agents:
                neighbor = environment.agents[neighbor_id]
                neighbor_type = neighbor.metadata.get("agent_type", "empty")
                neighbor_types[neighbor_type] += 1
                total_neighbors += 1
        
        # Calculate happiness
        if total_neighbors > 0:
            same_type_neighbors = neighbor_types[self.agent_type.value]
            self.happiness = same_type_neighbors / total_neighbors
        else:
            self.happiness = 0.0  # No neighbors = unhappy
        
        # Update metadata
        self.metadata["happiness"] = self.happiness
        self.metadata["neighbor_types"] = neighbor_types
        self.metadata["total_neighbors"] = total_neighbors
        
        return Observation(
            agent_id=self.agent_id,
            position=self.position,
            neighbors=neighbors,
            paradigm="grid",
            data={
                "agent_type": self.agent_type.value,
                "happiness": self.happiness,
                "neighbor_types": neighbor_types,
                "total_neighbors": total_neighbors,
                "moves_made": self.moves_made
            }
        )
    
    def decide(self, observation: Observation, engine) -> Action:
        """Decide whether to move based on happiness"""
        happiness = observation.data["happiness"]
        moves_made = observation.data["moves_made"]
        
        # If happy enough or moved too much, stay
        if happiness >= 0.3 or moves_made >= self.max_moves:
            return Action(agent_id=self.agent_id, action_type="stay")
        
        # If unhappy, try to move to a better location
        return Action(
            agent_id=self.agent_id,
            action_type="move",
            parameters={"reason": "unhappy", "happiness": happiness}
        )
    
    def execute_action(self, action: Action, environment: GridEnvironment) -> bool:
        """Execute the decided action"""
        if action.action_type == "stay":
            return True
        
        elif action.action_type == "move":
            # Find a random empty position
            empty_positions = environment.get_empty_positions()
            if not empty_positions:
                return False  # No empty positions available
            
            # Choose random empty position
            new_position = empty_positions[np.random.randint(len(empty_positions))]
            
            # Attempt to move
            success = environment.move_agent(self.agent_id, new_position)
            if success:
                self.moves_made += 1
                self.metadata["moves_made"] = self.moves_made
                return True
        
        return False


class SchellingEnvironment(GridEnvironment):
    """Environment for Schelling segregation simulation"""
    
    def __init__(self, schelling_params: SchellingParameters, **kwargs):
        super().__init__(dimensions=(schelling_params.width, schelling_params.height), **kwargs)
        self.schelling_params = schelling_params
        self.step_count = 0
        
        # Initialize agents
        self._create_agents()
        
        # Statistics tracking
        self.segregation_history = []
        self.happiness_history = []
        self.mobility_history = []
    
    def _create_agents(self):
        """Create agents with random types and positions"""
        total_cells = self.schelling_params.width * self.schelling_params.height
        num_occupied = int(total_cells * self.schelling_params.occupancy_rate)
        
        # Create positions
        all_positions = [(x, y) for x in range(self.schelling_params.width) 
                        for y in range(self.schelling_params.height)]
        occupied_positions = np.random.choice(len(all_positions), size=num_occupied, replace=False)
        
        # Create agents
        agent_id = 0
        for i, pos_idx in enumerate(occupied_positions):
            position = all_positions[pos_idx]
            
            # Assign agent type
            if i < num_occupied // 2:
                agent_type = AgentType.TYPE_A
            else:
                agent_type = AgentType.TYPE_B
            
            # Create agent
            agent = SchellingAgent(agent_id, position, agent_type)
            self.add_agent(agent)
            agent_id += 1
    
    def get_empty_positions(self) -> List[Position]:
        """Get all empty positions in the grid"""
        empty_positions = []
        for x in range(self.schelling_params.width):
            for y in range(self.schelling_params.height):
                if not self.is_position_occupied((x, y)):
                    empty_positions.append((x, y))
        return empty_positions
    
    def move_agent(self, agent_id: AgentID, new_position: Position) -> bool:
        """Move agent to new position"""
        if self.is_position_occupied(new_position):
            return False
        
        # Get current position
        old_position = self.agents[agent_id].position
        
        # Update agent position
        self.agents[agent_id].position = new_position
        
        # Update spatial index
        if self.spatial_index is not None:
            self.spatial_index.update_agent(agent_id, old_position, new_position)
        
        return True
    
    def update_state(self):
        """Update environment state and collect statistics"""
        self.step_count += 1
        
        # Calculate segregation index
        segregation_index = self._calculate_segregation_index()
        self.segregation_history.append(segregation_index)
        
        # Calculate average happiness
        total_happiness = 0
        total_agents = 0
        total_moves = 0
        
        for agent in self.agents.values():
            if hasattr(agent, 'happiness'):
                total_happiness += agent.happiness
                total_agents += 1
                total_moves += agent.moves_made
        
        avg_happiness = total_happiness / total_agents if total_agents > 0 else 0
        avg_moves = total_moves / total_agents if total_agents > 0 else 0
        
        self.happiness_history.append(avg_happiness)
        self.mobility_history.append(avg_moves)
    
    def _calculate_segregation_index(self) -> float:
        """Calculate Duncan's segregation index"""
        if len(self.agents) == 0:
            return 0.0
        
        # Count agents by type
        type_counts = {"A": 0, "B": 0}
        for agent in self.agents.values():
            agent_type = agent.metadata.get("agent_type", "A")
            type_counts[agent_type] += 1
        
        total_agents = sum(type_counts.values())
        if total_agents == 0:
            return 0.0
        
        # Calculate segregation for each agent
        segregation_sum = 0
        for agent in self.agents.values():
            agent_type = agent.metadata.get("agent_type", "A")
            
            # Get neighbors
            neighbors = self.get_neighbors(agent.agent_id, radius=1.5)
            if len(neighbors) == 0:
                continue
            
            # Count same-type neighbors
            same_type_neighbors = 0
            for neighbor_id in neighbors:
                if neighbor_id in self.agents:
                    neighbor = self.agents[neighbor_id]
                    neighbor_type = neighbor.metadata.get("agent_type", "A")
                    if neighbor_type == agent_type:
                        same_type_neighbors += 1
            
            # Calculate local segregation
            local_segregation = same_type_neighbors / len(neighbors)
            segregation_sum += local_segregation
        
        return segregation_sum / len(self.agents)
    
    def get_segregation_summary(self) -> Dict:
        """Get comprehensive segregation summary"""
        if not self.segregation_history:
            return {}
        
        return {
            "initial_segregation": self.segregation_history[0],
            "final_segregation": self.segregation_history[-1],
            "max_segregation": max(self.segregation_history),
            "min_segregation": min(self.segregation_history),
            "total_steps": self.step_count,
            "final_happiness": self.happiness_history[-1] if self.happiness_history else 0,
            "total_moves": sum(self.mobility_history) if self.mobility_history else 0,
            "agent_count": len(self.agents),
            "segregation_change": self.segregation_history[-1] - self.segregation_history[0]
        }


def run_schelling_simulation(schelling_params: SchellingParameters, 
                           max_steps: int = 100) -> Tuple[SchellingEnvironment, List[Dict]]:
    """Run a complete Schelling segregation simulation"""
    print(f"🏘️ Running Schelling Segregation Simulation")
    print(f"   Grid: {schelling_params.width}x{schelling_params.height}")
    print(f"   Agents: {schelling_params.num_agents} ({schelling_params.agent_types} types)")
    print(f"   Similarity Threshold: {schelling_params.similarity_threshold}")
    print(f"   Occupancy Rate: {schelling_params.occupancy_rate}")
    
    # Create environment
    environment = SchellingEnvironment(schelling_params)
    
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
            "segregation_index": environment.segregation_history[-1],
            "avg_happiness": environment.happiness_history[-1],
            "avg_moves": environment.mobility_history[-1],
            "agent_count": len(environment.agents)
        }
        results.append(step_data)
        
        # Check for convergence
        if step > 10 and len(environment.segregation_history) > 5:
            recent_changes = abs(environment.segregation_history[-1] - 
                               environment.segregation_history[-5])
            if recent_changes < 0.01:  # Very small change
                print(f"   Simulation converged at step {step}")
                break
    
    return environment, results


def visualize_schelling_results(environment: SchellingEnvironment, results: List[Dict], 
                               title: str = "Schelling Segregation Model"):
    """Visualize Schelling simulation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    steps = [r["step"] for r in results]
    segregation = [r["segregation_index"] for r in results]
    happiness = [r["avg_happiness"] for r in results]
    moves = [r["avg_moves"] for r in results]
    
    # Plot 1: Segregation index over time
    axes[0, 0].plot(steps, segregation, 'b-', linewidth=2)
    axes[0, 0].set_title('Segregation Index Over Time')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Segregation Index')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Threshold')
    axes[0, 0].legend()
    
    # Plot 2: Average happiness over time
    axes[0, 1].plot(steps, happiness, 'g-', linewidth=2)
    axes[0, 1].set_title('Average Happiness Over Time')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Average Happiness')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Threshold')
    axes[0, 1].legend()
    
    # Plot 3: Mobility over time
    axes[0, 2].plot(steps, moves, 'm-', linewidth=2)
    axes[0, 2].set_title('Average Moves per Agent Over Time')
    axes[0, 2].set_xlabel('Time Steps')
    axes[0, 2].set_ylabel('Average Moves')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Initial spatial distribution
    axes[1, 0].set_title('Initial Distribution')
    initial_grid = np.zeros((environment.schelling_params.height, 
                           environment.schelling_params.width))
    
    # Create initial state visualization
    for agent in environment.agents.values():
        x, y = agent.position
        agent_type = agent.metadata.get("agent_type", "A")
        if agent_type == "A":
            initial_grid[y, x] = 1
        else:
            initial_grid[y, x] = 2
    
    im1 = axes[1, 0].imshow(initial_grid, cmap='tab10', vmin=0, vmax=2)
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    
    # Plot 5: Final spatial distribution
    axes[1, 1].set_title('Final Distribution')
    final_grid = np.zeros((environment.schelling_params.height, 
                         environment.schelling_params.width))
    
    # Create final state visualization
    for agent in environment.agents.values():
        x, y = agent.position
        agent_type = agent.metadata.get("agent_type", "A")
        if agent_type == "A":
            final_grid[y, x] = 1
        else:
            final_grid[y, x] = 2
    
    im2 = axes[1, 1].imshow(final_grid, cmap='tab10', vmin=0, vmax=2)
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    
    # Plot 6: Segregation summary
    summary = environment.get_segregation_summary()
    summary_text = f"""
    Initial Segregation: {summary.get('initial_segregation', 0):.3f}
    Final Segregation: {summary.get('final_segregation', 0):.3f}
    Change: {summary.get('segregation_change', 0):.3f}
    Max Segregation: {summary.get('max_segregation', 0):.3f}
    Final Happiness: {summary.get('final_happiness', 0):.3f}
    Total Moves: {summary.get('total_moves', 0):.0f}
    Steps: {summary.get('total_steps', 0)}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 2].set_title('Segregation Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_similarity_thresholds():
    """Compare segregation across different similarity thresholds"""
    print("\n🔬 Comparing Similarity Thresholds")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    final_segregations = []
    
    for threshold in thresholds:
        print(f"   Testing threshold {threshold}...")
        params = SchellingParameters(
            width=20, height=20, num_agents=200,
            similarity_threshold=threshold, occupancy_rate=0.4
        )
        env, results = run_schelling_simulation(params, max_steps=50)
        summary = env.get_segregation_summary()
        final_segregations.append(summary.get('final_segregation', 0))
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, final_segregations, 'o-', linewidth=2, markersize=8)
    plt.title('Final Segregation vs Similarity Threshold')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Final Segregation Index')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(thresholds)), final_segregations, alpha=0.7)
    plt.title('Final Segregation by Threshold')
    plt.xlabel('Threshold Index')
    plt.ylabel('Final Segregation Index')
    plt.xticks(range(len(thresholds)), [f'{t:.1f}' for t in thresholds])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Threshold Comparison Summary:")
    for i, (threshold, segregation) in enumerate(zip(thresholds, final_segregations)):
        print(f"   Threshold {threshold}: Final Segregation = {segregation:.3f}")


def main():
    """Main function to run Schelling segregation examples"""
    print("🏘️ LAMB Framework: Schelling Segregation Model Example")
    print("=" * 60)
    
    # Example 1: Basic Schelling simulation
    print("\n1️⃣ Basic Schelling Simulation")
    schelling_params = SchellingParameters(
        width=30, height=30, num_agents=400,
        similarity_threshold=0.3, occupancy_rate=0.4,
        random_seed=42
    )
    
    environment, results = run_schelling_simulation(schelling_params, max_steps=100)
    visualize_schelling_results(environment, results, "Basic Schelling Model")
    
    # Print summary
    summary = environment.get_segregation_summary()
    print(f"\n📈 Segregation Summary:")
    print(f"   Initial Segregation: {summary.get('initial_segregation', 0):.3f}")
    print(f"   Final Segregation: {summary.get('final_segregation', 0):.3f}")
    print(f"   Change: {summary.get('segregation_change', 0):.3f}")
    print(f"   Final Happiness: {summary.get('final_happiness', 0):.3f}")
    print(f"   Total Moves: {summary.get('total_moves', 0):.0f}")
    
    # Example 2: Compare similarity thresholds
    print("\n2️⃣ Similarity Threshold Comparison")
    compare_similarity_thresholds()
    
    # Example 3: Different grid sizes
    print("\n3️⃣ Grid Size Comparison")
    print("   Testing different grid sizes...")
    
    grid_sizes = [(20, 20), (30, 30), (40, 40)]
    size_results = []
    
    for width, height in grid_sizes:
        print(f"   Testing {width}x{height} grid...")
        params = SchellingParameters(
            width=width, height=height, 
            num_agents=int(width * height * 0.4),
            similarity_threshold=0.3, occupancy_rate=0.4
        )
        env, results = run_schelling_simulation(params, max_steps=50)
        summary = env.get_segregation_summary()
        size_results.append({
            'size': f'{width}x{height}',
            'segregation': summary.get('final_segregation', 0),
            'happiness': summary.get('final_happiness', 0)
        })
    
    # Plot grid size comparison
    plt.figure(figsize=(12, 5))
    
    sizes = [r['size'] for r in size_results]
    segregations = [r['segregation'] for r in size_results]
    happinesses = [r['happiness'] for r in size_results]
    
    plt.subplot(1, 2, 1)
    plt.bar(sizes, segregations, alpha=0.7, color='blue')
    plt.title('Final Segregation by Grid Size')
    plt.xlabel('Grid Size')
    plt.ylabel('Final Segregation Index')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(sizes, happinesses, alpha=0.7, color='green')
    plt.title('Final Happiness by Grid Size')
    plt.xlabel('Grid Size')
    plt.ylabel('Final Happiness')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Schelling Segregation Model Examples Complete!")
    print("   - Demonstrated emergent segregation from individual preferences")
    print("   - Compared different similarity thresholds")
    print("   - Analyzed grid size effects")
    print("   - Showed realistic segregation dynamics")


if __name__ == "__main__":
    main()
