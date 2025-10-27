#!/usr/bin/env python3
"""
Basic Sugarscape Example

This example demonstrates how to create a simple Sugarscape simulation
using the LAMB framework. Sugarscape is a classic agent-based model
where agents move around a grid to collect sugar resources.

Author: LAMB Development Team
Date: 2024-10-27
"""

import sys
import os
import time
from typing import List, Tuple

# Add the lamb package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lamb'))

from lamb.api import ResearchAPI
from lamb.config import SimulationConfig, ParadigmType


def create_sugarscape_simulation():
    """
    Create a basic Sugarscape simulation.
    
    Returns:
        ResearchAPI: Configured API instance
    """
    print("Creating Sugarscape simulation...")
    
    # Create API instance
    api = ResearchAPI()
    
    # Configure simulation
    config = SimulationConfig(
        paradigm=ParadigmType.GRID,
        num_agents=100,
        engine_type="rule",
        max_steps=1000,
        grid_config={
            "dimensions": (50, 50),
            "boundary_condition": "wrap",
            "cell_size": 1.0,
            "max_agents_per_cell": 1
        },
        agent_config={
            "vision_range": 3,
            "movement_range": 1,
            "initial_energy": 50.0,
            "metabolism": 1.0
        }
    )
    
    # Create simulation
    api.create_simulation(config=config)
    
    print(f"Simulation created with {len(api.agents)} agents")
    print(f"Grid size: {api.environment.dimensions}")
    
    return api


def run_simulation(api: ResearchAPI, steps: int = 500):
    """
    Run the simulation for specified number of steps.
    
    Args:
        api: ResearchAPI instance
        steps: Number of steps to run
    """
    print(f"Running simulation for {steps} steps...")
    
    start_time = time.time()
    results = api.run_simulation(steps=steps)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Total steps: {len(results)}")
    
    return results


def analyze_results(results: List[dict]):
    """
    Analyze simulation results.
    
    Args:
        results: List of step results from simulation
    """
    print("\nAnalyzing results...")
    
    # Calculate basic statistics
    total_steps = len(results)
    if total_steps == 0:
        print("No results to analyze")
        return
    
    # Get final step data
    final_step = results[-1]
    
    print(f"Final step: {final_step['step']}")
    print(f"Total agents: {final_step['num_agents']}")
    print(f"Successful actions: {final_step['successful_actions']}")
    print(f"Failed actions: {final_step['failed_actions']}")
    
    # Calculate average step time
    total_time = sum(step['step_time'] for step in results)
    avg_step_time = total_time / total_steps
    
    print(f"Average step time: {avg_step_time:.4f} seconds")
    
    # Environment metrics
    env_metrics = final_step.get('environment_metrics', {})
    if env_metrics:
        print(f"Environment metrics: {env_metrics}")


def main():
    """
    Main function to run the Sugarscape example.
    """
    print("LAMB Sugarscape Example")
    print("=" * 50)
    
    try:
        # Create simulation
        api = create_sugarscape_simulation()
        
        # Run simulation
        results = run_simulation(api, steps=500)
        
        # Analyze results
        analyze_results(results)
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
