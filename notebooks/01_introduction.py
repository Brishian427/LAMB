#!/usr/bin/env python3
"""
LAMB Framework Introduction

This script provides an introduction to the LAMB (LLM Agent Model Base) framework
for agent-based modeling with Large Language Model integration.

What is LAMB?
- A unified framework for building agent-based models
- Supports multiple simulation paradigms (Grid, Physics, Network)
- Integrates Large Language Models for agent behavior
- High-performance simulation capabilities
- Research-ready analysis tools

Author: LAMB Development Team
Date: 2024-10-27
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add the lamb package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lamb'))


def installation_check():
    """Check if LAMB is properly installed."""
    print("Checking LAMB installation...")
    
    try:
        from lamb.api import ResearchAPI
        from lamb.config import SimulationConfig, ParadigmType
        print("✓ LAMB successfully imported")
        print(f"✓ Available paradigms: {[p.value for p in ParadigmType]}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please ensure LAMB is properly installed")
        return False


def basic_usage_example():
    """Demonstrate basic LAMB usage."""
    print("\n" + "="*50)
    print("Basic Usage Example")
    print("="*50)
    
    # Create a simple grid simulation
    api = ResearchAPI()
    
    # Configure simulation
    api.create_simulation(
        paradigm="grid",
        num_agents=50,
        engine_type="rule",
        max_steps=100
    )
    
    print(f"Simulation created with {len(api.agents)} agents")
    print(f"Environment type: {type(api.environment).__name__}")
    print(f"Engine type: {type(api.engine).__name__}")
    
    return api


def run_simulation_example(api: ResearchAPI) -> List[Dict[str, Any]]:
    """Run simulation and return results."""
    print("\n" + "="*50)
    print("Running Simulation")
    print("="*50)
    
    print("Running simulation...")
    start_time = time.time()
    
    results = api.run_simulation(steps=50)
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Total steps: {len(results)}")
    
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze simulation results."""
    print("\n" + "="*50)
    print("Analyzing Results")
    print("="*50)
    
    if not results:
        print("No results to analyze")
        return
    
    final_step = results[-1]
    
    print("Final step analysis:")
    print(f"  Step number: {final_step['step']}")
    print(f"  Total agents: {final_step['num_agents']}")
    print(f"  Successful actions: {final_step['successful_actions']}")
    print(f"  Failed actions: {final_step['failed_actions']}")
    print(f"  Step time: {final_step['step_time']:.4f} seconds")
    
    # Calculate average performance
    total_time = sum(step['step_time'] for step in results)
    avg_step_time = total_time / len(results)
    
    print(f"\nPerformance metrics:")
    print(f"  Average step time: {avg_step_time:.4f} seconds")
    print(f"  Total simulation time: {total_time:.4f} seconds")


def main():
    """Main function to run the introduction example."""
    print("LAMB Framework Introduction")
    print("="*50)
    
    # Check installation
    if not installation_check():
        return 1
    
    # Basic usage
    api = basic_usage_example()
    
    # Run simulation
    results = run_simulation_example(api)
    
    # Analyze results
    analyze_results(results)
    
    # Next steps
    print("\n" + "="*50)
    print("Next Steps")
    print("="*50)
    print("This example demonstrated basic LAMB usage. Next steps:")
    print("1. 02_first_simulation.py: Building a complete Sugarscape model")
    print("2. 03_llm_integration.py: Adding LLM-powered agents")
    print("3. 04_analysis.py: Analyzing emergence patterns")
    print("4. 05_advanced_patterns.py: Complex multi-paradigm simulations")
    
    print("\nKey Concepts learned:")
    print("- ResearchAPI: The main interface for creating simulations")
    print("- Paradigms: Different simulation environments (Grid, Physics, Network)")
    print("- Engines: Different agent behavior systems (Rule, LLM, Hybrid)")
    print("- Composition: LAMB uses a composition-based architecture")
    
    print("\nResources:")
    print("- Documentation: https://lamb.readthedocs.io")
    print("- GitHub Repository: https://github.com/your-org/lamb")
    print("- Examples Gallery: https://lamb.readthedocs.io/en/latest/examples.html")
    print("- API Reference: https://lamb.readthedocs.io/en/latest/api_reference.html")
    
    print("\nExample completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
