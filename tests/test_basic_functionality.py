"""
Basic functionality tests for the LAMB framework.

This test suite validates core functionality of the LAMB framework
to ensure all components work together correctly.
"""

import pytest
import sys
import os

# Add the lamb package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lamb.config import SimulationConfig, ParadigmType, EngineType
from lamb.api import ResearchAPI
from lamb.core import BaseAgent, BaseEnvironment, BaseEngine, MockEngine
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.paradigms.physics import PhysicsAgent, PhysicsEnvironment
from lamb.paradigms.network import NetworkAgent, NetworkEnvironment
from lamb.spatial import GridIndex, KDTreeIndex, GraphIndex


class TestCoreComponents:
    """Test core framework components"""
    
    def test_mock_engine_creation(self):
        """Test MockEngine creation and basic functionality"""
        engine = MockEngine()
        assert engine.engine_type.value == "llm"  # Mock engine pretends to be LLM
        
        # Test metrics
        metrics = engine.get_performance_metrics()
        assert 'engine_type' in metrics
        assert 'total_decisions' in metrics
    
    def test_grid_environment_creation(self):
        """Test GridEnvironment creation"""
        env = GridEnvironment(dimensions=(10, 10))
        assert env.dimensions == (10, 10)
        assert len(env) == 0  # No agents initially
    
    def test_physics_environment_creation(self):
        """Test PhysicsEnvironment creation"""
        env = PhysicsEnvironment(world_bounds=((-10, -10), (10, 10)))
        assert env.world_bounds == ((-10, -10), (10, 10))
        assert len(env) == 0  # No agents initially
    
    def test_network_environment_creation(self):
        """Test NetworkEnvironment creation"""
        env = NetworkEnvironment()
        assert not env.is_directed  # Default is undirected
        assert len(env) == 0  # No agents initially


class TestSpatialIndexes:
    """Test spatial indexing structures"""
    
    def test_grid_index(self):
        """Test GridIndex functionality"""
        index = GridIndex(dimensions=(5, 5))
        
        # Add agent
        index.add_agent(0, (2, 2))
        assert 0 in index
        assert len(index) == 1
        
        # Get neighbors (should be empty for single agent)
        neighbors = index.get_neighbors(0, 1)
        assert len(neighbors) == 0
    
    def test_kdtree_index(self):
        """Test KDTreeIndex functionality"""
        index = KDTreeIndex()
        
        # Add agent
        index.add_agent(0, (0.0, 0.0))
        assert 0 in index
        assert len(index) == 1
        
        # Get neighbors (agent finds itself as neighbor)
        neighbors = index.get_neighbors(0, 1.0)
        assert len(neighbors) == 1  # Agent finds itself as neighbor
        assert 0 in neighbors  # Agent is in its own neighbor list
    
    def test_graph_index(self):
        """Test GraphIndex functionality"""
        index = GraphIndex()
        
        # Add agent
        index.add_agent(0, 0)
        assert 0 in index
        assert len(index) == 1
        
        # Get neighbors
        neighbors = index.get_neighbors(0, 1)
        assert len(neighbors) == 0  # No connections


class TestConfiguration:
    """Test configuration system"""
    
    def test_grid_config_creation(self):
        """Test grid configuration creation"""
        config = SimulationConfig(
            paradigm=ParadigmType.GRID,
            engine_type=EngineType.MOCK,
            num_agents=10
        )
        
        assert config.paradigm == ParadigmType.GRID
        assert config.engine_type == EngineType.MOCK
        assert config.num_agents == 10
        assert config.grid_config is not None
    
    def test_physics_config_creation(self):
        """Test physics configuration creation"""
        config = SimulationConfig(
            paradigm=ParadigmType.PHYSICS,
            engine_type=EngineType.MOCK,
            num_agents=5
        )
        
        assert config.paradigm == ParadigmType.PHYSICS
        assert config.physics_config is not None
    
    def test_network_config_creation(self):
        """Test network configuration creation"""
        config = SimulationConfig(
            paradigm=ParadigmType.NETWORK,
            engine_type=EngineType.MOCK,
            num_agents=8
        )
        
        assert config.paradigm == ParadigmType.NETWORK
        assert config.network_config is not None


class TestResearchAPI:
    """Test Research API functionality"""
    
    def test_api_creation(self):
        """Test ResearchAPI creation"""
        api = ResearchAPI()
        assert api.config is None
        assert api.environment is None
        assert api.engine is None
    
    def test_grid_simulation_setup(self):
        """Test setting up a grid simulation"""
        api = ResearchAPI()
        api.create_simulation(
            paradigm="grid",
            num_agents=5,
            engine_type="mock",
            max_steps=10
        )
        
        assert api.config is not None
        assert api.config.paradigm == ParadigmType.GRID
        assert api.config.num_agents == 5
        assert api.environment is not None
        assert api.engine is not None
        assert len(api.agents) == 5
    
    def test_physics_simulation_setup(self):
        """Test setting up a physics simulation"""
        api = ResearchAPI()
        api.create_simulation(
            paradigm="physics",
            num_agents=3,
            engine_type="mock",
            max_steps=5
        )
        
        assert api.config.paradigm == ParadigmType.PHYSICS
        assert api.config.num_agents == 3
        assert isinstance(api.environment, PhysicsEnvironment)
        assert len(api.agents) == 3
    
    def test_network_simulation_setup(self):
        """Test setting up a network simulation"""
        api = ResearchAPI()
        api.create_simulation(
            paradigm="network",
            num_agents=4,
            engine_type="mock",
            max_steps=8
        )
        
        assert api.config.paradigm == ParadigmType.NETWORK
        assert api.config.num_agents == 4
        assert isinstance(api.environment, NetworkEnvironment)
        assert len(api.agents) == 4


class TestIntegration:
    """Integration tests"""
    
    def test_simple_grid_simulation(self):
        """Test running a simple grid simulation"""
        api = ResearchAPI()
        api.create_simulation(
            paradigm="grid",
            num_agents=3,
            engine_type="mock",
            max_steps=2
        )
        
        # Run simulation
        result = api.run_simulation(steps=2)
        
        assert result.success
        assert result.steps_completed == 2
        assert len(result.agent_data) == 3
        assert result.total_time > 0
    
    def test_simple_physics_simulation(self):
        """Test running a simple physics simulation"""
        api = ResearchAPI()
        api.create_simulation(
            paradigm="physics",
            num_agents=2,
            engine_type="mock",
            max_steps=3
        )
        
        # Run simulation
        result = api.run_simulation(steps=3)
        
        assert result.success
        assert result.steps_completed == 3
        assert len(result.agent_data) == 2
    
    def test_simple_network_simulation(self):
        """Test running a simple network simulation"""
        api = ResearchAPI()
        api.create_simulation(
            paradigm="network",
            num_agents=4,
            engine_type="mock",
            max_steps=2
        )
        
        # Run simulation
        result = api.run_simulation(steps=2)
        
        assert result.success
        assert result.steps_completed == 2
        assert len(result.agent_data) == 4


if __name__ == "__main__":
    # Run basic tests
    print("Running LAMB Framework Basic Tests...")
    
    # Test core components
    print("Testing core components...")
    test_core = TestCoreComponents()
    test_core.test_mock_engine_creation()
    test_core.test_grid_environment_creation()
    test_core.test_physics_environment_creation()
    test_core.test_network_environment_creation()
    print("✓ Core components tests passed")
    
    # Test spatial indexes
    print("Testing spatial indexes...")
    test_spatial = TestSpatialIndexes()
    test_spatial.test_grid_index()
    test_spatial.test_kdtree_index()
    test_spatial.test_graph_index()
    print("✓ Spatial index tests passed")
    
    # Test configuration
    print("Testing configuration...")
    test_config = TestConfiguration()
    test_config.test_grid_config_creation()
    test_config.test_physics_config_creation()
    test_config.test_network_config_creation()
    print("✓ Configuration tests passed")
    
    # Test Research API
    print("Testing Research API...")
    test_api = TestResearchAPI()
    test_api.test_api_creation()
    test_api.test_grid_simulation_setup()
    test_api.test_physics_simulation_setup()
    test_api.test_network_simulation_setup()
    print("✓ Research API tests passed")
    
    # Test integration
    print("Testing integration...")
    test_integration = TestIntegration()
    test_integration.test_simple_grid_simulation()
    test_integration.test_simple_physics_simulation()
    test_integration.test_simple_network_simulation()
    print("✓ Integration tests passed")
    
    print("\n🎉 All basic tests passed! LAMB Framework is working correctly.")
    print("\nNext steps:")
    print("1. Install OpenAI library: pip install openai")
    print("2. Set OPENAI_API_KEY environment variable")
    print("3. Try LLM engine with engine_type='llm'")
    print("4. Run more comprehensive simulations")
