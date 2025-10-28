"""
Information Access Control in LAMB Framework

This demonstrates how to implement configurable information access
for different simulation types - from full information to limited access.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod
from enum import Enum

from lamb.core import BaseAgent, BaseEnvironment
from lamb.core.types import Observation, Action, AgentID


class InformationLevel(Enum):
    """Levels of information access for agents"""
    FULL = "full"           # See everything (current design)
    LIMITED = "limited"     # See only what's realistic
    PRIVACY = "privacy"     # Respect privacy boundaries
    RESTRICTED = "restricted" # Restricted access limitations


class InformationFilter(ABC):
    """Abstract base class for information filtering"""
    
    @abstractmethod
    def filter_observation(self, agent: BaseAgent, observation: Observation, 
                          environment: BaseEnvironment) -> Observation:
        """Filter observation based on agent's information access level"""
        pass


class FullInformationFilter(InformationFilter):
    """No filtering - agents see everything (current design)"""
    
    def filter_observation(self, agent: BaseAgent, observation: Observation, 
                          environment: BaseEnvironment) -> Observation:
        return observation


class LimitedInformationFilter(InformationFilter):
    """Limited information - realistic social boundaries"""
    
    def __init__(self, max_neighbor_distance: float = 2.0):
        self.max_neighbor_distance = max_neighbor_distance
    
    def filter_observation(self, agent: BaseAgent, observation: Observation, 
                          environment: BaseEnvironment) -> Observation:
        # Filter neighbors by distance
        filtered_neighbors = []
        for neighbor_id in observation.neighbors:
            if hasattr(environment, 'get_distance'):
                distance = environment.get_distance(agent.position, 
                                                  environment.get_agent_position(neighbor_id))
                if distance <= self.max_neighbor_distance:
                    filtered_neighbors.append(neighbor_id)
            else:
                filtered_neighbors.append(neighbor_id)
        
        # Create filtered observation
        filtered_obs = Observation(
            agent_id=observation.agent_id,
            position=observation.position,
            neighbors=filtered_neighbors,
            environment_state=observation.environment_state.copy(),
            paradigm=observation.paradigm,
            timestamp=observation.timestamp
        )
        
        return filtered_obs


class PrivacyAwareFilter(InformationFilter):
    """Privacy-aware filtering - respect social boundaries"""
    
    def __init__(self, social_network: Dict[AgentID, Set[AgentID]]):
        self.social_network = social_network
    
    def filter_observation(self, agent: BaseAgent, observation: Observation, 
                          environment: BaseEnvironment) -> Observation:
        # Only see agents in your social network
        agent_network = self.social_network.get(agent.agent_id, set())
        filtered_neighbors = [nid for nid in observation.neighbors if nid in agent_network]
        
        # Filter environment state to only include public information
        filtered_state = {}
        for key, value in observation.environment_state.items():
            if self._is_public_information(key):
                filtered_state[key] = value
        
        filtered_obs = Observation(
            agent_id=observation.agent_id,
            position=observation.position,
            neighbors=filtered_neighbors,
            environment_state=filtered_state,
            paradigm=observation.paradigm,
            timestamp=observation.timestamp
        )
        
        return filtered_obs
    
    def _is_public_information(self, key: str) -> bool:
        """Determine if information is public or private"""
        private_keys = {"private_wealth", "personal_health", "secret_strategy"}
        return key not in private_keys


class RestrictedFilter(InformationFilter):
    """Restricted filtering - access limitations"""
    
    def __init__(self, visibility_range: float = 5.0, intelligence_level: float = 0.7):
        self.visibility_range = visibility_range
        self.intelligence_level = intelligence_level
    
    def filter_observation(self, agent: BaseAgent, observation: Observation, 
                          environment: BaseEnvironment) -> Observation:
        # Filter by visibility range
        filtered_neighbors = []
        for neighbor_id in observation.neighbors:
            if hasattr(environment, 'get_distance'):
                distance = environment.get_distance(agent.position, 
                                                  environment.get_agent_position(neighbor_id))
                if distance <= self.visibility_range:
                    filtered_neighbors.append(neighbor_id)
        
        # Add noise to information based on intelligence level
        noisy_state = {}
        for key, value in observation.environment_state.items():
            if isinstance(value, (int, float)):
                # Add noise to numerical values
                noise_factor = 1.0 - self.intelligence_level
                noise = (hash(str(value)) % 100) / 100 * noise_factor
                noisy_value = value * (1 + noise)
                noisy_state[key] = noisy_value
            else:
                noisy_state[key] = value
        
        filtered_obs = Observation(
            agent_id=observation.agent_id,
            position=observation.position,
            neighbors=filtered_neighbors,
            environment_state=noisy_state,
            paradigm=observation.paradigm,
            timestamp=observation.timestamp
        )
        
        return filtered_obs


# ============================================================================
# EXAMPLE: CONFIGURABLE AGENT WITH INFORMATION FILTERING
# ============================================================================

class ConfigurableAgent(BaseAgent):
    """Agent with configurable information access"""
    
    def __init__(self, agent_id: AgentID, position: tuple, 
                 information_level: InformationLevel = InformationLevel.FULL,
                 information_filter: Optional[InformationFilter] = None):
        super().__init__(agent_id, position)
        self.information_level = information_level
        self.information_filter = information_filter or FullInformationFilter()
    
    def observe(self, environment: BaseEnvironment) -> Observation:
        """Observe with configurable information filtering"""
        # Get basic observation
        observation = super().observe(environment)
        
        # Add agent's own state
        observation.environment_state.update({
            "my_wealth": getattr(self, 'wealth', 100),
            "my_health": getattr(self, 'health', 100),
            "my_strategy": getattr(self, 'strategy', 'cooperative'),
        })
        
        # Apply information filtering
        filtered_observation = self.information_filter.filter_observation(
            self, observation, environment
        )
        
        return filtered_observation
    
    def act(self, action: Action, environment: BaseEnvironment):
        """Simple action execution"""
        pass
    
    def decide(self, observation: Observation) -> Action:
        """Simple decision making"""
        return Action(
            agent_id=observation.agent_id,
            action_type="wait",
            parameters={}
        )


# ============================================================================
# DEMONSTRATION: DIFFERENT INFORMATION ACCESS LEVELS
# ============================================================================

def demonstrate_information_access():
    """Demonstrate different information access levels"""
    
    print("🔍 Information Access Control in LAMB")
    print("=" * 50)
    
    # Create agents with different information levels
    agents = [
        ConfigurableAgent(0, (0, 0), InformationLevel.FULL),
        ConfigurableAgent(1, (1, 1), InformationLevel.LIMITED, 
                         LimitedInformationFilter(max_neighbor_distance=1.5)),
        ConfigurableAgent(2, (2, 2), InformationLevel.PRIVACY,
                         PrivacyAwareFilter({2: {0, 1}})),  # Agent 2 only sees agents 0,1
        ConfigurableAgent(3, (3, 3), InformationLevel.STRATEGIC,
                         StrategicFilter(visibility_range=2.0, intelligence_level=0.5))
    ]
    
    print("\n📊 Information Access Levels:")
    for agent in agents:
        print(f"  Agent {agent.agent_id}: {agent.information_level.value}")
    
    print("\n🎯 Key Design Question:")
    print("  Should agents ALWAYS see everything?")
    print("  Or should information access be configurable?")
    
    print("\n✅ Current LAMB Design: Always see everything")
    print("  - Simple and consistent")
    print("  - Good for most ABM simulations")
    print("  - Easy to implement and understand")
    
    print("\n🤔 Alternative Design: Configurable information access")
    print("  - More realistic for social simulations")
    print("  - Better for restricted access simulations")
    print("  - More complex to implement")
    print("  - Requires careful design decisions")
    
    print("\n💡 Recommendation:")
    print("  - Keep current design as DEFAULT (full information)")
    print("  - Add OPTIONAL information filtering for advanced use cases")
    print("  - Let researchers choose based on their simulation needs")


if __name__ == "__main__":
    demonstrate_information_access()
