"""
Factory for creating agents with different personalities.
"""

from typing import Dict, Any, List
import random
from ..paradigms.grid import GridAgent
from ..paradigms.physics import PhysicsAgent
from ..paradigms.network import NetworkAgent


class AgentFactory:
    """Factory for creating agents"""
    
    @staticmethod
    def create_agent(agent_type: str, agent_id: int, personality: str, position, metadata: Dict[str, Any] = None):
        """Create a single agent"""
        if agent_type == "cooperation_agent":
            return GridAgent(agent_id=agent_id, position=position, metadata=metadata or {})
        elif agent_type == "flocking_agent":
            return PhysicsAgent(agent_id=agent_id, position=position, metadata=metadata or {})
        elif agent_type == "social_agent":
            return NetworkAgent(agent_id=agent_id, position=position, metadata=metadata or {})
        else:
            return GridAgent(agent_id=agent_id, position=position, metadata=metadata or {})
    
    @staticmethod
    def create_batch(num_agents: int, config: Dict[str, Any]) -> List:
        """Create a batch of agents"""
        agents = []
        for i in range(num_agents):
            agent = AgentFactory.create_agent(
                agent_type="default",
                agent_id=i,
                personality="default",
                position=(i % 10, i // 10)
            )
            agents.append(agent)
        return agents
