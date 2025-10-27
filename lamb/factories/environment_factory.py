"""
Factory for creating environments for different paradigms.
"""

from typing import Dict, Any
from ..paradigms.grid import GridEnvironment
from ..paradigms.physics import PhysicsEnvironment
from ..paradigms.network import NetworkEnvironment


class EnvironmentFactory:
    """Factory for creating environments"""
    
    @staticmethod
    def create(world_type: str, config: Dict[str, Any]):
        """Create environment based on world type"""
        if world_type.startswith("grid"):
            return GridEnvironment(dimensions=config.get("grid_size", (20, 20)))
        elif world_type.startswith("physics"):
            return PhysicsEnvironment(world_bounds=config.get("world_bounds", ((-50, -50), (50, 50))))
        elif world_type.startswith("network"):
            return NetworkEnvironment()
        else:
            raise ValueError(f"Unknown world type: {world_type}")
