"""
Factory for creating executors for different paradigms.
"""

from typing import Dict, Any
from ..executors import GridExecutor, PhysicsExecutor, NetworkExecutor


class ExecutorFactory:
    """Factory for creating executors"""
    
    @staticmethod
    def create(world_type: str, config: Dict[str, Any]):
        """Create executor based on world type"""
        if world_type.startswith("grid"):
            return GridExecutor()
        elif world_type.startswith("physics"):
            return PhysicsExecutor()
        elif world_type.startswith("network"):
            return NetworkExecutor()
        else:
            return GridExecutor()
