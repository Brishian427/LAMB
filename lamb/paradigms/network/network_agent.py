"""
Network paradigm agent implementation for graph-based models.

Based on Technical_Specification.md Section 2.3: Network Paradigm Specification.
Designed for graph-based models where agents exist on network nodes
(SIR epidemics, opinion dynamics, social networks, information diffusion).

Performance characteristics:
- Memory usage: 50 bytes per agent (sparse graphs, degree < 10)
- Optimal: Network models with sparse connectivity
- Movement: O(1) edge traversal
- Observation: O(1) direct neighbors, O(d^h) for h-hop neighbors
"""

from typing import Dict, List, Set, Any, Optional
import random

from ...core.base_agent import BaseAgent
from ...core.base_environment import BaseEnvironment
from ...core.base_engine import BaseEngine
from ...core.types import (
    AgentID, NodeID, Observation, Action, ActionResult,
    InvalidActionError, EnvironmentConstraintError
)


class NetworkAgent(BaseAgent):
    """
    Agent for graph-based network models.
    
    Features:
    - Node-based positioning on graph topology
    - Edge traversal for movement
    - Multi-hop neighbor awareness
    - Node and edge attribute access
    - Support for directed and undirected graphs
    """
    
    def __init__(
        self,
        agent_id: AgentID,
        position: NodeID,
        metadata: Optional[Dict[str, Any]] = None,
        history_size: int = 100
    ):
        """
        Initialize NetworkAgent.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial node ID in network
            metadata: Agent-specific properties
            history_size: Size of action history buffer
        """
        super().__init__(agent_id, position, metadata, history_size)
        
        # Network-specific properties
        self.node_attributes = metadata.get('node_attributes', {}) if metadata else {}
        self.edge_weights_cache = {}  # Cache for edge weights to neighbors
        
        # Navigation and interaction ranges
        self.hop_range = metadata.get('hop_range', 1) if metadata else 1
        self.max_hops_per_step = metadata.get('max_hops_per_step', 1) if metadata else 1
        
        # Network behavior parameters
        self.movement_probability = metadata.get('movement_probability', 0.1) if metadata else 0.1
        self.interaction_probability = metadata.get('interaction_probability', 0.5) if metadata else 0.5
    
    def observe(self, environment: 'NetworkEnvironment') -> Observation:
        """
        Generate observation of network environment.
        
        Performance target: <0.0001s per call
        Includes: direct neighbors, multi-hop neighbors, node/edge attributes
        """
        if self.agent_id not in environment:
            raise AgentNotFoundError(f"Agent {self.agent_id} not in environment")
        
        # Get neighbors within hop range
        neighbors = environment.get_neighbors(self.agent_id, self.hop_range)
        
        # Get local network state
        local_state = self._observe_local_state(environment, neighbors)
        
        # Create observation
        observation = Observation(
            agent_id=self.agent_id,
            position=self.position,
            neighbors=neighbors,
            environment_state=local_state,
            paradigm="network"
        )
        
        return observation
    
    def _observe_local_state(self, environment: 'NetworkEnvironment', neighbors: List[AgentID]) -> Dict[str, Any]:
        """Observe local network state around agent"""
        # Get current node information
        current_node_info = environment.get_node_info(self.position)
        
        # Get neighbor information with edge weights
        neighbor_info = []
        direct_neighbors = environment.get_neighbors(self.agent_id, 1)  # Direct neighbors only
        
        for neighbor_id in direct_neighbors:
            if neighbor_id in environment.agent_registry:
                neighbor = environment.agent_registry[neighbor_id]
                neighbor_node = neighbor.position
                
                # Get edge weight if available
                edge_weight = environment.get_edge_weight(self.position, neighbor_node)
                
                neighbor_info.append({
                    'id': neighbor_id,
                    'node': neighbor_node,
                    'edge_weight': edge_weight,
                    'attributes': environment.get_node_attributes(neighbor_node)
                })
        
        # Get multi-hop neighbors (if hop_range > 1)
        multi_hop_neighbors = []
        if self.hop_range > 1:
            for hop in range(2, self.hop_range + 1):
                hop_neighbors = environment.get_neighbors_at_hops(self.position, hop)
                multi_hop_neighbors.extend([
                    {'id': nid, 'hops': hop} for nid in hop_neighbors
                    if nid in environment.agent_registry
                ])
        
        local_state = {
            'current_node': self.position,
            'node_attributes': current_node_info.get('attributes', {}),
            'node_degree': current_node_info.get('degree', 0),
            'direct_neighbors': neighbor_info,
            'multi_hop_neighbors': multi_hop_neighbors,
            'graph_properties': environment.get_graph_properties(),
            'agent_count': len(environment),
            'hop_range': self.hop_range
        }
        
        return local_state
    
    def decide(self, observation: Observation, engine: BaseEngine) -> Action:
        """
        Make decision based on observation using specified engine.
        
        Performance target: <0.456s per call (LLM mode)
        """
        if observation.paradigm != "network":
            raise IncompatibleEngineError("NetworkAgent requires network paradigm observation")
        
        # Use engine to make decision
        action = engine.process_single(observation)
        
        # Validate action is appropriate for network paradigm
        if not self._is_valid_network_action(action):
            # Create fallback action
            action = self._create_fallback_action()
        
        return action
    
    def _is_valid_network_action(self, action: Action) -> bool:
        """Validate that action is valid for network paradigm"""
        valid_actions = {'move', 'stay', 'interact', 'transmit', 'receive', 'create_edge', 'remove_edge'}
        return action.action_type in valid_actions
    
    def _create_fallback_action(self) -> Action:
        """Create safe fallback action"""
        return Action(
            agent_id=self.agent_id,
            action_type="stay",
            parameters={}
        )
    
    def act(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """
        Execute action in network environment.
        
        Performance target: <0.0001s per call
        """
        if action.agent_id != self.agent_id:
            raise InvalidActionError("Action agent_id doesn't match agent")
        
        try:
            if action.action_type == "move":
                return self._execute_move(action, environment)
            elif action.action_type == "stay":
                return self._execute_stay(action, environment)
            elif action.action_type == "interact":
                return self._execute_interact(action, environment)
            elif action.action_type == "transmit":
                return self._execute_transmit(action, environment)
            elif action.action_type == "receive":
                return self._execute_receive(action, environment)
            elif action.action_type == "create_edge":
                return self._execute_create_edge(action, environment)
            elif action.action_type == "remove_edge":
                return self._execute_remove_edge(action, environment)
            else:
                raise InvalidActionError(f"Unknown action type: {action.action_type}")
                
        except Exception as e:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message=str(e)
            )
    
    def _execute_move(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute movement to adjacent node"""
        target_node = action.parameters.get('target_node')
        direction = action.parameters.get('direction', 'random')
        
        if target_node is not None:
            # Direct node specification
            new_position = target_node
        else:
            # Choose target based on direction
            new_position = self._choose_movement_target(environment, direction)
        
        if new_position is None:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="No valid movement target found"
            )
        
        # Validate that target is adjacent
        if not environment.are_nodes_adjacent(self.position, new_position):
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Target node not adjacent"
            )
        
        # Validate movement constraints
        if not self._can_move_to_node(new_position, environment):
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Movement to target node not allowed"
            )
        
        # Attempt to update position in environment
        success = environment.update_agent_state(self.agent_id, new_position)
        
        if success:
            return ActionResult(
                agent_id=self.agent_id,
                success=True,
                new_position=new_position
            )
        else:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Movement blocked or invalid"
            )
    
    def _execute_stay(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute stay action (no movement)"""
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_interact(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute interaction with neighbor agents"""
        target_agents = action.parameters.get('target_agents', [])
        interaction_type = action.parameters.get('interaction_type', 'default')
        
        if not target_agents:
            # Interact with all direct neighbors
            target_agents = environment.get_neighbors(self.agent_id, 1)
        
        # Process interactions
        interaction_results = []
        for target_id in target_agents:
            if target_id in environment.agent_registry:
                result = environment.process_agent_interaction(
                    self.agent_id, target_id, interaction_type
                )
                interaction_results.append(result)
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_transmit(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute information transmission"""
        message = action.parameters.get('message', {})
        target_nodes = action.parameters.get('target_nodes', [])
        transmission_type = action.parameters.get('transmission_type', 'broadcast')
        
        if not target_nodes:
            # Default to direct neighbors
            target_nodes = [
                environment.agent_registry[nid].position 
                for nid in environment.get_neighbors(self.agent_id, 1)
                if nid in environment.agent_registry
            ]
        
        # Transmit message
        success = environment.transmit_message(
            self.agent_id, message, target_nodes, transmission_type
        )
        
        return ActionResult(
            agent_id=self.agent_id,
            success=success,
            new_position=self.position,
            error_message=None if success else "Transmission failed"
        )
    
    def _execute_receive(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute message reception"""
        # Check for pending messages
        messages = environment.get_pending_messages(self.agent_id)
        
        # Process messages (simple implementation)
        for message in messages:
            self._process_received_message(message)
        
        return ActionResult(
            agent_id=self.agent_id,
            success=True,
            new_position=self.position
        )
    
    def _execute_create_edge(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute edge creation"""
        target_node = action.parameters.get('target_node')
        edge_weight = action.parameters.get('edge_weight', 1.0)
        
        if target_node is None:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Target node not specified"
            )
        
        # Create edge in environment
        success = environment.create_edge(self.position, target_node, edge_weight)
        
        return ActionResult(
            agent_id=self.agent_id,
            success=success,
            new_position=self.position,
            error_message=None if success else "Edge creation failed"
        )
    
    def _execute_remove_edge(self, action: Action, environment: 'NetworkEnvironment') -> ActionResult:
        """Execute edge removal"""
        target_node = action.parameters.get('target_node')
        
        if target_node is None:
            return ActionResult(
                agent_id=self.agent_id,
                success=False,
                error_message="Target node not specified"
            )
        
        # Remove edge in environment
        success = environment.remove_edge(self.position, target_node)
        
        return ActionResult(
            agent_id=self.agent_id,
            success=success,
            new_position=self.position,
            error_message=None if success else "Edge removal failed"
        )
    
    def _choose_movement_target(self, environment: 'NetworkEnvironment', direction: str) -> Optional[NodeID]:
        """Choose movement target based on direction strategy"""
        direct_neighbors = environment.get_neighbors(self.agent_id, 1)
        
        if not direct_neighbors:
            return None
        
        # Get neighbor nodes
        neighbor_nodes = [
            environment.agent_registry[nid].position 
            for nid in direct_neighbors
            if nid in environment.agent_registry
        ]
        
        # Add unoccupied adjacent nodes
        adjacent_nodes = environment.get_adjacent_nodes(self.position)
        for node in adjacent_nodes:
            if node not in neighbor_nodes:
                neighbor_nodes.append(node)
        
        if not neighbor_nodes:
            return None
        
        if direction == 'random':
            return random.choice(neighbor_nodes)
        elif direction == 'highest_degree':
            # Move to node with highest degree
            return max(neighbor_nodes, key=lambda n: environment.get_node_degree(n))
        elif direction == 'lowest_degree':
            # Move to node with lowest degree
            return min(neighbor_nodes, key=lambda n: environment.get_node_degree(n))
        elif direction == 'highest_weight':
            # Move along edge with highest weight
            if environment.spatial_index.weighted:
                weights = [environment.get_edge_weight(self.position, node) for node in neighbor_nodes]
                max_idx = weights.index(max(weights))
                return neighbor_nodes[max_idx]
            else:
                return random.choice(neighbor_nodes)
        else:
            # Default to random
            return random.choice(neighbor_nodes)
    
    def _can_move_to_node(self, target_node: NodeID, environment: 'NetworkEnvironment') -> bool:
        """Check if agent can move to target node"""
        # Check movement constraints (can be customized)
        node_capacity = environment.get_node_capacity(target_node)
        current_occupancy = len([
            aid for aid, agent in environment.agent_registry.items()
            if agent.position == target_node
        ])
        
        return current_occupancy < node_capacity
    
    def _process_received_message(self, message: Dict[str, Any]) -> None:
        """Process received message (can be overridden by subclasses)"""
        # Simple implementation - store in node attributes
        if 'received_messages' not in self.node_attributes:
            self.node_attributes['received_messages'] = []
        
        self.node_attributes['received_messages'].append(message)
        
        # Keep only recent messages
        if len(self.node_attributes['received_messages']) > 10:
            self.node_attributes['received_messages'] = self.node_attributes['received_messages'][-10:]
    
    def get_network_properties(self) -> Dict[str, Any]:
        """Get network-specific agent properties"""
        return {
            'position': self.position,
            'node_attributes': self.node_attributes,
            'hop_range': self.hop_range,
            'max_hops_per_step': self.max_hops_per_step,
            'movement_probability': self.movement_probability,
            'interaction_probability': self.interaction_probability,
            'edge_weights_cache': self.edge_weights_cache,
            'memory_usage_bytes': self._estimate_memory_usage()
        }
    
    def set_node_attribute(self, key: str, value: Any) -> None:
        """Set node attribute"""
        self.node_attributes[key] = value
    
    def get_node_attribute(self, key: str, default: Any = None) -> Any:
        """Get node attribute"""
        return self.node_attributes.get(key, default)
    
    def update_edge_weight_cache(self, neighbor_node: NodeID, weight: float) -> None:
        """Update cached edge weight"""
        self.edge_weights_cache[neighbor_node] = weight
    
    def get_cached_edge_weight(self, neighbor_node: NodeID) -> Optional[float]:
        """Get cached edge weight"""
        return self.edge_weights_cache.get(neighbor_node)
    
    def __repr__(self) -> str:
        return (f"NetworkAgent(id={self.agent_id}, node={self.position}, "
                f"hop_range={self.hop_range})")
