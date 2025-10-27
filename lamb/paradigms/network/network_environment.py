"""
Network environment implementation for graph-based models.

Based on Technical_Specification.md Section 2.3: Network Paradigm Specification.
Manages graph topology with agents positioned on nodes and interactions along edges.

Performance characteristics:
- Agent lookup: O(1) - <0.0001s
- Direct neighbor query: O(1) - <0.001s for 1-hop neighbors
- Multi-hop query: O(d^h) - <0.01s for 2-hop neighbors
- State update: O(1) per agent - <0.0001s
- Memory: 50 bytes per agent (sparse graphs, degree < 10)
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import time
from collections import defaultdict, deque

from ...core.base_environment import BaseEnvironment
from ...core.types import (
    AgentID, NodeID, Action, ActionResult, BoundaryCondition,
    ConflictError, EnvironmentConstraintError
)
from ...spatial.graph_index import GraphIndex
from .network_agent import NetworkAgent


class NetworkEnvironment(BaseEnvironment):
    """
    Environment for graph-based network models.
    
    Features:
    - Graph topology management (directed/undirected, weighted/unweighted)
    - Agent positioning on network nodes
    - Multi-hop neighbor queries
    - Message passing and information diffusion
    - Dynamic topology changes (edge creation/removal)
    - Network metrics and analysis
    """
    
    def __init__(
        self,
        is_directed: bool = False,
        weighted: bool = False,
        default_node_capacity: int = 10,
        enable_message_passing: bool = True
    ):
        """
        Initialize NetworkEnvironment.
        
        Args:
            is_directed: Whether graph is directed
            weighted: Whether edges have weights
            default_node_capacity: Default maximum agents per node
            enable_message_passing: Whether to enable message passing
        """
        super().__init__()
        
        self.is_directed = is_directed
        self.weighted = weighted
        self.default_node_capacity = default_node_capacity
        self.enable_message_passing = enable_message_passing
        
        # Initialize graph spatial index
        self.spatial_index = GraphIndex(
            is_directed=is_directed,
            weighted=weighted
        )
        
        # Node properties and constraints
        self.node_attributes: Dict[NodeID, Dict[str, Any]] = defaultdict(dict)
        self.node_capacities: Dict[NodeID, int] = defaultdict(lambda: default_node_capacity)
        
        # Message passing system
        self.message_queue: Dict[AgentID, List[Dict[str, Any]]] = defaultdict(list)
        self.message_history: List[Dict[str, Any]] = []
        
        # Network dynamics tracking
        self.topology_changes: List[Dict[str, Any]] = []
        self.network_metrics_cache: Dict[str, Any] = {}
        self.metrics_cache_valid = False
    
    def get_neighbors(self, agent_id: AgentID, radius: float = 1.0) -> List[AgentID]:
        """
        Get neighboring agents within radius (hops).
        
        Performance target: <0.001s for 1-hop, <0.01s for 2-hop
        Uses GraphIndex for O(1) direct neighbors, O(d^h) for multi-hop
        """
        if agent_id not in self.agent_registry:
            return []
        
        agent_node = self.agent_registry[agent_id].position
        return self.spatial_index.get_neighbors_at_position(agent_node, radius)
    
    def get_neighbors_at_hops(self, node_id: NodeID, hops: int) -> List[AgentID]:
        """Get agents at exactly specified number of hops"""
        if hops <= 0:
            return []
        
        # Get all agents within hops
        agents_within_hops = self.spatial_index.get_neighbors_at_position(node_id, hops)
        
        if hops == 1:
            return agents_within_hops
        
        # Subtract agents within (hops-1) to get exactly at hops distance
        agents_within_prev_hops = self.spatial_index.get_neighbors_at_position(node_id, hops - 1)
        
        return [aid for aid in agents_within_hops if aid not in agents_within_prev_hops]
    
    def _is_valid_position(self, position: NodeID) -> bool:
        """Validate if position (node) exists in network"""
        return position in self.spatial_index
    
    def _get_target_position(self, action: Action) -> NodeID:
        """Get target position for action"""
        if action.action_type == "move":
            target_node = action.parameters.get('target_node')
            if target_node is not None:
                return target_node
        
        # For non-movement actions, target is current position
        agent = self.get_agent(action.agent_id)
        return agent.position
    
    def _execute_action(self, action: Action) -> ActionResult:
        """Execute single action"""
        agent = self.get_agent(action.agent_id)
        return agent.act(action, self)
    
    def _resolve_position_conflict(self, actions: List[Action]) -> List[ActionResult]:
        """
        Resolve conflicts for node occupancy.
        
        Strategy: First-come-first-served with capacity limits
        """
        results = []
        node_occupancy = defaultdict(int)
        
        # Count current occupancy
        for agent in self.agent_registry.values():
            node_occupancy[agent.position] += 1
        
        for action in actions:
            target_node = self._get_target_position(action)
            capacity = self.node_capacities[target_node]
            
            if node_occupancy[target_node] < capacity:
                # Space available
                result = self._execute_action(action)
                if result.success and result.new_position:
                    node_occupancy[result.new_position] += 1
                results.append(result)
            else:
                # Node at capacity
                results.append(ActionResult(
                    agent_id=action.agent_id,
                    success=False,
                    error_message=f"Node {target_node} at capacity"
                ))
        
        return results
    
    def _update_environment_state(self) -> None:
        """Update network-specific environment state"""
        # Process message queue
        if self.enable_message_passing:
            self._process_message_queue()
        
        # Update network metrics cache periodically
        if self.step_counter % 10 == 0:  # Every 10 steps
            self._update_network_metrics()
        
        # Clean up old topology changes
        if len(self.topology_changes) > 1000:
            self.topology_changes = self.topology_changes[-500:]
        
        # Clean up old message history
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-500:]
    
    def _process_message_queue(self) -> None:
        """Process pending messages"""
        # Simple message processing - messages are delivered immediately
        # In more complex models, this could include delays, routing, etc.
        pass
    
    def _update_network_metrics(self) -> None:
        """Update cached network metrics"""
        self.network_metrics_cache = self.spatial_index.compute_network_metrics()
        self.metrics_cache_valid = True
    
    def are_nodes_adjacent(self, node1: NodeID, node2: NodeID) -> bool:
        """Check if two nodes are directly connected"""
        if node1 not in self.spatial_index.adjacency:
            return False
        return node2 in self.spatial_index.adjacency[node1]
    
    def get_adjacent_nodes(self, node_id: NodeID) -> List[NodeID]:
        """Get all nodes adjacent to given node"""
        return list(self.spatial_index.adjacency.get(node_id, set()))
    
    def get_node_degree(self, node_id: NodeID) -> int:
        """Get degree of node"""
        return len(self.spatial_index.adjacency.get(node_id, set()))
    
    def get_node_info(self, node_id: NodeID) -> Dict[str, Any]:
        """Get comprehensive information about a node"""
        adjacent_nodes = self.get_adjacent_nodes(node_id)
        agents_on_node = [
            aid for aid, agent in self.agent_registry.items()
            if agent.position == node_id
        ]
        
        return {
            'node_id': node_id,
            'degree': len(adjacent_nodes),
            'adjacent_nodes': adjacent_nodes,
            'agents': agents_on_node,
            'agent_count': len(agents_on_node),
            'capacity': self.node_capacities[node_id],
            'attributes': dict(self.node_attributes[node_id])
        }
    
    def get_edge_weight(self, node1: NodeID, node2: NodeID) -> float:
        """Get weight of edge between two nodes"""
        if not self.weighted:
            return 1.0 if self.are_nodes_adjacent(node1, node2) else 0.0
        
        edge_key = self.spatial_index._get_edge_key(node1, node2)
        return self.spatial_index.edge_weights.get(edge_key, 1.0)
    
    def create_edge(self, node1: NodeID, node2: NodeID, weight: float = 1.0) -> bool:
        """Create edge between two nodes"""
        success = self.spatial_index.add_edge(node1, node2, weight)
        
        if success:
            # Record topology change
            self.topology_changes.append({
                'type': 'edge_created',
                'nodes': (node1, node2),
                'weight': weight,
                'step': self.step_counter,
                'timestamp': time.time()
            })
            
            # Invalidate metrics cache
            self.metrics_cache_valid = False
        
        return success
    
    def remove_edge(self, node1: NodeID, node2: NodeID) -> bool:
        """Remove edge between two nodes"""
        success = self.spatial_index.remove_edge(node1, node2)
        
        if success:
            # Record topology change
            self.topology_changes.append({
                'type': 'edge_removed',
                'nodes': (node1, node2),
                'step': self.step_counter,
                'timestamp': time.time()
            })
            
            # Invalidate metrics cache
            self.metrics_cache_valid = False
        
        return success
    
    def create_node(self, node_id: NodeID, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """Create new node in network"""
        if node_id in self.spatial_index:
            return False  # Node already exists
        
        # Add to spatial index
        self.spatial_index.add_agent(node_id, node_id)  # For network, position = node_id
        
        # Set attributes
        if attributes:
            self.node_attributes[node_id].update(attributes)
        
        # Record topology change
        self.topology_changes.append({
            'type': 'node_created',
            'node': node_id,
            'attributes': attributes or {},
            'step': self.step_counter,
            'timestamp': time.time()
        })
        
        # Invalidate metrics cache
        self.metrics_cache_valid = False
        
        return True
    
    def remove_node(self, node_id: NodeID) -> bool:
        """Remove node from network (and all its edges)"""
        if node_id not in self.spatial_index:
            return False
        
        # Remove all agents from this node first
        agents_to_remove = [
            aid for aid, agent in self.agent_registry.items()
            if agent.position == node_id
        ]
        
        for agent_id in agents_to_remove:
            self.remove_agent(agent_id)
        
        # Remove from spatial index (this also removes all edges)
        self.spatial_index.remove_agent(node_id)
        
        # Clean up node data
        if node_id in self.node_attributes:
            del self.node_attributes[node_id]
        if node_id in self.node_capacities:
            del self.node_capacities[node_id]
        
        # Record topology change
        self.topology_changes.append({
            'type': 'node_removed',
            'node': node_id,
            'step': self.step_counter,
            'timestamp': time.time()
        })
        
        # Invalidate metrics cache
        self.metrics_cache_valid = False
        
        return True
    
    def get_node_capacity(self, node_id: NodeID) -> int:
        """Get capacity of node"""
        return self.node_capacities[node_id]
    
    def set_node_capacity(self, node_id: NodeID, capacity: int) -> None:
        """Set capacity of node"""
        self.node_capacities[node_id] = max(1, capacity)
    
    def get_node_attributes(self, node_id: NodeID) -> Dict[str, Any]:
        """Get attributes of node"""
        return dict(self.node_attributes[node_id])
    
    def set_node_attribute(self, node_id: NodeID, key: str, value: Any) -> None:
        """Set attribute of node"""
        self.node_attributes[node_id][key] = value
    
    def transmit_message(
        self,
        sender_id: AgentID,
        message: Dict[str, Any],
        target_nodes: List[NodeID],
        transmission_type: str = "broadcast"
    ) -> bool:
        """Transmit message to agents on target nodes"""
        if not self.enable_message_passing:
            return False
        
        # Add sender and timestamp to message
        message_with_metadata = {
            'sender': sender_id,
            'content': message,
            'transmission_type': transmission_type,
            'timestamp': time.time(),
            'step': self.step_counter
        }
        
        # Deliver to agents on target nodes
        delivered_count = 0
        for node_id in target_nodes:
            agents_on_node = [
                aid for aid, agent in self.agent_registry.items()
                if agent.position == node_id
            ]
            
            for agent_id in agents_on_node:
                if agent_id != sender_id:  # Don't send to self
                    self.message_queue[agent_id].append(message_with_metadata)
                    delivered_count += 1
        
        # Record in message history
        self.message_history.append({
            'sender': sender_id,
            'target_nodes': target_nodes,
            'message': message_with_metadata,
            'delivered_count': delivered_count
        })
        
        return delivered_count > 0
    
    def get_pending_messages(self, agent_id: AgentID) -> List[Dict[str, Any]]:
        """Get and clear pending messages for agent"""
        messages = self.message_queue[agent_id].copy()
        self.message_queue[agent_id].clear()
        return messages
    
    def process_agent_interaction(
        self,
        agent1_id: AgentID,
        agent2_id: AgentID,
        interaction_type: str
    ) -> Dict[str, Any]:
        """Process interaction between two agents"""
        if agent1_id not in self.agent_registry or agent2_id not in self.agent_registry:
            return {'success': False, 'error': 'Agent not found'}
        
        agent1 = self.agent_registry[agent1_id]
        agent2 = self.agent_registry[agent2_id]
        
        # Check if agents are on adjacent nodes or same node
        if agent1.position == agent2.position:
            distance = 0  # Same node
        elif self.are_nodes_adjacent(agent1.position, agent2.position):
            distance = 1  # Adjacent nodes
        else:
            return {'success': False, 'error': 'Agents not adjacent'}
        
        # Process interaction
        interaction_result = {
            'success': True,
            'agent1': agent1_id,
            'agent2': agent2_id,
            'interaction_type': interaction_type,
            'distance': distance,
            'timestamp': time.time(),
            'step': self.step_counter
        }
        
        # Log interaction in global state
        if 'interactions' not in self.global_state:
            self.global_state['interactions'] = []
        
        self.global_state['interactions'].append(interaction_result)
        
        return interaction_result
    
    def get_graph_properties(self) -> Dict[str, Any]:
        """Get current graph properties"""
        if not self.metrics_cache_valid:
            self._update_network_metrics()
        
        return self.network_metrics_cache.copy()
    
    def _validate_paradigm_state(self) -> bool:
        """Validate network-specific state consistency"""
        try:
            # Validate spatial index consistency
            if not self.spatial_index.validate_consistency():
                return False
            
            # Check that all agents are on valid nodes
            for agent_id, agent in self.agent_registry.items():
                if not self._is_valid_position(agent.position):
                    return False
                
                # Check that agent is in spatial index
                if agent_id not in self.spatial_index:
                    return False
            
            # Check node capacity constraints
            node_occupancy = defaultdict(int)
            for agent in self.agent_registry.values():
                node_occupancy[agent.position] += 1
            
            for node_id, count in node_occupancy.items():
                if count > self.node_capacities[node_id]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network-specific statistics"""
        stats = self.spatial_index.get_graph_statistics()
        
        # Add environment-specific stats
        stats.update({
            'default_node_capacity': self.default_node_capacity,
            'enable_message_passing': self.enable_message_passing,
            'pending_messages': sum(len(msgs) for msgs in self.message_queue.values()),
            'message_history_length': len(self.message_history),
            'topology_changes': len(self.topology_changes),
            'metrics_cache_valid': self.metrics_cache_valid,
            'step_counter': self.step_counter
        })
        
        # Add cached network metrics if available
        if self.metrics_cache_valid:
            stats.update(self.network_metrics_cache)
        
        return stats
    
    def create_agent(self, agent_id: AgentID, position: NodeID, **kwargs) -> NetworkAgent:
        """
        Create and add a new NetworkAgent to the environment.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial node ID
            **kwargs: Additional agent properties
            
        Returns:
            Created NetworkAgent instance
        """
        # Ensure node exists
        if not self._is_valid_position(position):
            # Create node if it doesn't exist
            self.create_node(position)
        
        # Check capacity
        current_occupancy = len([
            aid for aid, agent in self.agent_registry.items()
            if agent.position == position
        ])
        
        if current_occupancy >= self.node_capacities[position]:
            raise EnvironmentConstraintError(f"Node {position} at capacity")
        
        # Create agent
        agent = NetworkAgent(agent_id, position, metadata=kwargs)
        
        # Add to environment
        self.add_agent(agent)
        
        return agent
    
    def __repr__(self) -> str:
        return (f"NetworkEnvironment(nodes={self.spatial_index.node_count}, "
                f"edges={self.spatial_index.edge_count}, agents={len(self.agent_registry)}, "
                f"directed={self.is_directed}, weighted={self.weighted}, step={self.step_counter})")
