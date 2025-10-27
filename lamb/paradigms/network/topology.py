"""
Network topology generation and analysis utilities.

Based on Technical_Specification.md Section 2.3: Network Paradigm Specification.
Provides utilities for generating common network topologies and analyzing
network properties for agent-based models.

Performance characteristics:
- Topology generation: O(V + E) where V = nodes, E = edges
- Network analysis: O(V + E) to O(V³) depending on metric
- Small-world rewiring: O(E) for edge rewiring
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import random
import math
from collections import deque

from ...core.types import NodeID


class NetworkTopology:
    """
    Utilities for generating and analyzing network topologies.
    
    Provides implementations of common network models:
    - Regular lattices (grid, ring, complete)
    - Random networks (Erdős-Rényi)
    - Small-world networks (Watts-Strogatz)
    - Scale-free networks (Barabási-Albert)
    - Social network patterns (community structures)
    """
    
    @staticmethod
    def create_complete_graph(n_nodes: int) -> List[Tuple[NodeID, NodeID]]:
        """
        Create complete graph where every node connects to every other node.
        
        Time complexity: O(n²)
        Edge count: n(n-1)/2 for undirected, n(n-1) for directed
        
        Args:
            n_nodes: Number of nodes
            
        Returns:
            List of edge tuples
        """
        edges = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edges.append((i, j))
        
        return edges
    
    @staticmethod
    def create_ring_lattice(n_nodes: int, k_neighbors: int) -> List[Tuple[NodeID, NodeID]]:
        """
        Create ring lattice where each node connects to k nearest neighbors.
        
        Time complexity: O(n * k)
        Used as starting point for Watts-Strogatz small-world networks.
        
        Args:
            n_nodes: Number of nodes
            k_neighbors: Number of neighbors each node connects to (must be even)
            
        Returns:
            List of edge tuples
        """
        if k_neighbors % 2 != 0:
            raise ValueError("k_neighbors must be even")
        
        edges = []
        half_k = k_neighbors // 2
        
        for i in range(n_nodes):
            for j in range(1, half_k + 1):
                # Connect to j-th neighbor on each side
                neighbor = (i + j) % n_nodes
                edges.append((i, neighbor))
        
        return edges
    
    @staticmethod
    def create_grid_lattice(
        width: int, 
        height: int, 
        periodic: bool = False
    ) -> List[Tuple[NodeID, NodeID]]:
        """
        Create 2D grid lattice topology.
        
        Time complexity: O(width * height)
        
        Args:
            width: Grid width
            height: Grid height
            periodic: Whether to use periodic boundary conditions (torus)
            
        Returns:
            List of edge tuples
        """
        edges = []
        
        def node_id(x: int, y: int) -> NodeID:
            return y * width + x
        
        for y in range(height):
            for x in range(width):
                current = node_id(x, y)
                
                # Right neighbor
                if x < width - 1:
                    edges.append((current, node_id(x + 1, y)))
                elif periodic:
                    edges.append((current, node_id(0, y)))
                
                # Down neighbor
                if y < height - 1:
                    edges.append((current, node_id(x, y + 1)))
                elif periodic:
                    edges.append((current, node_id(x, 0)))
        
        return edges
    
    @staticmethod
    def create_erdos_renyi_graph(n_nodes: int, p_edge: float) -> List[Tuple[NodeID, NodeID]]:
        """
        Create Erdős-Rényi random graph.
        
        Time complexity: O(n²)
        Each possible edge exists with probability p_edge.
        
        Args:
            n_nodes: Number of nodes
            p_edge: Probability of edge existence (0 to 1)
            
        Returns:
            List of edge tuples
        """
        edges = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < p_edge:
                    edges.append((i, j))
        
        return edges
    
    @staticmethod
    def create_watts_strogatz_graph(
        n_nodes: int, 
        k_neighbors: int, 
        p_rewire: float
    ) -> List[Tuple[NodeID, NodeID]]:
        """
        Create Watts-Strogatz small-world network.
        
        Time complexity: O(n * k)
        Starts with ring lattice, then rewires edges with probability p_rewire.
        
        Args:
            n_nodes: Number of nodes
            k_neighbors: Number of neighbors in initial ring (must be even)
            p_rewire: Probability of rewiring each edge (0 to 1)
            
        Returns:
            List of edge tuples
        """
        # Start with ring lattice
        edges = NetworkTopology.create_ring_lattice(n_nodes, k_neighbors)
        
        # Rewire edges
        rewired_edges = []
        edge_set = set()
        
        for i, j in edges:
            if random.random() < p_rewire:
                # Rewire edge
                new_target = random.randint(0, n_nodes - 1)
                
                # Avoid self-loops and duplicate edges
                attempts = 0
                while (new_target == i or (i, new_target) in edge_set or 
                       (new_target, i) in edge_set) and attempts < 100:
                    new_target = random.randint(0, n_nodes - 1)
                    attempts += 1
                
                if attempts < 100:
                    rewired_edges.append((i, new_target))
                    edge_set.add((i, new_target))
                else:
                    # Keep original edge if can't find valid rewiring
                    rewired_edges.append((i, j))
                    edge_set.add((i, j))
            else:
                # Keep original edge
                rewired_edges.append((i, j))
                edge_set.add((i, j))
        
        return rewired_edges
    
    @staticmethod
    def create_barabasi_albert_graph(n_nodes: int, m_edges: int) -> List[Tuple[NodeID, NodeID]]:
        """
        Create Barabási-Albert scale-free network using preferential attachment.
        
        Time complexity: O(n * m)
        Generates power-law degree distribution.
        
        Args:
            n_nodes: Number of nodes
            m_edges: Number of edges to attach from each new node
            
        Returns:
            List of edge tuples
        """
        if m_edges >= n_nodes:
            raise ValueError("m_edges must be less than n_nodes")
        
        edges = []
        degree = [0] * n_nodes
        
        # Start with complete graph of m+1 nodes
        for i in range(m_edges + 1):
            for j in range(i + 1, m_edges + 1):
                edges.append((i, j))
                degree[i] += 1
                degree[j] += 1
        
        # Add remaining nodes with preferential attachment
        for new_node in range(m_edges + 1, n_nodes):
            # Calculate attachment probabilities
            total_degree = sum(degree)
            probabilities = [d / total_degree for d in degree[:new_node]]
            
            # Select m_edges nodes to connect to
            targets = set()
            while len(targets) < m_edges:
                # Weighted random selection
                r = random.random()
                cumulative = 0
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        targets.add(i)
                        break
            
            # Add edges
            for target in targets:
                edges.append((new_node, target))
                degree[new_node] += 1
                degree[target] += 1
        
        return edges
    
    @staticmethod
    def create_community_graph(
        communities: List[int],
        p_within: float,
        p_between: float
    ) -> List[Tuple[NodeID, NodeID]]:
        """
        Create network with community structure.
        
        Args:
            communities: List of community sizes
            p_within: Probability of edge within community
            p_between: Probability of edge between communities
            
        Returns:
            List of edge tuples
        """
        edges = []
        node_to_community = {}
        current_node = 0
        
        # Assign nodes to communities
        for comm_id, comm_size in enumerate(communities):
            for _ in range(comm_size):
                node_to_community[current_node] = comm_id
                current_node += 1
        
        total_nodes = sum(communities)
        
        # Generate edges
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                comm_i = node_to_community[i]
                comm_j = node_to_community[j]
                
                if comm_i == comm_j:
                    # Same community
                    if random.random() < p_within:
                        edges.append((i, j))
                else:
                    # Different communities
                    if random.random() < p_between:
                        edges.append((i, j))
        
        return edges
    
    @staticmethod
    def analyze_network_properties(edges: List[Tuple[NodeID, NodeID]]) -> Dict[str, Any]:
        """
        Analyze basic network properties.
        
        Args:
            edges: List of edge tuples
            
        Returns:
            Dictionary of network properties
        """
        if not edges:
            return {
                'node_count': 0,
                'edge_count': 0,
                'density': 0.0,
                'average_degree': 0.0
            }
        
        # Build adjacency list
        adjacency = {}
        nodes = set()
        
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)
            
            if i not in adjacency:
                adjacency[i] = set()
            if j not in adjacency:
                adjacency[j] = set()
            
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        node_count = len(nodes)
        edge_count = len(edges)
        
        # Calculate properties
        density = 2 * edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
        average_degree = 2 * edge_count / node_count if node_count > 0 else 0
        
        # Degree distribution
        degrees = [len(adjacency.get(node, set())) for node in nodes]
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'density': density,
            'average_degree': average_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'degree_distribution': degrees
        }
    
    @staticmethod
    def calculate_clustering_coefficient(edges: List[Tuple[NodeID, NodeID]]) -> float:
        """
        Calculate global clustering coefficient.
        
        Time complexity: O(V * d²) where d = average degree
        """
        if not edges:
            return 0.0
        
        # Build adjacency list
        adjacency = {}
        nodes = set()
        
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)
            
            if i not in adjacency:
                adjacency[i] = set()
            if j not in adjacency:
                adjacency[j] = set()
            
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        total_clustering = 0
        valid_nodes = 0
        
        for node in nodes:
            neighbors = adjacency.get(node, set())
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2 and n2 in adjacency.get(n1, set()):
                        triangles += 1
            
            if possible_triangles > 0:
                total_clustering += triangles / possible_triangles
                valid_nodes += 1
        
        return total_clustering / valid_nodes if valid_nodes > 0 else 0.0
    
    @staticmethod
    def calculate_average_path_length(edges: List[Tuple[NodeID, NodeID]]) -> float:
        """
        Calculate average shortest path length using BFS.
        
        Time complexity: O(V * (V + E))
        """
        if not edges:
            return 0.0
        
        # Build adjacency list
        adjacency = {}
        nodes = set()
        
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)
            
            if i not in adjacency:
                adjacency[i] = set()
            if j not in adjacency:
                adjacency[j] = set()
            
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        total_path_length = 0
        path_count = 0
        
        for start_node in nodes:
            # BFS from start_node
            distances = {start_node: 0}
            queue = deque([start_node])
            
            while queue:
                current = queue.popleft()
                current_distance = distances[current]
                
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in distances:
                        distances[neighbor] = current_distance + 1
                        queue.append(neighbor)
            
            # Add distances to total
            for distance in distances.values():
                if distance > 0:  # Exclude self-distance
                    total_path_length += distance
                    path_count += 1
        
        return total_path_length / path_count if path_count > 0 else float('inf')
    
    @staticmethod
    def find_connected_components(edges: List[Tuple[NodeID, NodeID]]) -> List[List[NodeID]]:
        """
        Find connected components using DFS.
        
        Time complexity: O(V + E)
        """
        if not edges:
            return []
        
        # Build adjacency list
        adjacency = {}
        nodes = set()
        
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)
            
            if i not in adjacency:
                adjacency[i] = set()
            if j not in adjacency:
                adjacency[j] = set()
            
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        visited = set()
        components = []
        
        def dfs(node: NodeID, component: List[NodeID]) -> None:
            visited.add(node)
            component.append(node)
            
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in nodes:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    @staticmethod
    def add_edge_weights(
        edges: List[Tuple[NodeID, NodeID]], 
        weight_distribution: str = "uniform",
        weight_range: Tuple[float, float] = (0.1, 1.0)
    ) -> List[Tuple[NodeID, NodeID, float]]:
        """
        Add weights to edges.
        
        Args:
            edges: List of edge tuples
            weight_distribution: "uniform", "normal", or "exponential"
            weight_range: (min_weight, max_weight)
            
        Returns:
            List of (node1, node2, weight) tuples
        """
        weighted_edges = []
        min_weight, max_weight = weight_range
        
        for i, j in edges:
            if weight_distribution == "uniform":
                weight = random.uniform(min_weight, max_weight)
            elif weight_distribution == "normal":
                mean = (min_weight + max_weight) / 2
                std = (max_weight - min_weight) / 6  # 99.7% within range
                weight = max(min_weight, min(max_weight, random.gauss(mean, std)))
            elif weight_distribution == "exponential":
                # Exponential with lambda = 1, then scale to range
                weight = random.expovariate(1.0)
                weight = min_weight + (weight / 5.0) * (max_weight - min_weight)
                weight = min(max_weight, weight)
            else:
                weight = 1.0  # Default weight
            
            weighted_edges.append((i, j, weight))
        
        return weighted_edges
