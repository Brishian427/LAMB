"""
Grid space utilities and neighborhood functions.

Based on Technical_Specification.md Section 2.1: Grid Paradigm Specification.
Provides utilities for Moore and Von Neumann neighborhoods, distance calculations,
and grid topology operations.

Performance characteristics:
- Moore neighborhood: O(r²) where r = radius
- Von Neumann neighborhood: O(4r) where r = radius  
- Distance calculations: O(1)
"""

from typing import List, Tuple, Set, Iterator, Optional
import math
from enum import Enum

from ...core.types import GridCoord, BoundaryCondition


class NeighborhoodType(Enum):
    """Types of grid neighborhoods"""
    MOORE = "moore"          # 8-connected (includes diagonals)
    VON_NEUMANN = "von_neumann"  # 4-connected (no diagonals)


class GridSpace:
    """
    Utilities for grid space operations and neighborhood calculations.
    
    Provides efficient implementations of common grid operations:
    - Neighborhood generation (Moore, Von Neumann)
    - Distance calculations (Manhattan, Euclidean, Chebyshev)
    - Path finding utilities
    - Boundary condition handling
    """
    
    @staticmethod
    def get_moore_neighborhood(
        center: GridCoord, 
        radius: int, 
        dimensions: Optional[Tuple[int, int]] = None,
        boundary_condition: BoundaryCondition = BoundaryCondition.INFINITE
    ) -> List[GridCoord]:
        """
        Get Moore neighborhood (8-connected) around center position.
        
        Performance: O(r²) where r = radius
        
        Args:
            center: Center position (x, y)
            radius: Neighborhood radius
            dimensions: Grid dimensions for boundary checking
            boundary_condition: How to handle boundaries
            
        Returns:
            List of positions in Moore neighborhood
        """
        x, y = center
        neighbors = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip center
                
                neighbor_pos = (x + dx, y + dy)
                
                # Apply boundary condition
                actual_pos = GridSpace._apply_boundary_condition(
                    neighbor_pos, dimensions, boundary_condition
                )
                
                if actual_pos is not None:
                    neighbors.append(actual_pos)
        
        return neighbors
    
    @staticmethod
    def get_von_neumann_neighborhood(
        center: GridCoord,
        radius: int,
        dimensions: Optional[Tuple[int, int]] = None,
        boundary_condition: BoundaryCondition = BoundaryCondition.INFINITE
    ) -> List[GridCoord]:
        """
        Get Von Neumann neighborhood (4-connected) around center position.
        
        Performance: O(4r) where r = radius (~25% faster than Moore)
        
        Args:
            center: Center position (x, y)
            radius: Neighborhood radius
            dimensions: Grid dimensions for boundary checking
            boundary_condition: How to handle boundaries
            
        Returns:
            List of positions in Von Neumann neighborhood
        """
        x, y = center
        neighbors = []
        
        for distance in range(1, radius + 1):
            for dx in range(-distance, distance + 1):
                dy = distance - abs(dx)  # Manhattan distance constraint
                
                for dy_sign in ([-1, 1] if dy != 0 else [0]):
                    neighbor_pos = (x + dx, y + dy * dy_sign)
                    
                    # Apply boundary condition
                    actual_pos = GridSpace._apply_boundary_condition(
                        neighbor_pos, dimensions, boundary_condition
                    )
                    
                    if actual_pos is not None:
                        neighbors.append(actual_pos)
        
        return neighbors
    
    @staticmethod
    def get_neighborhood(
        center: GridCoord,
        radius: int,
        neighborhood_type: NeighborhoodType = NeighborhoodType.MOORE,
        dimensions: Optional[Tuple[int, int]] = None,
        boundary_condition: BoundaryCondition = BoundaryCondition.INFINITE
    ) -> List[GridCoord]:
        """
        Get neighborhood of specified type around center position.
        
        Args:
            center: Center position
            radius: Neighborhood radius
            neighborhood_type: Type of neighborhood (Moore or Von Neumann)
            dimensions: Grid dimensions
            boundary_condition: Boundary handling
            
        Returns:
            List of neighbor positions
        """
        if neighborhood_type == NeighborhoodType.MOORE:
            return GridSpace.get_moore_neighborhood(
                center, radius, dimensions, boundary_condition
            )
        elif neighborhood_type == NeighborhoodType.VON_NEUMANN:
            return GridSpace.get_von_neumann_neighborhood(
                center, radius, dimensions, boundary_condition
            )
        else:
            raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")
    
    @staticmethod
    def manhattan_distance(pos1: GridCoord, pos2: GridCoord) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Performance: O(1)
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def euclidean_distance(pos1: GridCoord, pos2: GridCoord) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Performance: O(1)
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def chebyshev_distance(pos1: GridCoord, pos2: GridCoord) -> int:
        """
        Calculate Chebyshev distance (max of x,y differences).
        
        Performance: O(1)
        Useful for Moore neighborhood distance calculations.
        """
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
    
    @staticmethod
    def get_positions_within_distance(
        center: GridCoord,
        max_distance: float,
        distance_type: str = "euclidean",
        dimensions: Optional[Tuple[int, int]] = None,
        boundary_condition: BoundaryCondition = BoundaryCondition.INFINITE
    ) -> List[GridCoord]:
        """
        Get all positions within specified distance of center.
        
        Args:
            center: Center position
            max_distance: Maximum distance
            distance_type: Type of distance ("manhattan", "euclidean", "chebyshev")
            dimensions: Grid dimensions
            boundary_condition: Boundary handling
            
        Returns:
            List of positions within distance
        """
        positions = []
        
        # Use appropriate radius based on distance type
        if distance_type == "manhattan":
            radius = int(max_distance)
        elif distance_type == "chebyshev":
            radius = int(max_distance)
        else:  # euclidean
            radius = int(math.ceil(max_distance))
        
        # Get candidate positions using Moore neighborhood
        candidates = GridSpace.get_moore_neighborhood(
            center, radius, dimensions, boundary_condition
        )
        
        # Filter by actual distance
        distance_func = {
            "manhattan": GridSpace.manhattan_distance,
            "euclidean": GridSpace.euclidean_distance,
            "chebyshev": GridSpace.chebyshev_distance
        }.get(distance_type, GridSpace.euclidean_distance)
        
        for pos in candidates:
            if distance_func(center, pos) <= max_distance:
                positions.append(pos)
        
        return positions
    
    @staticmethod
    def get_line_positions(start: GridCoord, end: GridCoord) -> List[GridCoord]:
        """
        Get positions along line from start to end using Bresenham's algorithm.
        
        Useful for line-of-sight calculations and pathfinding.
        """
        positions = []
        
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        
        x, y = x0, y0
        
        while True:
            positions.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
            
            if error2 < dx:
                error += dx
                y += y_step
        
        return positions
    
    @staticmethod
    def get_adjacent_positions(
        position: GridCoord,
        neighborhood_type: NeighborhoodType = NeighborhoodType.MOORE
    ) -> List[GridCoord]:
        """
        Get immediately adjacent positions (radius=1).
        
        Args:
            position: Center position
            neighborhood_type: Type of neighborhood
            
        Returns:
            List of adjacent positions
        """
        return GridSpace.get_neighborhood(position, 1, neighborhood_type)
    
    @staticmethod
    def is_adjacent(pos1: GridCoord, pos2: GridCoord, neighborhood_type: NeighborhoodType = NeighborhoodType.MOORE) -> bool:
        """
        Check if two positions are adjacent.
        
        Args:
            pos1: First position
            pos2: Second position
            neighborhood_type: Type of adjacency to check
            
        Returns:
            True if positions are adjacent
        """
        if neighborhood_type == NeighborhoodType.MOORE:
            return GridSpace.chebyshev_distance(pos1, pos2) == 1
        elif neighborhood_type == NeighborhoodType.VON_NEUMANN:
            return GridSpace.manhattan_distance(pos1, pos2) == 1
        else:
            return False
    
    @staticmethod
    def _apply_boundary_condition(
        position: GridCoord,
        dimensions: Optional[Tuple[int, int]],
        boundary_condition: BoundaryCondition
    ) -> Optional[GridCoord]:
        """Apply boundary condition to position"""
        if dimensions is None:
            return position  # No boundary constraints
        
        x, y = position
        width, height = dimensions
        
        if boundary_condition == BoundaryCondition.WRAP:
            # Toroidal topology - wrap around edges
            wrapped_x = x % width
            wrapped_y = y % height
            return (wrapped_x, wrapped_y)
        
        elif boundary_condition == BoundaryCondition.WALL:
            # Reflective boundaries - reject out-of-bounds positions
            if x < 0 or x >= width or y < 0 or y >= height:
                return None  # Out of bounds
            return (x, y)
        
        elif boundary_condition == BoundaryCondition.INFINITE:
            # Infinite grid - all coordinates valid
            return (x, y)
        
        else:
            return (x, y)
    
    @staticmethod
    def get_grid_bounds(positions: List[GridCoord]) -> Tuple[GridCoord, GridCoord]:
        """
        Get bounding box of a list of positions.
        
        Args:
            positions: List of grid positions
            
        Returns:
            Tuple of (min_position, max_position)
        """
        if not positions:
            return ((0, 0), (0, 0))
        
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        return ((min_x, min_y), (max_x, max_y))
    
    @staticmethod
    def positions_in_rectangle(
        top_left: GridCoord,
        bottom_right: GridCoord
    ) -> List[GridCoord]:
        """
        Get all positions within a rectangular region.
        
        Args:
            top_left: Top-left corner of rectangle
            bottom_right: Bottom-right corner of rectangle
            
        Returns:
            List of positions in rectangle
        """
        positions = []
        
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                positions.append((x, y))
        
        return positions
    
    @staticmethod
    def get_direction_vector(from_pos: GridCoord, to_pos: GridCoord) -> Tuple[int, int]:
        """
        Get unit direction vector from one position to another.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            Direction vector (-1, 0, or 1 for each axis)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Normalize to unit vector
        unit_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        unit_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        
        return (unit_x, unit_y)
    
    @staticmethod
    def rotate_position(
        position: GridCoord,
        center: GridCoord,
        angle_degrees: int
    ) -> GridCoord:
        """
        Rotate position around center by specified angle.
        
        Args:
            position: Position to rotate
            center: Center of rotation
            angle_degrees: Rotation angle (90, 180, 270 degrees supported)
            
        Returns:
            Rotated position
        """
        # Translate to origin
        x = position[0] - center[0]
        y = position[1] - center[1]
        
        # Rotate
        if angle_degrees == 90:
            new_x, new_y = -y, x
        elif angle_degrees == 180:
            new_x, new_y = -x, -y
        elif angle_degrees == 270:
            new_x, new_y = y, -x
        else:
            new_x, new_y = x, y  # No rotation for other angles
        
        # Translate back
        return (new_x + center[0], new_y + center[1])
