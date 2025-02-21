import numpy as np
from typing import List, Tuple, Dict, Any
from numpy.typing import NDArray

class GeometryUtils:
    @staticmethod
    def get_ordered_polygon(boundaries: Dict[str, Dict[str, Any]]) -> List[Tuple[float, float]]:
        """
        Reconstructs the ordered list of vertices from boundary segments.
        Returns a list of vertices (points) in order.
        """
        segments = []
        for boundary_data in boundaries.values():
            coords = np.column_stack((boundary_data["x"], boundary_data["y"]))
            segments.extend(zip(coords[:-1], coords[1:]))

        segments = [((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))) 
                   for p1, p2 in segments]

        adj = {}
        for p1, p2 in segments:
            adj.setdefault(p1, set()).add(p2)
            adj.setdefault(p2, set()).add(p1)

        start = next(iter(adj))
        polygon = [start]
        prev = None
        current = start
        
        while True:
            neighbors = list(adj[current])
            if prev is not None and prev in neighbors:
                neighbors.remove(prev)
            if not neighbors:
                break
            next_pt = neighbors[0]
            if next_pt == start:
                break
            polygon.append(next_pt)
            prev, current = current, next_pt
        
        return polygon

    @staticmethod
    def is_point_inside(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Vectorized implementation of ray-casting algorithm to check if point is inside polygon.
        Handles vertical edges and division by zero cases.
        """
        x, y = point
        polygon = np.array(polygon)
        
        vertices = polygon
        next_vertices = np.roll(vertices, -1, axis=0)
        
        x1, y1 = vertices[:, 0], vertices[:, 1]
        x2, y2 = next_vertices[:, 0], next_vertices[:, 1]
        
        vertical_edges = np.abs(y2 - y1) < 1e-10
        non_vertical = ~vertical_edges
        
        intersect = np.zeros_like(y1, dtype=bool)
        
        if np.any(non_vertical):
            y_cond = (y1[non_vertical] > y) != (y2[non_vertical] > y)
            slope = (x2[non_vertical] - x1[non_vertical]) / (y2[non_vertical] - y1[non_vertical])
            x_intersect = x1[non_vertical] + slope * (y - y1[non_vertical])
            intersect[non_vertical] = y_cond & (x < x_intersect)
        
        if np.any(vertical_edges):
            y_between = ((y >= np.minimum(y1[vertical_edges], y2[vertical_edges])) & 
                        (y <= np.maximum(y1[vertical_edges], y2[vertical_edges])))
            intersect[vertical_edges] = (x < x1[vertical_edges]) & y_between
        
        return np.sum(intersect) % 2 == 1

    @staticmethod
    def check_points_in_domain(points: NDArray[np.float32], 
                             boundaries: Dict[str, Dict[str, Any]], 
                             interior_boundaries: Dict[str, Dict[str, Any]]) -> NDArray[np.float32]:
        """Check if points are inside domain and outside interior boundaries."""
        try:
            polygon = GeometryUtils.get_ordered_polygon(boundaries)
            
            valid_mask = np.array([
                GeometryUtils.is_point_inside((point[0], point[1]), polygon)
                for point in points
            ])
            
            valid_points = points[valid_mask]

            if interior_boundaries:
                for boundary_data in interior_boundaries.values():
                    x_int = boundary_data['x']
                    y_int = boundary_data['y']
                    interior_polygon = list(zip(x_int[:-1].astype(float), y_int[:-1].astype(float)))
                    
                    interior_mask = np.array([
                        not GeometryUtils.is_point_inside((x, y), interior_polygon)
                        for x, y in valid_points[:, :2]
                    ])
                    valid_points = valid_points[interior_mask]

            return valid_points

        except Exception as e:
            print(f"Debug: Error checking points in domain: {str(e)}")
            raise
