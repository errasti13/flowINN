import numpy as np
from typing import Optional, List, Tuple
from scipy.spatial import Delaunay

class Sampler:
    @staticmethod
    def _sampleRandomlyWithinBoundary(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                     z_boundary: Optional[np.ndarray], Nx: int, Ny: int,
                                     Nz: Optional[int]) -> None:
        """
        Samples points randomly within the defined boundary.

        Args:
            x_boundary (np.ndarray): x-coordinates of the boundary.
            y_boundary (np.ndarray): y-coordinates of the boundary.
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.

        Raises:
            ValueError: If boundary coordinates contain NaN values.
        """
        if mesh.is2D:
            Sampler._sampleRandom2D(mesh, x_boundary, y_boundary, Nx, Ny)
        else:
            Sampler._sampleRandom3D(mesh, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)


    @staticmethod
    def _sampleRandom2D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                         Nx: int, Ny: int) -> None:
        
        try:
            x_boundary = np.asarray(x_boundary, dtype=np.float32)
            y_boundary = np.asarray(y_boundary, dtype=np.float32)

            if np.any(np.isnan(x_boundary)) or np.any(np.isnan(y_boundary)):
                raise ValueError("Boundary coordinates contain NaN values")

            Nt = Nx * Ny

            samples: List[np.ndarray] = []
            while len(samples) < Nt:
                x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
                y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)

                points = np.column_stack((x_rand, y_rand))

                valid_points = Sampler._check_points_in_domain(mesh, points, x_boundary, y_boundary)
                samples.extend(valid_points)

            samples = np.array(samples)[:Nt]

            mesh._x = samples[:, 0].reshape(Nx, Ny)
            mesh._y = samples[:, 1].reshape(Nx, Ny)

        except Exception as e:
            print(f"Debug: Error during random sampling: {str(e)}")
            raise
        
    @staticmethod
    def _sampleRandom3D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                      z_boundary: np.ndarray, Nx: int, Ny: int,
                                      Nz: int) -> None:
        try:
            x_boundary = np.asarray(x_boundary, dtype=np.float32)
            y_boundary = np.asarray(y_boundary, dtype=np.float32)
            z_boundary = np.asarray(z_boundary, dtype=np.float32)

            if np.any(np.isnan(x_boundary)) or np.any(np.isnan(y_boundary)) or \
                (np.any(np.isnan(z_boundary))):
                raise ValueError("Boundary coordinates contain NaN values")

            Nt = Nx * Ny * Nz

            samples: List[np.ndarray] = []
            while len(samples) < Nt:
                x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
                y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)
                z_rand = np.random.uniform(np.min(z_boundary), np.max(z_boundary), size=Nt)
                points = np.column_stack((x_rand, y_rand, z_rand))

                samples.extend(points)

            samples = np.array(samples)[:Nt]

            mesh._x = samples[:, 0].reshape(Nx, Ny, Nz)
            mesh._y = samples[:, 1].reshape(Nx, Ny, Nz)
            mesh._z = samples[:, 2].reshape(Nx, Ny, Nz)

        except Exception as e:
            print(f"Debug: Error during random sampling: {str(e)}")
            raise

    @staticmethod
    def _sampleUniformlyWithinBoundary(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                      z_boundary: Optional[np.ndarray], Nx: int, Ny: int,
                                      Nz: Optional[int]) -> None:
        """
        Samples points uniformly within the defined boundary.

        Args:
            x_boundary (np.ndarray): x-coordinates of the boundary.
            y_boundary (np.ndarray): y-coordinates of the boundary.
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.
        """
        if mesh.is2D:
            Sampler._sampleUniform2D(mesh, x_boundary, y_boundary, Nx, Ny)
        else:
            Sampler._sampleUniform3D(mesh, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)

    @staticmethod
    def _sampleUniform2D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                    Nx: int, Ny: int) -> None:
        
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)
        # Create uniform grid
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, Nx),
            np.linspace(y_min, y_max, Ny)
        )
        
        # Stack coordinates for testing
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        
        # Use point-in-polygon test to filter points
        valid_points = Sampler._check_points_in_domain(mesh, grid_points, x_boundary, y_boundary)
        
        mesh._x, mesh._y = valid_points[:, 0].astype(np.float32), valid_points[:, 1].astype(np.float32)
    
        
    @staticmethod
    def _sampleUniform3D(mesh: 'Mesh', x_boundary: np.ndarray, y_boundary: np.ndarray,
                                      z_boundary: np.ndarray, Nx: int, Ny: int,
                                      Nz: int) -> None:
        """
        Samples points uniformly within the defined boundary.

        Args:
            x_boundary (np.ndarray): x-coordinates of the boundary.
            y_boundary (np.ndarray): y-coordinates of the boundary.
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.
        """
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)
        z_min, z_max = (np.min(z_boundary), np.max(z_boundary)) if z_boundary is not None else (None, None)

        x_grid, y_grid, z_grid = np.meshgrid(
            np.linspace(x_min, x_max, Nx),
            np.linspace(y_min, y_max, Ny),
            np.linspace(z_min, z_max, Nz) if z_boundary is not None else [0]
        )
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())) if not mesh.is2D else np.column_stack((x_grid.flatten(), y_grid.flatten()))

        points = np.column_stack((x_boundary, y_boundary, z_boundary)) if not mesh.is2D else np.column_stack((x_boundary, y_boundary))
        triangulation = Delaunay(points)

        inside = triangulation.find_simplex(grid_points) >= 0
        inside_points = grid_points[inside]

        if mesh._interiorBoundaries:
            for boundary_data in mesh._interiorBoundaries.values():
                x_int = boundary_data['x'].flatten()
                y_int = boundary_data['y'].flatten()
                if not mesh.is2D:
                    z_int = boundary_data['z'].flatten()
                    interior_points = np.column_stack((x_int, y_int, z_int))
                else:
                    interior_points = np.column_stack((x_int, y_int))

                interior_tri = Delaunay(interior_points)

                inside_interior = interior_tri.find_simplex(inside_points) >= 0

                inside_points = inside_points[~inside_interior]

        mesh._x, mesh._y, mesh._z = inside_points[:, 0].astype(np.float32), inside_points[:, 1].astype(np.float32), inside_points[:, 2].astype(np.float32)

    @staticmethod
    def _check_points_in_domain(mesh: 'Mesh', points: np.ndarray) -> np.ndarray:
        """
        Checks if points are inside the domain and outside interior boundaries.
        
        Args:
            points (np.ndarray): Points to check
            x_boundary (np.ndarray): x-coordinates of the boundary
            y_boundary (np.ndarray): y-coordinates of the boundary
            z_boundary (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes)
            
        Returns:
            np.ndarray: Valid points inside the domain
        """
        try:
            polygon = Sampler._get_ordered_polygon(mesh)
            
            valid_mask = np.array([
                Sampler._is_point_inside((point[0], point[1]), polygon)
                for point in points
            ])
            
            valid_points = points[valid_mask]

            if mesh._interiorBoundaries:
                for boundary_data in mesh._interiorBoundaries.values():
                    x_int = boundary_data['x']
                    y_int = boundary_data['y']
                    interior_polygon = list(zip(x_int[:-1].astype(float), y_int[:-1].astype(float)))
                    
                    # Vectorized interior check
                    interior_mask = np.array([
                        not Sampler._is_point_inside((x, y), interior_polygon)
                        for x, y in valid_points[:, :2]
                    ])
                    valid_points = valid_points[interior_mask]

            return valid_points

        except Exception as e:
            print(f"Debug: Error checking points in domain: {str(e)}")
            raise

    @staticmethod
    def _get_ordered_polygon(mesh: 'Mesh') -> List[Tuple[float, float]]:
        """
        Reconstructs the ordered list of vertices from boundary segments.
        Returns a list of vertices (points) in order.
        """

        # Extract segments using NumPy
        segments = []
        for boundary_data in mesh.boundaries.values():
            coords = np.column_stack((boundary_data["x"], boundary_data["y"]))
            segments.extend(zip(coords[:-1], coords[1:]))

        # Convert tuples to float tuples
        segments = [((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))) for p1, p2 in segments]

        # Build adjacency dictionary efficiently
        adj = {}
        for p1, p2 in segments:
            adj.setdefault(p1, set()).add(p2)
            adj.setdefault(p2, set()).add(p1)

        # Construct ordered polygon
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
    def _is_point_inside(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Vectorized implementation of ray-casting algorithm to check if point is inside polygon.
        Handles vertical edges and division by zero cases.
        """
        x, y = point
        polygon = np.array(polygon)
        
        # Get current and next vertices (with wrap-around)
        vertices = polygon
        next_vertices = np.roll(vertices, -1, axis=0)
        
        # Extract coordinates
        x1, y1 = vertices[:, 0], vertices[:, 1]
        x2, y2 = next_vertices[:, 0], next_vertices[:, 1]
        
        # Handle vertical edges first
        vertical_edges = np.abs(y2 - y1) < 1e-10  # Small threshold for floating-point comparison
        non_vertical = ~vertical_edges
        
        # Initialize intersection mask
        intersect = np.zeros_like(y1, dtype=bool)
        
        # Handle non-vertical edges
        if np.any(non_vertical):
            y_cond = (y1[non_vertical] > y) != (y2[non_vertical] > y)
            slope = (x2[non_vertical] - x1[non_vertical]) / (y2[non_vertical] - y1[non_vertical])
            x_intersect = x1[non_vertical] + slope * (y - y1[non_vertical])
            intersect[non_vertical] = y_cond & (x < x_intersect)
        
        # Handle vertical edges
        if np.any(vertical_edges):
            # For vertical edges, check if point's x is to the left of the edge
            # and y is between the edge endpoints
            y_between = ((y >= np.minimum(y1[vertical_edges], y2[vertical_edges])) & 
                        (y <= np.maximum(y1[vertical_edges], y2[vertical_edges])))
            intersect[vertical_edges] = (x < x1[vertical_edges]) & y_between
        
        # Return true if number of intersections is odd
        return np.sum(intersect) % 2 == 1