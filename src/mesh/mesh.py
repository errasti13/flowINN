import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, Tuple, List
from src.mesh.meshio import MeshIO


class Mesh:
    """
    A class for generating and managing computational meshes.

    Attributes:
        is2D (bool): Flag indicating whether the mesh is 2D or 3D.
        meshio (MeshIO): Instance of MeshIO for handling mesh I/O operations.
    """

    def __init__(self, is2D: bool = True) -> None:
        """
        Initializes a new Mesh object.

        Args:
            is2D (bool): Flag indicating whether the mesh is 2D or 3D. Defaults to True.
        """
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None
        self._solutions: Dict[str, np.ndarray] = {}
        self._boundaries: Dict[str, Dict[str, np.ndarray]] = {}
        self._interiorBoundaries: Dict[str, Dict[str, np.ndarray]] = {}
        self._is2D: bool = is2D
        self.meshio: Optional[MeshIO] = None

    def _create_meshio(self) -> None:
        """
        Creates a MeshIO instance if it doesn't exist.
        """
        if self.meshio is None:
            self.meshio = MeshIO(self)

    @property
    def x(self) -> Optional[np.ndarray]:
        """
        Returns the x-coordinates of the mesh points.
        """
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        """
        Sets the x-coordinates of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the x-coordinates.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("x must be a numpy array")
        self._x = value

    @property
    def y(self) -> Optional[np.ndarray]:
        """
        Returns the y-coordinates of the mesh points.
        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        """
        Sets the y-coordinates of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the y-coordinates.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("y must be a numpy array")
        self._y = value

    @property
    def z(self) -> Optional[np.ndarray]:
        """
        Returns the z-coordinates of the mesh points.
        """
        return self._z

    @z.setter
    def z(self, value: np.ndarray) -> None:
        """
        Sets the z-coordinates of the mesh points.

        Args:
            value (np.ndarray): A numpy array containing the z-coordinates.

        Raises:
            TypeError: If value is not a numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("z must be a numpy array")
        self._z = value

    @property
    def solutions(self) -> Dict[str, np.ndarray]:
        """
        Returns the solutions dictionary.
        """
        return self._solutions

    @solutions.setter
    def solutions(self, value: Dict[str, np.ndarray]) -> None:
        """
        Sets the solutions dictionary.

        Args:
            value (dict): A dictionary containing solution data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._solutions = value

    @property
    def boundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the boundaries dictionary.
        """
        return self._boundaries

    @boundaries.setter
    def boundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the boundaries dictionary.

        Args:
            value (dict): A dictionary containing boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("boundaries must be a dictionary")
        self._boundaries = value

    @property
    def interiorBoundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the interior boundaries dictionary.
        """
        return self._interiorBoundaries

    @interiorBoundaries.setter
    def interiorBoundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the interior boundaries dictionary.

        Args:
            value (dict): A dictionary containing interior boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("interiorBoundaries must be a dictionary")
        self._interiorBoundaries = value

    @property
    def is2D(self) -> bool:
        """
        Returns the is2D flag.
        """
        return self._is2D

    @is2D.setter
    def is2D(self, value: bool) -> None:
        """
        Sets the is2D flag.

        Args:
            value (bool): A boolean value indicating if the mesh is 2D.

        Raises:
            TypeError: If value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("is2D must be a boolean")
        self._is2D = value

    def generateMesh(self, Nx: int = 100, Ny: int = 100, Nz: Optional[int] = None, sampling_method: str = 'random') -> None:
        """
        Generates a mesh within a domain defined by boundary data.

        Args:
            Nx (int): Number of points in the x-dimension for structured sampling. Defaults to 100.
            Ny (int): Number of points in the y-dimension for structured sampling. Defaults to 100.
            Nz (Optional[int]): Number of points in the z-dimension for structured sampling. Defaults to None.
            sampling_method (str): Sampling method ('random', 'uniform'). Defaults to 'random'.

        Raises:
            ValueError: If input parameters are invalid or mesh generation fails.
        """
        if not self.boundaries:
            raise ValueError("No boundaries defined. Use setBoundary() to define boundaries before generating mesh")

        try:
            self._generateMeshFromBoundary(sampling_method, Nx, Ny, Nz)
        except Exception as e:
            raise ValueError(f"Mesh generation failed: {str(e)}")

    def _generateMeshFromBoundary(self, sampling_method: str, Nx: int, Ny: int, Nz: Optional[int]) -> None:
        """
        Generates the mesh from the defined boundaries using the specified sampling method.

        Args:
            sampling_method (str): Sampling method ('random', 'uniform').
            Nx (int): Number of points in the x-dimension.
            Ny (int): Number of points in the y-dimension.
            Nz (Optional[int]): Number of points in the z-dimension.

        Raises:
            ValueError: If boundary data is invalid or sampling method is unsupported.
        """
        for boundary_name, boundary_data in self.boundaries.items():
            if 'x' not in boundary_data or 'y' not in boundary_data:
                raise ValueError(f"Boundary '{boundary_name}' must contain 'x' and 'y' coordinates.")
            if not self.is2D and 'z' not in boundary_data:
                raise ValueError(f"3D mesh requires z coordinate for boundary {boundary_name}")

        try:
            x_boundary = np.concatenate([np.asarray(boundary_data['x'], dtype=np.float32).flatten()
                                         for boundary_data in self.boundaries.values()])
            y_boundary = np.concatenate([np.asarray(boundary_data['y'], dtype=np.float32).flatten()
                                         for boundary_data in self.boundaries.values()])

            if not self.is2D:
                z_boundary = np.concatenate([np.asarray(boundary_data['z'], dtype=np.float32).flatten()
                                             for boundary_data in self.boundaries.values()])
            else:
                z_boundary = None

        except Exception as e:
            print(f"Debug: Error during boundary concatenation: {str(e)}")
            raise

        if sampling_method == 'random':
            self._sampleRandomlyWithinBoundary(x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)
        elif sampling_method == 'uniform':
            self._sampleUniformlyWithinBoundary(x_boundary, y_boundary, z_boundary, Nx, Ny, Nz)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")
        
        self.check_closed_curve()

    def _sampleRandomlyWithinBoundary(self, x_boundary: np.ndarray, y_boundary: np.ndarray,
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
        try:
            x_boundary = np.asarray(x_boundary, dtype=np.float32)
            y_boundary = np.asarray(y_boundary, dtype=np.float32)
            if z_boundary is not None:
                z_boundary = np.asarray(z_boundary, dtype=np.float32)

            if np.any(np.isnan(x_boundary)) or np.any(np.isnan(y_boundary)) or \
               (z_boundary is not None and np.any(np.isnan(z_boundary))):
                raise ValueError("Boundary coordinates contain NaN values")

            Nt = Nx * Ny * (Nz if not self.is2D and Nz is not None else 1)

            samples: List[np.ndarray] = []
            while len(samples) < Nt:
                x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
                y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)

                if not self.is2D and z_boundary is not None:
                    z_rand = np.random.uniform(np.min(z_boundary), np.max(z_boundary), size=Nt)
                    points = np.column_stack((x_rand, y_rand, z_rand))
                else:
                    points = np.column_stack((x_rand, y_rand))


                valid_points = self._check_points_in_domain(points, x_boundary, y_boundary, z_boundary)
                samples.extend(valid_points)

            samples = np.array(samples)[:Nt]
            if not self.is2D:
                self._x = samples[:, 0].reshape(Nx, Ny, Nz)
                self._y = samples[:, 1].reshape(Nx, Ny, Nz)
                self._z = samples[:, 2].reshape(Nx, Ny, Nz)
            else:
                self._x = samples[:, 0].reshape(Nx, Ny)
                self._y = samples[:, 1].reshape(Nx, Ny)

        except Exception as e:
            print(f"Debug: Error during random sampling: {str(e)}")
            raise

    def _check_points_in_domain(self, points: np.ndarray, x_boundary: np.ndarray,
                                y_boundary: np.ndarray, z_boundary: Optional[np.ndarray] = None) -> np.ndarray:
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
            # Get ordered polygon representation of the domain
            polygon = self._get_ordered_polygon()
            
            # Check each point using ray-casting
            valid_mask = np.array([
                self._is_point_inside((point[0], point[1]), polygon)
                for point in points
            ])
            
            valid_points = points[valid_mask]

            # Check interior boundaries if they exist
            if self._interiorBoundaries:
                for boundary_data in self._interiorBoundaries.values():
                    interior_polygon = []
                    x_int = boundary_data['x']
                    y_int = boundary_data['y']
                    
                    for i in range(len(x_int) - 1):
                        interior_polygon.append((float(x_int[i]), float(y_int[i])))
                    
                    # Remove points that are inside interior boundaries
                    valid_points = np.array([
                        point for point in valid_points
                        if not self._is_point_inside((point[0], point[1]), interior_polygon)
                    ])

            return valid_points

        except Exception as e:
            print(f"Debug: Error checking points in domain: {str(e)}")
            raise

    def _get_ordered_polygon(self) -> List[Tuple[float, float]]:
        """
        Reconstructs the ordered list of vertices from boundary segments.
        Returns a list of vertices (points) in order.
        """
        # Create segments from boundaries
        segments = []
        for boundary_data in self.boundaries.values():
            x_coords = boundary_data['x']
            y_coords = boundary_data['y']
            for i in range(len(x_coords) - 1):
                pt1 = (float(x_coords[i]), float(y_coords[i]))
                pt2 = (float(x_coords[i + 1]), float(y_coords[i + 1]))
                segments.append((pt1, pt2))
        
        # Build adjacency dictionary
        adj = {}
        for seg in segments:
            p1, p2 = seg
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

    def _is_point_inside(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
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

    def _sampleUniformlyWithinBoundary(self, x_boundary: np.ndarray, y_boundary: np.ndarray,
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
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)
        z_min, z_max = (np.min(z_boundary), np.max(z_boundary)) if z_boundary is not None else (None, None)

        # Create uniform grid
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, Nx),
            np.linspace(y_min, y_max, Ny)
        )
        
        # Stack coordinates for testing
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        
        # Use point-in-polygon test to filter points
        valid_points = self._check_points_in_domain(grid_points, x_boundary, y_boundary)
        
        if self.is2D:
            self._x, self._y = valid_points[:, 0].astype(np.float32), valid_points[:, 1].astype(np.float32)
        else:
            # For 3D, add z coordinates after filtering x,y points
            z_points = np.linspace(z_min, z_max, Nz)
            valid_points_3d = np.array([
                [x, y, z] for x, y in valid_points for z in z_points
            ])
            self._x = valid_points_3d[:, 0].astype(np.float32)
            self._y = valid_points_3d[:, 1].astype(np.float32)
            self._z = valid_points_3d[:, 2].astype(np.float32)

    def setBoundary(self, boundary_name: str, xBc: np.ndarray, yBc: np.ndarray, interior: bool = False,
                    **boundary_conditions: Dict[str, np.ndarray]) -> None:
        """
        Sets multiple boundary conditions at once.

        Args:
            boundary_name (str): Name of the boundary.
            xBc (np.ndarray): x-coordinates of the boundary.
            yBc (np.ndarray): y-coordinates of the boundary.
            interior (bool): Flag indicating if this is an interior boundary. Defaults to False.
            **boundary_conditions (Dict[str, np.ndarray]): Variable names and their values.
        """
        for var_name, values in boundary_conditions.items():
            self.setBoundaryCondition(xBc, yBc, values, var_name, boundary_name, interior=interior)

    def setBoundaryCondition(self, xCoord: np.ndarray, yCoord: np.ndarray, value: np.ndarray, varName: str,
                             boundaryName: str, zCoord: Optional[np.ndarray] = None, interior: bool = False,
                             bc_type: Optional[str] = None) -> None:
        """
        Sets boundary conditions for either exterior or interior boundaries.

        Args:
            xCoord (np.ndarray): x-coordinates of the boundary.
            yCoord (np.ndarray): y-coordinates of the boundary.
            value (np.ndarray): Value of the boundary condition.
            varName (str): Name of the variable.
            boundaryName (str): Name of the boundary.
            zCoord (Optional[np.ndarray]): z-coordinates of the boundary (for 3D meshes).
            interior (bool): Flag indicating if this is an interior boundary. Defaults to False.
            bc_type (Optional[str]): Type of the boundary condition.
        """
        boundary_dict = self._interiorBoundaries if interior else self._boundaries

        if boundaryName not in boundary_dict:
            boundary_dict[boundaryName] = {}

        boundary_dict[boundaryName]['x'] = np.asarray(xCoord, dtype=np.float32)
        boundary_dict[boundaryName]['y'] = np.asarray(yCoord, dtype=np.float32)

        if not self.is2D:
            if zCoord is None:
                raise ValueError(f"z coordinate required for 3D mesh in boundary {boundaryName}")
            boundary_dict[boundaryName]['z'] = np.asarray(zCoord, dtype=np.float32)

        if value is not None:
            boundary_dict[boundaryName][varName] = np.asarray(value, dtype=np.float32)
            boundary_dict[boundaryName][f'{varName}_type'] = bc_type

        boundary_dict[boundaryName]['isInterior'] = interior

    def showMesh(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Visualizes the mesh with proper dimensional scaling.

        Args:
            figsize (Tuple[int, int]): Size of the figure. Defaults to (8, 6).

        Raises:
            ValueError: If the mesh has not been generated yet.
        """
        if self.x is None or self.y is None:
            raise ValueError("Mesh has not been generated yet")

        x_min, x_max = np.min(self.x), np.max(self.x)
        y_min, y_max = np.min(self.y), np.max(self.y)

        if not self.is2D and self.z is not None:
            z_min, z_max = np.min(self.z), np.max(self.z)
            domain_size = f"L×H×W: {x_max-x_min:.1f}×{y_max-y_min:.1f}×{z_max-z_min:.1f}"

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            z_mid = (z_max + z_min) / 2

            ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
            ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
            ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

            scatter = ax.scatter(self.x, self.y, self.z,
                               c='black', s=1, alpha=0.5)

            for boundary_data in self.boundaries.values():
                x_boundary = boundary_data['x']
                y_boundary = boundary_data['y']
                z_boundary = boundary_data['z']
                ax.plot3D(x_boundary, y_boundary, z_boundary,
                         'b-', linewidth=1, alpha=0.5)

            ax.set_xlabel(f'X')
            ax.set_ylabel(f'Y')
            ax.set_zlabel(f'Z')

            ax.set_box_aspect([x_range/max_range,
                             y_range/max_range,
                             z_range/max_range])

        else:
            plt.figure(figsize=figsize)
            domain_size = f"L×H: {x_max-x_min:.1f}×{y_max-y_min:.1f}"

            plt.scatter(self.x, self.y, c='black', s=1, alpha=0.5)

            for boundary_data in self.boundaries.values():
                plt.plot(boundary_data['x'], boundary_data['y'],
                        'b-', linewidth=1, alpha=0.5)

            plt.xlabel(f'X ({x_max-x_min:.1f})')
            plt.ylabel(f'Y ({y_max-y_min:.1f})')
            plt.axis('equal')

        plt.title(f'Mesh Visualization\n{domain_size}')
        plt.tight_layout()
        plt.show()

    def write_tecplot(self, filename: str) -> None:
        """
        Writes the solution to a Tecplot file using MeshIO.

        Args:
            filename (str): The name of the Tecplot file to write.
        """
        self._create_meshio()
        self.meshio.write_tecplot(filename)

    def _is_domain_closed(self) -> bool:
        """
        Checks if all boundaries together form a single closed domain.
        Returns True if every vertex appears exactly twice and all boundaries are connected.
        """
        # Collect all segments from all boundaries
        segments = []
        for boundary_data in self.boundaries.values():
            x_coords = boundary_data['x']
            y_coords = boundary_data['y']
            
            for i in range(len(x_coords) - 1):
                pt1 = (float(x_coords[i]), float(y_coords[i]))
                pt2 = (float(x_coords[i + 1]), float(y_coords[i + 1]))
                segments.append((pt1, pt2))
        
        # Count vertex appearances in the entire domain
        endpoints = {}
        for seg in segments:
            for pt in seg:
                endpoints[pt] = endpoints.get(pt, 0) + 1
        
        # Check if each vertex appears exactly twice
        for pt, count in endpoints.items():
            if count != 2:
                print(f"Debug: Vertex {pt} appears {count} times (should be 2)")
                return False

        # Build adjacency graph
        adj = {}
        for seg in segments:
            p1, p2 = seg
            adj.setdefault(p1, set()).add(p2)
            adj.setdefault(p2, set()).add(p1)
        
        # DFS to check connectivity
        visited = set()
        def dfs(pt):
            visited.add(pt)
            for neighbor in adj[pt]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        # Start DFS from any vertex
        start = next(iter(adj))
        dfs(start)
        
        # Check if all vertices are connected
        if len(visited) != len(adj):
            print(f"Debug: Domain is not fully connected. Found {len(visited)} vertices, expected {len(adj)}")
            return False
            
        return True

    def check_closed_curve(self) -> bool:
        """
        Checks if all boundaries together form a closed domain.
        Returns True if the domain is properly closed, False otherwise.
        """
        try:
            if not self.boundaries:
                print("Debug: No boundaries defined")
                return False
                
            if self._is_domain_closed():
                print("Debug: Domain is properly closed")
                return True
            else:
                print("Debug: Domain is not properly closed")
                return False

        except Exception as e:
            print(f"Debug: Error checking closed curve: {str(e)}")
            return False
