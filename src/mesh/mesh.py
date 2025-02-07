import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt  # Add this import at the top
from typing import Dict, Optional, Union, Tuple
from src.mesh.meshio import MeshIO # Import MeshIO

class Mesh:
    def __init__(self, is2D: bool = True) -> None:
        # Private attributes
        self._x: np.ndarray = None
        self._y: np.ndarray = None
        self._z: np.ndarray = None
        self._solutions: dict = {}
        self._boundaries: dict = {}
        self._interiorBoundaries: dict = {}
        self._is2D: bool = is2D
        self.meshio = MeshIO(self) # Create MeshIO instance

    # Coordinate properties
    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("x must be a numpy array")
        self._x = value

    @property
    def y(self) -> np.ndarray:
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("y must be a numpy array")
        self._y = value

    @property
    def z(self) -> np.ndarray:
        return self._z

    @z.setter
    def z(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("z must be a numpy array")
        self._z = value

    # Other properties
    @property
    def solutions(self) -> dict:
        return self._solutions

    @solutions.setter
    def solutions(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._solutions = value

    @property
    def boundaries(self) -> dict:
        return self._boundaries

    @boundaries.setter
    def boundaries(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("boundaries must be a dictionary")
        self._boundaries = value

    @property
    def interiorBoundaries(self) -> dict:
        return self._interiorBoundaries

    @interiorBoundaries.setter
    def interiorBoundaries(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("interiorBoundaries must be a dictionary")
        self._interiorBoundaries = value
        
    @property
    def is2D(self) -> bool:
        return self._is2D

    @is2D.setter
    def is2D(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("is2D must be a boolean")
        self._is2D = value

    def generateMesh(self, Nx=100, Ny=100, Nz=None, sampling_method='random'):
        """
        Generate a mesh within a domain defined by boundary data.

        Parameters:
        - Nx, Ny, Nz: Number of points in each dimension for structured sampling.
        - sampling_method: Sampling method ('random', 'uniform').

        Raises:
        - ValueError: If input parameters are invalid or mesh generation fails
        """
        # Check if boundaries are defined
        if not self.boundaries:
            raise ValueError("No boundaries defined. Use setBoundary() to define boundaries before generating mesh")

        try:
            self._generateMeshFromBoundary(sampling_method, Nx, Ny, Nz)
        except Exception as e:
            raise ValueError(f"Mesh generation failed: {str(e)}")
        
    def _generateMeshFromBoundary(self, sampling_method, Nx, Ny, Nz):
        # Validate boundaries and print their content
        for boundary_name, boundary_data in self.boundaries.items():
            if 'x' not in boundary_data or 'y' not in boundary_data:
                raise ValueError(f"Boundary '{boundary_name}' must contain 'x' and 'y' coordinates.")
            if not self.is2D and 'z' not in boundary_data:
                raise ValueError(f"3D mesh requires z coordinate for boundary {boundary_name}")

        # Convert and combine boundary coordinates
        try:
            x_boundary = np.concatenate([np.asarray(boundary_data['x'], dtype=np.float32).flatten() 
                                    for boundary_data in self.boundaries.values()])
            y_boundary = np.concatenate([np.asarray(boundary_data['y'], dtype=np.float32).flatten() 
                                    for boundary_data in self.boundaries.values()])
            
            # Handle z coordinates for 3D case
            if not self.is2D:
                z_boundary = np.concatenate([np.asarray(boundary_data['z'], dtype=np.float32).flatten() 
                                        for boundary_data in self.boundaries.values()])
            else:
                z_boundary = None
                
        except Exception as e:
            print(f"Debug: Error during boundary concatenation: {str(e)}")
            raise

        if sampling_method == 'random':
            self._sampleRandomlyWithinBoundary(x_boundary, y_boundary, z_boundary if not self.is2D else None, 
                                             Nx, Ny, Nz)
        elif sampling_method == 'uniform':
            self._sampleUniformlyWithinBoundary(x_boundary, y_boundary, z_boundary if not self.is2D else None,
                                              Nx, Ny, Nz)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")

    def _sampleRandomlyWithinBoundary(self, x_boundary: np.ndarray, y_boundary: np.ndarray, z_boundary: Optional[np.ndarray],
                                    Nx: int, Ny: int, Nz: Optional[int]) -> None:
        """Sample points randomly within boundary."""
        try:
            # Convert inputs to numpy arrays
            x_boundary = np.asarray(x_boundary, dtype=np.float32)
            y_boundary = np.asarray(y_boundary, dtype=np.float32)
            if z_boundary is not None:
                z_boundary = np.asarray(z_boundary, dtype=np.float32)

            # Check for NaN values
            if np.any(np.isnan(x_boundary)) or np.any(np.isnan(y_boundary)) or \
               (z_boundary is not None and np.any(np.isnan(z_boundary))):
                raise ValueError("Boundary coordinates contain NaN values")

            # Calculate total number of points
            Nt = Nx * Ny * (Nz if not self.is2D and Nz is not None else 1)
            
            # Generate samples
            samples = []
            while len(samples) < Nt:
                # Generate random points
                x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
                y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)
                
                if not self.is2D and z_boundary is not None:
                    z_rand = np.random.uniform(np.min(z_boundary), np.max(z_boundary), size=Nt)
                    points = np.column_stack((x_rand, y_rand, z_rand))
                else:
                    points = np.column_stack((x_rand, y_rand))

                # Check if points are inside the domain
                valid_points = self._check_points_in_domain(points, x_boundary, y_boundary, z_boundary)
                samples.extend(valid_points)

            # Trim to exact number needed and reshape
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

    def _check_points_in_domain(self, points, x_boundary, y_boundary, z_boundary=None):
        """Check if points are inside the domain and outside interior boundaries."""
        try:
            # First check if points are inside exterior boundary
            boundary_points = np.column_stack((x_boundary, y_boundary))
            exterior_tri = Delaunay(boundary_points)
            inside_exterior = exterior_tri.find_simplex(points) >= 0
            valid_points = points[inside_exterior]

            # Now check interior boundaries and remove points inside them
            if self._interiorBoundaries:
                for boundary_data in self._interiorBoundaries.values():
                    x_int = boundary_data['x'].flatten()
                    y_int = boundary_data['y'].flatten()
                    interior_points = np.column_stack((x_int, y_int))
                    
                    # Create Delaunay triangulation for interior boundary
                    interior_tri = Delaunay(interior_points)
                    
                    # Remove points that are inside this interior boundary
                    inside_interior = interior_tri.find_simplex(valid_points) >= 0
                    valid_points = valid_points[~inside_interior]

            return valid_points

        except Exception as e:
            print(f"Debug: Error checking points in domain: {str(e)}")
            raise

    def _sampleUniformlyWithinBoundary(self, x_boundary, y_boundary, z_boundary, Nx, Ny, Nz):
        # Create a regular grid and keep points inside the boundary
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)
        z_min, z_max = np.min(z_boundary), np.max(z_boundary) if z_boundary is not None else (None, None)

        x_grid, y_grid, z_grid = np.meshgrid(
            np.linspace(x_min, x_max, Nx), np.linspace(y_min, y_max, Ny), np.linspace(z_min, z_max, Nz) if z_boundary is not None else [0]
        )
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())) if not self.is2D else np.column_stack((x_grid.flatten(), y_grid.flatten()))
        
        # Create a Delaunay triangulation
        points = np.column_stack((x_boundary, y_boundary, z_boundary)) if not self.is2D else np.column_stack((x_boundary, y_boundary))
        triangulation = Delaunay(points)
        
        # Check which grid points are inside the triangulation
        inside = triangulation.find_simplex(grid_points) >= 0
        inside_points = grid_points[inside]
        
        if self.is2D:
            self._x, self._y = inside_points[:, 0].astype(np.float32), inside_points[:, 1].astype(np.float32)
        else:
            self._x, self._y, self._z = inside_points[:, 0].astype(np.float32), inside_points[:, 1].astype(np.float32), inside_points[:, 2].astype(np.float32)


    def setBoundary(self, boundary_name, xBc, yBc, interior=False, **boundary_conditions):
        """
        Set multiple boundary conditions at once.
        
        Parameters:
        - boundary_name: name of the boundary
        - xBc: x coordinates of the boundary
        - yBc: y coordinates of the boundary
        - interior: boolean flag to indicate if this is an interior boundary
        - **boundary_conditions: variable names and their values
        """
        for var_name, values in boundary_conditions.items():
            self.setBoundaryCondition(xBc, yBc, values, var_name, boundary_name, interior=interior)

    def setBoundaryCondition(self, xCoord, yCoord, value, varName, boundaryName, zCoord=None, interior=False, bc_type=None):
        """Set boundary conditions for either exterior or interior boundaries."""
        # Select appropriate boundary dictionary
        boundary_dict = self._interiorBoundaries if interior else self._boundaries
        
        # Initialize boundary if it doesn't exist
        if boundaryName not in boundary_dict:
            boundary_dict[boundaryName] = {}
        
        # Set coordinates
        boundary_dict[boundaryName]['x'] = np.asarray(xCoord, dtype=np.float32)
        boundary_dict[boundaryName]['y'] = np.asarray(yCoord, dtype=np.float32)
        
        if not self.is2D:
            if zCoord is None:
                raise ValueError(f"z coordinate required for 3D mesh in boundary {boundaryName}")
            boundary_dict[boundaryName]['z'] = np.asarray(zCoord, dtype=np.float32)
        
        # Set boundary condition value and type
        if value is not None:
            boundary_dict[boundaryName][varName] = np.asarray(value, dtype=np.float32)
            boundary_dict[boundaryName][f'{varName}_type'] = bc_type
        
        # Set type flag
        boundary_dict[boundaryName]['isInterior'] = interior

    def showMesh(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """Visualize the mesh with proper dimensional scaling."""
        if self.x is None or self.y is None:
            raise ValueError("Mesh has not been generated yet")
            
        # Calculate domain dimensions
        x_min, x_max = np.min(self.x), np.max(self.x)
        y_min, y_max = np.min(self.y), np.max(self.y)
        
        if not self.is2D and self.z is not None:
            z_min, z_max = np.min(self.z), np.max(self.z)
            domain_size = f"L×H×W: {x_max-x_min:.1f}×{y_max-y_min:.1f}×{z_max-z_min:.1f}"
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Set equal scaling for all axes
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            z_mid = (z_max + z_min) / 2
            
            # Set limits to maintain proper scaling
            ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
            ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
            ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
            
            # Plot mesh points
            scatter = ax.scatter(self.x, self.y, self.z, 
                               c='black', s=1, alpha=0.5)
            
            # Plot boundaries
            for boundary_data in self.boundaries.values():
                x_boundary = boundary_data['x']
                y_boundary = boundary_data['y']
                z_boundary = boundary_data['z']
                ax.plot3D(x_boundary, y_boundary, z_boundary, 
                         'b-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel(f'X')
            ax.set_ylabel(f'Y')
            ax.set_zlabel(f'Z')
            
            # Set aspect ratio to be equal
            ax.set_box_aspect([x_range/max_range, 
                             y_range/max_range, 
                             z_range/max_range])
            
        else:
            plt.figure(figsize=figsize)
            domain_size = f"L×H: {x_max-x_min:.1f}×{y_max-y_min:.1f}"
            
            # Plot mesh points
            plt.scatter(self.x, self.y, c='black', s=1, alpha=0.5)
            
            # Plot boundaries
            for boundary_data in self.boundaries.values():
                plt.plot(boundary_data['x'], boundary_data['y'], 
                        'b-', linewidth=1, alpha=0.5)
            
            plt.xlabel(f'X ({x_max-x_min:.1f})')
            plt.ylabel(f'Y ({y_max-y_min:.1f})')
            plt.axis('equal')
        
        plt.title(f'Mesh Visualization\n{domain_size}')
        plt.tight_layout()
        plt.show()

    def write_tecplot(self, filename: str):
        """Write solution using MeshIO."""
        self.meshio.write_tecplot(filename)
