import numpy as np
from scipy.spatial import Delaunay

class Mesh:
    def __init__(self, is2D: bool = True) -> None:
        # Private attributes
        self._x: np.ndarray = None
        self._y: np.ndarray = None
        self._z: np.ndarray = None
        self._solutions: dict = {}
        self._boundaries: dict = {}
        self._is2D: bool = is2D

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
        # Validate that all boundaries contain 'x' and 'y' coordinates
        for boundary_name, boundary_data in self.boundaries.items():
            if 'x' not in boundary_data or 'y' not in boundary_data:
                raise ValueError(f"Boundary '{boundary_name}' must contain 'x' and 'y' coordinates.")

        # Combine boundary coordinates into a single array for validation
        x_boundary = np.concatenate([np.array(boundary_data['x']).flatten() for boundary_data in self.boundaries.values()])
        y_boundary = np.concatenate([np.array(boundary_data['y']).flatten() for boundary_data in self.boundaries.values()])

        # Sampling logic

        if sampling_method == 'random':
            self._sampleRandomlyWithinBoundary(x_boundary, y_boundary, Nx, Ny, Nz)
        elif sampling_method == 'uniform':
            self._sampleUniformlyWithinBoundary(x_boundary, y_boundary, Nx, Ny, Nz)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")


    def _sampleRandomlyWithinBoundary(self, x_boundary, y_boundary, Nx, Ny, Nz):

        Nt = Nx * Ny * Nz if not self.is2D else Nx * Ny

        # Create a Delaunay triangulation
        points = np.column_stack((x_boundary, y_boundary))
        triangulation = Delaunay(points)
        
        samples = []
        while len(samples) < Nt: 
            x_rand = np.random.uniform(np.min(x_boundary), np.max(x_boundary), size=Nt)
            y_rand = np.random.uniform(np.min(y_boundary), np.max(y_boundary), size=Nt)
            random_points = np.column_stack((x_rand, y_rand))
            
            # Check which points are inside the triangulation
            inside = triangulation.find_simplex(random_points) >= 0
            samples.extend(random_points[inside])
        
        samples = np.array(samples)  # Keep only the first 1000 points
        self._x, self._y = samples[:, 0].astype(np.float32), samples[:, 1].astype(np.float32)

    def _sampleUniformlyWithinBoundary(self, x_boundary, y_boundary, Nx, Ny, Nz):
        # Create a regular grid and keep points inside the boundary
        x_min, x_max = np.min(x_boundary), np.max(x_boundary)
        y_min, y_max = np.min(y_boundary), np.max(y_boundary)

        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, Nx), np.linspace(y_min, y_max, Ny)
        )
        grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        
        # Create a Delaunay triangulation
        points = np.column_stack((x_boundary, y_boundary))
        triangulation = Delaunay(points)
        
        # Check which grid points are inside the triangulation
        inside = triangulation.find_simplex(grid_points) >= 0
        inside_points = grid_points[inside]
        
        self._x, self._y = inside_points[:, 0].astype(np.float32), inside_points[:, 1].astype(np.float32)


    def setBoundary(self, boundary_name, xBc, yBc, **boundary_conditions):
        # Initialize boundary if not exists
        if boundary_name not in self._boundaries:
            self._boundaries[boundary_name] = {}

        # Set x and y coordinates for the boundary
        self._boundaries[boundary_name]['x'] = xBc
        self._boundaries[boundary_name]['y'] = yBc

        # Iterate over all specified boundary conditions
        for var_name, values in boundary_conditions.items():
            self.setBoundaryCondition(xBc, yBc, values, var_name, boundary_name)

    def setBoundaryCondition(self, xCoord, yCoord, value, varName, boundaryName, zCoord=None):
        if boundaryName not in self._boundaries:
            raise ValueError(f"Boundary name '{boundaryName}' is not valid. Available boundaries are: {list(self._boundaries.keys())}")

        if varName not in self._boundaries[boundaryName]:
            self._boundaries[boundaryName][varName] = {}

        self._boundaries[boundaryName]['x'] = xCoord
        self._boundaries[boundaryName]['y'] = yCoord

        if not self.is2D:
            self._boundaries[boundaryName]['z'] = zCoord

        self._boundaries[boundaryName][varName] = value

