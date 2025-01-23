import numpy as np

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

    def generateMesh(self, x_range, y_range, z_range=None, Nx=100, Ny=100, Nz=None, sampling_method='random'):
        # Validate inputs
        if not isinstance(x_range, (tuple, list)) or len(x_range) != 2:
            raise ValueError("x_range must be a tuple or list with two elements (min, max).")
        if not isinstance(y_range, (tuple, list)) or len(y_range) != 2:
            raise ValueError("y_range must be a tuple or list with two elements (min, max).")

        # Check for z_range consistency with Nz
        if z_range is not None and Nz is None:
            raise ValueError("If z_range is provided, Nz must also be provided.")
        if z_range is None and Nz is not None:
            raise ValueError("If Nz is provided, z_range must also be provided.")

        self.is2D = z_range is None

        x_min, x_max = x_range
        y_min, y_max = y_range

        if not self.is2D:
            if not isinstance(z_range, (tuple, list)) or len(z_range) != 2:
                raise ValueError("z_range must be a tuple or list with two elements (min, max) for 3D meshes.")
            z_min, z_max = z_range

        # Default to 2D if Nz is None
        if Nz is None:
            Nz = 1

        nPoints = Nx * Ny * Nz

        # Sampling logic
        if sampling_method == 'random':
            self._x = (np.random.rand(nPoints) * (x_max - x_min) + x_min).astype(np.float32)
            self._y = (np.random.rand(nPoints) * (y_max - y_min) + y_min).astype(np.float32)

            if not self.is2D:
                self._z = (np.random.rand(nPoints) * (z_max - z_min) + z_min).astype(np.float32)

        elif sampling_method == 'uniform':
            x_lin = np.linspace(x_min, x_max, Nx)
            y_lin = np.linspace(y_min, y_max, Ny)

            if self.is2D:
                self._x, self._y = np.meshgrid(x_lin, y_lin)
                self._x, self._y = self._x.flatten().astype(np.float32), self._y.flatten().astype(np.float32)
            else:
                z_lin = np.linspace(z_min, z_max, Nz)
                self._x, self._y, self._z = np.meshgrid(x_lin, y_lin, z_lin)
                self._x, self._y, self._z = (
                    self._x.flatten().astype(np.float32),
                    self._y.flatten().astype(np.float32),
                    self._z.flatten().astype(np.float32),
                )

        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")

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

