import numpy as np

class Mesh:
    def __init__(self):
        self.X = None
        self.Y = None
        self.Z = None
        self.is2D = True  # Default is 2D until explicitly set to 3D during mesh generation

        self.boundaries = {
            'left': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'right': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
        }

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
            self.X = (np.random.rand(nPoints) * (x_max - x_min) + x_min).astype(np.float32)
            self.Y = (np.random.rand(nPoints) * (y_max - y_min) + y_min).astype(np.float32)

            # For 3D, generate Z points
            if Nz > 1:
                self.Z = (np.random.rand(nPoints) * (z_max - z_min) + z_min).astype(np.float32)


        elif sampling_method == 'uniform':
            self.X = np.linspace(x_min, x_max, nPoints, dtype=np.float32)
            self.Y = np.linspace(y_min, y_max, nPoints, dtype=np.float32)
            if not self.is2D:
                self.Z = np.linspace(z_min, z_max, nPoints, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")


        return 
    

    def setBoundaryCodition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):

        if sampling_method == 'random':
            # Random sampling of boundary points
            self.boundaries['left']['x'] = np.full((N0, 1), x_min, dtype=np.float32)
            self.boundaries['left']['y'] = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            self.boundaries['right']['x'] = np.full((N0, 1), x_max, dtype=np.float32)
            self.boundaries['right']['y'] = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            self.boundaries['bottom']['y'] = np.full((N0, 1), y_min, dtype=np.float32)
            self.boundaries['bottom']['x'] = np.random.rand(N0, 1) * (x_max - x_min) + x_min

            self.boundaries['top']['y'] = np.full((N0, 1), y_max, dtype=np.float32)
            self.boundaries['top']['x'] = np.random.rand(N0, 1) * (x_max - x_min) + x_min

        elif sampling_method == 'uniform':
            # Uniform grid of boundary points
            yBc = np.linspace(y_min, y_max, N0)[:, None].astype(np.float32)
            xBc = np.linspace(x_min, x_max, N0)[:, None].astype(np.float32)

            self.boundaries['left']['x'] = np.full_like(yBc, x_min, dtype=np.float32)
            self.boundaries['left']['y'] = yBc

            self.boundaries['right']['x'] = np.full_like(yBc, x_max, dtype=np.float32)
            self.boundaries['right']['y'] = yBc

            self.boundaries['bottom']['y'] = np.full_like(xBc, y_min, dtype=np.float32)
            self.boundaries['bottom']['x'] = xBc

            self.boundaries['top']['y'] = np.full_like(xBc, y_max, dtype=np.float32)
            self.boundaries['top']['x'] = xBc
        else:
            raise ValueError("sampling_method should be 'random' or 'uniform'")

        for key in self.boundaries:
            self.boundaries[key]['u'] = np.zeros_like(self.boundaries[key]['x'], dtype=np.float32) 
            self.boundaries[key]['v'] = np.zeros_like(self.boundaries[key]['y'], dtype=np.float32)

        self.boundaries['top']['u'] = np.ones_like(self.boundaries['top']['x'], dtype=np.float32)

        return
