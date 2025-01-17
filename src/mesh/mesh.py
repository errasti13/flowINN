import numpy as np

class Mesh:
    def __init__(self, is2D = True):
        self.X = None
        self.Y = None
        self.Z = None

        self.solutions = {}
        self.is2D = is2D  # Default is 2D until explicitly set to 3D during mesh generation

        self.boundaries = {}

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
            
            if Nz > 1:
                self.Z = (np.random.rand(nPoints) * (z_max - z_min) + z_min).astype(np.float32)

        elif sampling_method == 'uniform':
            self.X = np.linspace(x_min, x_max, nPoints)[:, None].astype(np.float32)
            self.Y = np.linspace(y_min, y_max, nPoints)[:, None].astype(np.float32)
            if not self.is2D:
                self.Z = np.linspace(z_min, z_max, nPoints)[:, None].astype(np.float32)

        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")
        
        return 
    

    def setBoundaryCodition(self, xCoord, yCoord, value, varName, boundaryName, zCoord = None):
        if boundaryName not in self.boundaries:
            raise ValueError(f"Boundary name '{boundaryName}' is not valid. Available boundaries are: {list(self.boundaries.keys())}")
        
        if varName not in self.boundaries[boundaryName]:
            raise ValueError(f"Variable name '{varName}' is not valid for boundary '{boundaryName}'. Available variables are: {list(self.boundaries[boundaryName].keys())}")
    
        self.boundaries[boundaryName]['x'] = xCoord
        self.boundaries[boundaryName]['y'] = yCoord

        if self.is2D is not True:
            self.boundaries[boundaryName]['z'] = zCoord

        self.boundaries[boundaryName][varName] = value
        return
