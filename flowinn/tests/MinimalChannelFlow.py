import numpy as np
import logging
from typing import Tuple
from flowinn.mesh.mesh import Mesh
from flowinn.models.model import PINN
from flowinn.training.loss import NavierStokesLoss
from flowinn.plot.plot import Plot
from flowinn.physics.boundary_conditions import InletBC, OutletBC, WallBC

class MinimalChannelFlow:
    def __init__(self, caseName: str, xRange: Tuple[float, float], 
                 yRange: Tuple[float, float], zRange: Tuple[float, float]):
        """
        Initialize MinimalChannelFlow simulation.
        
        Args:
            caseName: Name of the simulation case
            xRange: Tuple of (min_x, max_x) domain bounds
            yRange: Tuple of (min_y, max_y) domain bounds
            zRange: Tuple of (min_z, max_z) domain bounds
        """
        if not isinstance(caseName, str):
            raise TypeError("caseName must be a string")
        if not all(isinstance(x, (int, float)) for x in [*xRange, *yRange, *zRange]):
            raise TypeError("xRange, yRange and zRange must be numeric")
            
        self.logger = logging.getLogger(__name__)
        self.is2D = False
        self.problemTag = caseName
        self.mesh = Mesh(self.is2D)
        self.model = PINN(input_shape=(3,), output_shape=4, eq=self.problemTag, layers=[30,60,90,60,30])

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange

        # Initialize boundary condition objects
        self.inlet_bc = InletBC("inlet")
        self.outlet_bc = OutletBC("outlet")
        self.wall_bc = WallBC("wall")

    def generateMesh(self, Nx: int = 50, Ny: int = 50, Nz: int = 50, 
                    NBoundary: int = 100, sampling_method: str = 'random'):
        """Generate 3D mesh for channel flow."""
        try:
            if not all(isinstance(x, int) and x > 0 for x in [Nx, Ny, Nz, NBoundary]):
                raise ValueError("Nx, Ny, Nz, and NBoundary must be positive integers")
            if sampling_method not in ['random', 'uniform']:
                raise ValueError("sampling_method must be 'random' or 'uniform'")
            
            # Initialize boundaries first
            self._initialize_boundaries()
            
            # Create and validate boundary points
            x_top = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_top = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            z_top = np.full_like(x_top, self.zRange[1])
            
            x_bottom = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_bottom = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            z_bottom = np.full_like(x_bottom, self.zRange[0])
            
            x_inlet = np.full(NBoundary, self.xRange[0])
            y_inlet = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            z_inlet = np.linspace(self.zRange[0], self.zRange[1], NBoundary)
            
            x_outlet = np.full(NBoundary, self.xRange[1])
            y_outlet = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            z_outlet = np.linspace(self.zRange[0], self.zRange[1], NBoundary)
            
            x_front = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_front = np.full_like(x_front, self.yRange[0])
            z_front = np.linspace(self.zRange[0], self.zRange[1], NBoundary)
            
            x_back = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_back = np.full_like(x_back, self.yRange[1])
            z_back = np.linspace(self.zRange[0], self.zRange[1], NBoundary)

            # Validate coordinates
            all_coords = [
                (x_top, y_top, z_top),
                (x_bottom, y_bottom, z_bottom),
                (x_inlet, y_inlet, z_inlet),
                (x_outlet, y_outlet, z_outlet),
                (x_front, y_front, z_front),
                (x_back, y_back, z_back)
            ]
            
            if any(np.any(np.isnan(coord)) for coords in all_coords for coord in coords):
                raise ValueError("NaN values detected in boundary coordinates")

            # Update boundaries
            boundaries_map = {
                'Top': (x_top, y_top, z_top),
                'Bottom': (x_bottom, y_bottom, z_bottom),
                'Inlet': (x_inlet, y_inlet, z_inlet),
                'Outlet': (x_outlet, y_outlet, z_outlet),
                'Front': (x_front, y_front, z_front),
                'Back': (x_back, y_back, z_back)
            }

            for name, coords in boundaries_map.items():
                if name not in self.mesh.boundaries:
                    raise KeyError(f"Boundary '{name}' not initialized")
                self.mesh.boundaries[name].update({
                    'x': coords[0].astype(np.float32),
                    'y': coords[1].astype(np.float32),
                    'z': coords[2].astype(np.float32)
                })

            # Validate boundary conditions before mesh generation
            self._validate_boundary_conditions()

            # Generate the mesh
            self.mesh.generateMesh(
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                sampling_method=sampling_method
            )
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {str(e)}")
            raise

    def _initialize_boundaries(self):
        """Initialize boundary dictionaries with proper BC system."""
        self.mesh.boundaries = {
            'Inlet': {
                'x': None, 'y': None, 'z': None,
                'conditions': {
                    'u': {'value': 1.0},
                    'v': {'value': 0.0},
                    'w': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'x'}
                },
                'bc_type': self.inlet_bc
            },
            'Outlet': {
                'x': None, 'y': None, 'z': None,
                'conditions': {
                    'u': {'gradient': 0.0, 'direction': 'x'},
                    'v': {'gradient': 0.0, 'direction': 'x'},
                    'w': {'gradient': 0.0, 'direction': 'x'},
                    'p': {'value': 0.0}
                },
                'bc_type': self.outlet_bc
            },
            'Bottom': {
                'x': None, 'y': None, 'z': None,
                'conditions': {
                    'u': {'value': 0.0},
                    'v': {'value': 0.0},
                    'w': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'z'}
                },
                'bc_type': self.wall_bc
            },
            'Top': {
                'x': None, 'y': None, 'z': None,
                'conditions': {
                    'u': {'value': 1.0},
                    'v': {'value': 0.0},
                    'w': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'z'}
                },
                'bc_type': self.wall_bc
            },
            'Front': {
                'x': None, 'y': None, 'z': None,
                'conditions': {
                    'u': {'gradient': 0.0, 'direction': 'y'},
                    'v': {'gradient': 0.0, 'direction': 'y'},
                    'w': {'gradient': 0.0, 'direction': 'y'},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.wall_bc
            },
            'Back': {
                'x': None, 'y': None, 'z': None,
                'conditions': {
                    'u': {'gradient': 0.0, 'direction': 'y'},
                    'v': {'gradient': 0.0, 'direction': 'y'},
                    'w': {'gradient': 0.0, 'direction': 'y'},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.wall_bc
            }
        }

    def _validate_boundary_conditions(self):
        """Validate boundary conditions before mesh generation."""
        # Check exterior boundaries
        for name, boundary in self.mesh.boundaries.items():
            if any(key not in boundary for key in ['x', 'y', 'z', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in boundary {name}")
            if boundary['x'] is None or boundary['y'] is None or boundary['z'] is None:
                raise ValueError(f"Coordinates not set for boundary {name}")

    def _set_channel_boundaries(self, NBoundary: int):
        """Set boundary conditions for channel flow."""
        # Calculate number of points per side
        n_side = int(np.sqrt(NBoundary))
        
        # Create coordinate arrays for each boundary
        y_coords = np.linspace(self.yRange[0], self.yRange[1], n_side)
        z_coords = np.linspace(self.zRange[0], self.zRange[1], n_side)
        x_coords = np.linspace(self.xRange[0], self.xRange[1], n_side)
        
        # Create meshgrids for each face
        # YZ plane (for inlet and outlet)
        y_grid_yz, z_grid_yz = np.meshgrid(y_coords, z_coords)
        
        # XY plane (for front and back)
        x_grid_xy, y_grid_xy = np.meshgrid(x_coords, y_coords)
        
        # XZ plane (for top and bottom)
        x_grid_xz, z_grid_xz = np.meshgrid(x_coords, z_coords)
        
        n_face = n_side * n_side
        
        # Set each boundary with explicit coordinates and boundary conditions
        boundaries = {
            'Inlet': {
                'x': np.full(n_face, self.xRange[0], dtype=np.float32),
                'y': y_grid_yz.flatten(),
                'z': z_grid_yz.flatten(),
                'conditions': {
                    'u': np.ones(n_face),
                    'v': np.zeros(n_face),
                    'w': np.zeros(n_face)
                }
            },
            'Outlet': {
                'x': np.full(n_face, self.xRange[1], dtype=np.float32),
                'y': y_grid_yz.flatten(),
                'z': z_grid_yz.flatten(),
                'conditions': {}
            },
            'Bottom': {
                'x': x_grid_xy.flatten(),
                'y': y_grid_xy.flatten(),
                'z': np.full(n_face, self.zRange[0], dtype=np.float32),
                'conditions': {
                    'u': np.zeros(n_face),
                    'v': np.zeros(n_face),
                    'w': np.zeros(n_face)
                }
            },
            'Top': {
                'x': x_grid_xy.flatten(),
                'y': y_grid_xy.flatten(),
                'z': np.full(n_face, self.zRange[1], dtype=np.float32),
                'conditions': {
                    'u': np.ones(n_face),
                    'v': None,
                    'w': None
                }
            },
            'Front': {
                'x': x_grid_xz.flatten(),
                'y': np.full(n_face, self.yRange[0], dtype=np.float32),
                'z': z_grid_xz.flatten(),
                'conditions': {}
            },
            'Back': {
                'x': x_grid_xz.flatten(),
                'y': np.full(n_face, self.yRange[1], dtype=np.float32),
                'z': z_grid_xz.flatten(),
                'conditions': {}
            }
        }
        
        # Set all boundaries
        for name, data in boundaries.items():
            # First set the coordinates for the boundary
            self.mesh.setBoundary(
                boundary_name=name,
                xBc=data['x'],
                yBc=data['y'],
                zBc=data['z']  # Always pass z coordinates for 3D mesh
            )
            
            # Then set any boundary conditions
            for var_name, value in data.get('conditions', {}).items():
                if value is not None:
                    value_array = np.asarray(value).astype(np.float32)
                    if np.isscalar(value):
                        value_array = np.full_like(data['x'], value_array)
                    
                    self.mesh.setBoundary(
                        boundary_name=name,
                        xBc=data['x'],
                        yBc=data['y'],
                        zBc=data['z'],
                        **{var_name: value_array}
                    )

    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model)
    
    def train(self, epochs=10000, print_interval=100, autosaveInterval=10000, num_batches=10):
        """Train the model with batch support."""
        self.getLossFunction()
        self.model.train(
            self.loss.loss_function,
            self.mesh,
            epochs=epochs,
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches
        )

    def predict(self) -> None:
        """Predict flow solution and generate plots."""
        try:
            X = np.column_stack((self.mesh.x.flatten(), 
                               self.mesh.y.flatten(),
                               self.mesh.z.flatten()))
            sol = self.model.predict(X)

            self.mesh.solutions['u'] = sol[:, 0]
            self.mesh.solutions['v'] = sol[:, 1]
            self.mesh.solutions['w'] = sol[:, 2]
            self.mesh.solutions['p'] = sol[:, 3]

            self.generate_plots()
            self.write_solution()
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def write_solution(self, filename=None):
        """Write the solution to a CSV format file."""
        if filename is None:
            filename = f"{self.problemTag}.csv"
        elif not filename.endswith('.csv'):
            filename += '.csv'
        
        try:
            if any(key not in self.mesh.solutions for key in ['u', 'v', 'w', 'p']):
                raise ValueError("Missing required solution components (u, v, w, p)")
            
            self.mesh.write_tecplot(filename)
            
        except Exception as e:
            print(f"Error writing solution: {str(e)}")
            raise

    def generate_plots(self):
        self.Plot = Plot(self.mesh)

    def plot(self, solkey='u'):
        self.Plot.scatterPlot(solkey)
        if not self.is2D:
            self.Plot.plotSlices(solkey)

    def load_model(self):
        self.model.load(self.problemTag)
