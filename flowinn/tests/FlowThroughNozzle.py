import numpy as np
import logging
from typing import Tuple
from flowinn.mesh.mesh import Mesh
from flowinn.models.model import PINN
from flowinn.training.loss import NavierStokesLoss
from flowinn.plot.plot import Plot
from flowinn.physics.boundary_conditions import InletBC, OutletBC, WallBC

class FlowThroughNozzle:
    def __init__(self, caseName: str, xRange: Tuple[float, float], yRange: Tuple[float, float]):
        """
        Initialize FlowThroughNozzle simulation.

        Args:
            caseName: Name of the simulation case
            xRange: Tuple of (min_x, max_x) domain bounds
            yRange: Tuple of (min_y, max_y) domain bounds
        """
        if not isinstance(caseName, str):
            raise TypeError("caseName must be a string")
        if not all(isinstance(x, (int, float)) for x in [*xRange, *yRange]):
            raise TypeError("xRange and yRange must be numeric")

        self.logger = logging.getLogger(__name__)
        self.is2D = True
        self.problemTag = caseName
        self.mesh = Mesh(self.is2D)
        self.model = PINN(input_shape=(2,), output_shape=3, eq = self.problemTag, layers=[20,40,60,40,20])

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange

        # Initialize boundary condition objects
        self.inlet_bc = InletBC("inlet")
        self.outlet_bc = OutletBC("outlet")
        self.wall_bc = WallBC("wall")

        return

    def generate_nozzle_geometry(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate coordinates for a convergent-divergent nozzle using a combination
        of quadratic functions for smooth transitions.
        """
        x = np.linspace(self.xRange[0], self.xRange[1], N)
        
        # Nozzle geometry parameters
        inlet_width = 1.0    # Width at inlet
        throat_width = 0.3   # Width at throat (narrowest point)
        outlet_width = 1.2   # Width at outlet (slightly larger than inlet for supersonic expansion)
        
        # Position parameters
        x_length = self.xRange[1] - self.xRange[0]
        throat_position = self.xRange[0] + 0.4 * x_length  # Throat at 40% of length
        
        # Generate width profile
        width = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= throat_position:
                # Convergent section (inlet to throat)
                x_norm = (xi - self.xRange[0]) / (throat_position - self.xRange[0])
                width[i] = inlet_width + (throat_width - inlet_width) * x_norm**2
            else:
                # Divergent section (throat to outlet)
                x_norm = (xi - throat_position) / (self.xRange[1] - throat_position)
                width[i] = throat_width + (outlet_width - throat_width) * x_norm**2
        
        # Generate top and bottom profiles
        y_top = width
        y_bottom = -width
        
        return x, y_top, y_bottom

    def generateMesh(self, Nx: int = 100, Ny: int = 100, NBoundary: int = 100, sampling_method: str = 'random'):
        try:
            if not all(isinstance(x, int) and x > 0 for x in [Nx, Ny, NBoundary]):
                raise ValueError("Nx, Ny, and NBoundary must be positive integers")
            if sampling_method not in ['random', 'uniform']:
                raise ValueError("sampling_method must be 'random' or 'uniform'")
                
            # Initialize boundaries first
            self._initialize_boundaries()
            
            # Generate nozzle geometry for boundaries
            x_boundary, y_top, y_bottom = self.generate_nozzle_geometry(NBoundary)
            
            # Create inlet/outlet points that match the nozzle height at those positions
            inlet_height = y_top[0] - y_bottom[0]  # Height at inlet
            outlet_height = y_top[-1] - y_bottom[-1]  # Height at outlet
            
            x_inlet = np.full(NBoundary, self.xRange[0])
            y_inlet = np.linspace(y_bottom[0], y_top[0], NBoundary)  # Match nozzle height at inlet
            
            x_outlet = np.full(NBoundary, self.xRange[1])
            y_outlet = np.linspace(y_bottom[-1], y_top[-1], NBoundary)  # Match nozzle height at outlet

            # Update exterior boundaries with nozzle geometry
            for name, coords in [
                ('top', (x_boundary, y_top)),
                ('bottom', (x_boundary, y_bottom)),
                ('Inlet', (x_inlet, y_inlet)),
                ('Outlet', (x_outlet, y_outlet))
            ]:
                if name not in self.mesh.boundaries:
                    raise KeyError(f"Boundary '{name}' not initialized")
                self.mesh.boundaries[name].update({
                    'x': coords[0].astype(np.float32),
                    'y': coords[1].astype(np.float32)
                })

            # Validate boundary conditions before mesh generation
            self._validate_boundary_conditions()

            # Generate the mesh
            self.mesh.generateMesh(
                Nx=Nx,
                Ny=Ny,
                sampling_method=sampling_method
            )
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed: {str(e)}")
            raise

    def _initialize_boundaries(self):
        """Initialize boundaries with proper BC system."""
        # Initialize exterior boundaries
        self.mesh.boundaries = {
            'Inlet': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': 1.0},  # Example inlet velocity
                    'v': None,
                    'p': {'value': 1.2}
                },
                'bc_type': self.inlet_bc
            },
            'Outlet': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': None,
                    'v': None,
                    'p': {'value': 1.0}  # Example outlet pressure
                },
                'bc_type': self.outlet_bc
            },
            'top': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': 0.0},  # No-slip condition
                    'v': {'value': 0.0},  # No-slip condition
                    'p': {'gradient': 0.0, 'direction': 'normal'}  # Zero pressure gradient normal to wall
                },
                'bc_type': self.wall_bc
            },
            'bottom': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': 0.0},  # No-slip condition
                    'v': {'value': 0.0},  # No-slip condition
                    'p': {'gradient': 0.0, 'direction': 'normal'}  # Zero pressure gradient normal to wall
                },
                'bc_type': self.wall_bc
            }
        }

    def _validate_boundary_conditions(self):
        """Validate boundary conditions before mesh generation."""
        for name, boundary in self.mesh.boundaries.items():
            if any(key not in boundary for key in ['x', 'y', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in boundary {name}")
            if boundary['x'] is None or boundary['y'] is None:
                raise ValueError(f"Coordinates not set for boundary {name}")

    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model)
    
    def train(self, epochs=10000, print_interval=100, autosaveInterval=10000, num_batches=10):
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
            X = (np.hstack((self.mesh.x.flatten()[:, None], self.mesh.y.flatten()[:, None])))
            sol = self.model.predict(X)

            self.mesh.solutions['u'] = sol[:, 0]
            self.mesh.solutions['v'] = sol[:, 1]
            self.mesh.solutions['p'] = sol[:, 2]

            self.generate_plots()
            
            # Write solution to Tecplot file
            self.write_solution()

            return
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
            # Ensure solutions are properly shaped before writing
            if any(key not in self.mesh.solutions for key in ['u', 'v', 'p']):
                raise ValueError("Missing required solution components (u, v, p)")
            
            self.mesh.write_tecplot(filename)
            
        except Exception as e:
            print(f"Error writing solution: {str(e)}")
            raise
    
    def generate_plots(self):
        self.Plot = Plot(self.mesh)

    def plot(self, solkey = 'u', streamlines = False):
        self.Plot.scatterPlot(solkey)

    def load_model(self):
        self.model.load(self.problemTag)
