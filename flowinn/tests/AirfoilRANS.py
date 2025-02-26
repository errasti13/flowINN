import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple
from flowinn.mesh.mesh import Mesh
from flowinn.models.model import PINN
from flowinn.training.loss import NavierStokesLoss
from flowinn.plot.plot import Plot
from flowinn.physics.boundary_conditions import InletBC, OutletBC, WallBC 

class AirfoilRANS:
    def __init__(self, caseName: str, xRange: Tuple[float, float], yRange: Tuple[float, float], AoA: float = 0.0):
        """
        Initialize FlowOverAirfoil simulation.
        
        Args:
            caseName: Name of the simulation case
            xRange: Tuple of (min_x, max_x) domain bounds
            yRange: Tuple of (min_y, max_y) domain bounds
            AoA: Angle of attack in degrees
        """
        if not isinstance(caseName, str):
            raise TypeError("caseName must be a string")
        if not all(isinstance(x, (int, float)) for x in [*xRange, *yRange, AoA]):
            raise TypeError("xRange, yRange and AoA must be numeric")
            
        self.logger = logging.getLogger(__name__)
        self.is2D = True
        self.problemTag = caseName
        self.mesh = Mesh(self.is2D)
        self.model = PINN(input_shape=(2,), output_shape=6, eq = self.problemTag, layers=[20,40,60,40,20])

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange

        self.AoA = AoA
        self.c   = 1.0  #Airfoil Chord
        self.x0  = 0.0  #Airfoil leading edge x coordinate
        self.y0  = 0.0  #Airfoil leading edge y coordinate

        self.generate_airfoil_coords()

        # Initialize boundary condition objects
        self.inlet_bc = InletBC("inlet")
        self.outlet_bc = OutletBC("outlet")
        self.wall_bc = WallBC("wall")

        return
    
    def generate_airfoil_coords(self, N=100, thickness=0.12):
        """
        Generate NACA 4-digit airfoil coordinates for both upper and lower surfaces.
        The lower surface is just the negative of the upper surface.
        """
        # Generate x coordinates
        x = np.linspace(self.x0, self.x0 + self.c, N)
        x_normalized = x / self.c
        
        # Generate upper surface
        y_upper = self.y0 + 5 * thickness * (0.2969 * np.sqrt(x_normalized) 
                                - 0.1260 * x_normalized 
                                - 0.3516 * x_normalized**2 
                                + 0.2843 * x_normalized**3 
                                - 0.1015 * x_normalized**4)
        
        # Generate lower surface
        y_lower = self.y0 - y_upper
        
        # Combine coordinates in the correct order (counterclockwise)
        self.xAirfoil = np.concatenate([x, np.flip(x)]).reshape(-1, 1)
        self.yAirfoil = np.concatenate([y_upper, np.flip(y_lower)]).reshape(-1, 1)
        
        return
    
    def generateMesh(self, Nx: int = 100, Ny: int = 100, NBoundary: int = 100, sampling_method: str = 'random'):
        try:
            if not all(isinstance(x, int) and x > 0 for x in [Nx, Ny, NBoundary]):
                raise ValueError("Nx, Ny, and NBoundary must be positive integers")
            if sampling_method not in ['random', 'uniform']:
                raise ValueError("sampling_method must be 'random' or 'uniform'")
                
            # Initialize boundaries first
            self._initialize_boundaries()
            
            # Create and validate boundary points
            x_top = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_top = np.full_like(x_top, self.yRange[1])
            
            x_bottom = np.linspace(self.xRange[0], self.xRange[1], NBoundary)
            y_bottom = np.full_like(x_bottom, self.yRange[0])
            
            x_inlet = np.full(NBoundary, self.xRange[0])
            y_inlet = np.linspace(self.yRange[0], self.yRange[1], NBoundary)
            
            x_outlet = np.full(NBoundary, self.xRange[1])
            y_outlet = np.linspace(self.yRange[0], self.yRange[1], NBoundary)

            # Ensure airfoil coordinates are properly shaped
            x_airfoil = self.xAirfoil.flatten()
            y_airfoil = self.yAirfoil.flatten()

            # Validate coordinates
            all_coords = [x_top, y_top, x_bottom, y_bottom, x_inlet, y_inlet, 
                         x_outlet, y_outlet, x_airfoil, y_airfoil]
            
            if any(np.any(np.isnan(coord)) for coord in all_coords):
                raise ValueError("NaN values detected in boundary coordinates")

            # Update exterior boundaries
            for name, coords in [
                ('top', (x_top, y_top)),
                ('bottom', (x_bottom, y_bottom)),
                ('Inlet', (x_inlet, y_inlet)),
                ('Outlet', (x_outlet, y_outlet))
            ]:
                if name not in self.mesh.boundaries:
                    raise KeyError(f"Boundary '{name}' not initialized")
                self.mesh.boundaries[name].update({
                    'x': coords[0].astype(np.float32),
                    'y': coords[1].astype(np.float32)
                })

            # Update interior boundary (airfoil)
            if 'Airfoil' not in self.mesh.interiorBoundaries:
                raise KeyError("Airfoil boundary not initialized")
            
            self.mesh.interiorBoundaries['Airfoil'].update({
                'x': x_airfoil.astype(np.float32),
                'y': y_airfoil.astype(np.float32)
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
        # Calculate velocity components
        u_inf = float(np.cos(np.radians(self.AoA)))
        v_inf = float(np.sin(np.radians(self.AoA)))
        
        # Initialize exterior boundaries
        self.mesh.boundaries = {
            'Inlet': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': u_inf},
                    'v': {'value': v_inf},
                    'uu': {'value': 0.0},
                    'vv': {'value': 0.0},
                    'uv': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'x'}
                },
                'bc_type': self.inlet_bc
            },
            'Outlet': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': None,
                    'v': None,
                    'p': None
                },
                'bc_type': self.outlet_bc
            },
            'top': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': u_inf},
                    'v': {'value': v_inf},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.wall_bc
            },
            'bottom': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': u_inf},
                    'v': {'value': v_inf},     
                    'uu': {'value': 0.0},
                    'vv': {'value': 0.0},
                    'uv': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.wall_bc
            }
        }
        
        # Initialize interior boundary (airfoil) separately
        self.mesh.interiorBoundaries = {
            'Airfoil': {
                'x': None,
                'y': None,
                'conditions': {
                    'u': {'value': 0.0},  # No-slip condition
                    'v': {'value': 0.0},  # No-slip condition
                    'p': {'gradient': 0.0, 'direction': 'normal'}  # Zero pressure gradient normal to wall
                },
                'bc_type': self.wall_bc,
                'isInterior': True  # Explicitly mark as interior boundary
            }
        }

    def _validate_boundary_conditions(self):
        """Validate boundary conditions before mesh generation."""
        # Check exterior boundaries
        for name, boundary in self.mesh.boundaries.items():
            if any(key not in boundary for key in ['x', 'y', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in boundary {name}")
            if boundary['x'] is None or boundary['y'] is None:
                raise ValueError(f"Coordinates not set for boundary {name}")

        # Check interior boundaries
        for name, boundary in self.mesh.interiorBoundaries.items():
            if any(key not in boundary for key in ['x', 'y', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in interior boundary {name}")
            if boundary['x'] is None or boundary['y'] is None:
                raise ValueError(f"Coordinates not set for interior boundary {name}")

    def getLossFunction(self):
        self.loss = NavierStokesLoss(self.mesh, self.model, physics_model='RANS2D')
    
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
            # Prepare input points correctly
            points = np.column_stack((
                self.mesh.x.flatten(),
                self.mesh.y.flatten()
            ))
            
            # Make predictions
            sol = self.model.predict(points)

            # Store all RANS components
            self.mesh.solutions['U'] = sol[:, 0]
            self.mesh.solutions['V'] = sol[:, 1]
            self.mesh.solutions['P'] = sol[:, 2]
            self.mesh.solutions['uu'] = sol[:, 3]
            self.mesh.solutions['vv'] = sol[:, 4]
            self.mesh.solutions['uv'] = sol[:, 5]

            # Create aliases for backward compatibility
            self.mesh.solutions['u'] = self.mesh.solutions['U']
            self.mesh.solutions['v'] = self.mesh.solutions['V']
            self.mesh.solutions['p'] = self.mesh.solutions['P']

            self.generate_plots()
            self.write_solution()

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def write_solution(self, filename=None):
        """Write the solution to a CSV format file."""
        if filename is None:
            filename = f"{self.problemTag}_AoA_{self.AoA:.1f}.csv"
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
        """Initialize plot object with appropriate variable mappings."""
        self.Plot = Plot(self.mesh)
        # Define RANS variable mapping
        self.rans_variables = {
            'U': 'Mean X Velocity',
            'V': 'Mean Y Velocity',
            'P': 'Mean Pressure',
            'uu': 'Reynolds Stress <u\'u\'>',
            'vv': 'Reynolds Stress <v\'v\'>',
            'uv': 'Reynolds Stress <u\'v\'>'
        }

    def plot(self, solkey='U', streamlines=False):
        """
        Plot RANS flow field variables.

        Args:
            solkey (str): Solution key to plot. Options: 'U', 'V', 'P', 'uu', 'vv', 'uv'
            streamlines (bool): Whether to overlay streamlines (only for velocity)
        """
        if not self.Plot:
            self.generate_plots()

        if solkey not in self.rans_variables:
            raise ValueError(f"Invalid solution key. Must be one of: {list(self.rans_variables.keys())}")

        # Get proper title for the plot
        title = f"{self.rans_variables[solkey]} - AoA: {self.AoA}°"

        # Generate the scatter plot with appropriate title
        self.Plot.scatterPlot(solkey, title=title)

        # Add streamlines for mean velocity components if requested
        if streamlines and solkey in ['U', 'V']:
            self.Plot.addStreamlines(
                u_key='U',  # Use mean velocity components
                v_key='V',
                density=1.0,
                color='white',
                linewidth=0.5,
                arrowsize=1.0
            )

        plt.tight_layout()
        plt.show()

    def save_plots(self, directory="results"):
        """
        Save all RANS variable plots to files.
        
        Args:
            directory (str): Directory to save plots
        """
        import os
        os.makedirs(directory, exist_ok=True)

        for var in self.rans_variables.keys():
            self.plot(solkey=var)
            plt.savefig(os.path.join(directory, f"{self.problemTag}_{var}_AoA{self.AoA}.png"))
            plt.close()

    def plot(self, solkey = 'u', streamlines = False):
        self.Plot.scatterPlot(solkey)

    def load_model(self):
        self.model.load(self.problemTag)
