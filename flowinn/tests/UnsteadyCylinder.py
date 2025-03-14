import numpy as np
import logging
from typing import Tuple
from flowinn.mesh.mesh import Mesh
from flowinn.models.model import PINN
from flowinn.training.loss import NavierStokesLoss
from flowinn.plot.plot import Plot
from flowinn.physics.boundary_conditions import InletBC, OutletBC, WallBC 
import matplotlib.pyplot as plt
import os

class UnsteadyCylinder:
    def __init__(self, caseName: str, xRange: Tuple[float, float], yRange: Tuple[float, float], tRange: Tuple[float, float]):
        """
        Initialize FlowOverAirfoil simulation.
        
        Args:
            caseName: Name of the simulation case
            xRange: Tuple of (min_x, max_x) domain bounds
            yRange: Tuple of (min_y, max_y) domain bounds
            tRange: Tuple of (min_t, max_t) time bounds
            Re: Reynolds number
        """
        if not isinstance(caseName, str):
            raise TypeError("caseName must be a string")
        if not all(isinstance(x, (int, float)) for x in [*xRange, *yRange]):
            raise TypeError("xRange, yRange, tRange must be numeric")
            
        self.logger = logging.getLogger(__name__)
        self.is2D = True
        self.problemTag = caseName

        layerSizes = []
        for i in range(6):
            layerSizes.append(64)

        self.mesh = Mesh(self.is2D)
        self.model = PINN(input_shape=(3,), output_shape=3, eq = self.problemTag, layers=layerSizes)

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange
        self.tRange = tRange

        self.R   = 1.0  #Cylinder radius
        self.x0  = 0.0  #Cylinder center x coordinate
        self.y0  = 0.0  #Cylinder center y coordinate

        self.generate_cylinder()

        self.u_inf = 1.0 #Freestream velocity
        self.P_inf = 101325 #Freestream pressure
        self.rho = 1.225 #Freestream density
        self.nu = 0.1 #Kinematic viscosity
        self.L_ref = xRange[1] - xRange[0] #Length of domain

        self.Re = self.L_ref * self.u_inf / self.nu #Reynolds number
        print(f"Reynolds number: {self.Re}")

        self.normalize_data()

        # Initialize boundary condition objects
        self.inlet_bc = InletBC("inlet")
        self.outlet_bc = OutletBC("outlet")
        self.wall_bc = WallBC("wall")

        return
    
    def normalize_data(self):
        """
        Normalize the airfoil coordinates and freestream velocity.
        """
        self.xCylinder = (self.xCylinder) / self.L_ref
        self.yCylinder = (self.yCylinder) / self.L_ref

        self.xRange = (self.xRange[0] / self.L_ref, self.xRange[1] / self.L_ref)
        self.yRange = (self.yRange[0] / self.L_ref, self.yRange[1] / self.L_ref)
        self.tRange = (self.u_inf * self.tRange[0] / self.L_ref, 
                       self.u_inf * self.tRange[1] / self.L_ref)

        self.P_inf = self.P_inf / (self.rho * self.u_inf**2)

        return
    
    def generate_cylinder(self, N=100):
        """
        Generate cylinder coordinates for both upper and lower surfaces.
        """
        # Generate x coordinates from left to right edge of cylinder
        x = np.linspace(self.x0 - self.R, self.x0 + self.R, N)
        
        y_upper = self.y0 + np.sqrt(self.R**2 - (x - self.x0)**2)
        y_lower = self.y0 - np.sqrt(self.R**2 - (x - self.x0)**2)
        
        self.xCylinder = np.concatenate([np.flip(x), x]).reshape(-1, 1)
        self.yCylinder = np.concatenate([np.flip(y_upper), y_lower]).reshape(-1, 1)

        return
    
    def regenerate_cylinder(self, NBoundary=100):
        self.generate_cylinder(N=NBoundary//2)  # Half points for upper and lower
        self.normalize_data() #Normalize again for the new set of values.

    def generateMesh(self, Nx: int = 100, Ny: int = 100, Nt: int = 100, NBoundary: int = 100, sampling_method: str = 'random'):
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
            
            #Regenerate airfoil coordinates
            self.regenerate_cylinder(NBoundary=NBoundary)
            xCylinder, yCylinder = self.xCylinder, self.yCylinder

            t = np.linspace(self.tRange[0], self.tRange[1], NBoundary)

            # Validate coordinates
            all_coords = [x_top, y_top, x_bottom, y_bottom, x_inlet, y_inlet, 
                         x_outlet, y_outlet, xCylinder, yCylinder]
            
            if any(np.any(np.isnan(coord)) for coord in all_coords):
                raise ValueError("NaN values detected in boundary coordinates")

            # Update exterior boundaries
            for name, coords in [
                ('top', (x_top, y_top, t)),
                ('bottom', (x_bottom, y_bottom, t)),
                ('Inlet', (x_inlet, y_inlet, t)),
                ('Outlet', (x_outlet, y_outlet, t))
            ]:
                if name not in self.mesh.boundaries:
                    raise KeyError(f"Boundary '{name}' not initialized")
                self.mesh.boundaries[name].update({
                    'x': coords[0].astype(np.float32),
                    'y': coords[1].astype(np.float32),
                    't': coords[2].astype(np.float32)
                })

            # Update interior boundary (airfoil)
            if 'Cylinder' not in self.mesh.interiorBoundaries:
                raise KeyError("Cylinder boundary not initialized")
            
            self.mesh.interiorBoundaries['Cylinder'].update({
                'x': xCylinder.astype(np.float32),
                'y': yCylinder.astype(np.float32),
                't': t.astype(np.float32)
            })

            # Validate boundary conditions before mesh generation
            self._validate_boundary_conditions()

            # Generate the mesh
            self.mesh.generateMesh(
                Nx=Nx,
                Ny=Ny,
                sampling_method=sampling_method
            )

            self.mesh.generate_time_discretization(t_range=self.tRange, Nt=Nt)

            self.mesh.initialConditions = {
                'Initial': {
                    'x': self.mesh.x,
                    'y': self.mesh.y,
                    't': np.zeros_like(self.mesh.x),
                    'conditions': {
                        'u': {'value': self.u_inf},
                        'v': {'value': 0.0},
                        'p': None
                    }
                }
            }
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
                't': None,
                'conditions': {
                    'u': {'value': self.u_inf},
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'x'}
                },
                'bc_type': self.inlet_bc
            },
            'Outlet': {
                'x': None,
                'y': None,
                't': None,
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
                't': None,
                'conditions': {
                    'u': None,
                    'v': None,
                    'p': None
                },
                'bc_type': self.wall_bc
            },
            'bottom': {
                'x': None,
                'y': None,
                't': None,
                'conditions': {
                    'u': None,
                    'v': None,
                    'p': None
                },
                'bc_type': self.wall_bc
            }
        }
        
        # Initialize interior boundary (airfoil) separately
        self.mesh.interiorBoundaries = {
            'Cylinder': {
                'x': None,
                'y': None,
                't': None,
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
            if any(key not in boundary for key in ['x', 'y', 't', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in boundary {name}")
            if boundary['x'] is None or boundary['y'] is None or boundary['t'] is None:
                raise ValueError(f"Coordinates not set for boundary {name}")

        # Check interior boundaries
        for name, boundary in self.mesh.interiorBoundaries.items():
            if any(key not in boundary for key in ['x', 'y', 't', 'conditions', 'bc_type']):
                raise ValueError(f"Missing required fields in interior boundary {name}")
            if boundary['x'] is None or boundary['y'] is None or boundary['t'] is None:
                raise ValueError(f"Coordinates not set for interior boundary {name}")

    def getLossFunction(self):
        self.loss = NavierStokesLoss('unsteady', self.mesh, self.model, Re=self.Re)
    
    def train(self, epochs=10000, print_interval=100, autosaveInterval=10000, num_batches=10):
        self.getLossFunction()
        self.model.train(
            self.loss.loss_function,
            self.mesh,
            epochs=epochs,
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches,
            patience=1000,
            min_delta=1e-4
        )

    def predict(self) -> None:
        """Predict flow solution and generate plots."""
        try:
            nt = 10
            tPred = np.linspace(self.tRange[0], self.tRange[1], nt)

            x = self.mesh.x.flatten()  # Make sure x is flattened
            y = self.mesh.y.flatten()  # Make sure y is flattened

            # Store all results for later visualization
            all_u = []
            all_v = []
            all_p = []
            all_t = []

            for i, t in enumerate(tPred):
                # Create time vector
                tVec = t * np.ones_like(x)
                
                # Stack coordinates
                X = np.hstack((x[:, np.newaxis], y[:, np.newaxis], tVec[:, np.newaxis]))
                
                # Make predictions
                predictions = self.model.predict(X)
                
                # Store predictions for this time step
                self.mesh.solutions['u'] = predictions[:, 0]  # First column
                self.mesh.solutions['v'] = predictions[:, 1]  # Second column
                self.mesh.solutions['p'] = predictions[:, 2]  # Third column
                
                # Save for visualization
                all_u.append(predictions[:, 0].copy())
                all_v.append(predictions[:, 1].copy())
                all_p.append(predictions[:, 2].copy())
                all_t.append(t)
                
                # Generate plots with current time in the title
                self.generate_plots()
                self.plot()


            # Store all time steps for later use if needed
            self.time_series = {
                'u': all_u,
                'v': all_v, 
                'p': all_p,
                't': all_t
            }

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    
    def write_solution(self, filename=None):
        """Write the solution to a CSV format file."""
        if filename is None:
            filename = f"{self.problemTag}_solution.csv"
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
        """Initialize plotting object with custom styling."""
        self.Plot = Plot(self.mesh)
        self.Plot.set_style(
            figsize=(12, 8),
            cmap='viridis',
            scatter_size=15,
            scatter_alpha=0.7,
            font_size=12,
            title_size=14,
            colorbar_label_size=10,
            axis_label_size=11
        )

    def plot(self, solkey='u', plot_type='default', **kwargs):
        """
        Create various types of plots for the solution fields.
        
        Args:
            solkey (str): Solution field to plot ('u', 'v', 'p', 'vMag')
            plot_type (str): Type of plot to create:
                - 'default': Scatter plot with boundaries
                - 'quiver': Vector field plot showing flow direction
                - 'slices': Multiple y-plane slices (3D only)
            **kwargs: Additional plotting parameters
        """
        if not hasattr(self, 'Plot'):
            self.generate_plots()

        if plot_type == 'quiver':
            self.Plot.vectorField(xRange=self.xRange, yRange=self.yRange, **kwargs)
        elif plot_type == 'default':
            self.Plot.scatterPlot(solkey)
        elif plot_type == 'slices':
            if self.mesh.is2D:
                raise ValueError("Slice plotting is only available for 3D meshes")
            self.Plot.plotSlices(solkey, **kwargs)
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")

    def export_plots(self, directory="results", format=".png"):
        """
        Export all solution field plots to files.
        
        Args:
            directory (str): Directory to save plots
            format (str): File format ('.png', '.pdf', '.svg', etc.)
        """
        os.makedirs(directory, exist_ok=True)
        
        # Ensure all fields are available
        if not all(key in self.mesh.solutions for key in ['u', 'v', 'p', 'vMag']):
            raise ValueError(
                "Missing required solution components. "
                "Make sure to run predict() before exporting plots."
            )
        
        # Ensure Plot object exists
        if not hasattr(self, 'Plot'):
            self.generate_plots()
        
        # Export standard field plots
        for solkey in ['u', 'v', 'p', 'vMag']:
            plt.figure()
            self.Plot.scatterPlot(solkey)
            filename = os.path.join(directory, f"{self.problemTag}_AoA{self.AoA}_{solkey}{format}")
            plt.savefig(filename, bbox_inches='tight', dpi=self.Plot.style['dpi'])
            plt.close()
        
        # Export vector field plot
        plt.figure(figsize=self.Plot.style['figsize'])
        self.plot(plot_type='quiver')
        filename = os.path.join(directory, f"{self.problemTag}_AoA{self.AoA}_quiver{format}")
        plt.savefig(filename, bbox_inches='tight', dpi=self.Plot.style['dpi'])
        plt.close()

    def load_model(self):
        self.model.load(self.problemTag)
