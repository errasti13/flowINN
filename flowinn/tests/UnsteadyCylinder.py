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

        # Enhanced neural network architecture for unsteady flow
        # Use alternating layer sizes to better capture multi-scale features
        layerSizes = []
        for i in range(6):
            layerSizes.append(128)

        self.mesh = Mesh(self.is2D)
        # Create model with sine activation for better capturing of periodic phenomena
        self.model = PINN(
            input_shape=(3,), 
            output_shape=3, 
            eq=self.problemTag, 
            layers=layerSizes, 
            activation='swish',  # Swish activation for better gradient flow
            learning_rate=0.0005  # Lower learning rate for stability
        )

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

        print(f"\nNormalized data")
        print("xRange: ", self.xRange)
        print("yRange: ", self.yRange)
        print("tRange: ", self.tRange)

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
        self.tCylinder = np.linspace(self.tRange[0], self.tRange[1], 2*N).reshape(-1, 1)

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
            
            # Increase points on the cylinder boundary for better resolution
            self.regenerate_cylinder(NBoundary=NBoundary*2)
            xCylinder, yCylinder = self.xCylinder, self.yCylinder
            
            # Create time points with clustering at the beginning for capturing initial transients
            t_values = np.zeros(NBoundary)
            # Use a nonlinear mapping for better resolution at early times
            beta = 3.0  # Controls clustering (higher = more clustering at t=0)
            for i in range(NBoundary):
                normalized_idx = i / (NBoundary - 1)
                # This gives more points at the beginning of the time range
                t_values[i] = self.tRange[0] + (self.tRange[1] - self.tRange[0]) * (normalized_idx ** beta)
            
            tCylinder = np.repeat(t_values, len(xCylinder) // NBoundary + 1)[:len(xCylinder)].reshape(-1, 1)

            # Validate coordinates
            all_coords = [x_top, y_top, x_bottom, y_bottom, x_inlet, y_inlet, 
                         x_outlet, y_outlet, xCylinder, yCylinder]
            
            if any(np.any(np.isnan(coord)) for coord in all_coords):
                raise ValueError("NaN values detected in boundary coordinates")

            # Update exterior boundaries
            for name, coords in [
                ('top', (x_top, y_top, t_values)),
                ('bottom', (x_bottom, y_bottom, t_values)),
                ('Inlet', (x_inlet, y_inlet, t_values)),
                ('Outlet', (x_outlet, y_outlet, t_values))
            ]:
                if name not in self.mesh.boundaries:
                    raise KeyError(f"Boundary '{name}' not initialized")
                self.mesh.boundaries[name].update({
                    'x': coords[0].astype(np.float32),
                    'y': coords[1].astype(np.float32),
                    't': coords[2].astype(np.float32)
                })

            # Update interior boundary (cylinder)
            if 'Cylinder' not in self.mesh.interiorBoundaries:
                raise KeyError("Cylinder boundary not initialized")
            
            self.mesh.interiorBoundaries['Cylinder'].update({
                'x': xCylinder.astype(np.float32),
                'y': yCylinder.astype(np.float32),
                't': tCylinder.astype(np.float32)
            })

            # Validate boundary conditions before mesh generation
            self._validate_boundary_conditions()

            # Generate the mesh
            self.mesh.generateMesh(
                Nx=Nx,
                Ny=Ny,
                sampling_method=sampling_method
            )

            # Generate time discretization with refined points at start
            t_refined = np.zeros(Nt)
            for i in range(Nt):
                normalized_idx = i / (Nt - 1)
                # Nonlinear mapping for time points (more at the beginning)
                t_refined[i] = self.tRange[0] + (self.tRange[1] - self.tRange[0]) * (normalized_idx ** 2)
            
            self.mesh.t = t_refined
            # Set unsteady flag for proper batching
            self.mesh.is_unsteady = True
            
            # Set up initial conditions at t=0
            self.mesh.initialConditions = {
                'Initial': {
                    'x': self.mesh.x,
                    'y': self.mesh.y,
                    't': np.zeros_like(self.mesh.x),
                    'conditions': {
                        'u': {'value': self.u_inf},
                        'v': {'value': 0.0},
                        # Don't constrain pressure at t=0 - let physics determine it
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
                    'u': {'gradient': 0.0, 'direction': 'x'},
                    'v': {'gradient': 0.0, 'direction': 'x'},
                    'p': {'value': 0.0}  # Set pressure value at outlet
                },
                'bc_type': self.outlet_bc
            },
            'top': {
                'x': None,
                'y': None,
                't': None,
                'conditions': {
                    'u': {'gradient': 0.0, 'direction': 'y'},
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'y'}
                },
                'bc_type': self.wall_bc
            },
            'bottom': {
                'x': None,
                'y': None,
                't': None,
                'conditions': {
                    'u': {'gradient': 0.0, 'direction': 'y'},
                    'v': {'value': 0.0},
                    'p': {'gradient': 0.0, 'direction': 'y'}
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
        # Use specific weights for unsteady flow - emphasize physics more
        self.loss = NavierStokesLoss('unsteady', self.mesh, self.model, Re=self.Re, weights=[0.9, 0.1])
        
        # Access the underlying unsteady loss object to customize it
        unsteady_loss = self.loss
        
        # Set ReLoBraLo parameters for better balance during training
        unsteady_loss.lookback_window = 100  # More history for stability
        unsteady_loss.alpha = 0.05          # Smaller alpha for smoother adjustments
        unsteady_loss.min_weight = 0.2      # Higher minimum physics weight
        unsteady_loss.max_weight = 0.95     # Higher maximum physics weight
        
        return unsteady_loss
    
    def train(self, epochs=10000, print_interval=100, autosaveInterval=10000, num_batches=10, use_cpu=False):
        self.getLossFunction()
        
        # Calculate time window size based on expected flow oscillation period
        # For vortex shedding, we want to capture at least one shedding cycle in a window
        # Strouhal number for cylinder is ~0.2, so period T ≈ 5D/U for D=1 and U=1
        expected_period = 5.0  # Non-dimensional time units
        
        # Time window should be at least 1/4 of the period to capture temporal correlations
        target_window_fraction = 0.25
        
        # Calculate number of time steps corresponding to our target window
        dt = (self.tRange[1] - self.tRange[0]) / len(self.mesh.t)  # Approximate time step size
        time_window_size = max(3, int(target_window_fraction * expected_period / dt))
        
        # Make sure window size is reasonable
        time_window_size = min(time_window_size, 50)  # Cap window size to avoid CUDA memory issues
        
        # First stage: Focus on initial condition and early time steps with higher noise
        print(f"\n=== Stage 1: Training initial condition and early dynamics ===")
        print(f"Using time window size: {time_window_size} steps")
        
        # Save original time range
        original_tRange = self.mesh.t.copy()
        
        # Create a reduced time range for first training stage (first 30% of time)
        early_time_idx = int(len(self.mesh.t) * 0.3)
        self.mesh.t = self.mesh.t[:early_time_idx]
        
        # Initial stage with smaller window size for early dynamics
        early_window_size = min(time_window_size, len(self.mesh.t) // 3)
        early_window_size = max(3, early_window_size)  # Ensure at least 3 time steps
        
        # Start with higher noise level to encourage exploration
        initial_noise_level = 0.02
        
        # Train with focus on early time and higher noise
        self.model.train(
            self.loss.loss_function,
            self.mesh,
            epochs=int(epochs * 0.3),  # 30% of epochs for initial stage
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches,
            plot_loss=False,  # Never plot loss
            patience=2000,
            min_delta=1e-6,
            time_window_size=early_window_size,
            add_noise=False,
            noise_level=initial_noise_level,
            use_cpu=use_cpu  # Pass CPU flag to handle CUDA errors
        )
        
        # Second stage: Train on full time range with medium noise
        print(f"\n=== Stage 2: Training on full time range with medium noise ===")
        print(f"Using time window size: {time_window_size} steps")
        
        # Restore original time range
        self.mesh.t = original_tRange
        
        # Medium noise level for main training phase
        medium_noise_level = 0.01
        
        # Continue training with full time range, full window size, and medium noise
        self.model.train(
            self.loss.loss_function,
            self.mesh,
            epochs=int(epochs * 0.5),  # 50% of epochs for main training
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches,
            plot_loss=False,  # Never plot loss
            patience=2000,
            min_delta=1e-7,
            time_window_size=time_window_size,
            add_noise=True,
            noise_level=medium_noise_level,
            use_cpu=use_cpu  # Pass CPU flag to handle CUDA errors
        )
        
        # Third stage: Fine-tuning with minimal noise
        print(f"\n=== Stage 3: Fine-tuning with minimal noise ===")
        
        # Low noise level for fine-tuning
        final_noise_level = 0.001
        
        # Final fine-tuning with minimal noise
        self.model.train(
            self.loss.loss_function,
            self.mesh,
            epochs=int(epochs * 0.2),  # 20% of epochs for fine-tuning
            print_interval=print_interval,
            autosave_interval=autosaveInterval,
            num_batches=num_batches,
            plot_loss=False,  # Never plot loss
            patience=1000,
            min_delta=1e-8,
            time_window_size=time_window_size,
            add_noise=True,
            noise_level=final_noise_level,
            use_cpu=use_cpu  # Pass CPU flag to handle CUDA errors
        )

    def predict(self, use_cpu=False) -> None:
        """
        Predict flow solution and create animation frames using GPU.
        """
        try:
            # Force GPU usage - ignore use_cpu parameter
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Remove any restrictions
            
            import tensorflow as tf
            # Clear any existing session/graph
            tf.keras.backend.clear_session()
            
            # Set memory growth to avoid OOM errors
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Using GPU: {gpus[0].name}")
                except RuntimeError as e:
                    print(f"GPU memory config error: {e}")
            
            nt = 100  # Number of time steps for visualization
            tPred = np.linspace(self.tRange[0], self.tRange[1], nt)

            print("Predicting flow field from ", tPred[0], " to ", tPred[-1])
            print("Using GPU for prediction - no fallbacks to CPU")

            # Prepare mesh points
            x_min, x_max = self.xRange
            y_min, y_max = self.yRange
            
            # Create a denser uniform grid for visualization
            nx, ny = 150, 150  # Increased resolution for visualization
            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            
            # Create mesh grid
            X_mesh, Y_mesh = np.meshgrid(x, y)
            
            # Flatten for prediction
            X_flat = X_mesh.flatten()
            Y_flat = Y_mesh.flatten()
            
            # Create directory for animation frames
            os.makedirs('animation_frames', exist_ok=True)
            
            # Initialize plotting
            self.generate_plots()

            print("Generating animation frames...")
            
            # Create mask for cylinder
            cylinder_mask = np.zeros_like(X_flat, dtype=bool)
            for i in range(len(X_flat)):
                dist = np.sqrt((X_flat[i] - self.x0)**2 + (Y_flat[i] - self.y0)**2)
                cylinder_mask[i] = dist <= self.R
            
            # For GPU stability, process time steps in smaller chunks
            # to avoid memory issues while staying on GPU
            chunk_size = 10  # Process this many time steps at once
            
            for chunk_start in range(0, len(tPred), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(tPred))
                chunk_tPred = tPred[chunk_start:chunk_end]
                
                print(f"\nProcessing time chunk {chunk_start//chunk_size + 1}/{(len(tPred)+chunk_size-1)//chunk_size}")
                
                # Process each time step in this chunk
                for i, t in enumerate(chunk_tPred):
                    global_i = chunk_start + i
                    print(f"  Time step {global_i+1}/{len(tPred)}: t = {t:.4f}")
                    
                    # Create input tensor for current time
                    time_values = np.full_like(X_flat, t)
                    X_input = np.column_stack([X_flat, Y_flat, time_values])
                    
                    # Convert to TensorFlow tensor
                    X_input_tf = tf.convert_to_tensor(X_input, dtype=tf.float32)
                    
                    # Force prediction on GPU - direct model call to avoid any device switching
                    with tf.device('/device:GPU:0'):
                        predictions = self.model.model(X_input_tf, training=False).numpy()
                    
                    # Extract velocity components and pressure
                    u = predictions[:, 0].reshape(ny, nx)
                    v = predictions[:, 1].reshape(ny, nx)
                    p = predictions[:, 2].reshape(ny, nx)
                    
                    # Calculate velocity magnitude for this time step
                    vel_mag = np.sqrt(u**2 + v**2)
                    
                    # Calculate vorticity (curl of velocity field) for this time step
                    vorticity = self.calculate_vorticity(u, v, x, y)
                    
                    # Create mask for cylinder in 2D arrays
                    mask_2d = cylinder_mask.reshape(ny, nx)
                    
                    # Apply mask to fields (set values inside cylinder to NaN for visualization)
                    u = np.where(mask_2d, np.nan, u)
                    v = np.where(mask_2d, np.nan, v)
                    p = np.where(mask_2d, np.nan, p)
                    vel_mag = np.where(mask_2d, np.nan, vel_mag)
                    vorticity = np.where(mask_2d, np.nan, vorticity)
                    
                    # Plot and save velocity field
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
                    
                    # Velocity magnitude plot
                    cm1 = ax1.pcolormesh(X_mesh, Y_mesh, vel_mag, cmap='viridis', shading='auto')
                    plt.colorbar(cm1, ax=ax1, label='Velocity Magnitude')
                    
                    # Add velocity vectors (subsample for clarity)
                    skip = 8
                    ax1.quiver(X_mesh[::skip, ::skip], Y_mesh[::skip, ::skip], 
                              u[::skip, ::skip], v[::skip, ::skip], 
                              scale=25, alpha=0.7)
                    
                    # Draw cylinder
                    circle = plt.Circle((self.x0, self.y0), self.R, color='white', fill=True)
                    ax1.add_patch(circle)
                    ax1.set_aspect('equal')
                    ax1.set_title(f'Velocity Field at t = {t:.4f}')
                    ax1.set_xlabel('x')
                    ax1.set_ylabel('y')
                    
                    # Vorticity plot
                    cm2 = ax2.pcolormesh(X_mesh, Y_mesh, vorticity, cmap='RdBu_r', 
                                        shading='auto', vmin=-5, vmax=5)
                    plt.colorbar(cm2, ax=ax2, label='Vorticity')
                    
                    # Draw cylinder on vorticity plot
                    circle2 = plt.Circle((self.x0, self.y0), self.R, color='black', fill=True)
                    ax2.add_patch(circle2)
                    ax2.set_aspect('equal')
                    ax2.set_title(f'Vorticity at t = {t:.4f}')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                    
                    plt.tight_layout()
                    
                    # Save frame
                    plt.savefig(f'animation_frames/flow_t{global_i:04d}.png', dpi=150)
                    plt.close()
                
                # Free memory between chunks
                tf.keras.backend.clear_session()
                
                # Re-initialize model on GPU
                with tf.device('/device:GPU:0'):
                    tf.keras.backend.clear_session()
                
            # Create a simple animation command suggestion
            print("\nTo create an animation from the saved frames, you can use:")
            print("ffmpeg -framerate 10 -i animation_frames/flow_t%04d.png -c:v libx264 -pix_fmt yuv420p cylinder_flow.mp4")
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def calculate_vorticity(self, u, v, x_grid, y_grid):
        """Calculate vorticity (curl of velocity field) using finite differences."""
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        
        # Calculate partial derivatives
        # Use central differences for interior points
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        
        # Interior points using central difference
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
        
        # Edge points using forward/backward differences
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx
        
        # Vorticity (ω = dv/dx - du/dy)
        vorticity = dv_dx - du_dy
        
        return vorticity

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
