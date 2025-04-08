import numpy as np
import logging
from typing import Tuple
from flowinn.mesh.mesh import Mesh
from flowinn.models.model import PINN
from flowinn.training.loss import NavierStokesLoss
from flowinn.plot.plot import Plot
from flowinn.physics.boundary_conditions import InletBC, OutletBC, WallBC 
from flowinn.nn.architectures import create_mfn_model  # Import network architecture from source code
from flowinn.training.window_trainer import TimeWindowTrainer  # Import time window trainer from source code
import os

class UnsteadyCylinder:
    """
    Test case for unsteady flow around a cylinder using PINN with time window approach.
    This test demonstrates the Sequential Moving Time Windows approach.
    """
    def __init__(self, caseName: str, xRange: Tuple[float, float], yRange: Tuple[float, float], tRange: Tuple[float, float], 
                activation='gelu', architecture='mfn'):
        """
        Initialize UnsteadyCylinder simulation.
        
        Args:
            caseName: Name of the simulation case
            xRange: Tuple of (min_x, max_x) domain bounds
            yRange: Tuple of (min_y, max_y) domain bounds
            tRange: Tuple of (min_t, max_t) time bounds
            activation: Activation function to use ('gelu', 'tanh', 'swish')
            architecture: Neural network architecture ('mlp', 'mfn', 'fast_mfn')
        """
        if not isinstance(caseName, str):
            raise TypeError("caseName must be a string")
        if not all(isinstance(x, (int, float)) for x in [*xRange, *yRange]):
            raise TypeError("xRange, yRange, tRange must be numeric")
            
        self.logger = logging.getLogger(__name__)
        self.is2D = True
        self.problemTag = caseName
        self.architecture = architecture
        self.activation = activation
        
        # Initialize mesh
        self.mesh = Mesh(self.is2D)
        
        # Initialize model with appropriate architecture from source code module
        if architecture == 'mfn':
            # Regular MFN (more accurate but slower)
            self.model = create_mfn_model(
                input_shape=(3,),
                output_shape=3,
                eq=self.problemTag,
                activation=activation,
                fourier_dim=32,
                layer_sizes=[256] * 6,
                learning_rate=0.001,
                trainable_fourier=False,
                use_adaptive_activation=True
            )
        elif architecture == 'fast_mfn':
            # Fast MFN with reduced parameters for better performance
            print("Using performance-optimized MFN architecture")
            self.model = create_mfn_model(
                input_shape=(3,),
                output_shape=3,
                eq=self.problemTag,
                activation='swish',  # Faster activation function
                fourier_dim=16,      # Fewer Fourier features
                layer_sizes=[64] * 4,  # Smaller network
                learning_rate=0.001
            )
        else:
            self.model = PINN(
                input_shape=(3,), 
                output_shape=3, 
                eq=self.problemTag, 
                    layers=[128] * 6,
                    activation=activation,
                    learning_rate=0.001
            )

        self.loss = None
        self.Plot = None
        
        self.xRange = xRange
        self.yRange = yRange
        self.tRange = tRange

        self.R = 1.0  # Cylinder radius
        self.x0 = 0.0  # Cylinder center x coordinate
        self.y0 = 0.0  # Cylinder center y coordinate

        self.generate_cylinder()

        self.u_inf = 1.0  # Freestream velocity
        self.P_inf = 101325  # Freestream pressure
        self.rho = 1.225  # Freestream density
        self.nu = 1.0  # Kinematic viscosity
        self.L_ref = xRange[1] - xRange[0]  # Length of domain

        self.Re = self.L_ref * self.u_inf / self.nu  # Reynolds number
        print(f"Reynolds number: {self.Re}")

        # Initialize boundary condition objects
        self.inlet_bc = InletBC("inlet")
        self.outlet_bc = OutletBC("outlet")
        self.wall_bc = WallBC("wall")
        
        # Initialize time window trainer with parameters from paper
        self.trainer = TimeWindowTrainer(
            num_windows=3,
            initial_lr=0.001,
            lr_decay_factor=0.999947,
            max_epochs=100000
        )

        return
    
    def normalize_data(self):
        """
        Normalize the cylinder coordinates and freestream velocity.
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
        self.normalize_data() # Normalize again for the new set of values.

    def generateMesh(self, Nx: int = 100, Ny: int = 100, Nt: int = 100, NBoundary: int = 100, sampling_method: str = 'random'):
        """Generate mesh for the unsteady cylinder flow problem."""
        try:
            if not all(isinstance(x, int) and x > 0 for x in [Nx, Ny, NBoundary]):
                raise ValueError("Nx, Ny, and NBoundary must be positive integers")
            if sampling_method not in ['random', 'uniform']:
                raise ValueError("sampling_method must be 'random' or 'uniform'")

            # Increase points on the cylinder boundary for better resolution
            self.regenerate_cylinder(NBoundary=NBoundary*2)
            xCylinder, yCylinder = self.xCylinder, self.yCylinder
                
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
            
            # Create time points with clustering at the beginning for capturing initial transients
            t_values = np.zeros(NBoundary)
            # Use a nonlinear mapping for better resolution at early times
            beta = 3.0  # Controls clustering (higher = more clustering at t=0)
            for i in range(NBoundary):
                normalized_idx = i / (NBoundary - 1)
                # This gives more points at the beginning of the time range
                t_values[i] = self.tRange[0] + (self.tRange[1] - self.tRange[0]) * (normalized_idx ** beta)
            
            tCylinder = np.repeat(t_values, len(xCylinder) // NBoundary + 1)[:len(xCylinder)].reshape(-1, 1)

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
            
            # Validate boundary conditions
            self._validate_boundary_conditions()
            
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
                    'u': {'value': 1.0},
                    'v': {'value': 0.0},
                    'p': {'value': 0.0}  # Set pressure value at outlet
                },
                'bc_type': self.outlet_bc
            },
            'top': {
                'x': None,
                'y': None,
                't': None,
                'conditions': {
                    'u': {'value': 1.0},
                    'v': {'value': 0.0},
                    'p': None
                },
                'bc_type': self.wall_bc
            },
            'bottom': {
                'x': None,
                'y': None,
                't': None,
                'conditions': {
                    'u': {'value': 1.0},
                    'v': {'value': 0.0},
                    'p': None
                },
                'bc_type': self.wall_bc
            }
        }
        
        # Initialize interior boundary (cylinder) separately
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
                
            # For boundaries with coordinates set, check them
            if all(coord in boundary and boundary[coord] is not None for coord in ['x', 'y', 't']):
                # Check that all coordinate arrays have the same shape
                shapes = [np.asarray(boundary[coord]).shape for coord in ['x', 'y', 't']]
                if not all(shape == shapes[0] for shape in shapes):
                    raise ValueError(f"Coordinate shape mismatch in boundary {name}: {shapes}")

        # Check interior boundaries
        if hasattr(self.mesh, 'interiorBoundaries'):
            for name, boundary in self.mesh.interiorBoundaries.items():
                if any(key not in boundary for key in ['x', 'y', 't', 'conditions', 'bc_type']):
                    raise ValueError(f"Missing required fields in interior boundary {name}")
                    
                # For boundaries with coordinates set, check them
                if all(coord in boundary and boundary[coord] is not None for coord in ['x', 'y', 't']):
                    # Check that all coordinate arrays have the same shape
                    shapes = [np.asarray(boundary[coord]).shape for coord in ['x', 'y', 't']]
                    if not all(shape == shapes[0] for shape in shapes):
                        raise ValueError(f"Coordinate shape mismatch in interior boundary {name}: {shapes}")

        # Check initial conditions if they exist
        if hasattr(self.mesh, 'initialConditions'):
            for name, ic_data in self.mesh.initialConditions.items():
                if any(key not in ic_data for key in ['x', 'y', 't', 'conditions']):
                    raise ValueError(f"Missing required fields in initial condition {name}")
                
                # Check that all coordinate arrays have the same shape
                shapes = [np.asarray(ic_data[coord]).shape for coord in ['x', 'y', 't']]
                if not all(shape == shapes[0] for shape in shapes):
                    raise ValueError(f"Coordinate shape mismatch in initial condition {name}: {shapes}")

    def getLossFunction(self):
        """
        Configure the Navier-Stokes loss function for unsteady flow.
        
        Returns:
            A callable loss function
        """
        # Use specific weights for unsteady flow - emphasize physics more
        self.loss = NavierStokesLoss('unsteady', self.mesh, self.model, Re=self.Re, weights=[0.9, 0.1])
        
        # Set ReLoBraLo parameters for better balance during training if available
        if hasattr(self.loss, 'lookback_window'):
            self.loss.lookback_window = 100  # More history for stability
        if hasattr(self.loss, 'alpha'):
            self.loss.alpha = 0.05         # Smaller alpha for smoother adjustments
        if hasattr(self.loss, 'min_weight'):
            self.loss.min_weight = 0.2     # Higher minimum physics weight
        if hasattr(self.loss, 'max_weight'):
            self.loss.max_weight = 0.95    # Higher maximum physics weight
        
        # Make sure we're returning a callable function
        if hasattr(self.loss, 'loss_function') and callable(self.loss.loss_function):
            return self.loss.loss_function  # Return the method
        else:
            # Fallback: create a wrapper function if loss_obj is callable
            if callable(self.loss):
                return self.loss  # Return the object if it's callable
            else:
                # Last resort: create a wrapper around the loss object
                def loss_wrapper(batch_data=None):
                    return self.loss(batch_data)
                return loss_wrapper
    
    def train(self, epochs=None, print_interval=100, autosaveInterval=10000, num_batches=10, 
            num_spatial_batches=4, num_temporal_batches=3):
        """
        Train the model using moving time window approach from the paper.
        
        Args:
            epochs: Maximum epochs per window (default uses value from paper: 100000)
            print_interval: Interval for printing loss information
            autosaveInterval: Interval for autosaving the model
            num_batches: Number of batches for mini-batch training
            num_spatial_batches: Number of batches to divide spatial domain into
            num_temporal_batches: Number of batches to divide temporal domain into
        """
        try:
            # Get loss function
            loss_function = self.getLossFunction()
            
            print("\nStarting training with moving time window approach...")
            print(f"Using architecture: {self.architecture}, activation: {self.activation}")
            print(f"Max epochs per window: {epochs if epochs else 'default'}")
            print(f"Memory-efficient batching: {num_spatial_batches} spatial x {num_temporal_batches} temporal batches")
            
                
            # Use the TimeWindowTrainer to train the model with moving time windows
            # This uses the implementation from the paper with:
            # - 3 time windows
            # - Learning rate decay with gamma = 0.999947
            # - Initial conditions transferred between windows
            history = self.trainer.train(
                model=self.model,
                loss_function=loss_function,
                mesh=self.mesh,
                tRange=self.tRange,
                epochs=epochs,
                print_interval=print_interval,
                autosave_interval=autosaveInterval,
                num_batches=num_batches,
                save_name=self.problemTag,
                num_spatial_batches=num_spatial_batches,
                num_temporal_batches=num_temporal_batches,
                patience=20
            )
            
            print(f"Training completed.")
            
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Plot training history
            if hasattr(self.trainer, 'plot_loss_history'):
                self.trainer.plot_loss_history(save_path=f'plots/{self.problemTag}_loss_history.png')
                
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self):
        """Predict flow solution over the entire time domain."""
        try:
            # Initialize plotting object
            self.generate_plots()

            nt = 100
            t = np.linspace(self.tRange[0], self.tRange[1], nt)

            X_pred = self.mesh.x.flatten()[:, None]
            Y_pred = self.mesh.y.flatten()[:, None]

            nx, ny = int(np.sqrt(len(X_pred))), int(np.sqrt(len(X_pred)))
            
            # Initialize storage for all solutions
            self.all_solutions = {
                'u': np.zeros((nx, ny, nt)),
                'v': np.zeros((nx, ny, nt)),
                'p': np.zeros((nx, ny, nt)),
                'vMag': np.zeros((nx, ny, nt)),
                'vorticity': np.zeros((nx, ny, nt))
            }
            
            # Store prediction coordinates for later use
            self.pred_X_mesh = X_pred
            self.pred_Y_mesh = Y_pred
            self.pred_t = t
            
            # For each time step
            for i, current_t in enumerate(t):
                print(f"  Time step {i+1}/{nt}: t = {current_t:.4f}")
                
                # Create input tensor with current time value
                T_pred = np.full_like(X_pred, current_t)
                X = np.hstack((X_pred, Y_pred, T_pred))
                
                # Get solution from model
                sol = self.model.predict(X)

                # Store primary solution components for current timestep
                self.mesh.solutions['u'] = sol[:, 0]
                self.mesh.solutions['v'] = sol[:, 1]
                self.mesh.solutions['p'] = sol[:, 2]

                # Calculate and store velocity magnitude
                self.mesh.solutions['vMag'] = np.sqrt(
                    self.mesh.solutions['u']**2 + 
                    self.mesh.solutions['v']**2
                )
            
                # Reshape and store the solutions for this timestep
                self.all_solutions['u'][:, :, i] = sol[:, 0].reshape(nx, ny)
                self.all_solutions['v'][:, :, i] = sol[:, 1].reshape(nx, ny)
                self.all_solutions['p'][:, :, i] = sol[:, 2].reshape(nx, ny)
                self.all_solutions['vMag'][:, :, i] = self.mesh.solutions['vMag'].reshape(nx, ny)
            
            print("Prediction completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Prediction failed: {str(e)}") from e
    
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

    def plot(self, solkey='vMag', title='', savePath=None, show=False):
        """
        Plot the solution field at a specific time index.
        """
        self.Plot.scatterPlot(solkey, title=title, savePath=savePath, show=show)

    def animate_flow(self, solkey='vMag', start_idx=0, end_idx=None, skip=1, 
                   output_file='cylinder_flow.mp4', **kwargs):
        """
        Create an animation of the flow field.
        """
        # Check if prediction data exists in memory
        if not hasattr(self, 'all_solutions') or not self.all_solutions:
            print("Prediction data not found in memory. Run predict() first.")
            return
            
        # Extract time steps from stored prediction data
        t = self.pred_t
        num_time_steps = len(t)
        
        if end_idx is None:
            end_idx = num_time_steps
        else:
            end_idx = min(end_idx, num_time_steps)
            
        # Validate indices
        if not (0 <= start_idx < num_time_steps):
            print(f"Invalid start_idx {start_idx}. Must be between 0 and {num_time_steps-1}")
            return
            
        if end_idx <= start_idx:
            print(f"Invalid end_idx {end_idx}. Must be greater than start_idx ({start_idx})")
            return
        
        # Create animation_frames directory
        frames_dir = 'animation_frames'
        os.makedirs(frames_dir, exist_ok=True)
        
        # Generate frames
        frame_count = 0
        total_frames = (end_idx - start_idx + skip - 1) // skip
        for i in range(start_idx, end_idx, skip):
            frame_count += 1
            print(f"Generating frame {frame_count}/{total_frames}: time step {i}")
            
            # Get the solution data for this time step
            solution_at_time = self.all_solutions[solkey][:, :, i]
            current_time = self.pred_t[i]
            
            self.mesh.solutions = {solkey: solution_at_time.flatten()}
            
            # Set frame filename
            frame_filename = f'frame_{i:04d}.png'
            frame_path = os.path.join(frames_dir, frame_filename)
            
            # Create and save the frame using plot function
            title = f"{solkey} at t = {current_time:.4f}"
            self.plot(solkey=solkey, title=title, savePath=frame_path, show=False)
            
        # Clean up temporary data in mesh to avoid confusion
        self.mesh.solutions = {}
            
        # Display ffmpeg command if frames were generated
        if frame_count > 0:
            print(f"\nGenerated {frame_count} frames in '{frames_dir}/'")
            print("To create an animation (requires ffmpeg), run a command like this in your terminal:")
            ffmpeg_input_path = os.path.join(frames_dir, 'frame_%04d.png') 
            print(f"ffmpeg -framerate 10 -i {ffmpeg_input_path} -c:v libx264 -pix_fmt yuv420p {output_file}")

    def load_model(self):
        """
        Load a trained model with architecture-specific handling.
        """
        try:
            # Try to load model metadata to determine architecture
            import os
            import json
            
            metadata_path = f'trainedModels/{self.problemTag}_metadata.json'
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if architecture info is stored in metadata
                    stored_architecture = metadata.get('architecture_type', None)
                    if stored_architecture:
                        print(f"Loading model with architecture: {stored_architecture}")
                        self.architecture = stored_architecture
                except Exception as e:
                    print(f"Warning: Could not load architecture from metadata: {e}")
            
            # Load model based on the architecture
            if self.architecture in ('mfn', 'fast_mfn'):
                # For MFN architectures, we need to recreate the proper structure first
                # then load weights into it
                from flowinn.nn.architectures import create_mfn_model
                
                print(f"Re-creating {self.architecture} structure before loading...")
                
                # Create structure with appropriate parameters
                if self.architecture == 'fast_mfn':
                    self.model = create_mfn_model(
                        input_shape=(3,),
                        output_shape=3,
                        eq=self.problemTag,
                        activation='swish',
                        fourier_dim=16,
                        layer_sizes=[64] * 4,
                        learning_rate=0.001
                    )
                else:
                    self.model = create_mfn_model(
                        input_shape=(3,),
                        output_shape=3,
                        eq=self.problemTag,
                        activation=self.activation,
                        fourier_dim=32,
                        layer_sizes=[128] * 6,
                        learning_rate=0.001,
                        trainable_fourier=False,
                        use_adaptive_activation=True
                    )
                
                # Now load the trained weights into this structure
                model_path = f'trainedModels/{self.problemTag}.keras'
                if os.path.exists(model_path):
                    self.model.model.load_weights(model_path)
                    print(f"Successfully loaded weights into {self.architecture} model")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            else:
                # For standard PINN models, use the regular load method
                self.model.load(self.problemTag)
                
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_model(self, custom_name=None):
        """
        Save the model with architecture metadata.
        
        Args:
            custom_name: Optional custom name for the saved model
        """
        try:
            model_name = custom_name or self.problemTag
            
            # Create trainedModels directory if it doesn't exist
            os.makedirs('trainedModels', exist_ok=True)
            
            # For MFN architectures, we need special handling
            if self.architecture in ('mfn', 'fast_mfn'):
                # Save the underlying Keras model
                model_path = f'trainedModels/{model_name}.keras'
                self.model.model.save(model_path)
                
                # Save additional metadata with architecture information
                metadata = {
                    'activation': self.activation,
                    'learning_rate': 0.001,  # Default learning rate
                    'input_dim': 3,          # 3D input for unsteady problems (x, y, t)
                    'eq': self.problemTag,
                    'architecture_type': self.architecture,
                    'model_version': '1.0',
                    'reynolds_number': self.Re,
                    'domain': {
                        'x_range': self.xRange,
                        'y_range': self.yRange,
                        't_range': self.tRange
                    }
                }
                
                # Save metadata to a JSON file
                metadata_path = f'trainedModels/{model_name}_metadata.json'
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2)
                    
                print(f"Model successfully saved to {model_path}")
                print(f"Model metadata saved to {metadata_path}")
            else:
                # For standard PINN models, use the regular save method
                # Always save to trainedModels directory
                self.model.save(f'trainedModels/{model_name}')
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False
