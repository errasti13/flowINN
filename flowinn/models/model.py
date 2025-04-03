import os
import cupy as cp  # Use cp as alias to avoid confusion with numpy if it were also present
import numpy as np  # Change back to standard import
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from flowinn.plot.boundary_visualization import BoundaryVisualization
from flowinn.nn.architectures import FourierFeatureLayer
import random # Import Python's random module for list shuffling

class PINN:
    """
    Physics-Informed Neural Network (PINN) class with a Modified Fourier Network (MFN) architecture.
    """

    def __init__(self, input_shape: int = 2, output_shape: int = 1, layers: List[int] = [20, 20, 20],
                 activation: str = 'gelu', learning_rate: float = 0.01, eq: str = None) -> None:
        """
        Initialize a Physics-Informed Neural Network.
        
        Args:
            input_shape: Shape of input tensor - either an integer (number of dimensions) or a tuple
            output_shape: Number of output variables
            layers: List of hidden layer sizes
            activation: Activation function to use
            learning_rate: Initial learning rate
            eq: Equation identifier
        """
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Check if input_shape is a tuple or an integer
        if isinstance(input_shape, tuple):
            self.input_dim = input_shape[0] if len(input_shape) > 0 else 1
        else:
            self.input_dim = input_shape
            
        # Build the model with a Fourier feature mapping layer at the start
        self.model: tf.keras.Sequential = self.create_model(self.input_dim, output_shape, layers, activation)
        self.model.summary()
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_schedule(learning_rate))
        self.eq: str = eq
        self.boundary_visualizer: Optional[BoundaryVisualization] = None

    def create_model(self, input_dim: int, output_shape: int, layers: List[int], activation: str) -> tf.keras.Sequential:
        """
        Creates an MFN model that begins with a Fourier feature mapping layer.
        
        Args:
            input_dim: Number of input dimensions
            output_shape: Number of output dimensions
            layers: List of hidden layer sizes
            activation: Activation function to use
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))
        
        # Detect if this is likely an unsteady problem based on input dimensions
        is_unsteady = input_dim >= 3
        
        # Add Fourier feature mapping layer with temporal-specific settings
        if is_unsteady:
            # For unsteady problems, use a different scale for temporal dimension
            # Higher scale (5.0) for time dimension to better capture high-frequency oscillations
            model.add(FourierFeatureLayer(
                fourier_dim=64,  # More features for complex temporal dynamics
                scale=5.0,       # Lower overall scale for spatial dimensions
                temporal_scale=25.0,  # Higher scale for temporal dimension
                trainable=False   # Keep features fixed for stability
            ))
        else:
            # For steady problems, use standard Fourier features
            model.add(FourierFeatureLayer(
                fourier_dim=32, 
                scale=10.0,
                trainable=False
            ))
        
        # Follow with hidden Dense layers
        for i, units in enumerate(layers):
            # Add dropout after some layers to prevent overfitting
            if i > 0 and i % 2 == 0 and is_unsteady:
                model.add(tf.keras.layers.Dropout(0.1))
                
            # Add dense layer with specified activation
            model.add(tf.keras.layers.Dense(
                units, 
                activation=activation,
                kernel_initializer=tf.keras.initializers.GlorotNormal()
            ))
            model.add(tf.keras.layers.BatchNormalization())
            
        # Final output layer
        model.add(tf.keras.layers.Dense(output_shape))
        return model

    def learning_rate_schedule(self, initial_learning_rate: float) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create an advanced learning rate schedule suitable for PINNs.
        
        Args:
            initial_learning_rate: Starting learning rate
            
        Returns:
            A learning rate schedule appropriate for the problem type
        """
        
        # For unsteady problems, use a warm-up followed by cosine decay
        if hasattr(self, 'input_dim'):
            # Check if we're dealing with an unsteady problem (input dim >= 3)
            if self.input_dim >= 3:  # Likely an unsteady problem
                # Create a custom schedule for unsteady problems
                class CustomWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
                    def __init__(self, initial_lr, warmup_steps=500, decay_steps=20000, min_lr_ratio=0.01):
                        super().__init__()
                        self.initial_lr = initial_lr
                        self.warmup_steps = warmup_steps
                        self.decay_steps = decay_steps
                        self.min_lr = initial_lr * min_lr_ratio
                        
                    def __call__(self, step):
                        # Convert to float
                        step = tf.cast(step, tf.float32)
                        
                        # Linear warmup phase
                        warmup_lr = self.initial_lr * (step / self.warmup_steps)
                        
                        # Cosine decay phase
                        step_after_warmup = step - self.warmup_steps
                        decay_fraction = step_after_warmup / self.decay_steps
                        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * decay_fraction))
                        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
                        
                        # Use warmup_lr if in warmup phase, otherwise use decay_lr
                        return tf.cond(step < self.warmup_steps, 
                                      lambda: warmup_lr,
                                      lambda: decay_lr)
                    
                    def get_config(self):
                        return {
                            "initial_lr": self.initial_lr,
                            "warmup_steps": self.warmup_steps,
                            "decay_steps": self.decay_steps,
                            "min_lr_ratio": self.min_lr / self.initial_lr
                        }
                
                return CustomWarmupCosineDecay(initial_learning_rate)
        
        # Default: exponential decay for steady problems
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=False
        )

    def generate_batches(self, mesh, num_batches, time_window_size=3):
        """Generate batches using CuPy for GPU acceleration."""
        batches = []

        # --- Convert mesh data to CuPy arrays ---
        # Ensure conversion happens correctly at the start
        try:
            x_flat = cp.asarray(mesh.x.ravel(), dtype=cp.float32)
            y_flat = cp.asarray(mesh.y.ravel(), dtype=cp.float32)
            z_flat = None if mesh.is2D else cp.asarray(mesh.z.ravel(), dtype=cp.float32)
            t_coords = cp.asarray(mesh.t, dtype=cp.float32) if hasattr(mesh, 't') and mesh.t is not None else None
            is_unsteady = hasattr(mesh, 'is_unsteady') and mesh.is_unsteady and t_coords is not None
        except Exception as e:
            print(f"Error converting mesh data to CuPy arrays: {e}")
            # Fallback or raise error? For now, raise to highlight the issue.
            raise TypeError("Failed to convert mesh data to CuPy arrays.") from e

        if is_unsteady:
            spatial_points = x_flat.size # Use .size for CuPy arrays
            time_points = t_coords.size

            if time_points == 0:
                 print("Warning: No time points found in mesh.t for unsteady batch generation.")
                 return [] # Cannot generate batches without time

            time_window_size = min(time_window_size, time_points)
            if time_window_size <= 0:
                time_window_size = 1 # Ensure at least one time step

            total_points = spatial_points * time_points
            # Ensure points_per_batch is at least 1
            points_per_batch = max(1, total_points // num_batches)

            # Estimate spatial points needed, ensure positive result
            points_per_spatial_dim = int(cp.sqrt(max(1.0, float(points_per_batch) / time_window_size)))
            points_per_spatial_dim = max(1, points_per_spatial_dim) # Ensure at least 1

            num_windows = num_batches

            for batch_idx in range(num_batches):
                # --- Use CuPy for random sampling ---
                x_indices = cp.random.choice(spatial_points, size=points_per_spatial_dim, replace=True)
                y_indices = cp.random.choice(spatial_points, size=points_per_spatial_dim, replace=True)

                # Determine time window indices using CuPy
                max_start_idx = time_points - time_window_size
                if num_windows > 1 and max_start_idx > 0:
                    # Calculate start index proportionally
                    window_start = int((batch_idx / (num_windows - 1)) * max_start_idx)
                    # Ensure window_start is within bounds
                    window_start = cp.clip(window_start, 0, max_start_idx).item() # .item() to get scalar
                else:
                    window_start = 0

                # Ensure end index doesn't exceed time_points
                window_end = min(window_start + time_window_size, time_points)
                # Use CuPy arange for indices
                t_indices = cp.arange(window_start, window_end, dtype=cp.int32)

                # Check if t_indices is empty
                if t_indices.size == 0:
                    print(f"Warning: Skipping batch {batch_idx+1} due to empty time window.")
                    continue

                # --- Efficiently build coordinates using CuPy broadcasting ---
                # Get sampled spatial coordinates
                sampled_x = x_flat[x_indices]
                sampled_y = y_flat[y_indices]

                # Get sampled temporal coordinates
                sampled_t = t_coords[t_indices]

                # Create meshgrid using CuPy
                # Shapes: (spatial_dim, spatial_dim, time_window)
                xx, yy, tt = cp.meshgrid(sampled_x, sampled_y, sampled_t, indexing='ij')

                # Stack coordinates: Result shape (spatial_dim * spatial_dim * time_window, 3)
                batch_coords_cp = cp.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1)

                # Ensure correct data type
                batch_coords_cp = batch_coords_cp.astype(cp.float32)

                # Limit batch size if needed using CuPy indexing
                current_batch_size = batch_coords_cp.shape[0]
                if current_batch_size > points_per_batch:
                    indices = cp.random.choice(current_batch_size, size=points_per_batch, replace=False)
                    batch_coords_cp = batch_coords_cp[indices]
                elif current_batch_size == 0:
                    print(f"Warning: Batch {batch_idx+1} is empty after coordinate generation.")
                    continue # Skip empty batch

                # --- Convert final CuPy array to NumPy for TensorFlow ---
                try:
                    batch_coords_np = cp.asnumpy(batch_coords_cp)
                    batch_tensor = tf.convert_to_tensor(batch_coords_np, dtype=tf.float32)
                    batches.append(batch_tensor)
                except Exception as conversion_e:
                    print(f"Error converting batch {batch_idx+1} from CuPy to Tensor: {conversion_e}")
                    # Optionally skip this batch or raise error
                    continue

        else:
            # --- Steady State Logic (using CuPy) ---
            total_points = x_flat.size
            points_per_batch = max(1, total_points // num_batches) # Ensure at least 1

            # Use CuPy sqrt/cbrt, ensure float for division, handle potential zero points
            points_per_dim_float = cp.power(float(points_per_batch), 1.0/mesh.input_dim) if points_per_batch > 0 else 0.0
            points_per_dim = max(1, int(points_per_dim_float)) # Ensure at least 1

            for _ in range(num_batches):
                # Generate indices using CuPy
                indices_list = [cp.random.choice(s.size, size=points_per_dim, replace=True)
                                for s in [x_flat, y_flat] + ([] if mesh.is2D else [z_flat])]

                # Use CuPy meshgrid for indices
                meshgrid_indices = cp.meshgrid(*indices_list, indexing='ij')

                # Stack indices and get coordinates
                all_indices = cp.stack([idx.ravel() for idx in meshgrid_indices], axis=-1)

                coords_list = [
                    x_flat[all_indices[:, 0]],
                    y_flat[all_indices[:, 1]]
                ]
                if not mesh.is2D:
                    coords_list.append(z_flat[all_indices[:, 2]])

                batch_coords_cp = cp.stack(coords_list, axis=-1).astype(cp.float32)

                # Limit batch size using CuPy slicing/indexing
                current_batch_size = batch_coords_cp.shape[0]
                if current_batch_size > points_per_batch:
                     # Efficiently sample using CuPy
                    indices = cp.random.choice(current_batch_size, size=points_per_batch, replace=False)
                    batch_coords_cp = batch_coords_cp[indices]
                elif current_batch_size == 0:
                    print(f"Warning: Steady-state batch is empty.")
                    continue # Skip empty batch


                # --- Convert final CuPy array to NumPy for TensorFlow ---
                try:
                    batch_coords_np = cp.asnumpy(batch_coords_cp)
                    batch_tensor = tf.convert_to_tensor(batch_coords_np, dtype=tf.float32)
                    batches.append(batch_tensor)
                except Exception as conversion_e:
                     print(f"Error converting steady-state batch from CuPy to Tensor: {conversion_e}")
                     continue


        return batches

    @tf.function  # Re-applying decorator for performance
    def train_step(self, loss_function, batch_data) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = loss_function(batch_data=batch_data)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, loss_function, mesh, epochs: int = 1000, num_batches: int = 1,
              print_interval: int = 100, autosave_interval: int = 100,
              plot_loss: bool = False, bc_plot_interval: Optional[int] = None, 
              domain_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
              airfoil_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              output_dir: str = 'bc_plots', patience: int = 10000, min_delta: float = 1e-6,
              time_window_size: int = 5, use_cpu: bool = False, 
              batch_data: Optional[List[tf.Tensor]] = None) -> None:
        """
        Train the model using the provided loss function.
        
        Args:
            loss_function: Loss function to optimize
            mesh: Mesh containing the domain discretization
            epochs: Number of training epochs
            num_batches: Number of batches per epoch (ignored if batch_data is provided)
            print_interval: Interval for printing progress
            autosave_interval: Interval for model autosaving
            plot_loss: Whether to plot loss during training
            bc_plot_interval: Interval for plotting boundary conditions
            domain_range: Range of the domain for plotting
            airfoil_coords: Coordinates of airfoil for plotting
            output_dir: Directory for output plots
            patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            time_window_size: Size of temporal window for unsteady problems)
            use_cpu: Force CPU usage (set to True if GPU/CUDA errors occur)
            batch_data: Pre-generated batches for memory-efficient training
        """
        # Set device strategy based on use_cpu flag or auto-detect issues
        if use_cpu:
            print("Using CPU for training (GPU disabled)")
            with tf.device('/CPU:0'):
                return self._train_implementation(
                    loss_function, mesh, epochs, num_batches, 
                    print_interval, autosave_interval, plot_loss, 
                    bc_plot_interval, domain_range, airfoil_coords, 
                    output_dir, patience, min_delta, time_window_size, 
                     batch_data
                )
        else:
            try:
                # Try with default device
                return self._train_implementation(
                    loss_function, mesh, epochs, num_batches, 
                    print_interval, autosave_interval, plot_loss, 
                    bc_plot_interval, domain_range, airfoil_coords, 
                    output_dir, patience, min_delta, time_window_size, 
                    batch_data
                )
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError, 
                   tf.errors.FailedPreconditionError) as e:
                print(f"\nGPU error encountered: {e}")
                print("Falling back to CPU training...")
                
                # Clear any GPU memory
                tf.keras.backend.clear_session()
                
                # Recreate model architecture
                old_weights = self.model.get_weights()
                
                # Replace model with same architecture
                self.model = self.create_model(self.input_dim, self.model.output_shape[-1], 
                                              [layer.units for layer in self.model.layers 
                                               if isinstance(layer, tf.keras.layers.Dense)][:-1], 
                                              self.activation)
                
                # Transfer weights if possible
                try:
                    self.model.set_weights(old_weights)
                except:
                    print("Could not transfer weights - reinitializing...")
                
                # Reset optimizer
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate_schedule(self.learning_rate))
                
                # Run on CPU
                with tf.device('/CPU:0'):
                    return self._train_implementation(
                        loss_function, mesh, epochs, num_batches, 
                        print_interval, autosave_interval, plot_loss, 
                        bc_plot_interval, domain_range, airfoil_coords, 
                        output_dir, patience, min_delta, time_window_size, 
                        batch_data
                    )

    def _train_implementation(self, loss_function, mesh, epochs, num_batches,
                          print_interval, autosave_interval, plot_loss, 
                          bc_plot_interval, domain_range, airfoil_coords,
                          output_dir, patience, min_delta, time_window_size, 
                          batch_data=None):
        """Implementation of the training process, separated for device context management."""
        import time
        
        loss_history = []
        epoch_history = []
        time_history = []  # Track time per epoch
        last_loss = float('inf')
        patience_counter = 0
        total_training_time = 0.0

        if bc_plot_interval is not None:
            self.boundary_visualizer = BoundaryVisualization(output_dir=output_dir)

        if plot_loss:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.yaxis.get_major_formatter().set_useOffset(False)
            ax1.yaxis.get_major_formatter().set_scientific(False)
            line1, = ax1.semilogy([], [], label='Training Loss')
            ax1.legend()
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Time per Epoch (s)')
            line2, = ax2.plot([], [], label='Time/Epoch')
            ax2.legend()

        # Use the @tf.function decorated train_step directly
        train_step_fn = self.train_step

        for epoch in range(epochs):
            epoch_start_time = time.time()
            try:
                # Generate or use provided batches
                if batch_data is not None and len(batch_data) > 0:
                    # Use pre-generated batches (memory efficient)
                    batches = batch_data
                    # Shuffle batches (which is a List[tf.Tensor]) using Python's random
                    random.shuffle(batches)
                else:
                    # Generate batches on-the-fly
                    batches = self.generate_batches(
                        mesh, 
                        num_batches, 
                        time_window_size=time_window_size
                    )
                
                # Train on batches
                epoch_loss = 0.0
                for batch_idx, batch in enumerate(batches):
                    try:
                        # Process in smaller chunks if batch is large
                        if batch.shape[0] > 5000:
                            sub_batch_size = 2500  # Smaller sub-batches for memory efficiency
                            num_sub_batches = (batch.shape[0] + sub_batch_size - 1) // sub_batch_size
                            
                            sub_batch_loss = 0.0
                            for i in range(num_sub_batches):
                                start_idx = i * sub_batch_size
                                end_idx = min(start_idx + sub_batch_size, batch.shape[0])
                                sub_batch = batch[start_idx:end_idx]
                                
                                batch_loss = train_step_fn(loss_function, sub_batch)
                                sub_batch_loss += batch_loss.numpy() * (end_idx - start_idx) / batch.shape[0]
                                
                            epoch_loss += sub_batch_loss
                        else:
                            # Process smaller batches normally
                            batch_loss = train_step_fn(loss_function, batch)
                            epoch_loss += batch_loss.numpy() / len(batches)
                            
                    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError,\
                           tf.errors.FailedPreconditionError) as e:
                        print(f"Error in batch {batch_idx+1}/{len(batches)}: {e}")
                        print("Skipping batch and continuing...")
                        # Continue with next batch instead of failing completely
                        continue
                
            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}")
                raise

            # Calculate epoch time and update total
            epoch_time = time.time() - epoch_start_time
            total_training_time += epoch_time

            # Early stopping check
            if epoch_loss < last_loss - min_delta:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Total training time: {total_training_time:.2f}s")
                    print(f"Average time per epoch: {total_training_time/(epoch+1):.2f}s")
                    break
            
            last_loss = epoch_loss

            if (epoch + 1) % print_interval == 0:
                loss_history.append(epoch_loss)
                epoch_history.append(epoch + 1)
                time_history.append(epoch_time)
                avg_time = total_training_time / (epoch + 1)

                if plot_loss:
                    # Update loss plot
                    line1.set_xdata(epoch_history)
                    line1.set_ydata(loss_history)
                    ax1.relim()
                    ax1.autoscale_view()
                    
                    # Update timing plot
                    line2.set_xdata(epoch_history)
                    line2.set_ydata(time_history)
                    ax2.relim()
                    ax2.autoscale_view()
                    
                    plt.draw()
                    plt.pause(0.001)

                print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4e}, Time = {epoch_time:.2f}s, Avg Time = {avg_time:.2f}s")
            
            if self.boundary_visualizer is not None and bc_plot_interval is not None and (epoch + 1) % bc_plot_interval == 0:
                self.boundary_visualizer.plot_boundary_conditions(
                    self, mesh, epoch + 1, domain_range, airfoil_coords
                )

            if (epoch + 1) % autosave_interval == 0:
                os.makedirs('trainedModels', exist_ok=True)
                try:
                    self.model.save(f'trainedModels/{self.eq}.keras')
                except OSError as e:
                    print(f"Error saving model: {e}")
                    # Continue without failing the whole training

        # Print final timing statistics
        if epoch > 0:  # Only if we completed at least one epoch
            print(f"\nTraining completed:")
            print(f"Total training time: {total_training_time:.2f}s")
            print(f"Average time per epoch: {total_training_time/(epoch+1):.2f}s")
            print(f"Min epoch time: {min(time_history):.2f}s")
            print(f"Max epoch time: {max(time_history):.2f}s")

        if self.boundary_visualizer is not None:
            self.boundary_visualizer.plot_error_evolution()

        if plot_loss:
            plt.ioff()
            plt.close()
            
        # Create a history object to return with timing information
        history = type('History', (), {
            'history': {
                'loss': loss_history,
                'time_per_epoch': time_history,
                'total_time': total_training_time,
                'avg_time_per_epoch': total_training_time/(epoch+1) if epoch > 0 else 0
            }
        })
        return history

    def predict(self, X, use_cpu=False):
        """
        Make predictions using the model.
        
        Args:
            X: Input tensor of shape (..., input_dim)
            use_cpu: Whether to force CPU usage for prediction
            
        Returns:
            Predictions tensor of shape (..., output_dim)
        """
        # Convert input to tensor if it's numpy array
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        
        # Choose device based on parameter and availability
        try:
            if use_cpu:
                with tf.device('/CPU:0'):
                    return self.model(X, training=False).numpy()
            else:
                return self.model(X, training=False).numpy()
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError, 
               tf.errors.FailedPreconditionError) as e:
            print(f"GPU error during prediction: {e}")
            print("Falling back to CPU for prediction...")
            
            with tf.device('/CPU:0'):
                return self.model(X, training=False).numpy()

    def save(self, model_name: str) -> None:
        """
        Save the model to disk with metadata.
        
        Args:
            model_name: Name or path for the saved model (without extension)
        """
        try:
            # Ensure we're saving to the trainedModels directory
            if not model_name.startswith('trainedModels/'):
                save_path = f"trainedModels/{model_name}"
            else:
                save_path = model_name
                
            # Create directory if it doesn't exist
            os.makedirs('trainedModels', exist_ok=True)
                
            # Clean up path for saving
            if not save_path.endswith('.keras') and not save_path.endswith('.h5'):
                save_path = f"{save_path}.keras"
            
            # Save the underlying Keras model
            self.model.save(save_path)
            
            # Save additional metadata that isn't captured by Keras save
            metadata = {
                'activation': self.activation,
                'learning_rate': self.learning_rate,
                'input_dim': self.input_dim,
                'eq': self.eq,
                'architecture_type': 'pinn',
                'model_version': '1.0'
            }
            
            # Save metadata to a JSON file
            metadata_path = f"{save_path.rsplit('.', 1)[0]}_metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
                
            print(f"Model successfully saved to {save_path}")
            print(f"Model metadata saved to {metadata_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load(self, model_name: str) -> None:
        """
        Load a model from disk with metadata.
        
        Args:
            model_name: Name of the model to load (without extension)
        """
        # Import os here to ensure it's available
        import os
        import json
        
        # Determine file paths
        filepath = f'trainedModels/{model_name}.keras'
        metadata_path = f'trainedModels/{model_name}_metadata.json'
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified model file does not exist: {filepath}")
        
        try:
            # Try to load the model
            print(f"Loading model from {filepath}")
            self.model = tf.keras.models.load_model(filepath, compile=False)
            
            # Load metadata if available
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update object attributes from metadata
                    self.activation = metadata.get('activation', self.activation)
                    self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                    self.input_dim = metadata.get('input_dim', self.model.input_shape[-1])
                    self.eq = metadata.get('eq', model_name)
                    
                    print(f"Model metadata loaded from {metadata_path}")
                except Exception as e:
                    print(f"Warning: Could not load metadata: {e}")
            else:
                # Update input_dim from loaded model if no metadata
                self.input_dim = self.model.input_shape[-1]
                print("No metadata file found. Using basic model information.")
            
            # Always recompile the optimizer to ensure it's properly initialized
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_schedule(self.learning_rate))
            
            print(f"Model successfully loaded")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load with CPU-only...")
            
            # Clear session
            tf.keras.backend.clear_session()
            
            # Force CPU usage
            old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            try:
                with tf.device('/CPU:0'):
                    self.model = tf.keras.models.load_model(filepath, compile=False)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Update object attributes from metadata
                        self.activation = metadata.get('activation', self.activation)
                        self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                        self.input_dim = metadata.get('input_dim', self.model.input_shape[-1])
                        self.eq = metadata.get('eq', model_name)
                        
                        print(f"Model metadata loaded from {metadata_path}")
                    except Exception as e:
                        print(f"Warning: Could not load metadata: {e}")
                else:
                    # Update input_dim from loaded model if no metadata
                    self.input_dim = self.model.input_shape[-1]
                    print("No metadata file found. Using basic model information.")
                
                # Always recompile the optimizer to ensure it's properly initialized
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate_schedule(self.learning_rate))
                
                print(f"Model successfully loaded on CPU")
                
                # Restore original CUDA setting
                if old_cuda is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            except Exception as e2:
                if old_cuda is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                raise RuntimeError(f"Failed to load model: {e2}")
