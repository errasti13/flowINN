"""
Time window training approach for unsteady problems.

This module implements the Sequential Moving Time Windows (SMTW) approach described in the
academic literature, which divides the full time domain into smaller windows for better
training of unsteady problems.

References:
    - Wight & Zhao (2020) - "Time-dependent physics-informed neural networks"
    - Krishnapriyan et al. (2021) - "Characterizing possible failure modes in physics-informed neural networks"
"""

import os
import time
import numpy as np
import cupy as cp
import tensorflow as tf
import inspect
from datetime import datetime
import matplotlib.pyplot as plt


class TimeWindowTrainer:
    """
    Trainer for PINN models using Sequential Moving Time Windows approach.
    
    This approach divides the total time domain into smaller overlapping windows,
    training a model on each window sequentially and using the results from one window
    as initial conditions for the next.
    
    Attributes:
        num_windows: Number of time windows to use (default: 3)
        initial_lr: Initial learning rate (default: 0.001)
        lr_decay_factor: Learning rate decay factor between windows (default: 0.999947)
        max_epochs: Maximum number of training epochs per window (default: 100000)
    """
    
    def __init__(self, num_windows=3, initial_lr=0.001, lr_decay_factor=0.999947, max_epochs=100000):
        """
        Initialize the time window trainer.
        
        Args:
            num_windows: Number of time windows to use
            initial_lr: Initial learning rate
            lr_decay_factor: Learning rate decay factor between windows
            max_epochs: Maximum number of training epochs per window
        """
        self.num_windows = num_windows
        self.initial_lr = initial_lr
        self.lr_decay_factor = lr_decay_factor
        self.max_epochs = max_epochs
        
        # History storage
        self.training_history = {
            'loss': [],
            'physics_loss': [],
            'data_loss': [],
            'lr': [],
            'epoch_times': []
        }
        
        self.window_histories = []
        
    def set_learning_rate(self, model, learning_rate):
        """
        Set the learning rate for the model optimizer robustly.
        
        Args:
            model: The PINN model
            learning_rate: New learning rate value (float)
        """
        try:
            # First try to update model's own learning_rate Variable attribute
            if hasattr(model, 'learning_rate') and hasattr(model.learning_rate, 'assign'):
                model.learning_rate.assign(learning_rate)
                print(f"Model learning_rate Variable assigned to {learning_rate:.8f}")
                return
             
            # If we reach here, try optimizer's learning_rate
            optimizer = model.optimizer
            
            if hasattr(optimizer, 'learning_rate'):
                lr_object = optimizer.learning_rate
                
                # Handle our custom VariableLearningRate class
                if hasattr(lr_object, 'lr_variable') and hasattr(lr_object.lr_variable, 'assign'):
                    lr_object.lr_variable.assign(learning_rate)
                    print(f"Learning rate assigned via VariableLearningRate to {learning_rate:.8f}")
                    return
                
                # Standard assign if available
                if hasattr(lr_object, 'assign'):
                    lr_object.assign(learning_rate)
                    print(f"Learning rate assigned to {learning_rate:.8f}")
                    return
                    
                # Try backend set_value as a fallback
                try:
                    tf.keras.backend.set_value(lr_object, learning_rate)
                    print(f"Learning rate set via backend to {learning_rate:.8f}")
                    return
                except Exception as e_setval:
                    print(f"Warning: Could not update learning rate ({e_setval}). Creating new optimizer.")
            
            # If we get here, create a new optimizer as last resort
            print(f"Creating new Adam optimizer with learning rate {learning_rate:.8f}")
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        except Exception as e:
            print(f"Error setting learning rate: {str(e)}")
            print(f"Attempting to create new Adam optimizer with learning rate {learning_rate:.8f}")
            # Ensure recreation on any error
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
    def _generate_spatial_temporal_batches(self, mesh, num_spatial_batches, num_temporal_batches):
        """
        Generate batches by dividing both spatial and temporal domains into smaller chunks using CuPy.
        This dramatically reduces memory usage compared to standard batching.
        
        Args:
            mesh: The mesh object containing coordinates (assumed NumPy)
            num_spatial_batches: Number of spatial batches to divide the domain into
            num_temporal_batches: Number of temporal batches to divide the time domain into
            
        Returns:
            List of TensorFlow Tensors, each containing coordinates for training
        """
        # --- Convert initial mesh data to CuPy arrays ---
        try:
            x_flat_cp = cp.asarray(mesh.x.ravel(), dtype=cp.float32)
            y_flat_cp = cp.asarray(mesh.y.ravel(), dtype=cp.float32)
            # Ensure t_values_cp is created even if mesh.t is None
            t_values_cp = cp.asarray(mesh.t.ravel(), dtype=cp.float32) if hasattr(mesh, 't') and mesh.t is not None else cp.zeros(x_flat_cp.shape[0], dtype=cp.float32)
        except Exception as e:
            print(f"Error converting mesh data to CuPy in _generate_spatial_temporal_batches: {e}")
            raise TypeError("Failed to convert mesh data to CuPy arrays.") from e

        # Get sizes using CuPy
        num_spatial_points = x_flat_cp.size
        num_temporal_points = t_values_cp.size

        if num_temporal_points == 0:
            print("Warning: No time points provided for batch generation.")
            return []

        batches = []
        
        # Special case: if both spatial and temporal batches are 1, use all points with sampling
        if num_spatial_batches == 1 and num_temporal_batches == 1:
            print(f"Using all {num_spatial_points} spatial points with sampled time points")
            
            # Sample time points using CuPy
            if num_temporal_points > num_temporal_points:
                 # Simple random choice if many points
                 selected_t_indices = cp.random.choice(num_temporal_points, size=num_temporal_points, replace=False)
            else:
                 # Use all points if fewer than target samples
                 selected_t_indices = cp.arange(num_temporal_points, dtype=cp.int32)
            selected_t = t_values_cp[selected_t_indices]
            
            # Create combinations efficiently using CuPy meshgrid and stacking
            # We need to combine all x, all y with selected t
            # Use cp.tile and cp.repeat for efficient combination
            num_selected_t = selected_t.shape[0]
            
            tiled_x = cp.tile(x_flat_cp, num_selected_t)
            tiled_y = cp.tile(y_flat_cp, num_selected_t)
            repeated_t = cp.repeat(selected_t, num_spatial_points)
            
            batch_coords_cp = cp.stack([tiled_x, tiled_y, repeated_t], axis=-1)
            
            # Convert final CuPy array to NumPy, then to TensorFlow Tensor
            try:
                batch_coords_np = cp.asnumpy(batch_coords_cp)
                batches.append(tf.convert_to_tensor(batch_coords_np, dtype=tf.float32))
                print(f"Generated 1 spatial-temporal batch with {batch_coords_cp.shape[0]} points")
            except Exception as conversion_e:
                print(f"Error converting single large batch from CuPy to Tensor: {conversion_e}")
            
            return batches

        # Regular case with multiple batches
        spatial_points_per_batch  = num_spatial_points  // num_spatial_batches
        temporal_points_per_batch = num_temporal_points // num_temporal_batches

        # Generate combined spatial-temporal batches using CuPy
        for t_batch in range(num_temporal_batches):
            t_start_idx = t_batch * temporal_points_per_batch
            t_end_idx = min(t_start_idx + temporal_points_per_batch, num_temporal_points)
            if t_batch == num_temporal_batches - 1: # Ensure last batch covers remaining points
                t_end_idx = num_temporal_points

            num_t_in_chunk = t_end_idx - t_start_idx
            if num_t_in_chunk <= 0: continue # Skip if chunk is empty

            # Sample time points for this chunk using CuPy
            replace_t = num_t_in_chunk < temporal_points_per_batch
            t_indices_in_chunk = cp.random.choice(
                cp.arange(t_start_idx, t_end_idx, dtype=cp.int32), 
                size=min(temporal_points_per_batch, num_t_in_chunk), 
                replace=replace_t
            )
            batch_t_values = t_values_cp[t_indices_in_chunk]

            if batch_t_values.size == 0: continue # Skip if no time values selected

            for s_batch in range(num_spatial_batches):
                # Sample spatial indices for this batch using CuPy
                replace_s = num_spatial_points < spatial_points_per_batch
                spatial_indices = cp.random.choice(
                    num_spatial_points, 
                    size=spatial_points_per_batch, 
                    replace=replace_s
                )
                
                # Get spatial coordinates using CuPy indexing
                batch_x = x_flat_cp[spatial_indices]
                batch_y = y_flat_cp[spatial_indices]

                # Create combinations using CuPy meshgrid and stacking
                # Limit time points per spatial batch for memory
                num_time_points_in_batch = min(10, batch_t_values.size)
                if num_time_points_in_batch <= 0 : continue # Ensure we have time points
                
                # Use replace=False if possible, otherwise True
                replace_t_sample = batch_t_values.size < num_time_points_in_batch
                selected_t = cp.random.choice(batch_t_values, size=num_time_points_in_batch, replace=replace_t_sample)

                if selected_t.size == 0: continue # Skip if no time points after sampling
                
                # CuPy meshgrid for coords
                # Shapes: (spatial_batch_size, time_samples)
                xx, tt = cp.meshgrid(batch_x, selected_t, indexing='ij')
                # Need yy to match shape, use repeat
                # Shape: (spatial_batch_size, 1) -> (spatial_batch_size, time_samples)
                yy = cp.repeat(batch_y[:, cp.newaxis], selected_t.size, axis=1)
                
                # Ravel and stack
                # Shape: (spatial_batch_size * time_samples, 3)
                batch_coords_cp = cp.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1)

                # Ensure correct dtype
                batch_coords_cp = batch_coords_cp.astype(cp.float32)

                if batch_coords_cp.size == 0:
                    print(f"Warning: Batch t{t_batch+1}/s{s_batch+1} is empty.")
                    continue # Skip empty batch

                # Convert final CuPy array to NumPy, then to TensorFlow Tensor
                try:
                    batch_coords_np = cp.asnumpy(batch_coords_cp)
                    batches.append(tf.convert_to_tensor(batch_coords_np, dtype=tf.float32))
                except Exception as conversion_e:
                    print(f"Error converting batch t{t_batch+1}/s{s_batch+1} from CuPy to Tensor: {conversion_e}")
                    continue # Optionally skip this batch

        print(f"Generated {len(batches)} spatial-temporal batches using CuPy")
        return batches

    def train(self, model, loss_function, mesh, tRange, epochs=None, 
             print_interval=100, autosave_interval=10000, num_batches=10, 
             use_cpu=False, save_name=None, window_overlap=0.1, 
             physics_points_ratio=10, adaptive_sampling=True,
             num_spatial_batches=4, num_temporal_batches=3,
             patience = 100, min_delta = 1e-7):
        """
        Train the model using the moving time window approach.
        
        Args:
            model: The PINN model to train
            loss_function: Loss function object
            mesh: Mesh object containing domain information
            tRange: Time range tuple (t_start, t_end)
            epochs: Maximum number of epochs per window (None uses default)
            print_interval: Interval for printing loss information
            autosave_interval: Interval for autosaving the model
            num_batches: Number of batches for mini-batch training
            use_cpu: Whether to use CPU instead of GPU
            save_name: Base name for saving model checkpoints
            window_overlap: Fraction of overlap between windows (0.0-1.0)
            physics_points_ratio: Ratio of physics points to mesh points
            adaptive_sampling: Whether to use adaptive sampling of collocation points
            num_spatial_batches: Number of batches to divide spatial domain into
            num_temporal_batches: Number of batches to divide temporal domain into
        """            
        # Timing and setup
        start_time = time.time()
        
        # Use epochs from init if not specified
        if epochs is None:
            epochs = self.max_epochs
        
        # Print settings
        print(f"\n{'='*80}")
        print(f"TRAINING WITH SEQUENTIAL MOVING TIME WINDOWS")
        print(f"Number of windows: {self.num_windows}")
        print(f"Time range: {tRange}")
        print(f"Maximum epochs per window: {epochs}")
        print(f"Learning rate decay: γ = {self.lr_decay_factor}")
        print(f"{'='*80}\n")
        
        # Calculate time window sizes
        time_duration = tRange[1] - tRange[0]
        window_size = time_duration / self.num_windows
        
        # Store original time range
        original_t = mesh.t.copy() if hasattr(mesh, 't') else None
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{save_name}_{timestamp}" if save_name else f"logs/pinn_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create TensorBoard callback
        try:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='epoch'
            )
        except:
            # If TensorBoard not available
            tensorboard_callback = None
            
        # Configure GPU/CPU
        if use_cpu:
            # Use the already imported os module
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Using CPU for training (GPU disabled)")
        
        # For each time window
        for window_idx in range(self.num_windows):
            window_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"TRAINING TIME WINDOW {window_idx+1}/{self.num_windows}")
            print(f"{'='*60}")
            
            # Calculate time range for this window
            window_start = tRange[0] + window_idx * window_size
            window_end = window_start + window_size
            
            # Set the time range for this window with overlap
            if window_idx > 0:
                # Use overlap with previous window for continuity
                overlap = window_size * window_overlap
                window_start -= overlap
            
            
            print(f"Time window: [{window_start:.4f}, {window_end:.4f}] (Duration: {window_end-window_start:.4f})")
            
            # Set learning rate based on decay schedule
            if window_idx > 0:
                current_lr = self.initial_lr * (self.lr_decay_factor ** (window_idx * epochs))
                self.set_learning_rate(model, current_lr)
            else:
                # For first window, use initial learning rate
                current_lr = self.initial_lr
                print(f"Initial learning rate: {current_lr:.6f}")
            
            # Generate time points for this window - more points at boundaries
            num_time_points = len(mesh.t)
            
            # Use non-linear spacing for better resolution at boundaries
            beta = 1.5  # Controls clustering at boundaries
            t_window = np.zeros(num_time_points)
            
            for i in range(num_time_points):
                # This creates more points near the boundaries of the window
                normalized_idx = i / (num_time_points - 1)
                if normalized_idx < 0.5:
                    # First half: more points at beginning
                    t_window[i] = window_start + (window_end - window_start) * (2 * normalized_idx) ** beta / 2
                else:
                    # Second half: more points at end
                    t_window[i] = window_end - (window_end - window_start) * (2 * (1 - normalized_idx)) ** beta / 2
            
            # Update mesh time points
            if hasattr(mesh, 't'):
                # Create a deep copy of the t array to avoid modifying the original
                mesh_t_copy = np.copy(original_t) if original_t is not None else None
                
                # Filter and keep only time points within this window
                if mesh_t_copy is not None:
                    t_mask = (mesh_t_copy >= window_start) & (mesh_t_copy <= window_end)
                    if np.any(t_mask):
                        mesh.t = mesh_t_copy[t_mask]
                    else:
                        # If no original time points fall within this window, create new ones
                        mesh.t = t_window
                else:
                    mesh.t = t_window
            else:
                mesh.t = t_window
                
            # Ensure we have enough time points for this window
            if len(mesh.t) < 10:  # Minimum number of time points for stable training
                mesh.t = t_window  # Use the non-linear spacing we generated
            
            # Set initial conditions from previous window
            if window_idx > 0:
                print("\nSetting initial conditions from previous window...")
                # Use the final state of the previous window as initial condition
                prev_window_end = window_start
                
                # Sample multiple points at the previous window boundary for stability
                num_boundary_samples = 5
                boundary_times = np.linspace(
                    prev_window_end - 0.01 * window_size,  # Slightly before boundary
                    prev_window_end + 0.01 * window_size,  # Slightly after boundary
                    num_boundary_samples
                )
                
                # Create input points
                X_init = mesh.x.flatten()[:, None]
                Y_init = mesh.y.flatten()[:, None]
                
                # Average predictions across boundary samples for stability
                init_sols = []
                
                for b_time in boundary_times:
                    T_init = np.full_like(X_init, b_time)
                    X_input = np.hstack((X_init, Y_init, T_init))
                    init_sols.append(model.predict(X_input))
                
                # Average solutions
                init_sol = np.mean(init_sols, axis=0)
                
                # Set as initial condition for this window
                mesh.initialConditions = {
                    'Initial': {
                        'x': X_init.flatten(),
                        'y': Y_init.flatten(),
                        't': np.full_like(X_init.flatten(), window_start),
                        'conditions': {
                            'u': {'value': init_sol[:, 0]},
                            'v': {'value': init_sol[:, 1]},
                            'p': {'value': init_sol[:, 2]}
                        }
                    }
                }
                print(f"Set initial conditions at t={window_start} from {num_boundary_samples} boundary samples")
            
            # Calculate number of physics points
            if num_spatial_batches == 1 and num_temporal_batches == 1:
                # When using all points, set physics points to match the full mesh
                num_physics_points = len(mesh.x)
                print("Using all mesh points for physics evaluation")
            else:
                # Use physics_points_ratio to determine points in other cases
                num_physics_points = len(mesh.x) * physics_points_ratio
            
            # Initialize adaptive weights if using
            if adaptive_sampling:
                # Initialize with uniform weights
                mesh.adaptive_weights = np.ones(num_physics_points) / num_physics_points
            
            print(f"\nTraining configuration:")
            print(f"Mesh size: {len(mesh.x)} spatial points x {len(mesh.t)} time points")
            print(f"Number of physics points: {num_physics_points}")
            print(f"Spatial batches: {num_spatial_batches}, Temporal batches: {num_temporal_batches}")
            
            # Setup callbacks
            callbacks = []
            if tensorboard_callback:
                callbacks.append(tensorboard_callback)
            
            # Prepare all possible parameters
            all_params = {
                'loss_function': loss_function,
                'mesh': mesh,
                'epochs': epochs,
                'print_interval': print_interval,
                'autosave_interval': autosave_interval,
                'num_batches': num_batches,
                'plot_loss': False,
                'patience': patience,
                'min_delta': min_delta,
                'use_cpu': use_cpu
            }
            
            # Add callbacks if available
            if 'callbacks' in locals() and callbacks is not None:
                all_params['callbacks'] = callbacks
            
            # Generate spatial-temporal batches if batch_data parameter is supported
            if 'batch_data' in inspect.signature(model.train).parameters:
                # Use spatial-temporal batching to reduce memory usage
                batches = self._generate_spatial_temporal_batches(
                    mesh, 
                    num_spatial_batches, 
                    num_temporal_batches
                )
                all_params['batch_data'] = batches
            
            # Train the model for this time window
            try:
                # Get only the supported parameters for the model's train method
                model_train_params = inspect.signature(model.train).parameters
                supported_params = {k: v for k, v in all_params.items() if k in model_train_params}
                
                # Train the model using only supported parameters
                history = model.train(**supported_params)
                
                # Save window history
                if hasattr(history, 'history'):
                    self.window_histories.append(history.history)
                    
                    # Update overall history
                    for key in history.history:
                        if key in self.training_history:
                            self.training_history[key].extend(history.history[key])
            
            except Exception as e:
                print(f"Error during training window {window_idx+1}: {str(e)}")
                print("Check model.train method implementation for supported parameters.")
                # Continue with next window
                
            # Save model for this time window
            if save_name:
                window_model_name = f"{save_name}_window_{window_idx+1}"
                try:
                    # Try to use model's save method if available
                    model.save(window_model_name)
                    print(f"Model saved as {window_model_name}")
                except AttributeError:
                    # If model doesn't have a save method, print a message
                    print(f"Note: Unable to save model - no save method available")
                    # Could implement custom saving here if needed
                    # Example: tf.keras.models.save_model(model.model, window_model_name) if hasattr(model, 'model')
            
            # Calculate and print window training time
            window_time = time.time() - window_start_time
            hours, remainder = divmod(window_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nWindow {window_idx+1} training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            
            # Store epoch time
            self.training_history['epoch_times'].append(window_time)
        
        # Restore original time range for later prediction
        if original_t is not None:
            mesh.t = original_t
        
        # Save final model
        if save_name:
            try:
                # Try to use model's save method if available
                model.save(save_name)
                print(f"\nTraining completed! Final model saved as {save_name}")
            except AttributeError:
                # If model doesn't have a save method, print a message
                print(f"\nTraining completed! Note: Unable to save final model - no save method available")
                # Could implement custom saving here if needed
        
        # Calculate and print total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        return self.training_history
        
    def plot_loss_history(self, save_path=None):
        """
        Plot the loss history across all windows.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.window_histories:
            print("No training history available")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot total loss
        for i, history in enumerate(self.window_histories):
            if 'loss' in history:
                offset = sum(len(h.get('loss', [])) for h in self.window_histories[:i])
                epochs = np.arange(offset, offset + len(history['loss']))
                plt.semilogy(epochs, history['loss'], label=f'Window {i+1}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss History Across Time Windows')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss history plot saved to {save_path}")
            
        plt.show()