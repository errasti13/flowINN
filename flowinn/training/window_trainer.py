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
            optimizer = model.optimizer
            
            if hasattr(optimizer, 'learning_rate'):
                lr_object = optimizer.learning_rate
                if hasattr(lr_object, 'assign'):
                    # Best case: Assign directly if it's assignable (like a tf.Variable or some schedules)
                    lr_object.assign(learning_rate)
                    print(f"Learning rate assigned to {learning_rate:.8f}")
                else:
                    # Fallback: Try setting value using backend if direct assign fails
                    # This might work for simple tensors or some schedule objects
                    try:
                        tf.keras.backend.set_value(lr_object, learning_rate)
                        print(f"Learning rate set via backend to {learning_rate:.8f}")
                    except Exception as e_setval:
                        print(f"Warning: Could not assign or set_value learning rate ({e_setval}). Optimizer replacement may occur (risk of tf.function errors).")
                        # Last Resort (Risky): Recreate optimizer - try to avoid this
                        # config = optimizer.get_config()
                        # config['learning_rate'] = learning_rate
                        # model.optimizer = type(optimizer).from_config(config)
                        # print(f"Optimizer recreated with learning rate {learning_rate:.8f}")
                        # Force build the new optimizer IF it was replaced, outside tf.function
                        # if not optimizer.built:
                        #    optimizer.build(model.model.trainable_variables)
            else:
                 print(f"Warning: Optimizer {type(optimizer)} has no 'learning_rate' attribute.")
                 # Fallback: Create a new optimizer if LR can't be accessed/set
                 model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                 print(f"Created new Adam optimizer with learning rate {learning_rate:.8f}")
                 # Force build the new optimizer
                 # model.optimizer.build(model.model.trainable_variables)

        except Exception as e:
            print(f"Error setting learning rate: {str(e)}")
            print(f"Attempting to create new Adam optimizer with learning rate {learning_rate:.8f}")
            # Ensure recreation on any error
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # Force build the new optimizer
            # model.optimizer.build(model.model.trainable_variables) # Build needs variables

    def _get_supported_train_params(self, model, all_params):
        """
        Determine which parameters are supported by the model's train method.
        
        Args:
            model: The model to train
            all_params: Dictionary of all available parameters
            
        Returns:
            Dictionary containing only the parameters supported by model.train()
        """
        # Default parameters that should be available in most train methods
        default_params = [
            'loss_function', 'mesh', 'epochs', 'print_interval', 
            'autosave_interval', 'num_batches', 'plot_loss'
        ]
        
        # Try to get the signature of the train method
        try:
            if hasattr(model, 'train') and callable(model.train):
                # Get the signature of the train method
                sig = inspect.signature(model.train)
                
                # Get the parameter names
                param_names = list(sig.parameters.keys())
                
                # Filter params to include only those in the signature
                # First parameter is usually 'self', so skip it
                supported_params = {}
                for name in param_names[1:]:  # Skip 'self'
                    if name in all_params:
                        supported_params[name] = all_params[name]
                
                # Ensure loss_function is included if it's a required parameter
                if 'loss_function' in param_names and 'loss_function' not in supported_params:
                    supported_params['loss_function'] = all_params['loss_function']
                
                # Fix for loss_function: if it's an object with a loss_function method, use that method
                if 'loss_function' in supported_params:
                    loss_func = supported_params['loss_function']
                    if hasattr(loss_func, 'loss_function') and callable(loss_func.loss_function):
                        supported_params['loss_function'] = loss_func.loss_function
                
                return supported_params
            else:
                # If train method doesn't exist, use default parameters
                return {k: v for k, v in all_params.items() if k in default_params}
        except (TypeError, ValueError):
            # If we can't inspect the signature, use default parameters
            return {k: v for k, v in all_params.items() if k in default_params}

    def _monitor_memory(self):
        """
        Monitor GPU memory usage and print statistics.
        """
        try:
            # Try to get TensorFlow memory info
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if not gpus:
                print("No GPUs available for memory monitoring")
                return

            # Try to get memory info for the first GPU
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                if memory_info:
                    print("\nGPU Memory Usage:")
                    print(f"  Current allocated: {memory_info['current'] / (1024**2):.1f} MB")
                    print(f"  Peak allocated: {memory_info['peak'] / (1024**2):.1f} MB")
                    print(f"  Free memory: {memory_info['free'] / (1024**2):.1f} MB")
                    return
            except:
                # TF memory info not available, try using NVML
                pass

            # Try using nvidia-smi through subprocess
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'], 
                                       stdout=subprocess.PIPE, 
                                       text=True)
                if result.returncode == 0:
                    memory_used, memory_free, memory_total = map(int, result.stdout.strip().split(','))
                    print("\nGPU Memory Usage (nvidia-smi):")
                    print(f"  Used: {memory_used} MB")
                    print(f"  Free: {memory_free} MB")
                    print(f"  Total: {memory_total} MB")
                    print(f"  Utilization: {memory_used/memory_total*100:.1f}%")
                    return
            except:
                # nvidia-smi not available or failed
                pass

            # If all methods failed, use generic message
            print("\nGPU memory monitoring not available")
        except Exception as e:
            print(f"Memory monitoring error: {str(e)}")
            
    def _generate_spatial_temporal_batches(self, mesh, num_spatial_batches, num_temporal_batches, 
                                   add_noise=True, noise_level=0.01):
        """
        Generate batches by dividing both spatial and temporal domains into smaller chunks.
        This dramatically reduces memory usage compared to standard batching.
        
        Args:
            mesh: The mesh object containing coordinates
            num_spatial_batches: Number of spatial batches to divide the domain into
            num_temporal_batches: Number of temporal batches to divide the time domain into
            add_noise: Whether to add small noise to coordinates for regularization
            noise_level: Level of noise to add
            
        Returns:
            List of batches, each containing coordinates for training
        """
        # Get coordinates
        x_flat = mesh.x.flatten()
        y_flat = mesh.y.flatten() 
        t_values = mesh.t.flatten() if hasattr(mesh, 't') else np.zeros_like(x_flat)
        
        # Calculate batch sizes
        num_spatial_points = len(x_flat)
        num_temporal_points = len(t_values)
        
        # Special case: if both spatial and temporal batches are 1, use all points
        if num_spatial_batches == 1 and num_temporal_batches == 1:
            # Create combinations of all spatial and temporal points
            print(f"Using all {num_spatial_points} spatial points and {num_temporal_points} time points")
            
            # For memory efficiency, we'll generate a reasonable number of combinations
            # rather than a full Cartesian product which could be too large
            
            # Calculate how many time points we can use with all spatial points
            # to stay under max_points

            # Use all spatial points but sample from time points
            print(f"Using all spatial points with sampled time points to fit memory constraints")
            time_samples = max(3, num_temporal_points)
            print(f"Using all {num_spatial_points} spatial points with {time_samples} time samples")
            
            # Sample time points with preference for earlier times (for causality)
            # Use weighted sampling that favors earlier time points
            weights = np.linspace(1.0, 0.5, num_temporal_points)  # Higher weight for earlier times
            selected_t_indices = np.random.choice(
                range(num_temporal_points), 
                size=time_samples, 
                replace=False, 
                p=weights/np.sum(weights)
            )
            selected_t = t_values[selected_t_indices]
            
            # Create combinations with all spatial points and selected time points
            batch_coords = []
            for t_val in selected_t:
                for i in range(num_spatial_points):
                    batch_coords.append([x_flat[i], y_flat[i], t_val])
            
            # Convert to numpy array
            batch_coords = np.array(batch_coords, dtype=np.float32)
            
            # Add small noise for regularization if requested
            if add_noise:
                # Calculate noise scale for each dimension
                x_range = np.max(x_flat) - np.min(x_flat)
                y_range = np.max(y_flat) - np.min(y_flat)
                t_range = np.max(t_values) - np.min(t_values) if len(t_values) > 1 else 1.0
                
                # Add scaled noise to each dimension
                batch_coords[:, 0] += np.random.normal(0, noise_level * x_range, batch_coords.shape[0])
                batch_coords[:, 1] += np.random.normal(0, noise_level * y_range, batch_coords.shape[0])
                # Don't add noise to temporal dimension to preserve causality
            
            batches = [tf.convert_to_tensor(batch_coords, dtype=tf.float32)]
            print(f"Generated 1 spatial-temporal batch with {len(batch_coords)} points")
            return batches
        
        # Regular case with multiple batches
        # Define spatial batch size
        spatial_points_per_batch = max(10, num_spatial_points // num_spatial_batches)
        
        # Define temporal batch size
        temporal_points_per_batch = max(3, num_temporal_points // num_temporal_batches)
        
        # Track total batches
        batches = []
        
        # Generate combined spatial-temporal batches
        for t_batch in range(num_temporal_batches):
            # Select temporal chunk
            t_start_idx = t_batch * temporal_points_per_batch
            t_end_idx = min(t_start_idx + temporal_points_per_batch, num_temporal_points)
            
            # Handle edge case for last batch
            if t_batch == num_temporal_batches - 1:
                t_end_idx = num_temporal_points
            
            # Get time values for this batch
            if num_temporal_points > 1:
                batch_t_indices = np.random.choice(
                    range(t_start_idx, t_end_idx),
                    size=min(temporal_points_per_batch, t_end_idx - t_start_idx),
                    replace=(t_end_idx - t_start_idx < temporal_points_per_batch)
                )
                batch_t_values = t_values[batch_t_indices]
            else:
                batch_t_values = t_values
            
            for s_batch in range(num_spatial_batches):
                # Generate random spatial indices for this batch
                spatial_indices = np.random.choice(
                    num_spatial_points,
                    size=spatial_points_per_batch,
                    replace=(num_spatial_points < spatial_points_per_batch)
                )
                
                # Get coordinates for these indices
                batch_x = x_flat[spatial_indices]
                batch_y = y_flat[spatial_indices]
                
                # Create all combinations of spatial and temporal coordinates
                # This uses a memory-efficient approach to avoid large meshgrids
                batch_coords = []
                
                # Only keep a random subset of time points to reduce size
                num_time_points = min(10, len(batch_t_values))
                selected_t = np.random.choice(batch_t_values, size=num_time_points, replace=False)
                
                # Add points to batch
                for t_val in selected_t:
                    for i in range(len(batch_x)):
                        batch_coords.append([batch_x[i], batch_y[i], t_val])
                
                # Convert to numpy array
                batch_coords = np.array(batch_coords, dtype=np.float32)
                
                # Add small noise for regularization if requested
                if add_noise:
                    # Calculate noise scale for each dimension
                    x_range = np.max(x_flat) - np.min(x_flat)
                    y_range = np.max(y_flat) - np.min(y_flat)
                    t_range = np.max(t_values) - np.min(t_values) if len(t_values) > 1 else 1.0
                    
                    # Add scaled noise to each dimension
                    batch_coords[:, 0] += np.random.normal(0, noise_level * x_range, batch_coords.shape[0])
                    batch_coords[:, 1] += np.random.normal(0, noise_level * y_range, batch_coords.shape[0])
                    # Don't add noise to temporal dimension to preserve causality
                
                # Add the batch to the list
                batches.append(tf.convert_to_tensor(batch_coords, dtype=tf.float32))
                
        print(f"Generated {len(batches)} spatial-temporal batches")
        return batches

    def train(self, model, loss_function, mesh, tRange, epochs=None, 
             print_interval=100, autosave_interval=10000, num_batches=10, 
             use_cpu=False, save_name=None, window_overlap=0.1, 
             physics_points_ratio=10, adaptive_sampling=True, memory_limit=None,
             monitor_memory=True, num_spatial_batches=4, num_temporal_batches=3):
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
            memory_limit: GPU memory limit in MB (None for no limit)
            monitor_memory: Whether to monitor and print GPU memory usage
            num_spatial_batches: Number of batches to divide spatial domain into
            num_temporal_batches: Number of batches to divide temporal domain into
        """            
        # Monitor initial memory usage
        if monitor_memory and not use_cpu:
            print("\nInitial GPU memory state:")
            self._monitor_memory()
            
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
            num_time_points = 100  # Can be adjusted
            
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
            
            # Adjust time window size for batching
            time_window_size = min(20, len(mesh.t) // 4)
            time_window_size = max(3, time_window_size)  # Ensure at least 3 time steps
            
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
            print(f"Time window size: {time_window_size} steps")
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
                'patience': int(epochs/10),
                'min_delta': 1e-7,
                'time_window_size': time_window_size,
                'add_noise': True,
                'noise_level': 0.005 * (0.9 ** window_idx),
                'physics_points': num_physics_points,
                'adaptive_sampling': adaptive_sampling,
                'use_cpu': use_cpu,
                'callbacks': callbacks if 'callbacks' in locals() else None
            }
            
            # Generate spatial-temporal batches if batch_data parameter is supported
            if 'batch_data' in inspect.signature(model.train).parameters:
                # Use spatial-temporal batching to reduce memory usage
                batches = self._generate_spatial_temporal_batches(
                    mesh, 
                    num_spatial_batches, 
                    num_temporal_batches,
                    add_noise=True,
                    noise_level=0.005 * (0.9 ** window_idx)  # Decreasing noise level
                )
                all_params['batch_data'] = batches
            
            # Get only the parameters supported by the model.train method
            supported_params = self._get_supported_train_params(model, all_params)
            
            # Train the model for this time window
            try:
                # Use only supported parameters
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
            
            # After training, monitor memory if requested
            if monitor_memory and not use_cpu:
                print(f"\nGPU memory after window {window_idx+1}:")
                self._monitor_memory()
        
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