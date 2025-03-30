import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os
import numpy as np
from flowinn.plot.boundary_visualization import BoundaryVisualization

# Custom Fourier Feature Mapping layer
class FourierFeatureMapping(tf.keras.layers.Layer):
    def __init__(self, mapping_size: int = 64, scale: float = 10.0, temporal_scale: float = None, **kwargs):
        """
        Args:
            mapping_size (int): The number of random Fourier features.
            scale (float): Scale factor for the random weights.
            temporal_scale (float): Optional separate scale for temporal dimension.
                                   If None, uses the same scale for all dimensions.
        """
        super(FourierFeatureMapping, self).__init__(**kwargs)
        self.mapping_size = mapping_size
        self.scale = scale
        self.temporal_scale = temporal_scale

    def build(self, input_shape):
        # Create a fixed random projection matrix (non-trainable)
        if self.temporal_scale is not None and input_shape[-1] >= 3:
            # For space-time problems, use different scales for spatial and temporal dimensions
            # Create a diagonal matrix with different scales
            scales = tf.ones(input_shape[-1], dtype=tf.float32)
            # Set temporal dimension (assuming it's the last one) to have different scale
            scales = tf.tensor_scatter_nd_update(scales, [[input_shape[-1]-1]], [self.temporal_scale/self.scale])
            
            # Create random weights with proper scaling applied per dimension
            B_init = tf.random_normal_initializer(stddev=1.0)(shape=(input_shape[-1], self.mapping_size))
            # Apply the scaling factors to each dimension
            B_init = tf.einsum('i,ij->ij', scales, B_init) * self.scale
            
            self.B = self.add_weight(
                name='B',
                shape=(input_shape[-1], self.mapping_size),
                initializer=lambda *args, **kwargs: B_init,
                trainable=False
            )
        else:
            # Use the original implementation for non-temporal problems
            self.B = self.add_weight(
                name='B',
                shape=(input_shape[-1], self.mapping_size),
                initializer=tf.random_normal_initializer(stddev=self.scale),
                trainable=False
            )
        super(FourierFeatureMapping, self).build(input_shape)

    def call(self, inputs):
        x_proj = tf.matmul(inputs, self.B)
        # Concatenate sine and cosine features
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)

class PINN:
    """
    Physics-Informed Neural Network (PINN) class with a Modified Fourier Network (MFN) architecture.
    """

    def __init__(self, input_shape: int = 2, output_shape: int = 1, layers: List[int] = [20, 20, 20],
                 activation: str = 'gelu', learning_rate: float = 0.01, eq: str = None) -> None:
        # Build the model with a Fourier feature mapping layer at the start.
        self.model: tf.keras.Sequential = self.create_model(input_shape, output_shape, layers, activation)
        self.model.summary()
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_schedule(learning_rate))
        self.eq: str = eq
        self.boundary_visualizer: Optional[BoundaryVisualization] = None

    def create_model(self, input_shape: int, output_shape: int, layers: List[int], activation: str) -> tf.keras.Sequential:
        """
        Creates an MFN model that begins with a Fourier feature mapping layer.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=(input_shape)))
        
        # Detect if this is likely an unsteady problem based on input shape
        is_unsteady = input_shape >= 3
        
        # Add Fourier feature mapping layer with temporal-specific settings
        if is_unsteady:
            # For unsteady problems, use a different scale for temporal dimension
            # Higher scale (5.0) for time dimension to better capture high-frequency oscillations
            model.add(FourierFeatureMapping(
                mapping_size=64,  # More features for complex temporal dynamics
                scale=5.0,       # Lower overall scale for spatial dimensions
                temporal_scale=25.0  # Higher scale for temporal dimension
            ))
        else:
            # For steady problems, use standard Fourier features
            model.add(FourierFeatureMapping(mapping_size=32, scale=10.0))
        
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
            
        # Final output layer
        model.add(tf.keras.layers.Dense(output_shape))
        return model

    def learning_rate_schedule(self, initial_learning_rate: float) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """Create an advanced learning rate schedule suitable for PINNs."""
        
        # For unsteady problems, use a warm-up followed by cosine decay
        if hasattr(self, 'model') and self.model is not None:
            input_shape = self.model.input_shape[-1]
            if input_shape >= 3:  # Likely an unsteady problem if input_dim >= 3
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
        """Generate batches for training using a sliding time window approach."""
        batches = []

        # Flatten arrays for proper indexing
        x_flat = mesh.x.ravel()
        y_flat = mesh.y.ravel()
        z_flat = None if mesh.is2D else mesh.z.ravel()

        if mesh.is_unsteady:
            spatial_points = len(x_flat)
            time_points = len(mesh.t)
            total_points = spatial_points * time_points
            points_per_batch = total_points // num_batches

            points_per_dim = int(np.cbrt(points_per_batch))

            for _ in range(num_batches):
                # Sample indices in a vectorized way
                x_indices = np.random.choice(spatial_points, size=points_per_dim, replace=True)
                y_indices = np.random.choice(spatial_points, size=points_per_dim, replace=True)
                t_indices = np.random.choice(time_points, size=points_per_dim, replace=True)
                
                # Create meshgrid of all combinations
                xx, yy, tt = np.meshgrid(x_indices, y_indices, t_indices)
                
                # Flatten and stack to get all combinations
                all_indices = np.stack([xx.flatten(), yy.flatten(), tt.flatten()], axis=1)
                
                # Map indices to actual coordinate values
                x_coords = x_flat[all_indices[:, 0]].astype(np.float32)
                y_coords = y_flat[all_indices[:, 1]].astype(np.float32)
                t_coords = mesh.t[all_indices[:, 2]].astype(np.float32)
                
                # Stack to create final coordinates array
                batch_coords = np.stack([x_coords, y_coords, t_coords], axis=1)
                
                # Limit to points_per_batch
                batch_coords = batch_coords[:points_per_batch]
                
                # Convert to tensor and add to batches
                batches.append(tf.convert_to_tensor(batch_coords))

        else:
            total_points = len(x_flat)
            points_per_batch = total_points // num_batches
            
            points_per_dim = int(np.sqrt(points_per_batch) if mesh.is2D else np.cbrt(points_per_batch))

            for _ in range(num_batches):
                if mesh.is2D:
                    # Sample indices for 2D case
                    x_indices = np.random.choice(total_points, size=points_per_dim, replace=True)
                    y_indices = np.random.choice(total_points, size=points_per_dim, replace=True)
                    
                    # Create meshgrid of all combinations
                    xx, yy = np.meshgrid(x_indices, y_indices)
                    
                    # Flatten and stack
                    all_indices = np.stack([xx.flatten(), yy.flatten()], axis=1)
                    
                    # Map indices to actual coordinate values
                    x_coords = x_flat[all_indices[:, 0]].astype(np.float32)
                    y_coords = y_flat[all_indices[:, 1]].astype(np.float32)
                    
                    # Stack to create final coordinates array
                    batch_coords = np.stack([x_coords, y_coords], axis=1)
                else:
                    # Sample indices for 3D case
                    x_indices = np.random.choice(total_points, size=points_per_dim, replace=True)
                    y_indices = np.random.choice(total_points, size=points_per_dim, replace=True)
                    z_indices = np.random.choice(total_points, size=points_per_dim, replace=True)
                    
                    # Create meshgrid of all combinations
                    xx, yy, zz = np.meshgrid(x_indices, y_indices, z_indices)
                    
                    # Flatten and stack
                    all_indices = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
                    
                    # Map indices to actual coordinate values
                    x_coords = x_flat[all_indices[:, 0]].astype(np.float32)
                    y_coords = y_flat[all_indices[:, 1]].astype(np.float32)
                    z_coords = z_flat[all_indices[:, 2]].astype(np.float32)
                    
                    # Stack to create final coordinates array
                    batch_coords = np.stack([x_coords, y_coords, z_coords], axis=1)
                
                # Limit to points_per_batch if needed
                batch_coords = batch_coords[:points_per_batch]

                batch_coords = np.array(batch_coords, dtype=np.float32)
                batch_coords = batch_coords[:points_per_batch]
                batches.append(tf.convert_to_tensor(batch_coords))

        return batches

    @tf.function(jit_compile=True)
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
              output_dir: str = 'bc_plots', patience: int = 10000, min_delta: float = 1e-6) -> None:
        loss_history = []
        epoch_history = []
        last_loss = float('inf')
        patience_counter = 0

        if bc_plot_interval is not None:
            self.boundary_visualizer = BoundaryVisualization(output_dir=output_dir)

        if plot_loss:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_scientific(False)
            line, = ax.semilogy([], [], label='Training Loss')
            plt.legend()

        for epoch in range(epochs):
            batches = self.generate_batches(mesh, num_batches)
            epoch_loss = 0.0
            
            for batch_data in batches:
                batch_loss = self.train_step(loss_function, batch_data)
                epoch_loss += batch_loss

            epoch_loss = epoch_loss / num_batches

            # Early stopping check
            if epoch_loss < last_loss - min_delta:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            last_loss = epoch_loss

            if (epoch + 1) % print_interval == 0:
                loss_history.append(epoch_loss.numpy())
                epoch_history.append(epoch + 1)

                if plot_loss:
                    line.set_xdata(epoch_history)
                    line.set_ydata(loss_history)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

                print(f"Epoch {epoch + 1}: Loss = {epoch_loss.numpy()}")
            
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
                    raise

        if self.boundary_visualizer is not None:
            self.boundary_visualizer.plot_error_evolution()

        if plot_loss:
            plt.ioff()
            plt.close()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def load(self, model_name: str) -> None:
        filepath: str = f'trainedModels/{model_name}.keras'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified file does not exist: {filepath}")
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model successfully loaded from {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {e}")
