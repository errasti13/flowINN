import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import os
import numpy as np
from flowinn.plot.boundary_visualization import BoundaryVisualization

# Custom Fourier Feature Mapping layer
class FourierFeatureMapping(tf.keras.layers.Layer):
    def __init__(self, mapping_size: int = 64, scale: float = 10.0, **kwargs):
        """
        Args:
            mapping_size (int): The number of random Fourier features.
            scale (float): Scale factor for the random weights.
        """
        super(FourierFeatureMapping, self).__init__(**kwargs)
        self.mapping_size = mapping_size
        self.scale = scale

    def build(self, input_shape):
        # Create a fixed random projection matrix (non-trainable)
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
        # Add Fourier feature mapping layer (adjust mapping_size and scale as needed)
        model.add(FourierFeatureMapping(mapping_size=64, scale=10.0))
        # Follow with hidden Dense layers
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        # Final output layer
        model.add(tf.keras.layers.Dense(output_shape))
        return model

    def learning_rate_schedule(self, initial_learning_rate: float) -> tf.keras.optimizers.schedules.ExponentialDecay:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )

    def generate_batches(self, mesh, num_batches, time_window_size=3):
        """Generate batches for training using a sliding time window approach."""
        
        # Use unsteady branch if time data exists
        if mesh.is2D and (mesh.t is not None):  # Changed condition here
            spatial_points = len(mesh.x)  # Nx*Ny points
            time_points = len(mesh.t)     # Nt points
            total_points = spatial_points * time_points

            # Create full space-time grid
            x_grid = np.repeat(mesh.x, time_points)
            y_grid = np.repeat(mesh.y, time_points)
            t_grid = np.tile(mesh.t, spatial_points)

            # Calculate valid indices considering the time window
            max_start_idx = total_points - (time_window_size * spatial_points)
            valid_start_indices = np.arange(0, max_start_idx, spatial_points)
            
            # Shuffle the start indices
            np.random.shuffle(valid_start_indices)
            
            # Calculate batch size
            indices_per_batch = max(1, len(valid_start_indices) // num_batches)
            
            batches = []
            for i in range(num_batches):
                start = i * indices_per_batch
                end = min((i + 1) * indices_per_batch, len(valid_start_indices))
                batch_start_indices = valid_start_indices[start:end]
                
                # Collect all indices for the time windows
                all_indices = []
                for start_idx in batch_start_indices:
                    window_indices = []
                    for t in range(time_window_size):
                        window_indices.extend(range(
                            start_idx + t * spatial_points,
                            start_idx + (t + 1) * spatial_points
                        ))
                    all_indices.extend(window_indices)
                
                all_indices = np.array(all_indices)
                
                # Stack coordinates along the last axis
                x_tensor = tf.reshape(tf.convert_to_tensor(x_grid[all_indices], dtype=tf.float32), [-1, 1])
                y_tensor = tf.reshape(tf.convert_to_tensor(y_grid[all_indices], dtype=tf.float32), [-1, 1])
                t_tensor = tf.reshape(tf.convert_to_tensor(t_grid[all_indices], dtype=tf.float32), [-1, 1])
                
                # Create combined input tensor with time as third feature
                batch = tf.concat([x_tensor, y_tensor, t_tensor], axis=-1)
                batches.append(batch)
                
        else:
            total_points = len(mesh.x)
            indices = np.random.permutation(total_points)
            batch_size = total_points // num_batches
            
            batches = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_points)
                batch_indices = indices[start_idx:end_idx]

                if mesh.is2D:
                    x_tensor = tf.reshape(tf.convert_to_tensor(mesh.x[batch_indices], dtype=tf.float32), [-1, 1])
                    y_tensor = tf.reshape(tf.convert_to_tensor(mesh.y[batch_indices], dtype=tf.float32), [-1, 1])
                    batch = tf.concat([x_tensor, y_tensor], axis=-1)
                else:
                    x_tensor = tf.reshape(tf.convert_to_tensor(mesh.x[batch_indices], dtype=tf.float32), [-1, 1])
                    y_tensor = tf.reshape(tf.convert_to_tensor(mesh.y[batch_indices], dtype=tf.float32), [-1, 1])
                    z_tensor = tf.reshape(tf.convert_to_tensor(mesh.z[batch_indices], dtype=tf.float32), [-1, 1])
                    batch = tf.concat([x_tensor, y_tensor, z_tensor], axis=-1)
                batches.append(batch)

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
              output_dir: str = 'bc_plots', patience: int = 1000, min_delta: float = 1e-6) -> None:
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
        filepath: str = f'trainedModels/{model_name}.tf'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified file does not exist: {filepath}")
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model successfully loaded from {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {e}")
