import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Tuple
import os
import numpy as np


class PINN:
    """
    Physics-Informed Neural Network (PINN) class.

    This class implements a PINN model for solving partial differential equations.

    Attributes:
        model (tf.keras.Sequential): The neural network model.
        optimizer (tf.keras.optimizers.Adam): The optimizer used for training.
        eq (str): The name of the equation being solved.
    """

    def __init__(self, input_shape: int = 2, output_shape: int = 1, layers: List[int] = [20, 20, 20],
                 activation: str = 'tanh', learning_rate: float = 0.01, eq: str = 'LidDrivenCavity') -> None:
        """
        Initializes a new PINN object.

        Args:
            input_shape (int): The shape of the input data. Defaults to 2.
            output_shape (int): The shape of the output data. Defaults to 1.
            layers (List[int]): A list of integers representing the number of units in each hidden layer. Defaults to [20, 20, 20].
            activation (str): The activation function to use in the hidden layers. Defaults to 'tanh'.
            learning_rate (float): The learning rate for the optimizer. Defaults to 0.01.
            eq (str): The name of the equation being solved. Defaults to 'LidDrivenCavity'.
        """
        self.model: tf.keras.Sequential = self.create_model(input_shape, output_shape, layers, activation)
        self.model.summary()
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate_schedule(learning_rate))
        self.eq: str = eq

    def create_model(self, input_shape: int, output_shape: int, layers: List[int], activation: str) -> tf.keras.Sequential:
        """
        Creates a TensorFlow Keras Sequential model.

        Args:
            input_shape (int): The shape of the input data.
            output_shape (int): The shape of the output data.
            layers (List[int]): A list of integers representing the number of units in each hidden layer.
            activation (str): The activation function to use in the hidden layers.

        Returns:
            tf.keras.Sequential: A TensorFlow Keras Sequential model.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dense(output_shape))  # Output layer
        return model

    def learning_rate_schedule(self, initial_learning_rate: float) -> tf.keras.optimizers.schedules.ExponentialDecay:
        """
        Creates an exponential decay learning rate schedule.

        Args:
            initial_learning_rate (float): The initial learning rate.

        Returns:
            tf.keras.optimizers.schedules.ExponentialDecay: An exponential decay learning rate schedule.
        """
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )

    @tf.function(jit_compile=True)
    def train_step(self, loss_function: Callable[[], tf.Tensor]) -> tf.Tensor:
        """
        Performs a single training step.

        Args:
            loss_function (Callable[[], tf.Tensor]): A callable that computes the loss.

        Returns:
            tf.Tensor: The computed loss.
        """
        with tf.GradientTape() as tape:
            loss: tf.Tensor = loss_function()  # Call the loss function here to compute the loss
        gradients: List[tf.Tensor] = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, loss_function: Callable[[], tf.Tensor], epochs: int = 1000, print_interval: int = 100,
              autosave_interval: int = 100, plot_loss: bool = False) -> None:
        """
        Trains the PINN model.

        Args:
            loss_function (Callable[[], tf.Tensor]): A callable that computes the loss.
            epochs (int): The number of epochs to train for. Defaults to 1000.
            print_interval (int): The interval at which to print the loss. Defaults to 100.
            autosave_interval (int): The interval at which to save the model. Defaults to 100.
            plot_loss (bool): Whether to plot the loss during training. Defaults to False.

        Raises:
            OSError: If there is an error saving the model.
        """
        loss_history: List[float] = []
        epoch_history: List[int] = []

        if plot_loss:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_scientific(False)
            line, = ax.semilogy([], [], label='Training Loss')
            plt.legend()

        for epoch in range(epochs):
            loss: tf.Tensor = self.train_step(loss_function)

            if (epoch + 1) % print_interval == 0:
                loss_history.append(loss.numpy())
                epoch_history.append(epoch + 1)

                if plot_loss:
                    line.set_xdata(epoch_history)
                    line.set_ydata(loss_history)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

                print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")

            if (epoch + 1) % autosave_interval == 0:
                os.makedirs('trainedModels', exist_ok=True)
                try:
                    self.model.save(f'trainedModels/{self.eq}.keras')
                except OSError as e:
                    print(f"Error saving model: {e}")
                    print("Check disk space and permissions.")
                    raise  # Re-raise the exception so the program doesn't continue with a potentially corrupted model.

        if plot_loss:
            plt.ioff()  # Turn off interactive mode
            plt.close()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return self.model.predict(X)

    def load(self, model_name: str) -> None:
        """
        Loads a trained model from the specified file.

        Args:
            model_name (str): The name of the model to load.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            RuntimeError: If there is an error loading the model.
        """
        filepath: str = f'trainedModels/{model_name}.tf'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified file does not exist: {filepath}")

        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model successfully loaded from {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {e}")
