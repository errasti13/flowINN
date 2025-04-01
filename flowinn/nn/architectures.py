"""
Neural network architectures for Physics-Informed Neural Networks (PINNs).

This module implements various neural network architectures that can be used
as the backbone for PINNs, including:
- Modified Fourier Networks (MFN): A network with Fourier feature mapping to address spectral bias
- Adaptive activation functions: Allow the network to learn activation function shapes
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
from flowinn.models.model import PINN


class FourierFeatureLayer(layers.Layer):
    """
    Fourier Feature mapping layer to overcome spectral bias in MLPs.
    
    This layer projects the input into a higher-dimensional space using
    trainable or fixed frequency matrices and sine/cosine activations.
    
    References:
        - Tancik et al. (2020) - "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
        - NVIDIA Modulus library
    """
    def __init__(self, fourier_dim=32, scale=10.0, temporal_scale=None, trainable=False, **kwargs):
        """
        Args:
            fourier_dim (int): The number of random Fourier features.
            scale (float): Scale factor for the random weights.
            temporal_scale (float): Optional separate scale for temporal dimension.
                                  If None, uses the same scale for all dimensions.
            trainable (bool): Whether the Fourier features should be trainable.
        """
        super().__init__(**kwargs)
        self.fourier_dim = fourier_dim
        self.scale = scale
        self.temporal_scale = temporal_scale
        self.trainable = trainable
        
    def build(self, input_shape):
        # Create a fixed or trainable random projection matrix
        if self.temporal_scale is not None and input_shape[-1] >= 3:
            # For space-time problems, use different scales for spatial and temporal dimensions
            scales = tf.ones(input_shape[-1], dtype=tf.float32)
            # Set temporal dimension (assuming it's the last one) to have different scale
            scales = tf.tensor_scatter_nd_update(scales, [[input_shape[-1]-1]], [self.temporal_scale/self.scale])
            
            # Create random weights with proper scaling applied per dimension
            B_init = tf.random_normal_initializer(stddev=1.0)(shape=(input_shape[-1], self.fourier_dim))
            # Apply the scaling factors to each dimension
            B_init = tf.einsum('i,ij->ij', scales, B_init) * self.scale
            
            self.B = self.add_weight(
                name='fourier_frequencies',
                shape=(input_shape[-1], self.fourier_dim),
                initializer=lambda *args, **kwargs: B_init,
                trainable=self.trainable
            )
        else:
            # Use standard initialization for non-temporal problems
            self.B = self.add_weight(
                name='fourier_frequencies',
                shape=(input_shape[-1], self.fourier_dim),
                initializer=tf.random_normal_initializer(stddev=self.scale),
                trainable=self.trainable
            )
        
    def call(self, inputs):
        # Project inputs to frequency space
        x_proj = tf.matmul(inputs, self.B)
        
        # Apply sin and cos to get periodic features
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "fourier_dim": self.fourier_dim,
            "scale": self.scale,
            "temporal_scale": self.temporal_scale,
            "trainable": self.trainable
        })
        return config


class AdaptiveActivation(layers.Layer):
    """
    Adaptive activation function layer.
    
    This layer implements an activation function with trainable parameters
    that can adapt during training. It combines a base activation function
    with a learnable linear component.
    
    References:
        - Jagtap et al. (2020) - "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks"
    """
    def __init__(self, base_activation='gelu', initial_param=0.1, **kwargs):
        super().__init__(**kwargs)
        self.base_activation = base_activation
        self.initial_param = initial_param
        
    def build(self, input_shape):
        # Trainable parameter that controls activation shape
        self.a = self.add_weight(
            'activation_param',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_param),
            trainable=True
        )
        
    def call(self, inputs):
        # Base activation function
        if self.base_activation == 'tanh':
            base = tf.tanh(inputs)
        elif self.base_activation == 'gelu':
            base = tf.nn.gelu(inputs)
        elif self.base_activation == 'swish':
            base = inputs * tf.sigmoid(inputs)
        else:
            base = tf.tanh(inputs)  # Default to tanh
            
        # Adaptive version: mix linear and non-linear components
        return self.a * inputs + (1 - self.a) * base
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "base_activation": self.base_activation,
            "initial_param": self.initial_param
        })
        return config


def create_mfn_model(input_shape, output_shape, eq, activation='gelu', fourier_dim=32, 
                    layer_sizes=None, learning_rate=0.001, use_adaptive_activation=False):
    """
    Create a Modified Fourier Network (MFN) model.
    
    This architecture helps overcome spectral bias in standard MLPs by using Fourier
    feature mapping to better capture high-frequency components of the solution.
    
    Args:
        input_shape: Shape of the input tensor (e.g., (3,) for x,y,t)
        output_shape: Number of output variables
        eq: Equation name or identifier
        activation: Activation function to use ('gelu', 'tanh', 'swish')
        fourier_dim: Dimension of the Fourier feature mapping
        layer_sizes: List of hidden layer sizes, defaults to [128, 128, 128, 128]
        learning_rate: Initial learning rate for Adam optimizer
        use_adaptive_activation: Whether to use adaptive activation functions
        
    Returns:
        A PINN model with MFN architecture
    """
    if layer_sizes is None:
        layer_sizes = [128, 128, 128, 128]
    
    # Create a standard PINN model
    pinn_model = PINN(
        input_shape=input_shape,
        output_shape=output_shape,
        eq=eq,
        layers=layer_sizes,
        activation=activation,
        learning_rate=learning_rate
    )
    
    # Replace the default model with our MFN architecture
    # Get the Keras backend model
    keras_model = pinn_model.model
    
    # Create new MFN model - more efficient design
    input_layer = layers.Input(shape=input_shape)
    
    # Direct Fourier feature mapping (more efficient)
    x = FourierFeatureLayer(fourier_dim)(input_layer)
    
    # Skip the additional dense layers after Fourier mapping
    # This significantly reduces computational cost
    
    # Core network
    for units in layer_sizes:
        x = layers.Dense(
            units, 
            kernel_initializer='glorot_normal',
            use_bias=True
        )(x)
        
        # Use standard activations for better performance
        if use_adaptive_activation:
            x = AdaptiveActivation(base_activation=activation)(x)
        else:
            x = layers.Activation(activation)(x)
    
    # Output layer
    output_layer = layers.Dense(output_shape)(x)
    
    # Create model
    mfn_model = Model(inputs=input_layer, outputs=output_layer)
    
    # Use float32 precision for better performance
    mfn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    # Replace the default model in PINN with our MFN
    pinn_model.model = mfn_model
    
    return pinn_model 