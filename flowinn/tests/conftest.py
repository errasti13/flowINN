import pytest
import os
import tensorflow as tf

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment before each test"""
    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Configure TensorFlow to use CPU only for tests
    tf.config.set_visible_devices([], 'GPU')
