import tensorflow as tf
from tensorflow import keras
from keras import layers


class AddGaussianNoise(layers.Layer):
    """
    Custom layer to add Gaussian noise to its inputs.

    Attributes:
        stddev (float): The standard deviation of the Gaussian noise to be added.
    """

    def __init__(self, stddev, **kwargs):
        """
        Initialize the AddGaussianNoise layer.

        Args:
            stddev (float): The standard deviation of the Gaussian noise.
            **kwargs: Additional keyword arguments inherited from the parent class.
        """
        super(AddGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        """
        Forward pass for the AddGaussianNoise layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Whether the layer should behave in training mode or inference mode. Defaults to None.

        Returns:
            tf.Tensor: Output tensor after adding Gaussian noise during training.
        """
        if training:
            # Generate Gaussian noise with the same shape as the input
            noise = tf.random.normal(
                shape=tf.shape(inputs), mean=0.0, stddev=self.stddev
            )
            # Add the noise to the input
            return inputs + noise
        # If not in training mode, return the original input
        return inputs
