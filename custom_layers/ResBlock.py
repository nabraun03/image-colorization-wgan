import tensorflow as tf
from tensorflow import keras
from keras import layers


class ResBlock(tf.keras.layers.Layer):
    """
    Residual Block class for implementing residual connections.

    Attributes:
        conv1 (tf.keras.layers.Conv2D): First convolutional layer in the main pathway.
        bn1 (tf.keras.layers.BatchNormalization): Batch normalization layer following the first convolution.
        relu1 (tf.keras.layers.ReLU): ReLU activation following the first batch normalization.
        conv2 (tf.keras.layers.Conv2D): Second convolutional layer in the main pathway.
        bn2 (tf.keras.layers.BatchNormalization): Batch normalization layer following the second convolution.
        identity_map (tf.keras.layers.Conv2D, optional): Convolutional layer for the residual pathway, only used if spatial dimensions or channel count change.
        relu3 (tf.keras.layers.ReLU): ReLU activation for the output, applied after adding the residual connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the ResBlock layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
        """
        super(ResBlock, self).__init__()

        # Main pathway
        self.conv1 = layers.Conv2D(
            out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = layers.BatchNormalization()

        # Residual pathway
        # Only apply convolution if the spatial dimensions or channel count change
        if stride != 1 or in_channels != out_channels:
            self.identity_map = layers.Conv2D(
                out_channels, kernel_size=1, strides=stride
            )
        else:
            self.identity_map = None

        self.relu3 = layers.ReLU()

    def call(self, inputs, training=True):
        """
        Forward pass for the ResBlock layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Whether the layer should behave in training mode or inference mode. Defaults to True.

        Returns:
            tf.Tensor: Output tensor after applying the residual block.
        """
        # Main pathway
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Residual pathway
        if self.identity_map is not None:
            residual = self.identity_map(inputs)
        else:
            residual = inputs

        # Add the residual connection
        x += residual
        return self.relu3(x)
