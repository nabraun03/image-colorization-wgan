import tensorflow as tf
from tensorflow import keras
from keras import layers
from custom_layers.ResBlock import ResBlock


class Generator(keras.Model):
    """
    Generator class for the WGAN model. Creates a U-Net with residual upsampling layers.

    Attributes:
        config (dict): Configuration dictionary containing hyperparameters and settings.
        image_size (int): The size of the input image.
        model (keras.Model): The underlying Keras model.
    """

    def __init__(self, config):
        """
        Initialize the Generator class.

        Args:
            config (dict): Configuration dictionary containing hyperparameters and settings.
        """
        super(Generator, self).__init__()
        self.config = config
        self.image_size = config["IMAGE_SIZE"]

    def build(self, input_shapes):
        """
        Build the generator model.

        Args:
            input_shapes (tuple): Shape of the input tensor.
        """
        # Define the input layer with the shape of the input image
        inputs = layers.Input(shape=[self.image_size, self.image_size, 1])

        # Define the downsampling layers
        down_stack = [
            self.downsample(
                256, 4, False
            ),  # No batch normalization for the first layer
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(1024, 4),
        ]

        # Define the upsampling layers making use of residual blocks
        up_stack = [
            self.upsample_res(512, 4),
            self.upsample_res(512, 4, True),  # Apply dropout
            self.upsample_res(256, 4, True),  # Apply dropout
        ]

        # Final convolution layer to produce the output image with 2 channels
        last_conv = layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            activation="tanh",
        )

        # Forward pass through the network
        x = inputs
        # Add downsampling layers to list to be used in skip connections and downsample the input
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        # Reverse the list, excluding the last layer
        skips = reversed(skips[:-1])

        # Upsample the feature map produced by downsampling and create skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)  # Up-sample
            x = layers.Concatenate()([x, skip])

        # Final upsample and convolutional layer to produce output
        x = layers.UpSampling2D(size=2)(x)
        x = last_conv(x)

        self.model = keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs, training=True):
        """
        Forward pass for the generator.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: Output tensor. Represents generator's prediction of AB data channels given the grayscale image.
        """
        return self.model(inputs, training=training)

    def downsample(self, num_filters, size, apply_batchnorm=True):
        """
        Create a down-sampling layer.
        """
        # Initialize weights with a random normal distribution
        initializer = tf.random_normal_initializer(0, 0.02)

        # Initialize model
        model = keras.Sequential()

        # Convolutional layer to reduce dimensions and increase channels
        model.add(
            layers.Conv2D(
                num_filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            )
        )

        # Apply batch normalization and LeakyReLU activation if specified
        if apply_batchnorm:
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        return model

    def upsample_res(self, num_filters, size, apply_dropout=False):
        """
        Create an up-sampling layer with a residual block.

        Args:
            num_filters (int): Number of filters in the convolutional layer.
            size (int): Size of the convolutional kernel.
            apply_dropout (bool): Whether to apply dropout.

        Returns:
            keras.Sequential: A sequential model representing the up-sampling layer using a residual block.
        """
        # Initialize model
        model = keras.Sequential()

        # Transposed convolutional layer to increase dimensions and reduce channels
        model.add(
            layers.Conv2DTranspose(
                num_filters, size, strides=2, padding="same", use_bias=False
            )
        )

        # Use residual block while upsampling to ensure each layer is producing features
        model.add(ResBlock(num_filters, num_filters, stride=1))

        # Apply dropout if specified
        if apply_dropout:
            model.add(layers.Dropout(0.2))

        return model
