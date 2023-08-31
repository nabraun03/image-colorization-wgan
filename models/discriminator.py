import tensorflow as tf
from tensorflow import keras
from keras import layers


class Discriminator(keras.Model):
    """
    Discriminator class for the generative adversarial network. Creates a PatchGAN Discriminator.

    Attributes:
        config (dict): Configuration dictionary containing hyperparameters and settings.
        image_size (int): The size of the input images.
    """

    def __init__(self, config):
        """
        Initialize the Discriminator class.

        Args:
            config (dict): Configuration dictionary containing hyperparameters and settings.
        """
        super(Discriminator, self).__init__()
        self.config = config
        self.image_size = config["IMAGE_SIZE"]

    def build(self, input_shapes):
        """
        Build the discriminator model.
        """
        # Initialize weights with a random normal distribution
        initializer = tf.random_normal_initializer(0.0, 0.02)

        # Define the input layers for the grayscale image and the target image
        input = layers.Input(shape=[self.image_size, self.image_size, 1])
        target = layers.Input(shape=[self.image_size, self.image_size, 2])

        # Concatenate the input and target images along the channel dimension
        x = layers.concatenate([input, target])

        # Downsampling layers to produce abstraction of features from image
        down1 = self.downsample(64, 4)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        # Wide convolutional layer
        x = layers.Conv2D(
            filters=256,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="same",
            kernel_initializer=initializer,
        )(down3)

        # Padding layer to adjust dimensions
        zero_pad1 = layers.ZeroPadding2D()(x)

        # Wider convolutional layer
        conv = layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer, use_bias=False
        )(zero_pad1)

        # Layer normalization and activation
        norm_layer = tf.keras.layers.LayerNormalization()(conv)
        leaky_relu = layers.LeakyReLU()(norm_layer)

        # Final padding and convolutional layer to produce the output
        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
        last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        # Create the model
        self.model = keras.Model(inputs=[input, target], outputs=last)

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)

    def downsample(self, num_filters, size):
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

        # Apply layer normalization and LeakyReLU activation
        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU())

        return model
