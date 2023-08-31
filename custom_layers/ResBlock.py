import tensorflow as tf
from tensorflow import keras
from keras import layers

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # Main pathway
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride,
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # Residual pathway
        # Only apply convolution if the spatial dimensions or channel count change
        if stride != 1 or in_channels != out_channels:
            self.identity_map = layers.Conv2D(out_channels, kernel_size=1, strides=stride)
        else:
            self.identity_map = None

        self.relu3 = layers.ReLU()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.identity_map is not None:
            residual = self.identity_map(inputs)
        else:
            residual = inputs

        x += residual
        return self.relu3(x)
