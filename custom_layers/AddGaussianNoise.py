
import tensorflow as tf
from tensorflow import keras
from keras import layers

class AddGaussianNoise(layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(AddGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training = None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev)
            return inputs + noise
        return inputs
