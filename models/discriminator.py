import tensorflow as tf
from tensorflow import keras
from keras import layers

class Discriminator(keras.Model):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.image_size = config['IMAGE_SIZE']

    def build(self, input_shapes):
        initializer = tf.random_normal_initializer(0., 0.02)

        input = layers.Input(shape=[self.image_size, self.image_size, 1])
        target = layers.Input(shape=[self.image_size, self.image_size, 2])

        
        x = layers.concatenate([input, target])

        down1 = self.downsample(64, 4)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)

        x = layers.Conv2D(256, (4, 4,), strides=(1, 1), padding='same', kernel_initializer=initializer)(down3)

        zero_pad1 = layers.ZeroPadding2D()(x)
        conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

        norm_layer = tf.keras.layers.LayerNormalization()(conv)
        leaky_relu = layers.LeakyReLU()(norm_layer)
        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
        last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        self.model = keras.Model(inputs=[input, target], outputs=last)
    
    def call(self, inputs, training = True):
        return self.model(inputs, training = training)
    
    def downsample(self, num_filters, size):
        initializer = tf.random_normal_initializer(0, 0.02)

        model = keras.Sequential()
        model.add(layers.Conv2D(num_filters, size, strides = 2, padding = 'same', kernel_initializer = initializer, use_bias = False))

        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU())

        return model
        